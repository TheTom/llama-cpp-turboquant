import copy
import json
import tempfile
import unittest
import hashlib
import sys
import os
from pathlib import Path
from unittest.mock import patch

import torch
import numpy as np

from conversion import get_model_class
from conversion.base import ModelBase, gguf
from conversion.fish_omni import FishOmniBackboneModel
from conversion.qwen import Qwen3_5MoeTextModel
from external_audio import DenseAttention, move_decode_modules
from prepare_backbone import stage
import packed_to_gguf
from safetensors.torch import save_file


CONFIG = {
    "architectures": ["FishOmniModel"], "model_type": "fish_omni",
    "quantization_config": {"quant_method": "bsgemm-moe", "weight_block_size": [128, 128], "activation_scheme": "dynamic"},
    "text_config": {
        "num_hidden_layers": 4, "hidden_size": 32, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
        "num_experts": 2, "num_experts_per_tok": 1, "moe_intermediate_size": 8, "shared_expert_intermediate_size": 8,
        "max_position_embeddings": 256, "rms_norm_eps": 1e-6,
        "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 8, "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4, "vocab_size": 128, "rotary_interleaved": False, "use_qk_norm": True,
        "attn_output_gate": True, "layernorm_zero_centered_gamma": True, "shared_expert_gate": True,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"], "mlp_layer_types": ["moe"] * 4,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25},
    },
}


def source_receipt(source, receipt):
    files = list(source.iterdir())
    receipt.write_text(json.dumps({"source": str(source), "revision": "a" * 40, "file_count": len(files),
                                  "total_bytes": sum(path.stat().st_size for path in files),
                                  "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}))


class ConversionTests(unittest.TestCase):
    def model(self, config=None, cls=FishOmniBackboneModel):
        config = copy.deepcopy(CONFIG if config is None else config)
        config["architectures"] = ["FishOmniBackboneForCausalLM" if cls is FishOmniBackboneModel else "Qwen3_5MoeForCausalLM"]
        with patch.object(ModelBase, "index_tensors", return_value={}), patch.object(cls, "no_mtp", True):
            return cls(Path("."), gguf.LlamaFileType.MOSTLY_BF16, Path("unused.gguf"), hparams=config)

    def test_explicit_architecture_only(self):
        self.assertIs(get_model_class("FishOmniBackboneForCausalLM"), FishOmniBackboneModel)
        with self.assertRaises(NotImplementedError):
            get_model_class("FishOmniModel")

    def test_rope_and_linear_metadata(self):
        model = self.model()
        model.set_gguf_parameters()
        fields = model.gguf_writer.kv_data[0]
        self.assertEqual(fields["qwen35moe.rope.dimension_sections"].value, [0, 0, 0, 0])
        self.assertEqual(fields["qwen35moe.rope.dimension_count"].value, 4)
        self.assertEqual(fields["qwen35moe.full_attention_interval"].value, 4)
        qwen = self.model(cls=Qwen3_5MoeTextModel)
        qwen.set_gguf_parameters()
        self.assertEqual(qwen.gguf_writer.kv_data[0]["qwen35moe.rope.dimension_sections"].value, [11, 11, 10, 0])

    def test_zero_centered_norm_is_applied_once(self):
        model = self.model()
        weight = torch.tensor([0.0, 0.5, -0.5])
        [(name, value)] = list(model.modify_tensors(weight, "model.layers.0.input_layernorm.weight", 0))
        self.assertEqual(name, "blk.0.attn_norm.weight")
        torch.testing.assert_close(value, torch.tensor([1.0, 1.5, 0.5]))
        [(name, value)] = list(model.modify_tensors(weight, "model.layers.0.linear_attn.norm.weight", 0))
        torch.testing.assert_close(value, weight)

    def test_namespace_preserves_scale_pair(self):
        generator = lambda: torch.ones(1)
        for suffix in ("weight", "weight_scale_inv"):
            name = f"model.language_model.layers.0.mlp.experts.0.down_proj.{suffix}"
            renamed, value = FishOmniBackboneModel.filter_tensors((name, generator))
            self.assertEqual(renamed, f"model.layers.0.mlp.experts.0.down_proj.{suffix}")
            self.assertIs(value, generator)
        self.assertIsNone(FishOmniBackboneModel.filter_tensors(("audio_decoder.codebook_embeddings.weight", generator)))
        with self.assertRaises(ValueError):
            FishOmniBackboneModel.filter_tensors(("unknown.weight", generator))

    def test_rejects_incompatible_config(self):
        config = copy.deepcopy(CONFIG)
        config["text_config"]["rotary_interleaved"] = True
        with self.assertRaises(ValueError):
            self.model(config)
        config = copy.deepcopy(CONFIG)
        config["text_config"]["layer_types"][0] = "full_attention"
        with self.assertRaises(ValueError):
            self.model(config)

    def test_fp8_block_scales(self):
        model = self.model()
        model.hparams["quantization_config"]["quant_method"] = "fp8"
        name = "model.layers.0.mlp.experts.0.gate_proj.weight"
        weight = torch.ones((130, 129), dtype=torch.float32).to(torch.float8_e4m3fn)
        scale = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        model.model_tensors = {name: lambda: weight, name + "_scale_inv": lambda: scale}
        model.dequant_model()
        self.assertNotIn(name + "_scale_inv", model.model_tensors)
        actual = model.model_tensors[name]()
        self.assertEqual(float(actual[0, 0]), 1.0)
        self.assertEqual(float(actual[0, 128]), 2.0)
        self.assertEqual(float(actual[129, 0]), 3.0)
        self.assertEqual(float(actual[129, 128]), 4.0)

    def test_staging_links_without_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            source, destination = Path(temp) / "source", Path(temp) / "staged"
            source.mkdir()
            (source / "config.json").write_text(json.dumps(CONFIG))
            for filename in ("model-1.safetensors", "tokenizer.json", "tokenizer_config.json"):
                (source / filename).write_text("fixture")
            (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"lm_head.weight": "model-1.safetensors"}}))
            receipt = Path(temp) / "verified.json"
            source_receipt(source, receipt)
            stage(source, destination, receipt)
            self.assertTrue((destination / "model-1.safetensors").is_symlink())
            config = json.loads((destination / "config.json").read_text())
            self.assertEqual(config["quantization_config"]["quant_method"], "fp8")
            self.assertEqual(json.loads((source / "config.json").read_text()), CONFIG)
            with self.assertRaises(FileExistsError):
                stage(source, destination, receipt)

    def test_tower_cache_causal_and_replacement(self):
        attention = DenseAttention(2, 1, 2)
        attention.bind(torch.tensor([0, 1]))
        queries = torch.zeros((2, 4))
        keys = torch.zeros((2, 2))
        result = attention(0, queries, keys, torch.tensor([[2.0, 4.0], [10.0, 12.0]]))
        torch.testing.assert_close(result[0], torch.tensor([2.0, 4.0, 2.0, 4.0]))
        torch.testing.assert_close(result[1], torch.tensor([6.0, 8.0, 6.0, 8.0]))
        attention.bind(torch.tensor([1]))
        result = attention(0, queries[:1], keys[:1], torch.tensor([[6.0, 8.0]]))
        torch.testing.assert_close(result[0], torch.tensor([4.0, 6.0, 4.0, 6.0]))
        self.assertEqual(len(attention.cache[0]), 2)

    def test_codec_keeps_encode_only_weights_off_device(self):
        codec = torch.nn.Module()
        codec.encoder = torch.nn.Linear(2, 2)
        codec.decoder = torch.nn.Linear(2, 2)
        codec.quantizer = torch.nn.Module()
        for name in ("semantic_quantizer", "quantizer", "post_module", "upsample", "pre_module", "downsample"):
            setattr(codec.quantizer, name, torch.nn.Linear(2, 2))
        move_decode_modules(codec, "meta")
        self.assertEqual(codec.encoder.weight.device.type, "cpu")
        self.assertEqual(codec.quantizer.pre_module.weight.device.type, "cpu")
        self.assertEqual(codec.quantizer.downsample.weight.device.type, "cpu")
        self.assertEqual(codec.decoder.weight.device.type, "meta")
        for name in ("semantic_quantizer", "quantizer", "post_module", "upsample"):
            self.assertEqual(getattr(codec.quantizer, name).weight.device.type, "meta")
            self.assertEqual(getattr(codec.quantizer, name).weight.dtype, torch.float32)

    def test_packed_expert_gguf_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            for dtype, width, block in ((gguf.GGMLQuantizationType.TQ4_1S, 20, 32), (gguf.GGMLQuantizationType.Q4_K, 144, 256)):
                path = Path(temp) / f"{dtype.name}.gguf"
                packed = np.arange(2 * 4 * width, dtype=np.uint8).reshape(2, 4, width)
                writer = gguf.GGUFWriter(path, "qwen35moe", use_temp_file=True)
                writer.add_tensor("blk.2.ffn_gate_exps.weight", packed, raw_dtype=dtype)
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file()
                writer.close()
                [tensor] = gguf.GGUFReader(path).tensors
                self.assertEqual(tensor.tensor_type, dtype)
                self.assertEqual(list(tensor.shape), [block, 4, 2])
                self.assertEqual(hashlib.sha256(tensor.data).hexdigest(), hashlib.sha256(packed).hexdigest())

    def test_complete_toy_candidate_import(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, candidate = root / "source", root / "C0"
            source.mkdir()
            candidate.mkdir()
            config = copy.deepcopy(CONFIG)
            config["text_config"].update(num_hidden_layers=6, hidden_size=256, head_dim=128, moe_intermediate_size=256,
                                         shared_expert_intermediate_size=256, layer_types=["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 2,
                                         mlp_layer_types=["moe"] * 6)
            (source / "config.json").write_text(json.dumps(config))
            for filename in ("tokenizer.json", "tokenizer_config.json"):
                (source / filename).write_text("fixture")
            original, payload, entries = {}, {}, []
            shard = "model-1.safetensors"
            for layer in range(6):
                for expert in range(2):
                    for projection in ("gate_proj", "up_proj", "down_proj"):
                        name = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
                        original[name] = torch.ones((256, 256)).to(torch.float8_e4m3fn)
                        original[name + "_scale_inv"] = torch.tensor([[0.125, 0.25], [0.5, 1.0]])
                        entry = {"name": name, "shard": shard, "original_shape": [256, 256]}
                        if layer in (2, 3):
                            fmt, width = ("q4_k", 144) if projection == "down_proj" else ("tq4_1s", 160)
                            payload[name] = (torch.arange(256 * width, dtype=torch.int32) + expert).remainder(256).to(torch.uint8).reshape(256, width)
                            entry.update(format=fmt, storage_shape=[256, width], source_scale_consumed=True, source_scale_tensor=name + "_scale_inv")
                        else:
                            payload[name] = original[name]
                            payload[name + "_scale_inv"] = original[name + "_scale_inv"]
                            entry.update(format="fp8_e4m3", storage_shape=[256, 256], source_scale_tensor=name + "_scale_inv")
                            entries.append({"name": name + "_scale_inv", "shard": shard, "format": "f32", "original_shape": [2, 2], "storage_shape": [2, 2]})
                        entries.append(entry)
            for name, shape in (("model.language_model.embed_tokens.weight", (128, 256)), ("model.language_model.norm.weight", (256,))):
                original[name] = payload[name] = torch.ones(shape, dtype=torch.bfloat16)
                entries.append({"name": name, "shard": shard, "format": "bf16", "original_shape": list(shape), "storage_shape": list(shape)})
            name = "model.language_model.layers.0.mlp.gate.weight"
            original[name] = payload[name] = torch.full((2, 256), 0.1234567, dtype=torch.float32)
            entries.append({"name": name, "shard": shard, "format": "f32", "original_shape": [2, 256], "storage_shape": [2, 256]})
            save_file(original, source / shard)
            save_file(payload, candidate / shard)
            (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {name: shard for name in original}}))
            receipt = root / "verified.json"
            source_receipt(source, receipt)
            packed_path = candidate / shard
            manifest = {"schema_version": 1, "candidate": "C0", "model_type": "fish_omni", "source_revision": "a" * 40, "entries": entries,
                        "shards": [{"file": shard, "bytes": packed_path.stat().st_size, "sha256": hashlib.sha256(packed_path.read_bytes()).hexdigest()}]}
            (candidate / "packed-manifest.json").write_text(json.dumps(manifest))
            output = root / "C0.gguf"
            arguments = ["packed_to_gguf", str(source), str(candidate), str(output), "--verified-source-receipt", str(receipt),
                         "--staged-metadata", str(root / "staged"), "--temp-dir", str(root / "scratch")]
            with patch.object(sys, "argv", arguments), patch.object(FishOmniBackboneModel, "prepare_metadata", lambda model, **kwargs: model.set_gguf_parameters()), patch.object(tempfile, "tempdir", str(root / "scratch")), patch.dict(os.environ, {"TMPDIR": str(root / "scratch")}):
                packed_to_gguf.main()
            tensors = {tensor.name: tensor for tensor in gguf.GGUFReader(output).tensors}
            self.assertEqual(tensors["blk.2.ffn_gate_exps.weight"].tensor_type, gguf.GGMLQuantizationType.TQ4_1S)
            self.assertEqual(tensors["blk.3.ffn_down_exps.weight"].tensor_type, gguf.GGMLQuantizationType.Q4_K)
            boundary = tensors["blk.0.ffn_gate_exps.weight"]
            self.assertEqual(boundary.tensor_type, gguf.GGMLQuantizationType.BF16)
            actual = torch.from_numpy(boundary.data.copy().reshape(-1).view(np.uint16)).view(torch.bfloat16).float().reshape(2, 256, 256)
            self.assertEqual(float(actual[0, 0, 0]), 0.125)
            self.assertEqual(float(actual[1, 129, 129]), 1.0)
            router = tensors["blk.0.ffn_gate_inp.weight"]
            self.assertEqual(router.tensor_type, gguf.GGMLQuantizationType.F32)
            np.testing.assert_array_equal(router.data, np.full((2, 256), np.float32(0.1234567)))
            self.assertEqual(len(json.loads(output.with_suffix(".import.json").read_text())["packed_byte_roundtrip"]), 6)


if __name__ == "__main__":
    unittest.main()
