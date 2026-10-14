import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file
from runtime_assets import verified_runtime_assets


class RuntimeAssets(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        tensors = {f"audio_decoder.test_{index}.weight": torch.zeros((2, 2), dtype=torch.bfloat16) for index in range(192)}
        save_file(tensors, self.root / "audio.safetensors")
        (self.root / "audio.index.json").write_text(json.dumps({"weight_map": {name: "audio.safetensors" for name in tensors}}))
        (self.root / "config.json").write_text(json.dumps({"architectures": ["FishOmniModel"], "model_type": "fish_omni"}))
        for name in ("tokenizer.json", "tokenizer_config.json", "LICENSE", "codec.pth"):
            (self.root / name).write_text("fixture")
        self.gguf = self.root / "model.gguf"
        self.gguf.write_bytes(b"GGUF-fixture")
        def identity(path):
            return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        self.manifest = {"schema_version": 1, "artifact_kind": "drama3-runtime-assets", "source_repo": "fishaudio/drama-3",
                         "source_revision": "a" * 40, "audio_index": "audio.index.json", "audio_tensor_count": 192,
                         "audio_tensors": {name: {"shape": [2, 2], "dtype": "BF16"} for name in tensors},
                         "files": {p.name: identity(p) for p in self.root.iterdir() if p != self.gguf}, "gguf": identity(self.gguf)}
        self.path = self.root / "runtime-manifest.json"

    def tearDown(self):
        self.temporary.cleanup()

    def verify(self):
        self.path.write_text(json.dumps(self.manifest))
        return verified_runtime_assets(self.root, self.path, self.gguf)

    def test_audio_only_bundle(self):
        self.assertEqual(self.verify()["audio_tensor_count"], 192)

    def test_corrupt_assets_and_model(self):
        for filename in ("codec.pth", "model.gguf"):
            with self.subTest(filename=filename):
                path = self.root / filename
                before = path.read_bytes()
                path.write_bytes(b"x" * len(before))
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    self.verify()
                path.write_bytes(before)

    def test_census_shape_and_missing_tensor(self):
        name = next(iter(self.manifest["audio_tensors"]))
        self.manifest["audio_tensors"][name]["shape"] = [4]
        with self.assertRaisesRegex(ValueError, "shape or dtype"):
            self.verify()
        del self.manifest["audio_tensors"][name]
        with self.assertRaisesRegex(ValueError, "192-tensor census"):
            self.verify()

    def test_manifest_path_escape(self):
        self.manifest["files"]["../escape"] = self.manifest["gguf"]
        with self.assertRaisesRegex(ValueError, "basenames"):
            self.verify()

    def test_extra_and_missing_actual_tensor(self):
        original = {name: torch.zeros((2, 2), dtype=torch.bfloat16) for name in self.manifest["audio_tensors"]}
        for mutation in ("extra", "missing"):
            with self.subTest(mutation=mutation):
                tensors = dict(original)
                if mutation == "extra":
                    tensors["audio_decoder.unindexed.weight"] = torch.zeros((2, 2), dtype=torch.bfloat16)
                else:
                    del tensors[next(iter(tensors))]
                shard = self.root / "audio.safetensors"
                save_file(tensors, shard)
                self.manifest["files"][shard.name] = {"bytes": shard.stat().st_size, "sha256": hashlib.sha256(shard.read_bytes()).hexdigest()}
                with self.assertRaisesRegex(ValueError, "shard tensor census"):
                    self.verify()


if __name__ == "__main__":
    unittest.main()
