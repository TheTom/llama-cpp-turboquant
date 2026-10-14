import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import gguf


def fixture(path, embedding_type):
    embeddings = np.sin(np.arange(128 * 32, dtype=np.float32).reshape(128, 32) * 0.013)
    writer = gguf.GGUFWriter(path, "llama")
    writer.add_name("Drama bridge protocol fixture")
    writer.add_block_count(1)
    writer.add_context_length(64)
    writer.add_embedding_length(32)
    writer.add_feed_forward_length(64)
    writer.add_head_count(2)
    writer.add_head_count_kv(2)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_rope_dimension_count(16)
    writer.add_tokenizer_model("llama")
    writer.add_token_list([str(index) for index in range(128)])
    writer.add_token_scores([0.0] * 128)
    writer.add_token_types([1] * 128)
    writer.add_bos_token_id(0)
    writer.add_eos_token_id(1)
    writer.add_unk_token_id(2)
    encoded = gguf.quantize(embeddings, embedding_type)
    writer.add_tensor("token_embd.weight", encoded, raw_dtype=embedding_type)
    writer.add_tensor("output.weight", embeddings)
    output_norm = np.linspace(0.5, 1.5, 32, dtype=np.float32)
    writer.add_tensor("output_norm.weight", output_norm)
    for name in ("blk.0.attn_norm.weight", "blk.0.ffn_norm.weight"):
        writer.add_tensor(name, np.ones(32, dtype=np.float32))
    for name in ("attn_q", "attn_k", "attn_v", "attn_output"):
        writer.add_tensor(f"blk.0.{name}.weight", np.zeros((32, 32), dtype=np.float32))
    for name in ("ffn_gate", "ffn_up"):
        writer.add_tensor(f"blk.0.{name}.weight", np.zeros((64, 32), dtype=np.float32))
    writer.add_tensor("blk.0.ffn_down.weight", np.zeros((32, 64), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return gguf.dequantize(encoded, embedding_type), embeddings, output_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        for kind in (gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.Q8_0):
            path = Path(directory) / f"{kind.name}.gguf"
            expected_embeddings, head, norm_weights = fixture(path, kind)
            process = subprocess.Popen([str(args.binary.resolve()), str(path), "0", "32"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            try:
                ready = json.loads(process.stdout.readline())
                assert ready["ready"] and ready["n_embd"] == 32
                ids = [3, 17, 31]
                process.stdin.write(json.dumps({"token_ids": ids}) + "\n")
                process.stdin.flush()
                actual_embeddings = np.array(json.loads(process.stdout.readline())["token_embeddings"], dtype=np.float32)
                np.testing.assert_allclose(actual_embeddings, expected_embeddings[ids], rtol=0, atol=1e-7)
                process.stdin.write(json.dumps({"embeddings": actual_embeddings.tolist(), "candidate_ids": [2, 9]}) + "\n")
                process.stdin.flush()
                result = json.loads(process.stdout.readline())
                assert result["timings"]["decode_synchronized_seconds"] >= 0
                assert result["timings"]["output_gather_seconds"] >= 0
                # Every residual projection is zero, so the output is only the final RMS norm.
                expected_hidden = actual_embeddings / np.sqrt(np.mean(actual_embeddings ** 2, axis=-1, keepdims=True) + 1e-6) * norm_weights
                np.testing.assert_allclose(result["hidden_states"], expected_hidden, rtol=2e-6, atol=2e-6)
                np.testing.assert_allclose(result["candidate_logits"], expected_hidden @ head[[2, 9]].T, rtol=2e-5, atol=2e-5)
                process.stdin.write(json.dumps({"reset": True}) + "\n")
                process.stdin.flush()
                assert json.loads(process.stdout.readline()) == {"reset": True}
                print(f"{kind.name}: native embedding rows, final-norm hidden states, logits and reset passed")
            finally:
                process.stdin.close()
                process.wait(timeout=30)


if __name__ == "__main__":
    main()
