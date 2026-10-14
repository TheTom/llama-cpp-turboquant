"""Validate an audio-only community bundle independently of source receipts."""
import hashlib
import json
import re
from pathlib import Path

from safetensors import safe_open

AUDIO_PREFIXES = ("audio_decoder.", "audio_decoder_mtp_shared.", "audio_decoder_mtp.")


def digest_file(path):
    digest = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Artifact changed during verification: {path.name}")
    return digest.hexdigest()


def basename(value):
    if not isinstance(value, str) or not value or Path(value).name != value or value in (".", ".."):
        raise ValueError("Runtime manifest filenames must be basenames")
    return value


def check_file(path, entry):
    if not isinstance(entry.get("bytes"), int) or entry["bytes"] <= 0 or not re.fullmatch("[0-9a-f]{64}", entry.get("sha256", "")):
        raise ValueError(f"Invalid artifact identity: {path.name}")
    if path.stat().st_size != entry["bytes"] or digest_file(path) != entry["sha256"]:
        raise ValueError(f"Runtime artifact size or hash mismatch: {path.name}")


def verified_runtime_assets(source: Path, manifest_path: Path, gguf: Path | None = None):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("artifact_kind") != "drama3-runtime-assets":
        raise ValueError("Unsupported runtime manifest schema")
    if manifest.get("source_repo") != "fishaudio/drama-3" or not re.fullmatch("[0-9a-f]{40}", manifest.get("source_revision", "")):
        raise ValueError("Runtime manifest must identify the pinned Drama-3 source")
    files = manifest["files"]
    if not isinstance(files, dict):
        raise ValueError("Expected runtime files mapping")
    index_name = basename(manifest["audio_index"])
    required = {"config.json", "tokenizer.json", "tokenizer_config.json", "LICENSE", "codec.pth", index_name}
    if not required <= set(files):
        raise ValueError("Runtime bundle is missing required assets")
    for name, entry in files.items():
        check_file(source / basename(name), entry)
    if gguf is not None:
        check_file(gguf, manifest["gguf"])
    config = json.loads((source / "config.json").read_text())
    if config.get("architectures") != ["FishOmniModel"] or config.get("model_type") != "fish_omni":
        raise ValueError("Expected original FishOmniModel configuration")
    census = manifest["audio_tensors"]
    index = json.loads((source / index_name).read_text())["weight_map"]
    if manifest.get("audio_tensor_count") != 192 or not isinstance(census, dict) or len(census) != 192 or set(index) != set(census):
        raise ValueError("Runtime audio index and 192-tensor census differ")
    if any(not name.startswith(AUDIO_PREFIXES) for name in census):
        raise ValueError("Runtime audio bundle contains a non-audio namespace")
    for shard in set(index.values()):
        basename(shard)
        if shard not in files:
            raise ValueError("Audio shard has no hash in the runtime manifest")
        expected = {name for name, filename in index.items() if filename == shard}
        with safe_open(source / shard, framework="pt", device="cpu") as reader:
            if set(reader.keys()) != expected:
                raise ValueError("Audio shard tensor census differs from its index")
            for name in expected:
                tensor = reader.get_slice(name)
                entry = census[name]
                if entry.get("dtype") not in ("BF16", "F32") or tensor.get_dtype() != entry["dtype"] or tensor.get_shape() != entry["shape"]:
                    raise ValueError(f"Audio tensor shape or dtype mismatch: {name}")
    return manifest
