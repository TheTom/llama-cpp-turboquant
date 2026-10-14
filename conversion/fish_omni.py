from __future__ import annotations

from typing import Callable, TYPE_CHECKING

from .base import ModelBase, gguf, logger
from .qwen import _LinearAttentionVReorderBase

if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("FishOmniBackboneForCausalLM")
class FishOmniBackboneModel(_LinearAttentionVReorderBase):
    """Explicit backbone export; audio tensors remain in the source checkpoint."""

    model_arch = gguf.MODEL_ARCH.QWEN35MOE
    supports_mtp_export = False
    no_mtp = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        required = {
            "rotary_interleaved": False,
            "use_qk_norm": True,
            "attn_output_gate": True,
            "layernorm_zero_centered_gamma": True,
            "shared_expert_gate": True,
        }
        for key, value in required.items():
            if self.hparams.get(key) is not value:
                raise ValueError(f"Fish backbone requires {key}={value!r}")
        if self.rope_parameters.get("rope_type", "default") != "default" or "mrope_section" in self.rope_parameters:
            raise ValueError("Fish backbone requires default noninterleaved partial RoPE")
        layer_types = self.hparams.get("layer_types")
        if not isinstance(layer_types, list) or len(layer_types) != self.block_count:
            raise ValueError("Fish backbone requires one layer type per block")
        if not set(layer_types) <= {"linear_attention", "full_attention"}:
            raise ValueError("Unsupported Fish attention layer type")
        if "full_attention" not in layer_types:
            raise ValueError("Fish backbone requires a full attention layer")
        interval = layer_types.index("full_attention") + 1
        if any((kind == "full_attention") != ((index + 1) % interval == 0) for index, kind in enumerate(layer_types)):
            raise ValueError("Fish attention pattern must be periodic")
        self.hparams["full_attention_interval"] = interval
        if self.hparams.get("mlp_layer_types") != ["moe"] * self.block_count:
            raise ValueError("Fish backbone requires MoE in every block")
        logger.warning("Exporting the Fish backbone only; the audio tower and codec require an external runtime")

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, generator = item
        if name.startswith("model.language_model."):
            return "model." + name.removeprefix("model.language_model."), generator
        if name == "lm_head.weight":
            return item
        if name.startswith(("audio_decoder.", "audio_decoder_mtp_shared.", "audio_decoder_mtp.")):
            return None
        raise ValueError(f"Unknown Fish checkpoint namespace: {name}")

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_rope_dimension_sections([0, 0, 0, 0])
