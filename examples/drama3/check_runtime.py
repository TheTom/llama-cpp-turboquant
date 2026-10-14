"""Import the speech bridge dependencies without loading weights or using a GPU."""
import json
import platform
import torch
import torchaudio
from sglang_omni.models.fishaudio_drama3.audio_decoder import AudioDecoder
from sglang_omni.models.fishaudio_drama3.hf_config import FishOmniConfig
from sglang_omni.models.fishaudio_drama3.prompt import build_prompt_rows
from sglang_omni.models.fishaudio_drama3.weights import load_audio_decoder_weights
from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.dac import modded_dac
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tokenizers import Tokenizer
import soundfile

assert torch.__version__ == '2.13.0+cu130', torch.__version__
assert torchaudio.__version__ == '2.11.0+cu130', torchaudio.__version__
print(json.dumps({'imports': 'passed', 'architecture': platform.machine(),
                  'torch': torch.__version__, 'torchaudio': torchaudio.__version__,
                  'scope': 'CPU imports only; no model weights or GPU execution'}))
