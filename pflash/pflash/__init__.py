"""pflash helpers for prompt-compression experiments.

Two execution paths live here:

- the original dflash daemon client for the CUDA-only in-process runtime
- a backend-agnostic prompt compressor + llama.cpp CLI runner for ROCm/HIP
  targets such as Strix Halo
"""

from . import config
from .dflash_client import DflashClient
from .llama_cli import LlamaCliRunner, LlamaRunResult
from .platform import AcceleratorInfo, detect_accelerator, query_gpu_memory_mib
from .prompt_compressor import CompressionResult, PromptCompressor

__version__ = "0.4.0"
__all__ = [
    "AcceleratorInfo",
    "CompressionResult",
    "DflashClient",
    "LlamaCliRunner",
    "LlamaRunResult",
    "PromptCompressor",
    "config",
    "detect_accelerator",
    "query_gpu_memory_mib",
]
