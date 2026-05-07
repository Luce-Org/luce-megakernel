"""pflash, speculative-prefill harness around the dflash C++/CUDA daemon.

The daemon does the work — drafter forward, FlashPrefill scoring (BSA), and
spec-decode generation are all in-process C++/CUDA. This Python package is
just a thin client over the daemon's stdin/stdout protocol so reproduction
benches and external tooling can drive it.
"""
from .dflash_client import DflashClient
from . import config

__version__ = "0.3.0"
__all__ = ["DflashClient", "config"]
