"""No-park PFlash helper for the Qwen3.5-0.8B megakernel.

This talks to dflash's standalone ``pflash_daemon``. Unlike the dflash target
daemon, this process only hosts the Qwen3-0.6B PFlash drafter, so there is no
target/draft park-unpark cycle.
"""

from __future__ import annotations

import os
import struct
import subprocess
import tempfile
from pathlib import Path


def _read_i32_stream(fd: int) -> list[int]:
    out: list[int] = []
    while True:
        b = os.read(fd, 4)
        if not b or len(b) < 4:
            break
        v = struct.unpack("<i", b)[0]
        if v == -1:
            break
        out.append(v)
    return out


class PFlashCompressor:
    """Persistent PFlash sidecar for megakernel prompt compression."""

    def __init__(
        self,
        *,
        daemon_bin: str | Path,
        drafter_gguf: str | Path,
        tokenizer_id: str = "Qwen/Qwen3-0.6B",
        keep_ratio: float = 0.05,
        lookahead: int = 8,
        chunk_size: int = 32,
        pool_kernel: int = 13,
        env: dict[str, str] | None = None,
    ):
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0.0, 1.0]")
        self.daemon_bin = Path(daemon_bin)
        self.drafter_gguf = Path(drafter_gguf)
        if not self.daemon_bin.is_file():
            raise FileNotFoundError(f"pflash daemon not found: {self.daemon_bin}")
        if not self.drafter_gguf.is_file():
            raise FileNotFoundError(f"PFlash drafter GGUF not found: {self.drafter_gguf}")

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
        self.keep_ratio = float(keep_ratio)
        self.lookahead = int(lookahead)
        self.chunk_size = int(chunk_size)
        self.pool_kernel = int(pool_kernel)
        self._r_fd, self._w_fd = os.pipe()
        proc_env = os.environ.copy()
        proc_env.setdefault("DFLASH_FP_USE_BSA", "1")
        proc_env.setdefault("DFLASH_FP_ALPHA", "0.85")
        if env:
            proc_env.update(env)
        self.proc = subprocess.Popen(
            [str(self.daemon_bin), str(self.drafter_gguf), f"--stream-fd={self._w_fd}"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            pass_fds=(self._w_fd,),
            env=proc_env,
            bufsize=1,
        )
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("failed to open pflash daemon pipes")
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("pflash daemon exited before ready")
            if "[pflash-daemon] ready" in line:
                break

    def close(self) -> None:
        if getattr(self, "proc", None) is not None and self.proc.poll() is None:
            assert self.proc.stdin is not None
            self.proc.stdin.write("quit\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=30)
        for fd in (getattr(self, "_r_fd", None), getattr(self, "_w_fd", None)):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def __enter__(self) -> "PFlashCompressor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def compress_ids(self, ids: list[int]) -> list[int]:
        if not ids:
            return []
        fd, path = tempfile.mkstemp(suffix=".pflash.bin")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(struct.pack("<I", len(ids)))
                for tok in ids:
                    f.write(struct.pack("<i", int(tok)))
            keep_x1000 = int(round(self.keep_ratio * 1000))
            assert self.proc.stdin is not None
            self.proc.stdin.write(
                f"compress {keep_x1000} {self.lookahead} "
                f"{self.chunk_size} {self.pool_kernel} {path}\n")
            self.proc.stdin.flush()
            out = _read_i32_stream(self._r_fd)
            if not out:
                raise RuntimeError("pflash compression returned no tokens")
            return out
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def compress_text(self, text: str) -> str:
        ids = self.tokenizer(text, return_tensors=None, add_special_tokens=False)["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        compressed = self.compress_ids([int(t) for t in ids])
        return self.tokenizer.decode(compressed, skip_special_tokens=True)

