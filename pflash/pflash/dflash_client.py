"""Subprocess client for the patched dflash daemon (with park/unpark commands)."""
from __future__ import annotations
import logging
import os
import struct
import subprocess
import tempfile
import time
from typing import Optional

from . import config

log = logging.getLogger(__name__)


def _query_nvidia_vram_mib() -> Optional[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=5)
        return int(out.decode().splitlines()[0])
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        return None


class DflashClient:
    def __init__(self, bin_path: str, target_path: str, draft_path: str,
                 max_ctx: int = 16384, ddtree_budget: int = 16,
                 *,
                 ddtree_temp: Optional[float] = None,
                 chain_seed: bool = True,
                 fa_window: Optional[int] = None,
                 kv_tq3: Optional[bool] = None,
                 lm_head_fix: Optional[bool] = None,
                 boot_timeout_s: float = 60.0,
                 boot_vram_mib: int = 18000):
        """Spawn the patched dflash daemon as a subprocess.

        Defaults for fa_window / kv_tq3 / lm_head_fix come from
        ``config.DFLASH_REQUIRED_ENV`` and are the only flags pflash relies on.
        Override per-call only when you know what you're doing.
        """
        self.bin_path = bin_path
        self.target_path = target_path
        self.draft_path = draft_path
        self.max_ctx = max_ctx
        env_overrides = {
            "DFLASH27B_FA_WINDOW": str(0 if fa_window is None else fa_window),
            "DFLASH27B_KV_TQ3": "1" if (kv_tq3 if kv_tq3 is not None else True) else "0",
            "DFLASH27B_LM_HEAD_FIX": "1" if lm_head_fix else "0",
        }
        env = {**os.environ, **config.DFLASH_REQUIRED_ENV, **env_overrides}
        bin_dir = os.path.dirname(os.path.abspath(bin_path))
        ld_extra = [bin_dir, os.path.join(bin_dir, "bin")]
        env["LD_LIBRARY_PATH"] = ":".join(
            ld_extra + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
        self.r_pipe, self.w_pipe = os.pipe()
        cmd = [bin_path, target_path, draft_path,
               "--daemon", "--fast-rollback", "--ddtree",
               f"--ddtree-budget={ddtree_budget}",
               f"--max-ctx={max_ctx}",
               f"--stream-fd={self.w_pipe}"]
        if ddtree_temp is not None:
            cmd.append(f"--ddtree-temp={ddtree_temp}")
        if not chain_seed:
            cmd.append("--ddtree-no-chain-seed")
        log.info("spawning dflash daemon: %s", " ".join(cmd))
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     pass_fds=(self.w_pipe,), env=env)
        os.close(self.w_pipe)
        self._wait_until_loaded(timeout=boot_timeout_s, vram_mib=boot_vram_mib)
        # Park draft by default; user calls unpark when needed
        self._send("park draft\n")

    def _wait_until_loaded(self, timeout: float = 60.0, vram_mib: int = 18000):
        boot = time.time()
        telemetry_works = True
        while time.time() - boot < timeout:
            time.sleep(1)
            if self.proc.poll() is not None:
                raise RuntimeError(
                    "dflash daemon exited before weights finished loading. Check the daemon's stderr.")
            if telemetry_works:
                vram = _query_nvidia_vram_mib()
                if vram is None:
                    telemetry_works = False  # No nvidia-smi (e.g. ROCm); fall through to time-based check
                elif vram > vram_mib:
                    return
            if not telemetry_works and time.time() - boot >= min(timeout, 5.0):
                log.warning(
                    "no GPU memory telemetry; assuming daemon boot succeeded since the process is still alive")
                return
        raise RuntimeError(
            f"dflash daemon failed to load target weights within {timeout:.0f}s "
            f"(expected VRAM > {vram_mib} MiB). Check the daemon's stderr.")

    def _send(self, cmd: str):
        self.proc.stdin.write(cmd.encode())
        self.proc.stdin.flush()
        # Read until -1 sentinel
        while True:
            b = os.read(self.r_pipe, 4)
            if not b or len(b) < 4:
                break
            if struct.unpack("<i", b)[0] == -1:
                break

    def free_drafter(self): self._send("free drafter\n")
    def park_draft(self):    self._send("park draft\n")
    def unpark_draft(self):  self._send("unpark draft\n")
    def park_target(self):   self._send("park target\n")
    def unpark_target(self): self._send("unpark target\n")

    def compress(self, prompt_ids: list[int], keep_ratio: float, drafter_gguf: str,
                 drafter_arch: str = "qwen3-0.6b") -> list[int]:
        """C++ drafter score+compress via daemon. Returns compressed token ids.

        Daemon command: compress <bin> <keep_x1000> <drafter_gguf> <drafter_arch>
        """
        fd, path = tempfile.mkstemp(suffix=".bin")
        with os.fdopen(fd, "wb") as f:
            for t in prompt_ids:
                f.write(struct.pack("<i", int(t)))
        keep_x1000 = int(round(keep_ratio * 1000))
        self.proc.stdin.write(f"compress {path} {keep_x1000} {drafter_gguf} {drafter_arch}\n".encode())
        self.proc.stdin.flush()
        toks = []
        while True:
            b = os.read(self.r_pipe, 4)
            if not b or len(b) < 4:
                break
            v = struct.unpack("<i", b)[0]
            if v == -1:
                break
            toks.append(v)
        os.unlink(path)
        return toks

    def generate(self, prompt_ids: list[int], n_gen: int) -> list[int]:
        """Send prompt + n_gen request, return generated token ids."""
        fd, path = tempfile.mkstemp(suffix=".bin")
        with os.fdopen(fd, "wb") as f:
            for t in prompt_ids:
                f.write(struct.pack("<i", int(t)))
        self.proc.stdin.write(f"{path} {n_gen}\n".encode())
        self.proc.stdin.flush()
        toks = []
        while True:
            b = os.read(self.r_pipe, 4)
            if not b or len(b) < 4:
                break
            v = struct.unpack("<i", b)[0]
            if v == -1:
                break
            toks.append(v)
        os.unlink(path)
        return toks

    def close(self, timeout: float = 5.0):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
