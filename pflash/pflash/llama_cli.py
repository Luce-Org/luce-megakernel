"""Thin wrapper around llama.cpp's `llama-cli` binary."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import tempfile
import time


@dataclass(frozen=True)
class LlamaRunResult:
    command: list[str]
    returncode: int
    wall_s: float
    ttft_s: float | None
    stdout: str
    stderr: str


class LlamaCliRunner:
    def __init__(
        self,
        binary: str | os.PathLike[str],
        model: str | os.PathLike[str],
        *,
        ctx_size: int,
        n_predict: int = 64,
        n_gpu_layers: str | int | None = "all",
        flash_attn: str = "on",
        threads: int | None = None,
        threads_batch: int | None = None,
        batch_size: int | None = None,
        ubatch_size: int | None = None,
        cache_type_k: str | None = None,
        cache_type_v: str | None = None,
        log_disable: bool = True,
        extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.binary = Path(binary)
        self.model = Path(model)
        self.ctx_size = ctx_size
        self.n_predict = n_predict
        self.n_gpu_layers = n_gpu_layers
        self.flash_attn = flash_attn
        self.threads = threads
        self.threads_batch = threads_batch
        self.batch_size = batch_size
        self.ubatch_size = ubatch_size
        self.cache_type_k = cache_type_k
        self.cache_type_v = cache_type_v
        self.log_disable = log_disable
        self.extra_args = list(extra_args or [])
        self.env = dict(env or {})

    def build_command(self, prompt: str | None = None, prompt_file: str | os.PathLike[str] | None = None) -> list[str]:
        if (prompt is None) == (prompt_file is None):
            raise ValueError("exactly one of prompt or prompt_file must be provided")

        command = [
            str(self.binary),
            "-m",
            str(self.model),
            "-c",
            str(self.ctx_size),
            "-n",
            str(self.n_predict),
            "--temp",
            "0",
            "--top-k",
            "1",
            "--top-p",
            "1.0",
            "--repeat-penalty",
            "1.0",
            "--simple-io",
            "--single-turn",
            "--no-display-prompt",
        ]
        if prompt_file is not None:
            command.extend(["-f", str(prompt_file)])
        else:
            command.extend(["-p", prompt])
        if self.flash_attn:
            command.extend(["-fa", self.flash_attn])
        if self.n_gpu_layers is not None:
            command.extend(["-ngl", str(self.n_gpu_layers)])
        if self.threads is not None:
            command.extend(["-t", str(self.threads)])
        if self.threads_batch is not None:
            command.extend(["-tb", str(self.threads_batch)])
        if self.batch_size is not None:
            command.extend(["-b", str(self.batch_size)])
        if self.ubatch_size is not None:
            command.extend(["-ub", str(self.ubatch_size)])
        if self.cache_type_k:
            command.extend(["-ctk", self.cache_type_k])
        if self.cache_type_v:
            command.extend(["-ctv", self.cache_type_v])
        if self.log_disable:
            command.append("--log-disable")
        command.extend(self.extra_args)
        return command

    def generate(self, prompt: str) -> LlamaRunResult:
        env = {**os.environ, **self.env}
        use_prompt_file = len(prompt) > 4096 or "\n" in prompt

        with tempfile.TemporaryDirectory(prefix="pflash-llama-") as tmpdir:
            if use_prompt_file:
                prompt_path = Path(tmpdir) / "prompt.txt"
                prompt_path.write_text(prompt, encoding="utf-8")
                command = self.build_command(prompt_file=prompt_path)
            else:
                command = self.build_command(prompt=prompt)

            start = time.perf_counter()
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=False,
            )

            assert proc.stdout is not None
            stdout_chunks: list[bytes] = []
            ttft_s: float | None = None

            while True:
                chunk = proc.stdout.read(1 if ttft_s is None else 4096)
                if not chunk:
                    break
                if ttft_s is None:
                    ttft_s = time.perf_counter() - start
                stdout_chunks.append(chunk)

            returncode = proc.wait()
            wall_s = time.perf_counter() - start
            stderr = b""
            if proc.stderr is not None:
                stderr = proc.stderr.read()

            stdout = b"".join(stdout_chunks).decode("utf-8", errors="replace")
            stderr_text = stderr.decode("utf-8", errors="replace")
            return LlamaRunResult(
                command=command,
                returncode=returncode,
                wall_s=wall_s,
                ttft_s=ttft_s,
                stdout=stdout,
                stderr=stderr_text,
            )
