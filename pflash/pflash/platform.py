"""Platform helpers for CUDA, ROCm, and CPU execution paths."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any


@dataclass(frozen=True)
class AcceleratorInfo:
    backend: str
    torch_device: str
    name: str | None
    is_gpu: bool
    gfx_target: str | None = None


def _run_text(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except Exception:
        return None


def _run_text_candidates(commands: list[list[str]]) -> str | None:
    for command in commands:
        payload = _run_text(command)
        if payload:
            return payload
    return None


def _extract_first_int(text: str) -> int | None:
    match = re.search(r"(-?\d+)", text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _iter_json_scalars(obj: Any):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key, value
            yield from _iter_json_scalars(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_json_scalars(value)


def _json_memory_mib(payload: str) -> int | None:
    try:
        obj = json.loads(payload)
    except Exception:
        return None

    best: int | None = None
    for key, value in _iter_json_scalars(obj):
        if not isinstance(value, (int, float, str)):
            continue
        key_lower = str(key).lower()
        if "used" not in key_lower and "usage" not in key_lower:
            continue
        if "mem" not in key_lower and "vram" not in key_lower and "gtt" not in key_lower:
            continue

        if isinstance(value, str):
            parsed = _extract_first_int(value)
            if parsed is None:
                continue
            number = parsed
        else:
            number = int(value)

        mib = number
        if "byte" in key_lower or number > (1 << 30):
            mib = int(number / (1024 * 1024))

        if mib >= 0 and (best is None or mib > best):
            best = mib
    return best


def _sysfs_memory_mib() -> int | None:
    for pattern in (
        "/sys/class/drm/card*/device/mem_info_vram_used",
        "/sys/class/drm/card*/device/mem_info_vis_vram_used",
        "/sys/class/drm/card*/device/mem_info_gtt_used",
    ):
        for path in sorted(Path("/").glob(pattern.lstrip("/"))):
            try:
                raw = path.read_text().strip()
            except OSError:
                continue
            value = _extract_first_int(raw)
            if value is None:
                continue
            if value > (1 << 30):
                return int(value / (1024 * 1024))
            return value
    return None


def query_gpu_memory_mib(index: int = 0) -> int | None:
    """Return used GPU memory in MiB when a platform tool exposes it."""

    nvidia = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    if nvidia:
        lines = [line.strip() for line in nvidia.splitlines() if line.strip()]
        if index < len(lines):
            value = _extract_first_int(lines[index])
            if value is not None:
                return value

    for command in (
        ["rocm-smi", "--showmemuse", "--json"],
        ["amd-smi", "monitor", "--json"],
        ["amd-smi", "metric", "--json"],
    ):
        payload = _run_text(command)
        if payload:
            value = _json_memory_mib(payload)
            if value is not None:
                return value

    return _sysfs_memory_mib()


def _detect_rocm_gfx_target() -> str | None:
    payload = _run_text_candidates([
        ["rocminfo"],
        ["/opt/rocm/bin/rocminfo"],
        ["hipinfo"],
    ])
    if payload:
        match = re.search(r"gfx\d{4}", payload)
        if match:
            return match.group(0)
    override = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    if override:
        return override
    return None


def detect_accelerator() -> AcceleratorInfo:
    try:
        import torch
    except ImportError:
        return AcceleratorInfo(backend="cpu", torch_device="cpu", name=None, is_gpu=False)

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        if getattr(torch.version, "hip", None):
            return AcceleratorInfo(
                backend="rocm",
                torch_device="cuda",
                name=name,
                is_gpu=True,
                gfx_target=_detect_rocm_gfx_target(),
            )
        return AcceleratorInfo(backend="cuda", torch_device="cuda", name=name, is_gpu=True)

    if getattr(getattr(torch, "backends", None), "mps", None) and torch.backends.mps.is_available():
        return AcceleratorInfo(backend="mps", torch_device="mps", name="Apple MPS", is_gpu=True)

    return AcceleratorInfo(backend="cpu", torch_device="cpu", name=None, is_gpu=False)
