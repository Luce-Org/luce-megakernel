"""Runtime marker for DN phase2 kernel variant.

Reads MEGAKERNEL_DN_PHASE2_WMMA_RUNTIME (default "0").
  "0" -> scalar FP32 path (current default)
  "1" -> WMMA path (future; kernel not yet implemented)

This module is imported for its side-effect (the print) so the user can
confirm which variant is active without inspecting build flags.
"""
import os as _os

_val = _os.environ.get("MEGAKERNEL_DN_PHASE2_WMMA_RUNTIME", "0").strip()
DN_PHASE2_VARIANT = "wmma" if _val == "1" else "scalar"
print(f"[megakernel] DN phase2 variant = {DN_PHASE2_VARIANT}")
