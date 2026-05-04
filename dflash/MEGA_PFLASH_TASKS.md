# Experimental Mega PFlash Task List

## Current Status

- [x] Integrate native Qwen3.5-0.8B Mega PFlash loader/scorer into `test_dflash`.
- [x] Route `server.py` and `server_tools.py` through `--prefill-backend mega-native`.
- [x] Add BSA hdim256 support for Qwen3.5 full-attention layers.
- [x] Add 64K end-to-end benchmark harness for off, stock PFlash, Mega PFlash, and native Mega PFlash.
- [x] Add configurable prefill parking policy: `full`, `draft`, and `none`.
- [x] Make native Mega PFlash self-contained without modifying the BSA submodule.
- [x] Enable DeltaNet WMMA phase-2 by default for Mega PFlash.

## Next Work

- [ ] Make true no-park stable under the current 27B target by reducing DeltaNet scratch.
- [ ] Rewrite `dn_w_scratch` as BF16 or tiled storage without triggering BSA runtime allocation failures.
- [ ] Tile or avoid persistent `dn_pre_qkv` where possible.
- [ ] Preallocate/reuse BSA scratch to avoid runtime `cudaMalloc` spikes during Mega PFlash.
- [ ] Profile the remaining Mega PFlash forward kernels against stock PFlash's 64K compression path.
- [ ] Test smaller or more aggressively quantized 27B target GGUF variants for no-park headroom.
- [ ] Decide default backend policy after quality and stability A/B runs.
