# Contributing to Lucebox

Thanks for considering a contribution. Lucebox is a hub of self-contained optimization projects. Each one lives with its own README, benchmarks, and code, and the hub stays thin on purpose.

## What we accept

- **Kernel improvements** that preserve correctness and improve `tok/s`, `tok/J`, or memory footprint on the target hardware. Benchmark deltas required.
- **Speculative decoding algorithms** that improve our current SOTA performances
- **Benchmark harness work** under `benchmarks/` once that directory starts shipping code.
- **Doc fixes and writeups** — always welcome.


## What we don't accept (yet)

- Closed-source dependencies. Everything here has to be reproducible from public sources.

## Luce DFash Setup

### dflash

**Hardware:** NVIDIA sm_86+ GPU (RTX 3090, A10, A40, 4090) or Jetson AGX Thor sm_110, 24 GB VRAM. Thor requires CUDA 13+.

On Ubuntu 22.04 or 24.04, one script installs all system dependencies — `build-essential`, `cmake`, `git`, `git-lfs`, and the CUDA Toolkit from NVIDIA's repo:

```bash
sudo dflash/scripts/setup_system.sh
```

The script is idempotent and configures `nvcc` on PATH for both bash and zsh. For other distros see the [CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).

| Tool | Min version |
|------|------------|
| GCC / G++ | 11 |
| CMake | 3.18 |
| Git | 2.x |
| git-lfs | any |
| CUDA Toolkit | 12.0+ |
| huggingface-cli | any |

After setup:

```bash
git submodule update --init --recursive
cmake -B dflash/build -S dflash -DCMAKE_BUILD_TYPE=Release
cmake --build dflash/build --target test_dflash -j
```

> If cmake was previously run without CUDA, wipe the build directory first (`rm -rf dflash/build`) to avoid a stale compiler cache.

---

## Before you open a PR

1. **Benchmark before and after** on the same hardware, at the same power limit, with the same warmup. Numbers without methodology don't get merged.
2. **Run the existing correctness check** (`bench_pp_tg.py` for megakernel) and confirm your change doesn't regress output parity.
3. **One concern per PR.** Kernel/algorithms changes, docs, and build config go in separate commits or separate PRs.

## Commit message format

Conventional commits:

```
feat(megakernel): fused QKV+RoPE path cuts per-token launch by 1 kernel
fix(dflash): clamp int8 DeltaNet state update before dequant
docs(hub): add DVFS methodology link
```

Allowed types: `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `bench`, `chore`, `ci`.

## Hardware access

If you want to contribute benchmarks but don't have the hardware:

- We can run numbered runs on our RTX 3090 (24GB) or Ryzen 395 AI Max (128GB). Open an issue with the PR.
- Apple Silicon numbers need an M-series machine running `powermetrics`, not a remote box.

## Getting help

- [Discord](https://discord.gg/yHfswqZmJQ) — fastest feedback
- [Issues](https://github.com/Luce-Org/lucebox-hub/issues) — for bugs and proposals
- Mention `@Luce-Org/maintainers` on a PR when it's ready for review

## Licensing

By contributing you agree your work is MIT-licensed, same as the rest of the repo.
