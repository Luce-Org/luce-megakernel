import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _detect_arch():
    arch = os.environ.get("MEGAKERNEL_CUDA_ARCH")
    if arch:
        return arch
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            if major == 12 and minor in (0, 1):
                return f"sm_{major}{minor}a"
            return f"sm_{major}{minor}"
    except Exception:
        pass
    return "sm_86"


def _int_env(name, default):
    return str(int(os.environ.get(name, default)))


arch = _detect_arch()
is_blackwell = arch.startswith("sm_12")
num_blocks = _int_env("MEGAKERNEL_NUM_BLOCKS", 82)
block_size = _int_env("MEGAKERNEL_BLOCK_SIZE", 512)
lm_num_blocks = _int_env("MEGAKERNEL_LM_NUM_BLOCKS", 512)
lm_block_size = _int_env("MEGAKERNEL_LM_BLOCK_SIZE", 256)
dn_phase2_wmma = _int_env("MEGAKERNEL_DN_PHASE2_WMMA", 0)

sources = [
    "torch_bindings.cpp",
    "kernel.cu",
    "prefill.cu",
]
libraries = ["cublas"]
cxx_args = ["-O3"]
nvcc_args = [
    "-O3",
    f"-arch={arch}",
    "--use_fast_math",
    "-std=c++17",
    f"-DNUM_BLOCKS={num_blocks}",
    f"-DBLOCK_SIZE={block_size}",
    f"-DLM_NUM_BLOCKS={lm_num_blocks}",
    f"-DLM_BLOCK_SIZE={lm_block_size}",
    f"-DMEGAKERNEL_DN_PHASE2_WMMA={dn_phase2_wmma}",
]
# Expose to host compiler (torch_bindings.cpp, prefill.cu host-side) so it
# can also branch on the flag without needing a separate nvcc pass.
cxx_args.append(f"-DMEGAKERNEL_DN_PHASE2_WMMA={dn_phase2_wmma}")

if is_blackwell:
    sources.append("kernel_gb10_nvfp4.cu")
    sources.append("prefill_megakernel.cu")
    sources.append("prefill_bw.cu")
    # Exposed to both nvcc (for the Blackwell .cu files) and the host
    # compiler (so torch_bindings.cpp registers the NVFP4 ops).
    cxx_args.append("-DMEGAKERNEL_HAS_NVFP4")
    nvcc_args.append("-DMEGAKERNEL_HAS_NVFP4")
    nvcc_args.append("-DMEGAKERNEL_HAS_PREFILL_MEGA")
    libraries.append("cublasLt")

setup(
    name="qwen35_megakernel_bf16",
    ext_modules=[
        CUDAExtension(
            name="qwen35_megakernel_bf16_C",
            sources=sources,
            extra_compile_args={
                "cxx": cxx_args,
                "nvcc": nvcc_args,
            },
            libraries=libraries,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
