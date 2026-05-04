// Instantiate BSA's hdim=256 bf16 forward block kernel for Qwen3.5 FA layers.
// This is intentionally separate from the hdim=128 instantiation because the
// cutlass template compile is expensive and hdim=256 may need independent
// tuning/fallbacks.
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template<typename T, bool Is_causal>
void run_mha_fwd_block_hdim256(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_fwd_block<
            Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, false, T>,
            Is_dropout,
            Is_causal>(params, stream);
    });
}

template<>
void run_mha_fwd_block_<cutlass::bfloat16_t, 256, false>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_block_hdim256<cutlass::bfloat16_t, false>(params, stream);
}

} // namespace FLASH_NAMESPACE
