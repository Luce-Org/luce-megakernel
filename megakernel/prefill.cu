/**
 * BF16 Prefill: cuBLAS bf16 GEMM + standalone recurrence kernel.
 * Weights bf16, activations bf16, state f32. No quantization, no conversion.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

constexpr int HIDDEN = 1024;
constexpr int INTER = 3584;
constexpr int VOCAB = 248320;
constexpr float RMS_EPS = 1e-6f;

constexpr int FA_Q_HEADS = 8;
constexpr int FA_KV_HEADS = 2;
constexpr int FA_HEAD_DIM = 256;
constexpr int FA_GQA = FA_Q_HEADS / FA_KV_HEADS;
constexpr int FA_Q_SIZE = FA_Q_HEADS * FA_HEAD_DIM;
constexpr int FA_QPROJ_SIZE = FA_Q_SIZE * 2;
constexpr int FA_KV_SIZE = FA_KV_HEADS * FA_HEAD_DIM;
constexpr int FA_ROT_DIM = 64;
constexpr float FA_ROPE_THETA = 10000000.0f;

constexpr int DN_HEADS = 16;
constexpr int DN_KEY = 128;
constexpr int DN_VAL = 128;
constexpr int DN_CONV_K = 4;
constexpr int DN_QK_SIZE = DN_HEADS * DN_KEY;
constexpr int DN_V_SIZE = DN_HEADS * DN_VAL;
constexpr int DN_CONV_CH = DN_QK_SIZE * 2 + DN_V_SIZE;

constexpr int NUM_LAYERS = 24;
constexpr int LAYER_TYPE[24] = {0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1};

struct PFLayerWeights { int layer_type; int _pad[3]; void *ptrs[14]; };

__device__ __forceinline__ float pf_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o); return v;
}
__device__ __forceinline__ float pf_warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o)); return v;
}
__device__ __forceinline__ float pf_silu(float x) { return x / (1.0f + expf(-x)); }

static void cublas_bf16_gemm(cublasHandle_t h,
    const __nv_bfloat16 *A, const __nv_bfloat16 *B, __nv_bfloat16 *C,
    int S, int N, int K);

static void cublas_bf16_qk_scores(cublasHandle_t h,
    const __nv_bfloat16 *q, int q_stride,
    const __nv_bfloat16 *k, int k_stride,
    float *scores, int rows, int key_count, int dim);

static void cublas_bf16_probs_v(cublasHandle_t h,
    const __nv_bfloat16 *probs, int prob_stride,
    const __nv_bfloat16 *v, int v_stride,
    float *out, int rows, int key_count, int dim);

// Embedding
__global__ void pf_embed(const int *ids, const __nv_bfloat16 *embed, __nv_bfloat16 *out, int S) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= S * HIDDEN) return;
    out[idx] = embed[ids[idx / HIDDEN] * HIDDEN + idx % HIDDEN];
}

// Batched RMSNorm: bf16 in → bf16 out, saves bf16 residual
__global__ void pf_rmsnorm(const __nv_bfloat16 *in, const __nv_bfloat16 *w,
    __nv_bfloat16 *out, __nv_bfloat16 *res, int S, int D) {
    int s = blockIdx.x; if (s >= S) return;
    int tid = threadIdx.x, wid = tid/32, lid = tid%32;
    __shared__ float smem[32];
    const __nv_bfloat16 *ri = in + s*D;
    __nv_bfloat16 *ro = out + s*D, *rr = res + s*D;
    float sq = 0;
    for (int i = tid; i < D; i += blockDim.x) { float v = __bfloat162float(ri[i]); rr[i] = ri[i]; sq += v*v; }
    sq = pf_warp_sum(sq); if(lid==0) smem[wid]=sq; __syncthreads();
    if(wid==0){float v=(lid<blockDim.x/32)?smem[lid]:0;v=pf_warp_sum(v);if(lid==0)smem[0]=rsqrtf(v/D+RMS_EPS);}
    __syncthreads(); float rstd = smem[0];
    for (int i = tid; i < D; i += blockDim.x) {
        float v = __bfloat162float(ri[i]) * rstd * (1.0f + __bfloat162float(w[i]));
        ro[i] = __float2bfloat16(v);
    }
}

// bf16 matvec for tiny projections (beta/alpha)
__global__ void pf_bf16_matvec(const __nv_bfloat16 *in, const __nv_bfloat16 *w, float *out, int S, int K, int N) {
    int idx = blockIdx.x; if (idx >= S * N) return;
    int s = idx / N, n = idx % N, lid = threadIdx.x;
    const __nv_bfloat16 *ir = in + s*K, *wr = w + n*K;
    float sum = 0;
    for (int k = lid; k < K; k += 32) sum += __bfloat162float(ir[k]) * __bfloat162float(wr[k]);
    sum = pf_warp_sum(sum);
    if (lid == 0) out[idx] = sum;
}

// bf16 result + bf16 residual → bf16 output
__global__ void pf_add_residual_bf16(const __nv_bfloat16 *a, const __nv_bfloat16 *b, __nv_bfloat16 *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
}

// SiLU(gate) * up — bf16 inputs → bf16 output
__global__ void pf_silu_mul_bf16(const __nv_bfloat16 *gate, const __nv_bfloat16 *up, __nv_bfloat16 *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) { float g = __bfloat162float(gate[i]); out[i] = __float2bfloat16(pf_silu(g) * __bfloat162float(up[i])); }
}

// ===== Standalone DeltaNet recurrence (state-in-registers, bf16 I/O, f32 state) =====
__global__ void __launch_bounds__(512, 1)
pf_deltanet_recurrence(
    const __nv_bfloat16 *qkv_proj, const __nv_bfloat16 *z_proj,
    const float *beta_proj, const float *alpha_proj,
    const __nv_bfloat16 *conv_w, const __nv_bfloat16 *a_log,
    const __nv_bfloat16 *dt_bias, const __nv_bfloat16 *norm_w,
    float *state, float *conv_buf, __nv_bfloat16 *output, int S)
{
    int h = blockIdx.x; if (h >= DN_HEADS) return;
    int tid = threadIdx.x, wid = tid/32, lid = tid%32;
    constexpr int NWARPS = 16;
    constexpr float Q_SCALE = 1.0f / 11.313708498984761f;

    float a_log_val = __bfloat162float(a_log[h]);
    float dt_b = __bfloat162float(dt_bias[h]);

    __shared__ float s_q[DN_KEY], s_k[DN_KEY], s_v[DN_VAL];
    __shared__ float s_beta, s_decay;
    __shared__ float s_gnorm[NWARPS];

    float *my_state = state + h * DN_KEY * DN_VAL;

    // Load state into registers
    constexpr int CPW = DN_VAL / NWARPS;  // 8
    constexpr int RPL = DN_KEY / 32;       // 4
    float sreg[CPW * RPL];  // 32 floats

    for (int jj = 0; jj < CPW; jj++) {
        int j = wid * CPW + jj;
        for (int ii = 0; ii < RPL; ii++)
            sreg[jj*RPL+ii] = my_state[j*DN_KEY + lid+ii*32];
    }

    for (int t = 0; t < S; t++) {
        // Conv1d + SiLU (read bf16 proj, write f32 to shared)
        for (int c = tid; c < DN_KEY; c += 512) {
            int ch = h*DN_KEY + c;
            float h0=conv_buf[ch*DN_CONV_K+1],h1=conv_buf[ch*DN_CONV_K+2],h2=conv_buf[ch*DN_CONV_K+3];
            conv_buf[ch*DN_CONV_K]=h0;conv_buf[ch*DN_CONV_K+1]=h1;conv_buf[ch*DN_CONV_K+2]=h2;
            conv_buf[ch*DN_CONV_K+3]=__bfloat162float(qkv_proj[t*DN_CONV_CH+ch]);
            float co=0;for(int k=0;k<DN_CONV_K;k++)co+=conv_buf[ch*DN_CONV_K+k]*__bfloat162float(conv_w[ch*DN_CONV_K+k]);
            s_q[c]=pf_silu(co);
        }
        for (int c = tid; c < DN_KEY; c += 512) {
            int ch = DN_QK_SIZE + h*DN_KEY + c;
            float h0=conv_buf[ch*DN_CONV_K+1],h1=conv_buf[ch*DN_CONV_K+2],h2=conv_buf[ch*DN_CONV_K+3];
            conv_buf[ch*DN_CONV_K]=h0;conv_buf[ch*DN_CONV_K+1]=h1;conv_buf[ch*DN_CONV_K+2]=h2;
            conv_buf[ch*DN_CONV_K+3]=__bfloat162float(qkv_proj[t*DN_CONV_CH+ch]);
            float co=0;for(int k=0;k<DN_CONV_K;k++)co+=conv_buf[ch*DN_CONV_K+k]*__bfloat162float(conv_w[ch*DN_CONV_K+k]);
            s_k[c]=pf_silu(co);
        }
        for (int c = tid; c < DN_VAL; c += 512) {
            int ch = 2*DN_QK_SIZE + h*DN_VAL + c;
            float h0=conv_buf[ch*DN_CONV_K+1],h1=conv_buf[ch*DN_CONV_K+2],h2=conv_buf[ch*DN_CONV_K+3];
            conv_buf[ch*DN_CONV_K]=h0;conv_buf[ch*DN_CONV_K+1]=h1;conv_buf[ch*DN_CONV_K+2]=h2;
            conv_buf[ch*DN_CONV_K+3]=__bfloat162float(qkv_proj[t*DN_CONV_CH+ch]);
            float co=0;for(int k=0;k<DN_CONV_K;k++)co+=conv_buf[ch*DN_CONV_K+k]*__bfloat162float(conv_w[ch*DN_CONV_K+k]);
            s_v[c]=pf_silu(co);
        }
        __syncthreads();

        // L2 normalize
        if(wid==0){float sq=0;for(int i=lid;i<DN_KEY;i+=32)sq+=s_q[i]*s_q[i];sq=pf_warp_sum(sq);float n=rsqrtf(sq+1e-6f)*Q_SCALE;n=__shfl_sync(0xffffffff,n,0);for(int i=lid;i<DN_KEY;i+=32)s_q[i]*=n;}
        if(wid==1){float sq=0;for(int i=lid;i<DN_KEY;i+=32)sq+=s_k[i]*s_k[i];sq=pf_warp_sum(sq);float n=rsqrtf(sq+1e-6f);n=__shfl_sync(0xffffffff,n,0);for(int i=lid;i<DN_KEY;i+=32)s_k[i]*=n;}
        __syncthreads();

        if(tid==0){s_beta=1.f/(1.f+expf(-beta_proj[t*DN_HEADS+h]));float x=alpha_proj[t*DN_HEADS+h]+dt_b;float sp=(x>20.f)?x:logf(1.f+expf(x));s_decay=expf(-expf(a_log_val)*sp);}
        __syncthreads();
        float beta = s_beta, decay = s_decay;
        __nv_bfloat16 *out_h = output + t * DN_V_SIZE + h * DN_VAL;

        // State-in-registers recurrence
        for (int jj = 0; jj < CPW; jj++) {
            int j = wid * CPW + jj;
            float kv = 0;
            for (int ii = 0; ii < RPL; ii++) kv += sreg[jj*RPL+ii] * s_k[lid+ii*32];
            kv = pf_warp_sum(kv); kv = __shfl_sync(0xffffffff, kv, 0);
            float delta = (s_v[j] - decay * kv) * beta;
            float attn = 0;
            for (int ii = 0; ii < RPL; ii++) {
                sreg[jj*RPL+ii] = decay * sreg[jj*RPL+ii] + s_k[lid+ii*32] * delta;
                attn += sreg[jj*RPL+ii] * s_q[lid+ii*32];
            }
            attn = pf_warp_sum(attn);
            if (lid == 0) out_h[j] = __float2bfloat16(attn);
        }
        __syncthreads();

        // Gated RMSNorm → bf16 output
        const __nv_bfloat16 *z_h = z_proj + t*DN_V_SIZE + h*DN_VAL;
        float sq2=0;for(int i=tid;i<DN_VAL;i+=512){float v=__bfloat162float(out_h[i]);sq2+=v*v;}
        sq2=pf_warp_sum(sq2);if(lid==0)s_gnorm[wid]=sq2;__syncthreads();
        if(wid==0){float v=(lid<NWARPS)?s_gnorm[lid]:0;v=pf_warp_sum(v);if(lid==0)s_gnorm[0]=rsqrtf(v/DN_VAL+RMS_EPS);}
        __syncthreads();float rstd=s_gnorm[0];
        for(int i=tid;i<DN_VAL;i+=512){
            float n=__bfloat162float(out_h[i])*rstd*__bfloat162float(norm_w[i]);
            out_h[i]=__float2bfloat16(n*pf_silu(__bfloat162float(z_h[i])));
        }
        __syncthreads();
    }

    // Write state back
    for (int jj = 0; jj < CPW; jj++) {
        int j = wid * CPW + jj;
        for (int ii = 0; ii < RPL; ii++)
            my_state[j*DN_KEY + lid+ii*32] = sreg[jj*RPL+ii];
    }
}

// ===== QK norm + RoPE + KV cache =====
__global__ void pf_qk_norm_rope(
    __nv_bfloat16 *q, __nv_bfloat16 *k, const __nv_bfloat16 *v,
    const __nv_bfloat16 *qnw, const __nv_bfloat16 *knw,
    __nv_bfloat16 *k_cache, __nv_bfloat16 *v_cache, int S, int max_seq)
{
    int idx = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lid = threadIdx.x % 32;
    int total_q = S * FA_Q_HEADS, total_k = S * FA_KV_HEADS;
    if (idx < total_q) {
        int pos = idx / FA_Q_HEADS, head = idx % FA_Q_HEADS;
        __nv_bfloat16 *qh = q + pos * FA_QPROJ_SIZE + head * FA_HEAD_DIM * 2;
        float ss = 0; for (int i = lid; i < FA_HEAD_DIM; i += 32) { float v = __bfloat162float(qh[i]); ss += v*v; }
        ss = pf_warp_sum(ss); float sc = rsqrtf(ss/FA_HEAD_DIM+RMS_EPS); sc = __shfl_sync(0xffffffff,sc,0);
        for (int i = lid; i < FA_HEAD_DIM; i += 32) {
            float normed = __bfloat162float(qh[i])*sc*(1.f+__bfloat162float(qnw[i]));
            if (i < FA_ROT_DIM) {
                float fe=float(2*(i%(FA_ROT_DIM/2)))/FA_ROT_DIM; float freq=float(pos)/powf(FA_ROPE_THETA,fe);
                float cv=cosf(freq),sv=sinf(freq); int p=(i<FA_ROT_DIM/2)?i+FA_ROT_DIM/2:i-FA_ROT_DIM/2;
                float pv=__bfloat162float(qh[p])*sc*(1.f+__bfloat162float(qnw[p]));
                qh[i]=__float2bfloat16((i<FA_ROT_DIM/2)?(normed*cv-pv*sv):(pv*sv+normed*cv));
            } else qh[i]=__float2bfloat16(normed);
        }
    }
    int kidx = idx - total_q;
    if (idx >= total_q && kidx < total_k) {
        int pos = kidx / FA_KV_HEADS, head = kidx % FA_KV_HEADS;
        __nv_bfloat16 *kh = k + pos*FA_KV_SIZE + head*FA_HEAD_DIM;
        const __nv_bfloat16 *vh = v + pos*FA_KV_SIZE + head*FA_HEAD_DIM;
        __nv_bfloat16 *kc = k_cache + head*max_seq*FA_HEAD_DIM + pos*FA_HEAD_DIM;
        __nv_bfloat16 *vc = v_cache + head*max_seq*FA_HEAD_DIM + pos*FA_HEAD_DIM;
        float ss = 0; for (int i = lid; i < FA_HEAD_DIM; i += 32) { float v = __bfloat162float(kh[i]); ss += v*v; }
        ss = pf_warp_sum(ss); float sc = rsqrtf(ss/FA_HEAD_DIM+RMS_EPS); sc = __shfl_sync(0xffffffff,sc,0);
        for (int i = lid; i < FA_HEAD_DIM; i += 32) {
            float normed = __bfloat162float(kh[i])*sc*(1.f+__bfloat162float(knw[i])); float fk;
            if (i < FA_ROT_DIM) {
                float fe=float(2*(i%(FA_ROT_DIM/2)))/FA_ROT_DIM; float freq=float(pos)/powf(FA_ROPE_THETA,fe);
                float cv=cosf(freq),sv=sinf(freq); int p=(i<FA_ROT_DIM/2)?i+FA_ROT_DIM/2:i-FA_ROT_DIM/2;
                float pv=__bfloat162float(kh[p])*sc*(1.f+__bfloat162float(knw[p]));
                fk=(i<FA_ROT_DIM/2)?(normed*cv-pv*sv):(pv*sv+normed*cv);
            } else fk=normed;
            kh[i]=__float2bfloat16(fk); kc[i]=__float2bfloat16(fk); vc[i]=vh[i];
        }
    }
}

// ===== Causal attention (bf16 Q/K/V, f32 accumulation, bf16 output) =====
__global__ void pf_causal_attn(const __nv_bfloat16 *q, const __nv_bfloat16 *k,
    const __nv_bfloat16 *v, __nv_bfloat16 *out, int S)
{
    int idx = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lid = threadIdx.x % 32;
    if (idx >= S * FA_Q_HEADS) return;
    int pos = idx / FA_Q_HEADS, qh = idx % FA_Q_HEADS, kvh = qh / FA_GQA;
    float scale = 1.0f / sqrtf(float(FA_HEAD_DIM));
    constexpr int EPL = FA_HEAD_DIM / 32;
    const __nv_bfloat16 *qv = q + pos*FA_QPROJ_SIZE + qh*FA_HEAD_DIM*2;
    const __nv_bfloat16 *gv = qv + FA_HEAD_DIM;
    __nv_bfloat16 *ov = out + pos*FA_Q_SIZE + qh*FA_HEAD_DIM;
    float ql[EPL]; for(int e=0;e<EPL;e++) ql[e]=__bfloat162float(qv[lid*EPL+e]);
    float oa[EPL]={}; float mx=-1e30f, se=0;
    for (int kp = 0; kp <= pos; kp++) {
        const __nv_bfloat16 *kv=k+kp*FA_KV_SIZE+kvh*FA_HEAD_DIM;
        const __nv_bfloat16 *vv=v+kp*FA_KV_SIZE+kvh*FA_HEAD_DIM;
        float sc=0; for(int e=0;e<EPL;e++) sc+=ql[e]*__bfloat162float(kv[lid*EPL+e]);
        sc=pf_warp_sum(sc)*scale; sc=__shfl_sync(0xffffffff,sc,0);
        float om=mx; mx=fmaxf(mx,sc); float ed=expf(om-mx); se=se*ed+expf(sc-mx);
        float wt=expf(sc-mx); for(int e=0;e<EPL;e++) oa[e]=oa[e]*ed+wt*__bfloat162float(vv[lid*EPL+e]);
    }
    float rs=1.f/se;
    for(int e=0;e<EPL;e++){int i=lid*EPL+e;float g=1.f/(1.f+expf(-__bfloat162float(gv[i])));ov[i]=__float2bfloat16(oa[e]*rs*g);}
}

// Final norm
__global__ void pf_final_norm(const __nv_bfloat16 *hidden, const __nv_bfloat16 *w,
    __nv_bfloat16 *normed, __nv_bfloat16 *hidden_out, int S) {
    int tid=threadIdx.x, wid=tid/32, lid=tid%32;
    __shared__ float smem[16];
    const __nv_bfloat16 *row = hidden + (S-1)*HIDDEN;
    float sq=0; for(int i=tid;i<HIDDEN;i+=blockDim.x){float v=__bfloat162float(row[i]);sq+=v*v;}
    sq=pf_warp_sum(sq);if(lid==0)smem[wid]=sq;__syncthreads();
    if(wid==0){float v=(lid<blockDim.x/32)?smem[lid]:0;v=pf_warp_sum(v);if(lid==0)smem[0]=rsqrtf(v/HIDDEN+RMS_EPS);}
    __syncthreads();float rstd=smem[0];
    for(int i=tid;i<HIDDEN;i+=blockDim.x){
        float v=__bfloat162float(row[i]);
        normed[i]=__float2bfloat16(v*rstd*(1.f+__bfloat162float(w[i])));
        hidden_out[i]=row[i];
    }
}

// LM head: bf16 weight × bf16 hidden
__global__ void pf_lm_head(const __nv_bfloat16 *hidden, const __nv_bfloat16 *w,
    float *bmv, int *bmi, int N) {
    __shared__ __nv_bfloat16 s_h[HIDDEN];
    for(int i=threadIdx.x;i<HIDDEN;i+=blockDim.x) s_h[i]=hidden[i];
    __syncthreads();
    int wid=threadIdx.x/32, lid=threadIdx.x%32, nw=blockDim.x/32;
    int rpb=(N+gridDim.x-1)/gridDim.x, rs=blockIdx.x*rpb, re=min(rs+rpb,N);
    float lm=-1e30f; int li=-1;
    for(int m=rs+wid;m<re;m+=nw){const __nv_bfloat16 *wr=w+m*HIDDEN;float s=0;
        for(int k=lid*8;k<HIDDEN;k+=32*8){for(int i=0;i<8;i++)s+=__bfloat162float(wr[k+i])*__bfloat162float(s_h[k+i]);}
        s=pf_warp_sum(s);if(lid==0&&s>lm){lm=s;li=m;}}
    lm=__shfl_sync(0xffffffff,lm,0);li=__shfl_sync(0xffffffff,li,0);
    __shared__ float wm[32]; __shared__ int wi[32];
    if(lid==0){wm[wid]=lm;wi[wid]=li;}__syncthreads();
    if(wid==0){float mv=(lid<nw)?wm[lid]:-1e30f;int mi=(lid<nw)?wi[lid]:-1;
        for(int o=16;o>0;o>>=1){float ov=__shfl_down_sync(0xffffffff,mv,o);int oi=__shfl_down_sync(0xffffffff,mi,o);if(ov>mv){mv=ov;mi=oi;}}
        if(lid==0){bmv[blockIdx.x]=mv;bmi[blockIdx.x]=mi;}}
}
__global__ void pf_lm_reduce(const float *bmv, const int *bmi, int *out, int nb) {
    int tid=threadIdx.x; float best=-1e30f; int bi=-1;
    for(int i=tid;i<nb;i+=blockDim.x){float v=bmv[i];if(v>best){best=v;bi=bmi[i];}}
    __shared__ float sv[256]; __shared__ int si[256];
    sv[tid]=best;si[tid]=bi;__syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){if(tid<s&&sv[tid+s]>sv[tid]){sv[tid]=sv[tid+s];si[tid]=si[tid+s];}__syncthreads();}
    if(tid==0)*out=si[0];
}

// ===== V3: Chunk-parallel DeltaNet — phase 1 (intra-chunk, parallel) =====
// Launch: <<<dim3(N_CHUNKS, DN_HEADS), CHUNK_BLOCK>>>
// Per (chunk n, head h):
//   1. Load K[n, :, h, :], V[n, :, h, :] from f32 qkv_pre into shared.
//   2. Compute β_eff = sigmoid(beta_proj), g = -exp(a_log) * softplus(alpha_proj + dt_bias) for each position.
//   3. cs = cumsum(g). Write to global.
//   4. Build M[i,j] = β_eff[i] * exp(cs[i]-cs[j]) * (K[i]·K[j]) strict lower tri.
//   5. Initialize U = β * V, W = β * exp(cs) * K.
//   6. Forward substitute: u[i] = U[i] - Σ_{s<i} M[i,s] * u[s], same for w.
//   7. Write u, w to global.
// All math in f32. Output u_intra/w_intra are f32 [N, C, H, D].
constexpr int DN_CHUNK_C = 8;       // chunk size (last chunk may be partial)
constexpr int DN_CHUNK_BLOCK = 128; // threads per block

__global__ void pf_dn_chunk_phase1(
    const float *qkv_pre,        // [S, DN_CONV_CH] f32
    const float *beta_proj,      // [S, DN_HEADS] f32
    const float *alpha_proj,     // [S, DN_HEADS] f32
    const __nv_bfloat16 *a_log,
    const __nv_bfloat16 *dt_bias,
    float *u_out,                // [N, C, DN_HEADS, DN_VAL]
    float *w_out,                // [N, C, DN_HEADS, DN_KEY]
    float *cs_out,               // [N, C, DN_HEADS]
    int S)
{
    int n = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    int t_start = n * DN_CHUNK_C;

    // Alias buffers: K and w share storage, V and u share storage. After K is used for
    // building M and initializing w, we overwrite it with w. Same for V → u.
    __shared__ float s_K_w[DN_CHUNK_C * DN_KEY];
    __shared__ float s_V_u[DN_CHUNK_C * DN_VAL];
    __shared__ float s_beta[DN_CHUNK_C];
    __shared__ float s_cs[DN_CHUNK_C];
    __shared__ float s_M[DN_CHUNK_C * DN_CHUNK_C];
    float *s_K = s_K_w;
    float *s_V = s_V_u;
    float *s_u = s_V_u;
    float *s_w = s_K_w;

    float a_log_val = __bfloat162float(a_log[h]);
    float dt_b = __bfloat162float(dt_bias[h]);

    // Load K and V chunks
    for (int ci = tid; ci < DN_CHUNK_C * DN_KEY; ci += DN_CHUNK_BLOCK) {
        int c = ci / DN_KEY;
        int d = ci % DN_KEY;
        int t = t_start + c;
        s_K[ci] = (t < S) ? qkv_pre[t * DN_CONV_CH + DN_QK_SIZE + h * DN_KEY + d] : 0.f;
    }
    for (int ci = tid; ci < DN_CHUNK_C * DN_VAL; ci += DN_CHUNK_BLOCK) {
        int c = ci / DN_VAL;
        int d = ci % DN_VAL;
        int t = t_start + c;
        s_V[ci] = (t < S) ? qkv_pre[t * DN_CONV_CH + 2 * DN_QK_SIZE + h * DN_VAL + d] : 0.f;
    }

    // Compute beta_eff and g per chunk position (1 thread per position, C=8)
    if (tid < DN_CHUNK_C) {
        int t = t_start + tid;
        if (t < S) {
            s_beta[tid] = 1.f / (1.f + expf(-beta_proj[t * DN_HEADS + h]));
            float x = alpha_proj[t * DN_HEADS + h] + dt_b;
            float sp = (x > 20.f) ? x : logf(1.f + expf(x));
            s_cs[tid] = -expf(a_log_val) * sp;
        } else {
            s_beta[tid] = 0.f;
            s_cs[tid] = 0.f;
        }
    }
    __syncthreads();

    // Cumulative sum of g -> cs (sequential, thread 0, C small)
    if (tid == 0) {
        for (int i = 1; i < DN_CHUNK_C; i++) s_cs[i] += s_cs[i - 1];
        for (int i = 0; i < DN_CHUNK_C; i++) {
            int t = t_start + i;
            if (t < S) cs_out[(n * DN_CHUNK_C + i) * DN_HEADS + h] = s_cs[i];
        }
    }
    __syncthreads();

    // Compute M[i,j] for strict lower tri (j < i). Each thread handles one or more (i, j).
    for (int ij = tid; ij < DN_CHUNK_C * DN_CHUNK_C; ij += DN_CHUNK_BLOCK) {
        int i = ij / DN_CHUNK_C;
        int j = ij % DN_CHUNK_C;
        float val = 0.f;
        if (j < i) {
            float kk = 0.f;
            #pragma unroll
            for (int d = 0; d < DN_KEY; d++) {
                kk += s_K[i * DN_KEY + d] * s_K[j * DN_KEY + d];
            }
            val = s_beta[i] * expf(s_cs[i] - s_cs[j]) * kk;
        }
        s_M[ij] = val;
    }
    __syncthreads();

    // Initialize u = β * V, w = β * exp(cs) * K
    for (int ci = tid; ci < DN_CHUNK_C * DN_VAL; ci += DN_CHUNK_BLOCK) {
        int c = ci / DN_VAL;
        s_u[ci] = s_beta[c] * s_V[ci];
    }
    for (int ci = tid; ci < DN_CHUNK_C * DN_KEY; ci += DN_CHUNK_BLOCK) {
        int c = ci / DN_KEY;
        s_w[ci] = s_beta[c] * expf(s_cs[c]) * s_K[ci];
    }
    __syncthreads();

    // Forward substitute: u[i] -= Σ_{s<i} M[i,s] u[s]; same for w
    #pragma unroll
    for (int i = 1; i < DN_CHUNK_C; i++) {
        for (int d = tid; d < DN_VAL; d += DN_CHUNK_BLOCK) {
            float acc = 0.f;
            for (int s = 0; s < i; s++) {
                acc += s_M[i * DN_CHUNK_C + s] * s_u[s * DN_VAL + d];
            }
            s_u[i * DN_VAL + d] -= acc;
        }
        for (int d = tid; d < DN_KEY; d += DN_CHUNK_BLOCK) {
            float acc = 0.f;
            for (int s = 0; s < i; s++) {
                acc += s_M[i * DN_CHUNK_C + s] * s_w[s * DN_KEY + d];
            }
            s_w[i * DN_KEY + d] -= acc;
        }
        __syncthreads();
    }

    // Write u and w to global, layout [N, C, H, D]
    for (int ci = tid; ci < DN_CHUNK_C * DN_VAL; ci += DN_CHUNK_BLOCK) {
        int c = ci / DN_VAL;
        int d = ci % DN_VAL;
        int t = t_start + c;
        if (t < S) {
            u_out[((n * DN_CHUNK_C + c) * DN_HEADS + h) * DN_VAL + d] = s_u[ci];
        }
    }
    for (int ci = tid; ci < DN_CHUNK_C * DN_KEY; ci += DN_CHUNK_BLOCK) {
        int c = ci / DN_KEY;
        int d = ci % DN_KEY;
        int t = t_start + c;
        if (t < S) {
            w_out[((n * DN_CHUNK_C + c) * DN_HEADS + h) * DN_KEY + d] = s_w[ci];
        }
    }
}

// ===== V3: Chunk-parallel DeltaNet — phase 2 (inter-chunk, sequential per head) =====
// Launch: <<<dim3(DN_HEADS, J_SPLITS), PHASE2_BLOCK>>>
// Each block owns 1 head and a slice of DN_VAL rows (j). State slice in shared memory.
// Sequential loop over N chunks.
constexpr int DN_PHASE2_J_SPLITS = 4;                      // split DN_VAL across this many blocks per head
constexpr int DN_PHASE2_J_PER_BLOCK = DN_VAL / DN_PHASE2_J_SPLITS;   // 32
constexpr int DN_PHASE2_BLOCK = 128;                       // threads per block

__global__ void __launch_bounds__(DN_PHASE2_BLOCK, 1)
pf_dn_chunk_phase2(
    const float *u_in,           // [N, C, H, Dv]
    const float *w_in,           // [N, C, H, Dk]
    const float *cs_in,          // [N*C, H]
    const float *qkv_pre,        // [S, DN_CONV_CH]   (we need Q and K here, K is shared with phase1)
    float *state,                // [H, Dv, Dk] f32 — persistent across decode too
    __nv_bfloat16 *output,       // [S, Dv*H] bf16 (raw, before gated rmsnorm)
    int S, int N)
{
    int h = blockIdx.x;
    int js = blockIdx.y;
    int tid = threadIdx.x;
    int j_start = js * DN_PHASE2_J_PER_BLOCK;

    // Dynamic shared memory layout (+1 stride padding on Dk dim to avoid 32-way bank conflicts)
    constexpr int DK_S = DN_KEY + 1;   // 129
    extern __shared__ float smem[];
    float *s_state = smem;
    float *s_u     = s_state + DN_PHASE2_J_PER_BLOCK * DK_S;
    float *s_w     = s_u     + DN_CHUNK_C * DN_PHASE2_J_PER_BLOCK;
    float *s_Q     = s_w     + DN_CHUNK_C * DK_S;
    float *s_K     = s_Q     + DN_CHUNK_C * DK_S;
    float *s_d     = s_K     + DN_CHUNK_C * DK_S;
    float *s_qkt   = s_d     + DN_CHUNK_C * DN_PHASE2_J_PER_BLOCK;
    float *s_cs    = s_qkt   + DN_CHUNK_C * DN_CHUNK_C;
    float *s_decay_rem_buf = s_cs + DN_CHUNK_C;
    // (s_decay_total kept as a single __shared__ scalar below)

    // Load state slice for this head and j-range from global (pack into padded stride DK_S)
    for (int ji = tid; ji < DN_PHASE2_J_PER_BLOCK * DN_KEY; ji += DN_PHASE2_BLOCK) {
        int j = ji / DN_KEY;
        int i = ji % DN_KEY;
        s_state[j * DK_S + i] = state[((h * DN_VAL) + (j_start + j)) * DN_KEY + i];
    }
    __syncthreads();

    for (int n = 0; n < N; n++) {
        int t_start = n * DN_CHUNK_C;

        // Load u[n, :, h, j_start : j_start+J_per] -> s_u  [C, J_per]
        for (int ci = tid; ci < DN_CHUNK_C * DN_PHASE2_J_PER_BLOCK; ci += DN_PHASE2_BLOCK) {
            int c = ci / DN_PHASE2_J_PER_BLOCK;
            int j = ci % DN_PHASE2_J_PER_BLOCK;
            int t = t_start + c;
            if (t < S) {
                s_u[ci] = u_in[((n * DN_CHUNK_C + c) * DN_HEADS + h) * DN_VAL + j_start + j];
            } else {
                s_u[ci] = 0.f;
            }
        }
        // Load w[n, :, h, :] → s_w with padded stride DK_S
        for (int ci = tid; ci < DN_CHUNK_C * DN_KEY; ci += DN_PHASE2_BLOCK) {
            int c = ci / DN_KEY;
            int d = ci % DN_KEY;
            int t = t_start + c;
            if (t < S) {
                s_w[c * DK_S + d] = w_in[((n * DN_CHUNK_C + c) * DN_HEADS + h) * DN_KEY + d];
            } else {
                s_w[c * DK_S + d] = 0.f;
            }
        }
        // Load Q and K from qkv_pre → s_Q, s_K with padded stride DK_S
        for (int ci = tid; ci < DN_CHUNK_C * DN_KEY; ci += DN_PHASE2_BLOCK) {
            int c = ci / DN_KEY;
            int d = ci % DN_KEY;
            int t = t_start + c;
            if (t < S) {
                s_Q[c * DK_S + d] = qkv_pre[t * DN_CONV_CH + h * DN_KEY + d];
                s_K[c * DK_S + d] = qkv_pre[t * DN_CONV_CH + DN_QK_SIZE + h * DN_KEY + d];
            } else {
                s_Q[c * DK_S + d] = 0.f;
                s_K[c * DK_S + d] = 0.f;
            }
        }
        // Load cs for this chunk [C]
        if (tid < DN_CHUNK_C) {
            int t = t_start + tid;
            s_cs[tid] = (t < S) ? cs_in[(n * DN_CHUNK_C + tid) * DN_HEADS + h] : 0.f;
        }
        __syncthreads();

        // Compute d and QKt simultaneously (no cross-dependency → no sync needed between)
        // d[c, j] = u[c, j] - Σ_i w[c, i] * s_state[j, i]
        for (int ci = tid; ci < DN_CHUNK_C * DN_PHASE2_J_PER_BLOCK; ci += DN_PHASE2_BLOCK) {
            int c = ci / DN_PHASE2_J_PER_BLOCK;
            int j = ci % DN_PHASE2_J_PER_BLOCK;
            float acc = 0.f;
            #pragma unroll
            for (int i = 0; i < DN_KEY; i++) {
                acc += s_w[c * DK_S + i] * s_state[j * DK_S + i];
            }
            s_d[ci] = s_u[ci] - acc;
        }
        // QKt[c, s] = Q[c] @ K[s] (continues in same threadblock section, no sync between)
        for (int ij = tid; ij < DN_CHUNK_C * DN_CHUNK_C; ij += DN_PHASE2_BLOCK) {
            int c = ij / DN_CHUNK_C;
            int sp = ij % DN_CHUNK_C;
            float sum = 0.f;
            #pragma unroll
            for (int d = 0; d < DN_KEY; d++) {
                sum += s_Q[c * DK_S + d] * s_K[sp * DK_S + d];
            }
            s_qkt[ij] = sum;
        }
        __syncthreads();

        // Compute output o[c, j] = o_inter + o_intra
        //   o_inter[c, j] = exp(cs[c]) * (Q[c] · state[j, :])
        //   o_intra[c, j] = Σ_{s<=c} (QKt[c,s] * exp(cs[c]-cs[s])) * d[s, j]
        for (int ci = tid; ci < DN_CHUNK_C * DN_PHASE2_J_PER_BLOCK; ci += DN_PHASE2_BLOCK) {
            int c = ci / DN_PHASE2_J_PER_BLOCK;
            int j = ci % DN_PHASE2_J_PER_BLOCK;
            int t = t_start + c;
            if (t >= S) continue;

            // o_inter
            float qs = 0.f;
            #pragma unroll
            for (int i = 0; i < DN_KEY; i++) {
                qs += s_Q[c * DK_S + i] * s_state[j * DK_S + i];
            }
            float cs_c = s_cs[c];
            float o_inter = expf(cs_c) * qs;

            // o_intra: strictly-lower-plus-diag mask, s from 0..c
            float o_intra = 0.f;
            for (int sp = 0; sp <= c; sp++) {
                float l = expf(cs_c - s_cs[sp]);
                o_intra += s_qkt[c * DN_CHUNK_C + sp] * l * s_d[sp * DN_PHASE2_J_PER_BLOCK + j];
            }

            float o = o_inter + o_intra;
            output[t * DN_V_SIZE + h * DN_VAL + j_start + j] = __float2bfloat16(o);
        }
        __syncthreads();

        // Precompute decay_rem[c] = exp(cs_end - cs[c]) and decay_total = exp(cs_end) once.
        float cs_end = s_cs[DN_CHUNK_C - 1];
        __shared__ float s_decay_total_static;
        if (tid < DN_CHUNK_C) s_decay_rem_buf[tid] = expf(cs_end - s_cs[tid]);
        if (tid == 0) s_decay_total_static = expf(cs_end);
        __syncthreads();
        float s_decay_total = s_decay_total_static;

        // Premultiply d_scaled[c, j] = decay_rem[c] * d[c, j] (in-place on s_d is fine since d is done with)
        for (int ci = tid; ci < DN_CHUNK_C * DN_PHASE2_J_PER_BLOCK; ci += DN_PHASE2_BLOCK) {
            int c = ci / DN_PHASE2_J_PER_BLOCK;
            s_d[ci] *= s_decay_rem_buf[c];
        }
        __syncthreads();

        // State update: S[j, i] = decay_total * S[j, i] + Σ_c d_scaled[c, j] * K[c, i]
        for (int ji = tid; ji < DN_PHASE2_J_PER_BLOCK * DN_KEY; ji += DN_PHASE2_BLOCK) {
            int j = ji / DN_KEY;
            int i = ji % DN_KEY;
            int off = j * DK_S + i;
            float s_val = s_decay_total * s_state[off];
            #pragma unroll
            for (int c = 0; c < DN_CHUNK_C; c++) {
                s_val += s_d[c * DN_PHASE2_J_PER_BLOCK + j] * s_K[c * DK_S + i];
            }
            s_state[off] = s_val;
        }
        __syncthreads();
    }

    // Write state slice back to global
    for (int ji = tid; ji < DN_PHASE2_J_PER_BLOCK * DN_KEY; ji += DN_PHASE2_BLOCK) {
        int j = ji / DN_KEY;
        int i = ji % DN_KEY;
        state[((h * DN_VAL) + (j_start + j)) * DN_KEY + i] = s_state[j * DK_S + i];
    }
}

// ===== V3: Fused QK norm + RoPE + KV cache (single fused QKV buffer) =====
// The full attention Q/K/V live in one fused buffer with row stride (FA_QPROJ_SIZE + 2*FA_KV_SIZE).
// Q occupies cols [0, FA_QPROJ_SIZE), K cols [FA_QPROJ_SIZE, FA_QPROJ_SIZE+FA_KV_SIZE), V the rest.
__global__ void pf_qk_norm_rope_fused(
    __nv_bfloat16 *qkv_fused,
    const __nv_bfloat16 *qnw, const __nv_bfloat16 *knw,
    __nv_bfloat16 *k_cache, __nv_bfloat16 *v_cache, int S, int max_seq)
{
    constexpr int STRIDE = FA_QPROJ_SIZE + 2*FA_KV_SIZE;
    constexpr int K_COL = FA_QPROJ_SIZE;
    constexpr int V_COL = FA_QPROJ_SIZE + FA_KV_SIZE;
    int idx = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lid = threadIdx.x % 32;
    int total_q = S * FA_Q_HEADS, total_k = S * FA_KV_HEADS;
    if (idx < total_q) {
        int pos = idx / FA_Q_HEADS, head = idx % FA_Q_HEADS;
        __nv_bfloat16 *qh = qkv_fused + pos * STRIDE + head * FA_HEAD_DIM * 2;
        float ss = 0; for (int i = lid; i < FA_HEAD_DIM; i += 32) { float v = __bfloat162float(qh[i]); ss += v*v; }
        ss = pf_warp_sum(ss); float sc = rsqrtf(ss/FA_HEAD_DIM+RMS_EPS); sc = __shfl_sync(0xffffffff,sc,0);
        for (int i = lid; i < FA_HEAD_DIM; i += 32) {
            float normed = __bfloat162float(qh[i])*sc*(1.f+__bfloat162float(qnw[i]));
            if (i < FA_ROT_DIM) {
                float fe=float(2*(i%(FA_ROT_DIM/2)))/FA_ROT_DIM; float freq=float(pos)/powf(FA_ROPE_THETA,fe);
                float cv=cosf(freq),sv=sinf(freq); int p=(i<FA_ROT_DIM/2)?i+FA_ROT_DIM/2:i-FA_ROT_DIM/2;
                float pv=__bfloat162float(qh[p])*sc*(1.f+__bfloat162float(qnw[p]));
                qh[i]=__float2bfloat16((i<FA_ROT_DIM/2)?(normed*cv-pv*sv):(pv*sv+normed*cv));
            } else qh[i]=__float2bfloat16(normed);
        }
    }
    int kidx = idx - total_q;
    if (idx >= total_q && kidx < total_k) {
        int pos = kidx / FA_KV_HEADS, head = kidx % FA_KV_HEADS;
        __nv_bfloat16 *kh = qkv_fused + pos * STRIDE + K_COL + head * FA_HEAD_DIM;
        const __nv_bfloat16 *vh = qkv_fused + pos * STRIDE + V_COL + head * FA_HEAD_DIM;
        __nv_bfloat16 *kc = k_cache + head*max_seq*FA_HEAD_DIM + pos*FA_HEAD_DIM;
        __nv_bfloat16 *vc = v_cache + head*max_seq*FA_HEAD_DIM + pos*FA_HEAD_DIM;
        float ss = 0; for (int i = lid; i < FA_HEAD_DIM; i += 32) { float v = __bfloat162float(kh[i]); ss += v*v; }
        ss = pf_warp_sum(ss); float sc = rsqrtf(ss/FA_HEAD_DIM+RMS_EPS); sc = __shfl_sync(0xffffffff,sc,0);
        for (int i = lid; i < FA_HEAD_DIM; i += 32) {
            float normed = __bfloat162float(kh[i])*sc*(1.f+__bfloat162float(knw[i])); float fk;
            if (i < FA_ROT_DIM) {
                float fe=float(2*(i%(FA_ROT_DIM/2)))/FA_ROT_DIM; float freq=float(pos)/powf(FA_ROPE_THETA,fe);
                float cv=cosf(freq),sv=sinf(freq); int p=(i<FA_ROT_DIM/2)?i+FA_ROT_DIM/2:i-FA_ROT_DIM/2;
                float pv=__bfloat162float(kh[p])*sc*(1.f+__bfloat162float(knw[p]));
                fk=(i<FA_ROT_DIM/2)?(normed*cv-pv*sv):(pv*sv+normed*cv);
            } else fk=normed;
            kh[i]=__float2bfloat16(fk); kc[i]=__float2bfloat16(fk); vc[i]=vh[i];
        }
    }
}

// ===== V3: Causal attention over fused QKV buffer =====
__global__ void pf_causal_attn_fused(const __nv_bfloat16 *qkv_fused, __nv_bfloat16 *out, int S)
{
    constexpr int STRIDE = FA_QPROJ_SIZE + 2*FA_KV_SIZE;
    constexpr int K_COL = FA_QPROJ_SIZE;
    constexpr int V_COL = FA_QPROJ_SIZE + FA_KV_SIZE;
    int idx = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lid = threadIdx.x % 32;
    if (idx >= S * FA_Q_HEADS) return;
    int pos = idx / FA_Q_HEADS, qh = idx % FA_Q_HEADS, kvh = qh / FA_GQA;
    float scale = 1.0f / sqrtf(float(FA_HEAD_DIM));
    constexpr int EPL = FA_HEAD_DIM / 32;
    const __nv_bfloat16 *qv = qkv_fused + pos*STRIDE + qh*FA_HEAD_DIM*2;
    const __nv_bfloat16 *gv = qv + FA_HEAD_DIM;
    __nv_bfloat16 *ov = out + pos*FA_Q_SIZE + qh*FA_HEAD_DIM;
    float ql[EPL]; for(int e=0;e<EPL;e++) ql[e]=__bfloat162float(qv[lid*EPL+e]);
    float oa[EPL]={}; float mx=-1e30f, se=0;
    for (int kp = 0; kp <= pos; kp++) {
        const __nv_bfloat16 *kv=qkv_fused+kp*STRIDE+K_COL+kvh*FA_HEAD_DIM;
        const __nv_bfloat16 *vv=qkv_fused+kp*STRIDE+V_COL+kvh*FA_HEAD_DIM;
        float sc=0; for(int e=0;e<EPL;e++) sc+=ql[e]*__bfloat162float(kv[lid*EPL+e]);
        sc=pf_warp_sum(sc)*scale; sc=__shfl_sync(0xffffffff,sc,0);
        float om=mx; mx=fmaxf(mx,sc); float ed=expf(om-mx); se=se*ed+expf(sc-mx);
        float wt=expf(sc-mx); for(int e=0;e<EPL;e++) oa[e]=oa[e]*ed+wt*__bfloat162float(vv[lid*EPL+e]);
    }
    float rs=1.f/se;
    for(int e=0;e<EPL;e++){int i=lid*EPL+e;float g=1.f/(1.f+expf(-__bfloat162float(gv[i])));ov[i]=__float2bfloat16(oa[e]*rs*g);}
}

__global__ void pf_causal_softmax_to_bf16(
    const float *scores,
    __nv_bfloat16 *probs,
    int q_start,
    int rows,
    int key_count,
    int stride)
{
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int wid = tid / 32;
    int lid = tid % 32;
    int q_pos = q_start + row;
    const float scale = 1.0f / sqrtf(float(FA_HEAD_DIM));
    const float *score_row = scores + row * stride;
    __nv_bfloat16 *prob_row = probs + row * stride;

    float local_max = -3.402823466e+38f;
    for (int k = tid; k < key_count; k += blockDim.x) {
        const float score = (k <= q_pos) ? (score_row[k] * scale) : -3.402823466e+38f;
        local_max = fmaxf(local_max, score);
    }
    local_max = pf_warp_max(local_max);
    __shared__ float warp_vals[32];
    if (lid == 0) warp_vals[wid] = local_max;
    __syncthreads();
    float row_max = -3.402823466e+38f;
    if (wid == 0) {
        row_max = (lid < blockDim.x / 32) ? warp_vals[lid] : -3.402823466e+38f;
        row_max = pf_warp_max(row_max);
        if (lid == 0) warp_vals[0] = row_max;
    }
    __syncthreads();
    row_max = warp_vals[0];

    float local_sum = 0.0f;
    for (int k = tid; k < key_count; k += blockDim.x) {
        if (k <= q_pos) {
            local_sum += expf(score_row[k] * scale - row_max);
        }
    }
    local_sum = pf_warp_sum(local_sum);
    if (lid == 0) warp_vals[wid] = local_sum;
    __syncthreads();
    float row_sum = 0.0f;
    if (wid == 0) {
        row_sum = (lid < blockDim.x / 32) ? warp_vals[lid] : 0.0f;
        row_sum = pf_warp_sum(row_sum);
        if (lid == 0) warp_vals[0] = row_sum;
    }
    __syncthreads();
    row_sum = warp_vals[0];
    const float inv_sum = row_sum > 0.0f ? (1.0f / row_sum) : 0.0f;

    for (int k = tid; k < key_count; k += blockDim.x) {
        float p = 0.0f;
        if (k <= q_pos) {
            p = expf(score_row[k] * scale - row_max) * inv_sum;
        }
        prob_row[k] = __float2bfloat16(p);
    }
}

__global__ void pf_apply_attention_gate_bf16_fused(
    const float *attn,
    const __nv_bfloat16 *qkv_fused,
    __nv_bfloat16 *out,
    int q_start,
    int rows,
    int q_head)
{
    constexpr int STRIDE = FA_QPROJ_SIZE + 2*FA_KV_SIZE;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * FA_HEAD_DIM;
    if (idx >= total) return;
    int row = idx / FA_HEAD_DIM;
    int d = idx % FA_HEAD_DIM;
    const __nv_bfloat16 *gate = qkv_fused + (q_start + row) * STRIDE + q_head * FA_HEAD_DIM * 2 + FA_HEAD_DIM;
    __nv_bfloat16 *out_row = out + (q_start + row) * FA_Q_SIZE + q_head * FA_HEAD_DIM;
    float g = 1.0f / (1.0f + expf(-__bfloat162float(gate[d])));
    out_row[d] = __float2bfloat16(attn[row * FA_HEAD_DIM + d] * g);
}

static void pf_causal_attn_fused_tiled_cublas(
    cublasHandle_t h,
    const __nv_bfloat16 *qkv_fused,
    __nv_bfloat16 *prob_scratch,
    float *score_scratch,
    float *attn_scratch,
    __nv_bfloat16 *out,
    int S,
    int query_block_tokens,
    cudaStream_t stream)
{
    constexpr int STRIDE = FA_QPROJ_SIZE + 2*FA_KV_SIZE;
    constexpr int K_COL = FA_QPROJ_SIZE;
    constexpr int V_COL = FA_QPROJ_SIZE + FA_KV_SIZE;
    query_block_tokens = max(1, min(query_block_tokens, S));

    for (int qh = 0; qh < FA_Q_HEADS; ++qh) {
        const int kvh = qh / FA_GQA;
        const __nv_bfloat16 *k_head = qkv_fused + K_COL + kvh * FA_HEAD_DIM;
        const __nv_bfloat16 *v_head = qkv_fused + V_COL + kvh * FA_HEAD_DIM;

        for (int q0 = 0; q0 < S; q0 += query_block_tokens) {
            const int rows = min(query_block_tokens, S - q0);
            const int key_count = q0 + rows;
            const __nv_bfloat16 *q_head = qkv_fused + q0 * STRIDE + qh * FA_HEAD_DIM * 2;

            cublas_bf16_qk_scores(h, q_head, STRIDE, k_head, STRIDE, score_scratch, rows, key_count, FA_HEAD_DIM);
            pf_causal_softmax_to_bf16<<<rows, 512, 0, stream>>>(
                score_scratch, prob_scratch, q0, rows, key_count, key_count);
            cublas_bf16_probs_v(h, prob_scratch, key_count, v_head, STRIDE, attn_scratch, rows, key_count, FA_HEAD_DIM);
            pf_apply_attention_gate_bf16_fused<<<(rows * FA_HEAD_DIM + 255) / 256, 256, 0, stream>>>(
                attn_scratch, qkv_fused, out, q0, rows, qh);
        }
    }
}

// ===== V3: Fused SiLU(gate)*up from concatenated [S, 2*N] buffer =====
__global__ void pf_silu_mul_fused(const __nv_bfloat16 *fused, __nv_bfloat16 *out, int S, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= S * N) return;
    int s = idx / N;
    int n = idx % N;
    float gate_val = __bfloat162float(fused[s * 2 * N + n]);
    float up_val   = __bfloat162float(fused[s * 2 * N + N + n]);
    out[idx] = __float2bfloat16(pf_silu(gate_val) * up_val);
}

static void pf_mlp_chunked_fused(
    cublasHandle_t cublas,
    const __nv_bfloat16 *normalized,
    const __nv_bfloat16 *residual,
    __nv_bfloat16 *hidden,
    const __nv_bfloat16 *gate_up_w,
    const __nv_bfloat16 *down_w,
    __nv_bfloat16 *gate_up_buf,
    __nv_bfloat16 *mlp_buf,
    int S,
    int chunk_tokens,
    cudaStream_t stream)
{
    chunk_tokens = max(1, min(chunk_tokens, S));
    for (int offset = 0; offset < S; offset += chunk_tokens) {
        const int rows = min(chunk_tokens, S - offset);
        const __nv_bfloat16 *norm_chunk = normalized + static_cast<size_t>(offset) * HIDDEN;
        const __nv_bfloat16 *residual_chunk = residual + static_cast<size_t>(offset) * HIDDEN;
        __nv_bfloat16 *hidden_chunk = hidden + static_cast<size_t>(offset) * HIDDEN;

        cublas_bf16_gemm(cublas, norm_chunk, gate_up_w, gate_up_buf, rows, 2 * INTER, HIDDEN);
        pf_silu_mul_fused<<<(rows * INTER + 255) / 256, 256, 0, stream>>>(gate_up_buf, mlp_buf, rows, INTER);
        cublas_bf16_gemm(cublas, mlp_buf, down_w, gate_up_buf, rows, HIDDEN, INTER);
        pf_add_residual_bf16<<<(rows * HIDDEN + 255) / 256, 256, 0, stream>>>(
            gate_up_buf, residual_chunk, hidden_chunk, rows * HIDDEN);
    }
}

// ===== V3: Parallel pre-projection of DeltaNet Q/K/V =====
// Computes conv1d + SiLU for all (t, h, channel) in parallel, plus L2-norm for Q/K.
// Reads initial conv_buf for t<3. Does NOT update conv_buf — pf_deltanet_update_conv_buf does.
// Launch: <<<dim3(S, DN_HEADS), 128>>>. Output: f32 qkv_pre[S, DN_CONV_CH] row-major.
__global__ void pf_deltanet_preproject(
    const __nv_bfloat16 *qkv_proj,   // [S, DN_CONV_CH] bf16
    const __nv_bfloat16 *conv_w,     // [DN_CONV_CH, DN_CONV_K] bf16
    const float *conv_buf_init,      // [DN_CONV_CH, DN_CONV_K] f32
    float *qkv_pre,                  // [S, DN_CONV_CH] f32 output
    int S)
{
    int t = blockIdx.x;
    int h = blockIdx.y;
    if (t >= S) return;
    int tid = threadIdx.x, wid = tid/32, lid = tid%32;
    constexpr int NTHREADS = 128;
    constexpr int NWARPS = NTHREADS / 32;  // 4
    constexpr float Q_SCALE = 1.0f / 11.313708498984761f;

    __shared__ float s_q[DN_KEY];
    __shared__ float s_k[DN_KEY];
    __shared__ float s_v[DN_VAL];
    __shared__ float smem[NWARPS + 1];

    auto conv1d_silu = [&](int ch) -> float {
        float val = 0;
        #pragma unroll
        for (int k = 0; k < DN_CONV_K; k++) {
            int src_t = t - (DN_CONV_K - 1) + k;  // t-3, t-2, t-1, t
            float x;
            if (src_t >= 0) {
                x = __bfloat162float(qkv_proj[src_t * DN_CONV_CH + ch]);
            } else {
                // Initial conv_buf: slot (src_t + DN_CONV_K) holds in(src_t) from caller
                x = conv_buf_init[ch * DN_CONV_K + (src_t + DN_CONV_K)];
            }
            val += x * __bfloat162float(conv_w[ch * DN_CONV_K + k]);
        }
        return pf_silu(val);
    };

    // Q conv1d
    for (int c = tid; c < DN_KEY; c += NTHREADS) {
        s_q[c] = conv1d_silu(h * DN_KEY + c);
    }
    // K conv1d
    for (int c = tid; c < DN_KEY; c += NTHREADS) {
        s_k[c] = conv1d_silu(DN_QK_SIZE + h * DN_KEY + c);
    }
    // V conv1d
    for (int c = tid; c < DN_VAL; c += NTHREADS) {
        s_v[c] = conv1d_silu(2 * DN_QK_SIZE + h * DN_VAL + c);
    }
    __syncthreads();

    // L2 norm Q (full-block reduction)
    float sq = 0;
    for (int i = tid; i < DN_KEY; i += NTHREADS) sq += s_q[i] * s_q[i];
    sq = pf_warp_sum(sq);
    if (lid == 0) smem[wid] = sq;
    __syncthreads();
    if (wid == 0) {
        float v = (lid < NWARPS) ? smem[lid] : 0;
        v = pf_warp_sum(v);
        if (lid == 0) smem[NWARPS] = rsqrtf(v + 1e-6f) * Q_SCALE;
    }
    __syncthreads();
    float q_norm = smem[NWARPS];
    for (int i = tid; i < DN_KEY; i += NTHREADS) s_q[i] *= q_norm;

    // L2 norm K (full-block reduction)
    float sk = 0;
    for (int i = tid; i < DN_KEY; i += NTHREADS) sk += s_k[i] * s_k[i];
    sk = pf_warp_sum(sk);
    if (lid == 0) smem[wid] = sk;
    __syncthreads();
    if (wid == 0) {
        float v = (lid < NWARPS) ? smem[lid] : 0;
        v = pf_warp_sum(v);
        if (lid == 0) smem[NWARPS] = rsqrtf(v + 1e-6f);
    }
    __syncthreads();
    float k_norm = smem[NWARPS];
    for (int i = tid; i < DN_KEY; i += NTHREADS) s_k[i] *= k_norm;
    __syncthreads();

    // Write to qkv_pre (f32)
    float *out_t = qkv_pre + t * DN_CONV_CH;
    for (int c = tid; c < DN_KEY; c += NTHREADS) out_t[h * DN_KEY + c] = s_q[c];
    for (int c = tid; c < DN_KEY; c += NTHREADS) out_t[DN_QK_SIZE + h * DN_KEY + c] = s_k[c];
    for (int c = tid; c < DN_VAL; c += NTHREADS) out_t[2 * DN_QK_SIZE + h * DN_VAL + c] = s_v[c];
}

// ===== V3: Update conv_buf to final state after prefill =====
// Final conv_buf at position S is [in(S-4), in(S-3), in(S-2), in(S-1)].
// Each thread owns one channel; reads all 4 values (from qkv_proj or initial conv_buf) then writes.
__global__ void pf_deltanet_update_conv_buf(
    const __nv_bfloat16 *qkv_proj, float *conv_buf, int S)
{
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= DN_CONV_CH) return;
    float new_cb[DN_CONV_K];
    #pragma unroll
    for (int k = 0; k < DN_CONV_K; k++) {
        int src_t = S - DN_CONV_K + k;  // S-4, S-3, S-2, S-1
        float v;
        if (src_t >= 0) {
            v = __bfloat162float(qkv_proj[src_t * DN_CONV_CH + ch]);
        } else {
            // Use initial conv_buf: slot (src_t + DN_CONV_K) = in(src_t)
            v = conv_buf[ch * DN_CONV_K + (src_t + DN_CONV_K)];
        }
        new_cb[k] = v;
    }
    #pragma unroll
    for (int k = 0; k < DN_CONV_K; k++) {
        conv_buf[ch * DN_CONV_K + k] = new_cb[k];
    }
}

// ===== V2: Split-j DeltaNet recurrence =====
// Launch: <<<dim3(DN_HEADS, SPLIT), 128, 0, stream>>>
// Each block owns J_PER_BLOCK=DN_VAL/SPLIT j-channels of the state.
// Conv_buf kept in shared memory; block (h,0) writes Q/K back, each block writes own V slice.
// Gated RMSNorm is pulled out into a separate kernel (pf_deltanet_gated_rmsnorm).
constexpr int DN_SPLIT = 16;
constexpr int DN_J_PER_BLOCK = DN_VAL / DN_SPLIT;  // 8
constexpr int DN_V2_BLOCK = 128;
constexpr int DN_V2_NWARPS = DN_V2_BLOCK / 32;     // 4

__global__ void __launch_bounds__(DN_V2_BLOCK, 8)
pf_deltanet_recurrence_split(
    const float *qkv_pre,            // [S, DN_CONV_CH] f32 (post conv+silu+L2norm)
    const float *beta_proj, const float *alpha_proj,
    const __nv_bfloat16 *a_log, const __nv_bfloat16 *dt_bias,
    float *state, __nv_bfloat16 *output, int S)
{
    int h = blockIdx.x;
    int split_idx = blockIdx.y;
    int tid = threadIdx.x, wid = tid/32, lid = tid%32;
    constexpr int CPW = DN_J_PER_BLOCK / DN_V2_NWARPS;
    constexpr int RPL = DN_KEY / 32;
    int jstart = split_idx * DN_J_PER_BLOCK;

    float a_log_val = __bfloat162float(a_log[h]);
    float dt_b = __bfloat162float(dt_bias[h]);

    __shared__ float s_q[DN_KEY], s_k[DN_KEY];
    __shared__ float s_v[DN_J_PER_BLOCK];
    __shared__ float s_beta, s_decay;

    // Load state slice into registers
    float sreg[CPW * RPL];
    float *my_state = state + h * DN_KEY * DN_VAL;
    #pragma unroll
    for (int jj = 0; jj < CPW; jj++) {
        int j = jstart + wid * CPW + jj;
        #pragma unroll
        for (int ii = 0; ii < RPL; ii++)
            sreg[jj*RPL+ii] = my_state[j*DN_KEY + lid + ii*32];
    }

    for (int t = 0; t < S; t++) {
        const float *qkv_t = qkv_pre + t * DN_CONV_CH;
        for (int c = tid; c < DN_KEY; c += DN_V2_BLOCK)
            s_q[c] = qkv_t[h * DN_KEY + c];
        for (int c = tid; c < DN_KEY; c += DN_V2_BLOCK)
            s_k[c] = qkv_t[DN_QK_SIZE + h * DN_KEY + c];
        for (int c = tid; c < DN_J_PER_BLOCK; c += DN_V2_BLOCK)
            s_v[c] = qkv_t[2 * DN_QK_SIZE + h * DN_VAL + jstart + c];

        if (tid == 0) {
            s_beta = 1.f/(1.f+expf(-beta_proj[t*DN_HEADS+h]));
            float x = alpha_proj[t*DN_HEADS+h] + dt_b;
            float sp = (x > 20.f) ? x : logf(1.f+expf(x));
            s_decay = expf(-expf(a_log_val)*sp);
        }
        __syncthreads();
        float beta = s_beta, decay = s_decay;
        __nv_bfloat16 *out_h = output + t * DN_V_SIZE + h * DN_VAL;

        #pragma unroll
        for (int jj = 0; jj < CPW; jj++) {
            int j_local = wid * CPW + jj;
            int j = jstart + j_local;
            float kv = 0;
            #pragma unroll
            for (int ii = 0; ii < RPL; ii++) kv += sreg[jj*RPL+ii] * s_k[lid + ii*32];
            kv = pf_warp_sum(kv);
            kv = __shfl_sync(0xffffffff, kv, 0);
            float delta = (s_v[j_local] - decay * kv) * beta;
            float attn = 0;
            #pragma unroll
            for (int ii = 0; ii < RPL; ii++) {
                sreg[jj*RPL+ii] = decay * sreg[jj*RPL+ii] + s_k[lid + ii*32] * delta;
                attn += sreg[jj*RPL+ii] * s_q[lid + ii*32];
            }
            attn = pf_warp_sum(attn);
            if (lid == 0) out_h[j] = __float2bfloat16(attn);
        }
        __syncthreads();
    }

    // Write state slice back
    #pragma unroll
    for (int jj = 0; jj < CPW; jj++) {
        int j = jstart + wid * CPW + jj;
        #pragma unroll
        for (int ii = 0; ii < RPL; ii++)
            my_state[j*DN_KEY + lid + ii*32] = sreg[jj*RPL+ii];
    }
}

// ===== V2: Gated RMSNorm pulled out of the recurrence =====
// Launch: <<<dim3(S, DN_HEADS), 128, 0, stream>>>
// Reads raw per-head attn output, applies RMSNorm with z-gating, writes back in place.
__global__ void pf_deltanet_gated_rmsnorm(
    __nv_bfloat16 *out, const __nv_bfloat16 *z_proj, const __nv_bfloat16 *norm_w, int S)
{
    int t = blockIdx.x;
    int h = blockIdx.y;
    if (t >= S) return;
    int tid = threadIdx.x, wid = tid/32, lid = tid%32;
    __shared__ float smem[4];
    __nv_bfloat16 *out_h = out + t * DN_V_SIZE + h * DN_VAL;
    const __nv_bfloat16 *z_h = z_proj + t * DN_V_SIZE + h * DN_VAL;

    float sq = 0;
    for (int i = tid; i < DN_VAL; i += blockDim.x) {
        float v = __bfloat162float(out_h[i]);
        sq += v*v;
    }
    sq = pf_warp_sum(sq);
    if (lid == 0) smem[wid] = sq;
    __syncthreads();
    if (wid == 0) {
        float v = (lid < (blockDim.x/32)) ? smem[lid] : 0;
        v = pf_warp_sum(v);
        if (lid == 0) smem[0] = rsqrtf(v/DN_VAL + RMS_EPS);
    }
    __syncthreads();
    float rstd = smem[0];
    for (int i = tid; i < DN_VAL; i += blockDim.x) {
        float n = __bfloat162float(out_h[i]) * rstd * __bfloat162float(norm_w[i]);
        out_h[i] = __float2bfloat16(n * pf_silu(__bfloat162float(z_h[i])));
    }
}

// ===== cuBLAS bf16 GEMM =====
static void cublas_bf16_gemm(cublasHandle_t h,
    const __nv_bfloat16 *A, const __nv_bfloat16 *B, __nv_bfloat16 *C,
    int S, int N, int K) {
    float alpha = 1.0f, beta_val = 0.0f;
    cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, S, K,
        &alpha, B, CUDA_R_16BF, K, A, CUDA_R_16BF, K,
        &beta_val, C, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

static void cublas_bf16_qk_scores(cublasHandle_t h,
    const __nv_bfloat16 *q, int q_stride,
    const __nv_bfloat16 *k, int k_stride,
    float *scores, int rows, int key_count, int dim) {
    float alpha = 1.0f, beta_val = 0.0f;
    cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, key_count, rows, dim,
        &alpha, k, CUDA_R_16BF, k_stride, q, CUDA_R_16BF, q_stride,
        &beta_val, scores, CUDA_R_32F, key_count,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

static void cublas_bf16_probs_v(cublasHandle_t h,
    const __nv_bfloat16 *probs, int prob_stride,
    const __nv_bfloat16 *v, int v_stride,
    float *out, int rows, int key_count, int dim) {
    float alpha = 1.0f, beta_val = 0.0f;
    cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, dim, rows, key_count,
        &alpha, v, CUDA_R_16BF, v_stride, probs, CUDA_R_16BF, prob_stride,
        &beta_val, out, CUDA_R_32F, dim,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

// ===== Main orchestrator =====
extern "C" void launch_prefill_bf16(
    const int *token_ids, int seq_len, int *output_token,
    const __nv_bfloat16 *embed_weight, const PFLayerWeights *layers,
    const __nv_bfloat16 *final_norm_w, const __nv_bfloat16 *lm_head_w,
    __nv_bfloat16 *fa_k_cache, __nv_bfloat16 *fa_v_cache,
    float *dn_states, float *conv_bufs,
    __nv_bfloat16 *hidden, __nv_bfloat16 *residual, __nv_bfloat16 *normalized,
    __nv_bfloat16 *proj_buf, __nv_bfloat16 *proj_buf2,
    __nv_bfloat16 *attn_buf, __nv_bfloat16 *mlp_buf,
    __nv_bfloat16 *dn_out_buf,
    float *beta_buf, float *alpha_buf, float *dn_pre_qkv,
    float *dn_u_scratch, float *dn_w_scratch, float *dn_cs_scratch,
    const __nv_bfloat16 *fused_fa_qkv_base,
    const __nv_bfloat16 *fused_gate_up_base,
    __nv_bfloat16 *final_normed, __nv_bfloat16 *hidden_bf16_out,
    float *lm_bmv, int *lm_bmi,
    int max_seq_len,
    cudaStream_t stream)
{
    static cublasHandle_t cublas = nullptr;
    if (!cublas) cublasCreate(&cublas);
    cublasSetStream(cublas, stream);

    static PFLayerWeights hl_v2[NUM_LAYERS];
    static bool copied_v2 = false;
    if (!copied_v2) { cudaMemcpy(hl_v2, layers, NUM_LAYERS*sizeof(PFLayerWeights), cudaMemcpyDeviceToHost); copied_v2 = true; }

    int S = seq_len;
    int bk = (S*HIDDEN+255)/256;

    pf_embed<<<bk, 256, 0, stream>>>(token_ids, embed_weight, hidden, S);

    int fa_stride = FA_KV_HEADS * max_seq_len * FA_HEAD_DIM;
    int dn_stride = DN_HEADS * DN_KEY * DN_VAL;
    int fa_idx = 0, dn_idx = 0;

    for (int li = 0; li < NUM_LAYERS; li++) {
        const PFLayerWeights &lw = hl_v2[li];
        int lt = LAYER_TYPE[li];

        const __nv_bfloat16 *norm_w = (const __nv_bfloat16 *)lw.ptrs[0];
        pf_rmsnorm<<<S, 512, 0, stream>>>(hidden, norm_w, normalized, residual, S, HIDDEN);

        if (lt == 0) {
            const __nv_bfloat16 *qkv_w=(const __nv_bfloat16*)lw.ptrs[1];
            const __nv_bfloat16 *z_w=(const __nv_bfloat16*)lw.ptrs[2];
            const __nv_bfloat16 *beta_w=(const __nv_bfloat16*)lw.ptrs[3];
            const __nv_bfloat16 *alpha_w=(const __nv_bfloat16*)lw.ptrs[4];
            const __nv_bfloat16 *conv_w=(const __nv_bfloat16*)lw.ptrs[5];
            const __nv_bfloat16 *a_log=(const __nv_bfloat16*)lw.ptrs[6];
            const __nv_bfloat16 *dt_bias=(const __nv_bfloat16*)lw.ptrs[7];
            const __nv_bfloat16 *dn_norm=(const __nv_bfloat16*)lw.ptrs[8];
            const __nv_bfloat16 *out_w=(const __nv_bfloat16*)lw.ptrs[9];
            const __nv_bfloat16 *post_norm=(const __nv_bfloat16*)lw.ptrs[10];
            const __nv_bfloat16 *down_w=(const __nv_bfloat16*)lw.ptrs[13];

            cublas_bf16_gemm(cublas, normalized, qkv_w, proj_buf, S, DN_CONV_CH, HIDDEN);
            cublas_bf16_gemm(cublas, normalized, z_w, proj_buf2, S, DN_V_SIZE, HIDDEN);
            pf_bf16_matvec<<<S*DN_HEADS, 32, 0, stream>>>(normalized, beta_w, beta_buf, S, HIDDEN, DN_HEADS);
            pf_bf16_matvec<<<S*DN_HEADS, 32, 0, stream>>>(normalized, alpha_w, alpha_buf, S, HIDDEN, DN_HEADS);

            // V3: parallel pre-projection (conv1d + silu + L2 norm)
            float *layer_conv_buf = conv_bufs + dn_idx*DN_CONV_CH*DN_CONV_K;
            dim3 pre_grid(S, DN_HEADS);
            pf_deltanet_preproject<<<pre_grid, 128, 0, stream>>>(
                proj_buf, conv_w, layer_conv_buf, dn_pre_qkv, S);

            // V4: chunk-parallel DeltaNet — phase 1 (intra-chunk, parallel)
            int N_chunks = (S + DN_CHUNK_C - 1) / DN_CHUNK_C;
            dim3 p1_grid(N_chunks, DN_HEADS);
            pf_dn_chunk_phase1<<<p1_grid, DN_CHUNK_BLOCK, 0, stream>>>(
                dn_pre_qkv, beta_buf, alpha_buf, a_log, dt_bias,
                dn_u_scratch, dn_w_scratch, dn_cs_scratch, S);

            // V4: chunk-parallel DeltaNet — phase 2 (inter-chunk, sequential per head)
            // Compute dynamic shared memory size once.
            constexpr int DK_S = DN_KEY + 1;
            constexpr size_t P2_SMEM_FLOATS =
                DN_PHASE2_J_PER_BLOCK * DK_S         // s_state
                + DN_CHUNK_C * DN_PHASE2_J_PER_BLOCK // s_u
                + DN_CHUNK_C * DK_S                  // s_w
                + DN_CHUNK_C * DK_S                  // s_Q
                + DN_CHUNK_C * DK_S                  // s_K
                + DN_CHUNK_C * DN_PHASE2_J_PER_BLOCK // s_d
                + DN_CHUNK_C * DN_CHUNK_C            // s_qkt
                + DN_CHUNK_C                         // s_cs
                + DN_CHUNK_C;                        // s_decay_rem_buf
            constexpr size_t P2_SMEM_BYTES = P2_SMEM_FLOATS * sizeof(float);
            // Opt into >48KB shared (Ampere supports up to 100KB per block) — call once.
            static bool phase2_opted_in = false;
            if (!phase2_opted_in) {
                cudaFuncSetAttribute(pf_dn_chunk_phase2,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)P2_SMEM_BYTES);
                phase2_opted_in = true;
            }
            dim3 p2_grid(DN_HEADS, DN_PHASE2_J_SPLITS);
            pf_dn_chunk_phase2<<<p2_grid, DN_PHASE2_BLOCK, P2_SMEM_BYTES, stream>>>(
                dn_u_scratch, dn_w_scratch, dn_cs_scratch, dn_pre_qkv,
                dn_states + dn_idx*dn_stride, dn_out_buf, S, N_chunks);

            // V3: update conv_buf final state from qkv_proj last 4 positions
            int cb_blocks = (DN_CONV_CH + 127) / 128;
            pf_deltanet_update_conv_buf<<<cb_blocks, 128, 0, stream>>>(
                proj_buf, layer_conv_buf, S);

            // Separate gated rmsnorm kernel
            dim3 norm_grid(S, DN_HEADS);
            pf_deltanet_gated_rmsnorm<<<norm_grid, 128, 0, stream>>>(
                dn_out_buf, proj_buf2, dn_norm, S);

            cublas_bf16_gemm(cublas, dn_out_buf, out_w, proj_buf, S, HIDDEN, DN_V_SIZE);
            pf_add_residual_bf16<<<bk, 256, 0, stream>>>(proj_buf, residual, hidden, S*HIDDEN);

            pf_rmsnorm<<<S, 512, 0, stream>>>(hidden, post_norm, normalized, residual, S, HIDDEN);
            {
                const __nv_bfloat16 *gu_w = fused_gate_up_base + (size_t)li * (2*INTER) * HIDDEN;
                pf_mlp_chunked_fused(
                    cublas,
                    normalized,
                    residual,
                    hidden,
                    gu_w,
                    down_w,
                    proj_buf,
                    mlp_buf,
                    S,
                    4096,
                    stream);
            }

            dn_idx++;
        } else {
            const __nv_bfloat16 *q_nw=(const __nv_bfloat16*)lw.ptrs[4];
            const __nv_bfloat16 *k_nw=(const __nv_bfloat16*)lw.ptrs[5];
            const __nv_bfloat16 *o_w=(const __nv_bfloat16*)lw.ptrs[6];
            const __nv_bfloat16 *post_norm=(const __nv_bfloat16*)lw.ptrs[7];
            const __nv_bfloat16 *down_w=(const __nv_bfloat16*)lw.ptrs[10];

            // Fused QKV GEMM: output row stride FA_QPROJ_SIZE + 2*FA_KV_SIZE
            constexpr int FA_QKV_STRIDE = FA_QPROJ_SIZE + 2*FA_KV_SIZE;
            const __nv_bfloat16 *fa_qkv_w = fused_fa_qkv_base + (size_t)fa_idx * FA_QKV_STRIDE * HIDDEN;
            cublas_bf16_gemm(cublas, normalized, fa_qkv_w, proj_buf, S, FA_QKV_STRIDE, HIDDEN);

            int total_heads = S*(FA_Q_HEADS+FA_KV_HEADS);
            pf_qk_norm_rope_fused<<<(total_heads+15)/16, 512, 0, stream>>>(
                proj_buf, q_nw, k_nw,
                fa_k_cache + fa_idx*fa_stride, fa_v_cache + fa_idx*fa_stride, S, max_seq_len);

            pf_causal_attn_fused_tiled_cublas(
                cublas,
                proj_buf,
                mlp_buf,
                dn_pre_qkv,
                dn_pre_qkv + static_cast<size_t>(S) * max_seq_len,
                dn_out_buf,
                S,
                4096,
                stream);

            cublas_bf16_gemm(cublas, dn_out_buf, o_w, proj_buf, S, HIDDEN, FA_Q_SIZE);
            pf_add_residual_bf16<<<bk, 256, 0, stream>>>(proj_buf, residual, hidden, S*HIDDEN);

            pf_rmsnorm<<<S, 512, 0, stream>>>(hidden, post_norm, normalized, residual, S, HIDDEN);
            {
                const __nv_bfloat16 *gu_w = fused_gate_up_base + (size_t)li * (2*INTER) * HIDDEN;
                pf_mlp_chunked_fused(
                    cublas,
                    normalized,
                    residual,
                    hidden,
                    gu_w,
                    down_w,
                    proj_buf,
                    mlp_buf,
                    S,
                    4096,
                    stream);
            }

            fa_idx++;
        }
    }

    pf_final_norm<<<1, 512, 0, stream>>>(hidden, final_norm_w, final_normed, hidden_bf16_out, S);

    int lm_blocks = 512;
    pf_lm_head<<<lm_blocks, 256, 0, stream>>>(final_normed, lm_head_w, lm_bmv, lm_bmi, VOCAB);
    pf_lm_reduce<<<1, 256, 0, stream>>>(lm_bmv, lm_bmi, output_token, lm_blocks);
}
