// Shared CPU sampler chain used by both target arches.
//
// dflash27b daemon protocol embeds optional sampler params as a tail on each
// generate command: ` samp=temp,top_p,top_k,rep_pen,seed`. parse_sampler_token
// strips the tail in place and fills a SamplerCfg; sample_logits applies the
// chain rep_penalty -> top_k -> softmax(temp) -> top_p -> draw.
//
// Both test_dflash.cpp (qwen35 + DFlash + DDTree) and src/laguna_daemon.cpp
// include this header to keep behaviour identical across arches.

#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace dflash27b {

struct SamplerCfg {
    float    temp       = 0.0f;
    float    top_p      = 1.0f;
    int      top_k      = 0;
    float    rep_pen    = 1.0f;
    int      rep_window = 256;
    uint64_t seed       = 0;
};

// Returns the chosen token id. cfg.temp == 0 -> caller should use argmax;
// the chain assumes a positive temperature and falls back to a small floor.
int sample_logits(const float * logits_in,
                  int vocab,
                  const SamplerCfg & cfg,
                  const std::vector<int32_t> & history,
                  std::mt19937_64 & rng);

// Strip ` samp=...` tail from `line` (in place); return true when one was
// parsed. Out-of-band fields default to a permissive greedy-equivalent (top_p=1,
// top_k=0, rep_pen=1, seed=0).
bool parse_sampler_token(std::string & line, SamplerCfg & out);

}  // namespace dflash27b
