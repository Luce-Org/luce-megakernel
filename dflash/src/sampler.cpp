// Shared CPU sampler chain. See sampler.h for the protocol overview.

#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <unordered_set>
#include <utility>

namespace dflash27b {

int sample_logits(const float * logits_in,
                  int vocab,
                  const SamplerCfg & cfg,
                  const std::vector<int32_t> & history,
                  std::mt19937_64 & rng) {
    std::vector<std::pair<float, int>> cand(vocab);
    for (int i = 0; i < vocab; i++) cand[i] = {logits_in[i], i};

    if (cfg.rep_pen > 1.0f && !history.empty()) {
        const int win  = std::min((int)history.size(), cfg.rep_window);
        const int from = (int)history.size() - win;
        std::unordered_set<int> seen;
        for (int i = from; i < (int)history.size(); i++) seen.insert(history[i]);
        for (auto & c : cand) {
            if (seen.count(c.second)) {
                c.first = (c.first > 0.0f) ? c.first / cfg.rep_pen
                                           : c.first * cfg.rep_pen;
            }
        }
    }

    if (cfg.top_k > 0 && cfg.top_k < vocab) {
        std::partial_sort(cand.begin(), cand.begin() + cfg.top_k, cand.end(),
                          [](auto & a, auto & b){ return a.first > b.first; });
        cand.resize(cfg.top_k);
    } else {
        std::sort(cand.begin(), cand.end(),
                  [](auto & a, auto & b){ return a.first > b.first; });
    }

    const float inv_t = 1.0f / std::max(1e-3f, cfg.temp);
    const float maxv  = cand.front().first * inv_t;
    double Z = 0.0;
    std::vector<float> probs(cand.size());
    for (size_t i = 0; i < cand.size(); i++) {
        probs[i] = std::exp(cand[i].first * inv_t - maxv);
        Z       += probs[i];
    }
    for (auto & p : probs) p = (float)(p / Z);

    if (cfg.top_p > 0.0f && cfg.top_p < 1.0f) {
        double cum = 0.0;
        size_t cut = probs.size();
        for (size_t i = 0; i < probs.size(); i++) {
            cum += probs[i];
            if (cum >= cfg.top_p) { cut = i + 1; break; }
        }
        probs.resize(cut); cand.resize(cut);
        double zz = 0.0;
        for (auto p : probs) zz += p;
        for (auto & p : probs) p = (float)(p / zz);
    }

    std::uniform_real_distribution<double> u(0.0, 1.0);
    const double r   = u(rng);
    double       acc = 0.0;
    for (size_t i = 0; i < probs.size(); i++) {
        acc += probs[i];
        if (r <= acc) return cand[i].second;
    }
    return cand.back().second;
}

bool parse_sampler_token(std::string & line, SamplerCfg & out) {
    auto pos = line.find(" samp=");
    if (pos == std::string::npos) return false;
    auto end = line.find(' ', pos + 1);
    std::string tok = (end == std::string::npos)
                          ? line.substr(pos + 6)
                          : line.substr(pos + 6, end - (pos + 6));
    line.erase(pos, (end == std::string::npos ? std::string::npos : end - pos));
    float t = 0.0f, tp = 1.0f, rp = 1.0f;
    int   tk = 0;
    unsigned long long sd = 0;
    int n = std::sscanf(tok.c_str(), "%f,%f,%d,%f,%llu",
                        &t, &tp, &tk, &rp, &sd);
    if (n < 1) return false;
    out.temp    = t;
    out.top_p   = tp;
    out.top_k   = tk;
    out.rep_pen = rp;
    out.seed    = sd;
    return true;
}

}  // namespace dflash27b
