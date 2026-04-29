# DFlash Prefix Cache Intended Design

Status: design note, not implemented.

## Goal

Improve repeated coding-agent turns by avoiding full prefill when a later request can safely reuse token state from an earlier request.

The immediate serving default remains the safe text-only `agent-code-text` profile with prefix cache disabled. Prefix caching must be proven through token-level and chat-template equivalence tests before it becomes a default server behavior.

## Non-goals for v1

- No multi-slot trie cache.
- No cross-process persistence.
- No concurrent generations on one GPU worker.
- No arbitrary rollback to a shorter longest common prefix.
- No support for layer-segmented prefill reuse.
- No default-on OpenAI/Anthropic chat integration until real chat-template round trips are proven.

## Terminology

- `RUN`: current behavior. Reset daemon KV/SSM/conv state and prefill the full prompt.
- `RUN_PREFIX`: experimental behavior. Reuse daemon state only if the new prompt exactly extends the resident token sequence.
- `RESET`: destroy/recreate request state and invalidate resident metadata.
- `resident state`: daemon-side model state plus metadata describing the exact token sequence represented by that state.

## Required daemon protocol

The daemon should accept both the legacy protocol and an explicit protocol:

```text
/tmp/prompt.bin 512              # legacy, maps to RUN
RUN /tmp/prompt.bin 512          # reset + full prefill
RUN_PREFIX /tmp/prompt.bin 512   # exact-extension reuse attempt
RESET                            # clear resident state
```

Old clients must continue to work unchanged.

## Required daemon state model

Prefix reuse needs more than a vector of resident tokens. The daemon should track a state object similar to:

```cpp
enum class DaemonCommandKind {
    Run,
    RunPrefix,
    Reset,
};

struct DaemonCommand {
    DaemonCommandKind kind;
    std::string prompt_path;
    int n_gen = 0;
};

struct DaemonResidentState {
    bool valid = false;
    std::vector<int32_t> tokens;
    int committed = 0;
    int32_t last_tok = -1;
    bool cache_decode_ready = false;
};
```

The state must make cache lifecycle explicit:

- whether cache is prefill-only or decode-ready;
- whether `migrate_prefill_cache()` has already happened;
- what token position `committed` represents;
- what token `last_tok` should seed decode with;
- whether any error invalidated reuse.

## v1 reuse rule

`RUN_PREFIX` v1 may reuse only exact extensions:

```cpp
reuse_len = lcp_tokens(resident.tokens, prompt);
can_reuse = resident.valid
    && reuse_len == resident.tokens.size()
    && reuse_len <= prompt.size();
```

If `can_reuse` is false, the daemon must fall back to `RUN` semantics and log a miss.

This is intentionally not a general longest-prefix cache. It avoids rollback in v1.

## Generated-token caveat

Do not assume `resident.tokens = prompt + generated_tokens` is reusable for OpenAI/Anthropic chat traffic.

The Python server may:

- stop streaming when a stop token is observed;
- decode with `skip_special_tokens=True`;
- parse/normalize tool-call markup;
- omit hidden raw tokens from the visible assistant message.

Therefore the client’s next templated prompt may not exactly extend the daemon’s raw generated token sequence.

For chat-serving integration, prove token round-trip first:

1. Tokenize request A.
2. Run generation.
3. Build request B exactly as a real client would from visible assistant/tool output.
4. Tokenize request B.
5. Check whether request B starts with the daemon’s intended resident tokens.

If that assertion fails, Python must not use `RUN_PREFIX` for chat traffic.

## Prefill scope

Prefix reuse v1 should support only token-segmented prefill.

If `DFLASH27B_LAYER_PREFILL=1` and `RUN_PREFIX` is requested, the daemon should either:

- fall back to `RUN` and log `reason=layer-prefill`; or
- return a clear unsupported error during development.

Layer-segmented prefill allocates full-prompt activation buffers and should not be part of the first reuse implementation.

## Observability

Every daemon request should log one stable parseable prefix-cache line:

```text
[prefix-cache] mode=disabled reason=run prompt_tokens=12000
[prefix-cache] mode=miss reason=not-extension reuse_tokens=0 suffix_tokens=13023 prompt_tokens=13023
[prefix-cache] mode=reuse reuse_tokens=12345 suffix_tokens=678 prompt_tokens=13023
```

Failures in daemon mode must emit `-1` to the stream fd and invalidate resident state.

## Validation milestones

1. Protocol compatibility:
   - legacy `<prompt.bin> <n_gen>` still works;
   - `RUN` matches legacy behavior;
   - `RESET` clears state.

2. Token-file equivalence:
   - let B = A + suffix;
   - compare `RUN B` against `RESET; RUN_PREFIX A; RUN_PREFIX B` under deterministic decoding;
   - outputs must match exactly.

3. Chat-template equivalence:
   - perform the same comparison through `/v1/chat/completions` and `/v1/messages`;
   - include stop-token behavior, tool-call parsing, and `skip_special_tokens` behavior;
   - only then consider Python-side `--prefix-cache` wiring.

4. Default-on decision:
   - prefix cache may become default only after real coding-agent traces show reliable reuse and exact output equivalence.

## Relationship to `agent-code-text`

The safe `agent-code-text` profile does not depend on prefix caching.

Default profile behavior should remain:

- text-only;
- 48K max context;
- 42K prompt admission limit;
- one serialized request per daemon;
- conservative prefill ubatch;
- prefix cache disabled.
