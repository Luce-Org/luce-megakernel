# Agentic Coding Profile + Automatic Prefix Cache Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Make `dflash-server` reliable and fast for text-only agentic coding workflows by adding a 48K safe coding profile and automatic prefix reuse for repeated OpenAI/Anthropic chat requests.

**Architecture:** Ship the safety profile first in the Python FastAPI server, then extend the DFlash daemon protocol so a single serialized GPU worker can reuse the longest resident token prefix instead of rebuilding KV/SSM/conv state every request. Keep the first prefix cache deliberately simple: one active resident slot per daemon process, exact-token prefix matching, reset fallback on mismatch, and metrics that show whether reuse happened.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic, transformers tokenizer, C++ DFlash `test_dflash` daemon, existing token `.bin` prompt files, pytest, existing CMake/build flow.

---

## Current state summary

Relevant files:

- `dflash/src/dflash/server/__init__.py`
  - Starts `test_dflash --daemon`.
  - Serializes requests through `daemon_lock`.
  - Current CLI default is `--max-ctx 16384` despite the module docstring saying 128K.
  - For `--max-ctx > 6144`, sets `DFLASH27B_KV_TQ3=1` unless `--kv-f16` is passed.
  - Sends daemon commands as one line: `<prompt_bin> <gen_len>`.
  - Deletes prompt temp files after generation.
- `dflash/test/test_dflash.cpp`
  - Daemon keeps weights resident.
  - Between daemon requests, it currently destroys the step graph and recreates target cache, so it does not preserve prefix KV/SSM/conv state.
  - Has existing knobs: `--max-ctx`, `--stream-fd`, `--cache-type-k/-ctk`, `--cache-type-v/-ctv`, `DFLASH27B_PREFILL_UBATCH`, `DFLASH27B_LAYER_PREFILL`, `DFLASH27B_FA_WINDOW`.
- `dflash/src/qwen35_target_graph.cpp`
  - Has `fa_window` support, but it should stay off for the default coding profile until quality is proven.

Design decisions for this plan:

- Text-only means no vision/multimodal support and no image payloads in requests.
- Default coding-agent context is 48K, not max bootable context.
- Default usable prompt limit is below 48K to reserve generation and scratch headroom.
- Single GPU worker remains serialized: one request at a time.
- Prefix cache v1 is one resident slot, not a complex multi-tenant trie.
- Reuse requires exact token prefix match. If not matched, reset and run the full prompt.
- Do not enable FA sliding window by default.

---

## Phase 1: Safe text-only coding profile in Python server

### Task 1: Add Python unit tests for profile defaults

**Objective:** Lock in the intended `agent-code-text` defaults before changing CLI behavior.

**Files:**
- Create: `dflash/tests/test_server_profiles.py`
- Modify later: `dflash/src/dflash/server/__init__.py`

**Step 1: Create the test file**

Add tests for a pure helper that does not start the server:

```python
from dflash.server import AGENT_CODE_TEXT_PROFILE, resolve_server_profile


def test_agent_code_text_profile_defaults():
    profile = AGENT_CODE_TEXT_PROFILE
    assert profile.name == "agent-code-text"
    assert profile.max_ctx == 48_000
    assert profile.max_prompt_tokens == 42_000
    assert profile.reserved_generation_tokens == 4_096
    assert profile.prefill_ubatch == 384
    assert profile.layer_prefill is False
    assert profile.fa_window == 0
    assert profile.single_request_per_gpu is True
    assert profile.vision is False


def test_profile_can_be_overridden_explicitly():
    profile = resolve_server_profile(
        profile_name="agent-code-text",
        max_ctx=32_000,
        max_prompt_tokens=28_000,
        prefill_ubatch=256,
    )
    assert profile.max_ctx == 32_000
    assert profile.max_prompt_tokens == 28_000
    assert profile.prefill_ubatch == 256
```

**Step 2: Run the test and verify failure**

Run:

```bash
cd /home/erik/Projects/lucebox-hub/dflash
uv run pytest tests/test_server_profiles.py -q
```

Expected: FAIL because `AGENT_CODE_TEXT_PROFILE` and `resolve_server_profile` do not exist.

### Task 2: Add server profile data model and defaults

**Objective:** Add a named profile object without changing request handling yet.

**Files:**
- Modify: `dflash/src/dflash/server/__init__.py`
- Test: `dflash/tests/test_server_profiles.py`

**Implementation:**

Add near the top of `__init__.py` after constants:

```python
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ServerProfile:
    name: str
    max_ctx: int
    max_prompt_tokens: int
    reserved_generation_tokens: int
    prefill_ubatch: int
    layer_prefill: bool
    fa_window: int
    single_request_per_gpu: bool
    vision: bool


AGENT_CODE_TEXT_PROFILE = ServerProfile(
    name="agent-code-text",
    max_ctx=48_000,
    max_prompt_tokens=42_000,
    reserved_generation_tokens=4_096,
    prefill_ubatch=384,
    layer_prefill=False,
    fa_window=0,
    single_request_per_gpu=True,
    vision=False,
)


def resolve_server_profile(
    profile_name: str,
    *,
    max_ctx: int | None = None,
    max_prompt_tokens: int | None = None,
    reserved_generation_tokens: int | None = None,
    prefill_ubatch: int | None = None,
    fa_window: int | None = None,
) -> ServerProfile:
    if profile_name != AGENT_CODE_TEXT_PROFILE.name:
        raise ValueError(f"unknown server profile: {profile_name}")

    profile = AGENT_CODE_TEXT_PROFILE
    overrides = {}
    if max_ctx is not None:
        overrides["max_ctx"] = max_ctx
    if max_prompt_tokens is not None:
        overrides["max_prompt_tokens"] = max_prompt_tokens
    if reserved_generation_tokens is not None:
        overrides["reserved_generation_tokens"] = reserved_generation_tokens
    if prefill_ubatch is not None:
        overrides["prefill_ubatch"] = prefill_ubatch
    if fa_window is not None:
        overrides["fa_window"] = fa_window
    return replace(profile, **overrides)
```

**Step 3: Run the test**

```bash
cd /home/erik/Projects/lucebox-hub/dflash
uv run pytest tests/test_server_profiles.py -q
```

Expected: PASS.

### Task 3: Add text-only request validation tests

**Objective:** Ensure the coding profile rejects image/multimodal payloads instead of silently trying to template them.

**Files:**
- Modify: `dflash/tests/test_server_profiles.py`
- Modify later: `dflash/src/dflash/server/__init__.py`

**Test cases:**

```python
from dflash.server import request_contains_vision_payload


def test_string_content_is_text_only():
    req = {"messages": [{"role": "user", "content": "read src/main.py"}]}
    assert request_contains_vision_payload(req) is False


def test_openai_image_url_content_is_rejected():
    req = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "file://x.png"}},
                ],
            }
        ]
    }
    assert request_contains_vision_payload(req) is True
```

**Run:**

```bash
uv run pytest tests/test_server_profiles.py -q
```

Expected: FAIL because helper does not exist.

### Task 4: Implement text-only request validation helper

**Objective:** Add a reusable helper that works on raw dicts and Pydantic request objects.

**Files:**
- Modify: `dflash/src/dflash/server/__init__.py`
- Test: `dflash/tests/test_server_profiles.py`

**Implementation:**

```python
def request_contains_vision_payload(req) -> bool:
    data = req.model_dump() if hasattr(req, "model_dump") else req
    messages = data.get("messages") or []
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type in {"image_url", "input_image", "image"}:
                    return True
    return False
```

**Run:**

```bash
uv run pytest tests/test_server_profiles.py -q
```

Expected: PASS.

### Task 5: Add admission policy tests

**Objective:** Enforce safe prompt limits below max context.

**Files:**
- Modify: `dflash/tests/test_server_profiles.py`
- Modify later: `dflash/src/dflash/server/__init__.py`

**Test cases:**

```python
from dflash.server import admission_for_prompt


def test_admission_allows_prompt_below_profile_limit():
    decision = admission_for_prompt(prompt_tokens=12_000, requested_max_tokens=2_000, profile=AGENT_CODE_TEXT_PROFILE)
    assert decision.allowed is True
    assert decision.generation_tokens == 2_000


def test_admission_clamps_generation_to_reserved_space():
    decision = admission_for_prompt(prompt_tokens=41_000, requested_max_tokens=8_000, profile=AGENT_CODE_TEXT_PROFILE)
    assert decision.allowed is True
    assert decision.generation_tokens <= AGENT_CODE_TEXT_PROFILE.max_ctx - 41_000 - 20


def test_admission_rejects_prompt_above_soft_limit():
    decision = admission_for_prompt(prompt_tokens=43_000, requested_max_tokens=512, profile=AGENT_CODE_TEXT_PROFILE)
    assert decision.allowed is False
    assert "max_prompt_tokens" in decision.reason
```

**Run:**

```bash
uv run pytest tests/test_server_profiles.py -q
```

Expected: FAIL because helper does not exist.

### Task 6: Implement admission policy helper

**Objective:** Centralize prompt/generation safety rules for OpenAI and Anthropic endpoints.

**Files:**
- Modify: `dflash/src/dflash/server/__init__.py`
- Test: `dflash/tests/test_server_profiles.py`

**Implementation:**

```python
@dataclass(frozen=True)
class AdmissionDecision:
    allowed: bool
    generation_tokens: int
    reason: str = ""


def admission_for_prompt(prompt_tokens: int, requested_max_tokens: int, profile: ServerProfile) -> AdmissionDecision:
    if prompt_tokens > profile.max_prompt_tokens:
        return AdmissionDecision(
            allowed=False,
            generation_tokens=0,
            reason=(
                f"prompt_tokens={prompt_tokens} exceeds max_prompt_tokens="
                f"{profile.max_prompt_tokens} for profile {profile.name}"
            ),
        )

    hard_available = profile.max_ctx - prompt_tokens - 20
    reserved_available = profile.max_ctx - prompt_tokens - profile.reserved_generation_tokens
    available = max(0, max(hard_available, reserved_available))
    gen_tokens = min(requested_max_tokens, hard_available)
    if gen_tokens <= 0:
        return AdmissionDecision(
            allowed=False,
            generation_tokens=0,
            reason=f"prompt_tokens={prompt_tokens} leaves no generation room in max_ctx={profile.max_ctx}",
        )
    return AdmissionDecision(allowed=True, generation_tokens=gen_tokens)
```

Note: simplify after implementation review if `reserved_available` is unnecessary. The critical behavior is: reject above soft limit, never exceed hard `max_ctx`.

**Run:**

```bash
uv run pytest tests/test_server_profiles.py -q
```

Expected: PASS.

### Task 7: Wire profile into CLI and daemon environment

**Objective:** Make `agent-code-text` the default server profile and export DFlash env knobs.

**Files:**
- Modify: `dflash/src/dflash/server/__init__.py:523-585`
- Test: `dflash/tests/test_server_profiles.py`

**Implementation notes:**

- Add CLI args:
  - `--profile`, default `agent-code-text`
  - `--max-prompt-tokens`, default None
  - `--reserved-generation-tokens`, default None
  - `--prefill-ubatch`, default None
  - `--cache-type-k`, default None
  - `--cache-type-v`, default None
  - keep `--kv-f16` temporarily for compatibility, but mark help as legacy
- Resolve profile before `build_app`.
- Use `profile.max_ctx` as the default `--max-ctx` if user does not explicitly pass `--max-ctx`.
- Set env:
  - `DFLASH27B_PREFILL_UBATCH=str(profile.prefill_ubatch)`
  - `DFLASH27B_LAYER_PREFILL="1"` only if profile says true; otherwise `"0"`
  - `DFLASH27B_FA_WINDOW` only if profile `fa_window > 0`
- For now keep current TQ3 default behavior, but add explicit K/V passthrough as the preferred path:
  - if `--cache-type-k`, set `DFLASH27B_KV_K`
  - if `--cache-type-v`, set `DFLASH27B_KV_V`

**Build-app signature change:**

Change:

```python
def build_app(..., max_ctx: int, tokenizer: AutoTokenizer, stop_ids: set[int], model_name: str = "") -> FastAPI:
```

to:

```python
def build_app(..., profile: ServerProfile, tokenizer: AutoTokenizer, stop_ids: set[int], model_name: str = "") -> FastAPI:
```

Then use `profile.max_ctx` internally.

**Run:**

```bash
uv run pytest tests/test_server_profiles.py -q
```

Expected: PASS.

### Task 8: Apply admission and text-only checks to OpenAI endpoint

**Objective:** Make `/v1/chat/completions` reject unsafe or vision requests before calling DFlash.

**Files:**
- Modify: `dflash/src/dflash/server/__init__.py:233-304`
- Test: add endpoint-level test if FastAPI TestClient can instantiate without starting DFlash; otherwise keep helper-level tests in this phase.

**Implementation notes:**

In `chat_completions` after tool-choice validation and before tokenization or after tokenization as appropriate:

```python
if not profile.vision and request_contains_vision_payload(req):
    return JSONResponse(
        {"error": {"code": "unsupported_content", "message": "agent-code-text profile is text-only and does not accept image content"}},
        status_code=400,
    )
```

After tokenization:

```python
decision = admission_for_prompt(prompt_len, req.max_tokens, profile)
if not decision.allowed:
    cleanup prompt file
    return JSONResponse({"error": {"code": "context_limit", "message": decision.reason}}, status_code=400)
gen_len = decision.generation_tokens
```

**Run:**

```bash
uv run pytest tests/test_server_profiles.py -q
```

Expected: PASS.

### Task 9: Apply admission and text-only checks to Anthropic endpoint

**Objective:** Keep `/v1/messages` behavior aligned with `/v1/chat/completions`.

**Files:**
- Modify: `dflash/src/dflash/server/__init__.py:445-518`

**Implementation notes:**

- Anthropic content lists may include image blocks. Reject if `profile.vision is False` and any content block has type `image`/`image_url`/`input_image`.
- Use `admission_for_prompt(prompt_len, req.max_tokens, profile)`.
- Return Anthropic-style error JSON.

**Run:**

```bash
uv run pytest tests/test_server_profiles.py -q
```

Expected: PASS.

### Task 10: Update server startup logging and docs

**Objective:** Make it obvious which profile is running and what safety limits apply.

**Files:**
- Modify: `dflash/src/dflash/server/__init__.py:576-584`
- Modify: `dflash/README.md`

**Startup log should include:**

```text
profile   = agent-code-text
vision    = false
max_ctx   = 48000
max_prompt_tokens = 42000
prefill_ubatch = 384
fa_window = 0
```

**Docs should include default command:**

```bash
uv run dflash-server --profile agent-code-text
```

And tuning examples:

```bash
uv run dflash-server --profile agent-code-text --prefill-ubatch 256
uv run dflash-server --profile agent-code-text --cache-type-k q4_0 --cache-type-v q4_0
```

**Run:**

```bash
uv run pytest tests/test_server_profiles.py -q
```

Expected: PASS.

---

## Phase 2: Automatic prefix cache in daemon protocol (deferred)

Prefix-cache implementation is intentionally deferred until the design in `dflash/docs/prefix-cache-design.md` is implemented and validated. The safe `agent-code-text` profile must not depend on engine changes or default-on prefix caching.


### Task 11: Add C++ daemon protocol compatibility wrapper

**Objective:** Keep old daemon commands working while allowing a new explicit command format.

**Files:**
- Modify: `dflash/test/test_dflash.cpp:1219-1242`

**Current format:**

```text
/tmp/prompt.bin 512
```

**New accepted formats:**

```text
RUN /tmp/prompt.bin 512
RUN_PREFIX /tmp/prompt.bin 512
RESET
```

**Behavior:**

- Old two-field format maps to `RUN`.
- `RUN` resets cache/state before request, preserving current behavior.
- `RUN_PREFIX` attempts automatic prefix reuse.
- `RESET` destroys/recreates cache and clears resident token metadata, emits no generated tokens.

**Verification:**

Build `test_dflash`, start daemon, send old-format command, confirm it still works.

### Task 12: Add resident token metadata to C++ daemon loop

**Objective:** Track what token prefix the daemon state currently represents.

**Files:**
- Modify: `dflash/test/test_dflash.cpp`

**Add variables near daemon loop setup:**

```cpp
std::vector<int32_t> resident_tokens;
bool resident_valid = false;
```

**Rules:**

- After full prefill + generation completes successfully, set `resident_tokens = out_all` and `resident_valid = true`.
- On any error, set `resident_valid = false`.
- On `RESET`, clear tokens and set invalid.

**Verification:**

Add log lines gated by daemon mode:

```text
[prefix-cache] resident_tokens=N
```

### Task 13: Implement longest common prefix helper in C++

**Objective:** Compute exact token prefix reuse length.

**Files:**
- Modify: `dflash/test/test_dflash.cpp`

**Helper:**

```cpp
static size_t lcp_tokens(const std::vector<int32_t> & a, const std::vector<int32_t> & b) {
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) ++i;
    return i;
}
```

**Initial behavior:**

For v1, only reuse if the new prompt fully extends the current resident state:

```cpp
reuse_len = lcp_tokens(resident_tokens, prompt);
can_reuse = resident_valid && reuse_len == resident_tokens.size() && reuse_len <= prompt.size();
```

If `can_reuse` is false, fall back to full reset and full prompt prefill.

Rationale: this avoids needing KV/SSM rollback in the first implementation.

### Task 14: Split prefill path into reusable function

**Objective:** Make it possible to prefill only the suffix tokens after a reused prefix.

**Files:**
- Modify: `dflash/test/test_dflash.cpp:1262+` prefill section

**Refactor target:**

Create a helper or local function with this logical signature:

```cpp
bool prefill_tokens(
    const std::vector<int32_t> & tokens,
    int start_pos,
    int end_pos,
    TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    ...existing needed state...
);
```

**Do not change math yet.** First refactor so full prefill still calls:

```cpp
prefill_tokens(prompt, 0, prompt.size(), ...)
```

**Verification:**

Run the existing generation smoke test before enabling reuse. Output should be unchanged for a fixed prompt and greedy settings.

### Task 15: Prefill only suffix when `RUN_PREFIX` extends resident state

**Objective:** Avoid recomputing the resident prefix.

**Files:**
- Modify: `dflash/test/test_dflash.cpp`

**Behavior:**

If:

```cpp
resident_valid && prompt starts with resident_tokens
```

Then:

- Do not recreate cache.
- Set committed/current position to `resident_tokens.size()`.
- Prefill only `prompt[resident_tokens.size():]`.
- Continue generation normally.

If not:

- Recreate cache.
- Prefill full prompt.

**Logging:**

Emit one line per daemon request:

```text
[prefix-cache] mode=reuse reuse_tokens=12345 suffix_tokens=678 prompt_tokens=13023
```

or:

```text
[prefix-cache] mode=miss reason=not-extension reuse_tokens=0 suffix_tokens=13023 prompt_tokens=13023
```

**Important correctness check:**

If generated output differs between `RUN` and `RUN_PREFIX` for the same full prompt after a reset, disable reuse and debug before proceeding.

### Task 16: Add Python-side automatic prefix mode flag

**Objective:** Let the server request reuse without changing client APIs.

**Files:**
- Modify: `dflash/src/dflash/server/__init__.py`

**CLI:**

Add:

```bash
--prefix-cache / --no-prefix-cache
```

Default for `agent-code-text`: disabled until `dflash/docs/prefix-cache-design.md` validation milestones pass.

**Command selection:**

Where Python currently writes:

```python
daemon_proc.stdin.write(f"{prompt_bin} {gen_len}\n".encode())
```

Change to:

```python
cmd_name = "RUN_PREFIX" if profile.prefix_cache else "RUN"
daemon_proc.stdin.write(f"{cmd_name} {prompt_bin} {gen_len}\n".encode())
```

This requires adding `prefix_cache: bool` to `ServerProfile` and tests.

**Run:**

```bash
uv run pytest tests/test_server_profiles.py -q
```

Expected: PASS.

### Task 17: Add prefix-cache response/debug metadata

**Objective:** Make prefix reuse visible during development without breaking OpenAI clients.

**Files:**
- Modify: `dflash/src/dflash/server/__init__.py`
- Modify: `dflash/test/test_dflash.cpp`

**C++:**

Log prefix-cache stats to stderr/stdout with a stable parseable line.

**Python:**

Initially do not parse C++ logs. Just document that server logs contain prefix-cache stats.

Optional later: expose `/debug/prefix-cache` returning last observed stats if Python captures daemon stderr.

### Task 18: Add integration smoke test script for prefix reuse

**Objective:** Verify that the second request reuses the first request prefix.

**Files:**
- Create: `dflash/scripts/smoke_prefix_cache.py`

**Script behavior:**

- Send request 1 with a stable coding-system prompt and a short user message.
- Send request 2 whose prompt includes request 1 plus an appended tool-result-like block.
- Ask user/developer to inspect logs for:

```text
[prefix-cache] mode=reuse
```

**Future enhancement:** parse daemon logs and assert reuse automatically.

### Task 19: Correctness comparison test: full reset vs prefix reuse

**Objective:** Prove prefix reuse is functionally equivalent for deterministic generation.

**Files:**
- Create: `dflash/scripts/check_prefix_cache_equivalence.py`

**Script behavior:**

- Start or target a server with prefix cache disabled.
- Run prompt B, capture output.
- Start or target a fresh server with prefix cache enabled.
- Run prompt A, then prompt B where B extends A, capture output for B.
- Compare token/text output for prompt B.

Expected: exact match under greedy deterministic decoding.

### Task 20: Benchmark coding-agent profile

**Objective:** Establish the safe operating envelope for real coding-agent traces.

**Files:**
- Modify or reuse: `dflash/scripts/bench_server.py`
- Create optional fixture directory: `dflash/bench/prompts/agentic-coding/`

**Prompt cases:**

- 8K coding prompt: repo instructions + one file + small test failure.
- 16K coding prompt: multiple files + tool schema + command output.
- 32K coding prompt: large logs/diffs/tool output.
- 42K upper-limit prompt.
- Repeated-prefix pair: A = 16K, B = A + 4K tool result.

**Measure:**

- TTFT
- prefill tokens/sec if available
- decode tokens/sec
- peak VRAM using `nvidia-smi`
- prefix reuse mode/miss logs
- OOM/failure threshold

**Commands:**

```bash
cd /home/erik/Projects/lucebox-hub/dflash
uv run dflash-server --profile agent-code-text --prefill-ubatch 384
uv run python scripts/bench_server.py --base-url http://localhost:1236/v1 --profile agentic-coding
```

Adjust exact `bench_server.py` args to match the current script.

---

## Phase 3: KV/cache tuning after prefix cache works

### Task 21: Benchmark K/V cache choices under the coding profile

**Objective:** Pick the default KV setting empirically for coding agents.

**Candidates:**

- Current implicit TQ3 path: `DFLASH27B_KV_TQ3=1`
- Explicit `--cache-type-k q4_0 --cache-type-v q4_0`, if supported
- Explicit `--cache-type-k q8_0 --cache-type-v q8_0`, if supported
- F16 baseline via `--kv-f16`

**Benchmark matrix:**

```bash
uv run dflash-server --profile agent-code-text --prefill-ubatch 384 --kv-f16
uv run dflash-server --profile agent-code-text --prefill-ubatch 384 --cache-type-k q8_0 --cache-type-v q8_0
uv run dflash-server --profile agent-code-text --prefill-ubatch 384 --cache-type-k q4_0 --cache-type-v q4_0
uv run dflash-server --profile agent-code-text --prefill-ubatch 384
```

**Choose default by:**

1. No OOM on 42K prompt.
2. No quality/tool-call regression on coding traces.
3. Lowest TTFT among stable candidates.
4. Lowest peak VRAM among stable candidates.

### Task 22: Benchmark prefill ubatch under the coding profile

**Objective:** Find the best stable default for large coding prefills.

**Values:**

```text
256, 384, 512
```

**Expected:**

- 256: safest, slower TTFT.
- 384: likely balanced default.
- 512: try only if stable at 32K/42K.

Do not tune `DFLASH27B_FA_WINDOW` until the basic cache is correct.

---

## Done criteria

The implementation is done when:

1. `uv run dflash-server --profile agent-code-text` starts with:
   - max_ctx 48000
   - max_prompt_tokens 42000
   - vision false
   - prefill_ubatch 384
   - prefix cache disabled
2. `/v1/chat/completions` rejects image payloads under this profile.
3. `/v1/chat/completions` and `/v1/messages` reject prompts above the safe profile limit before calling C++.
4. Existing old daemon line protocol still works.
5. New `RUN_PREFIX` daemon command reuses the resident token prefix when the next prompt extends the previous resident state.
6. Cache miss falls back to full reset/full prefill.
7. Prefix-cache logs clearly show reuse/miss and token counts.
8. Deterministic full-reset vs prefix-reuse outputs match for an extension prompt.
9. Coding-agent benchmark shows improved second-turn TTFT/prefill behavior with no OOM at the 42K prompt limit.
10. README documents the new profile, tuning knobs, and that vision is intentionally unsupported.

---

## Non-goals for this implementation

- Multi-slot prefix trie.
- Cross-process cache persistence.
- Sharing one GPU across concurrent generations.
- Vision/multimodal support.
- FA-window default enablement.
- Raising default context above 48K.
- Supporting rollback to arbitrary LCP shorter than resident state in v1. If the prompt does not extend resident tokens exactly, reset and run full prompt.
