"""Phase A: single-point prefix cache.

Auto-detects the system-prompt boundary in token id streams via Qwen chat
template markers, hashes prefixes, and maintains an LRU map of hash → daemon
slot id. Daemon owns slot buffers; Python is the index.

Usage:
    bus = DaemonStdoutBus(daemon_proc.stdout)
    bus.start(loop)

    pc = PrefixCache(
        daemon_stdin=daemon_proc.stdin,
        await_reply=bus.await_reply,
        daemon_lock=lock,
        tokenizer=tokenizer,
        cap=4,
    )
    await pc.startup_sync()  # free orphaned slots from a previous daemon run

    # Per request (caller holds daemon_lock):
    hit = pc.lookup(prompt_ids, kv_k_type, fa_window)   # (slot_id, prefix_len) or None
    if hit:
        slot, prefix_len = hit
        # send "RESTORE <slot> <prompt_bin> <n_gen>" instead of bare line
        ...
    else:
        # send bare "<prompt_bin> <n_gen>"
        ...
        # after daemon finishes, snapshot for future cache hits:
        await pc.maybe_snapshot(prompt_ids, kv_k_type, fa_window)

Option 3 — full-compress-result cache:
    When pFlash compression is enabled, the prefix-cache path above silently
    no-ops because compressed tokens lack Qwen chat-template markers.  The
    full-cache path caches the compressed cur_bin keyed on the ORIGINAL raw
    prompt token IDs, so that an identical long prompt sent a second time skips
    BOTH the drafter compression dance AND the target prefill.

    full_hit = pc.lookup_full(prompt_ids)
    if full_hit:
        slot, cached_cur_bin, cur_ids_len = full_hit
        cmd_line = f"RESTORE {slot} {cached_cur_bin} {gen_len}\\n"
    else:
        cur_bin, cur_ids = _maybe_compress(...)
        if cur_bin != prompt_bin:          # compression actually fired
            prep = pc.prepare_full_snap(prompt_ids)
            if prep:
                slot, _ = prep
                cmd_line = f"{cur_bin} {gen_len} snap={len(cur_ids)}:{slot}\\n"
        # ...after response completes:
        pc.confirm_full_snap(slot, prompt_ids, cur_bin, len(cur_ids))
        # on exception:
        pc.abort_full_snap(slot)
"""
import asyncio
import hashlib
import os
import shutil
import struct
from collections import OrderedDict
from pathlib import Path


# ---------------------------------------------------------------------------
# DaemonStdoutBus
# ---------------------------------------------------------------------------

class DaemonStdoutBus:
    """Owns the read loop on daemon stdout.

    Lines that start with a registered prefix are routed to the waiting
    coroutine; everything else is printed as a log (with noise filtering).
    """

    # Prefixes that are too spammy to print in normal operation.
    _SUPPRESS_PREFIXES = (
        "[step ", "[timing]", "[dflash]", "[prompt]",
        "[prefill]", "[migrate]", "[dbg ", "  ",
    )

    def __init__(self, stdout):
        self.stdout = stdout
        self._waiters: list[tuple[str, asyncio.Future]] = []
        self._task: asyncio.Task | None = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._task = loop.create_task(self._run())

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, self.stdout.readline)
            if not line:
                # Daemon exited — wake all waiters with an error.
                for _, fut in self._waiters:
                    if not fut.done():
                        fut.set_exception(EOFError("daemon stdout closed"))
                self._waiters.clear()
                return
            decoded = line.decode("utf-8", errors="replace").rstrip()

            # Try to satisfy a waiter first.
            matched = False
            for i, (prefix, fut) in enumerate(self._waiters):
                if decoded.startswith(prefix) and not fut.done():
                    fut.set_result(decoded)
                    self._waiters.pop(i)
                    matched = True
                    break

            if not matched:
                # Log line — suppress very noisy prefixes.
                show_timing = os.environ.get("DFLASH_SERVER_SHOW_TIMING", "0") != "0"
                suppressed = False if show_timing else any(
                    decoded.startswith(p) for p in self._SUPPRESS_PREFIXES)
                if decoded and not suppressed:
                    print(f"  [daemon] {decoded}", flush=True)

    async def await_reply(self, prefix: str, timeout: float = 10.0) -> str:
        """Block until daemon emits a line starting with *prefix*."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[str] = loop.create_future()
        entry = (prefix, fut)
        self._waiters.append(entry)
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            # On timeout / cancellation the matcher loop never popped us;
            # remove ourselves so _waiters doesn't grow without bound.
            try: self._waiters.remove(entry)
            except ValueError: pass


# ---------------------------------------------------------------------------
# Qwen chat template helpers
# ---------------------------------------------------------------------------

def _qwen_marker_ids(tokenizer):
    """Resolve <|im_end|>, <|im_start|>, and 'system' token ids."""
    im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    im_start = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    system_t = tokenizer.encode("system", add_special_tokens=False)
    if len(im_end) != 1 or len(im_start) != 1:
        raise ValueError(
            f"Expected single-token chat markers; got "
            f"im_end={im_end} im_start={im_start}"
        )
    return im_end[0], im_start[0], system_t[0] if len(system_t) == 1 else None


def find_prefix_boundary(ids, im_end_id, im_start_id, system_token_id):
    """Return the index AFTER the FIRST end-of-system-message marker, or -1.

    Qwen's chat template renders to:

        <|im_start|>system\\nCONTENT<|im_end|>\\n<|im_start|>user\\n...

    so a `\\n` token sits BETWEEN ``<|im_end|>`` and the next ``<|im_start|>``.
    We allow up to 2 intervening tokens (covers `\\n` and similar separators).

    The cacheable prefix is the SYSTEM message: from index 0 through and
    including the ``<|im_start|>`` that begins the next role. Subsequent turns
    sharing this system message hash to the same key.

    Returns the index right after that ``<|im_start|>``, so ``ids[:boundary]``
    is the cached state and ``ids[boundary:]`` is the per-request suffix.
    Returns -1 if there is no recognizable system message.
    """
    # Find the first <|im_start|>system sequence.
    sys_idx = -1
    for i in range(len(ids) - 1):
        if ids[i] == im_start_id:
            if system_token_id is None or ids[i + 1] == system_token_id:
                sys_idx = i
                break
    if sys_idx < 0:
        return -1

    # Find the FIRST <|im_end|> after sys_idx, then locate the next <|im_start|>
    # within a small lookahead (handles a single-token newline separator).
    for i in range(sys_idx + 1, len(ids)):
        if ids[i] == im_end_id:
            for j in range(i + 1, min(i + 3, len(ids))):
                if ids[j] == im_start_id:
                    return j + 1   # boundary is one past <|im_start|>
            return -1   # malformed — im_end without subsequent im_start
    return -1


def find_all_boundaries(ids, im_end_id, im_start_id, system_token_id):
    """Return ascending list of candidate cut points for multi-slot caching.

    Each cut point is the index AFTER an ``<|im_start|>`` that begins a new
    role's content. The first cut is the system-prompt boundary (same as
    ``find_prefix_boundary``); subsequent cuts are at every following
    ``<|im_end|>`` + ``<|im_start|>`` pair.

    Returns an empty list if no recognizable system message is found.
    """
    boundaries = []

    # Locate the opening <|im_start|>system token.
    sys_idx = -1
    for i in range(len(ids) - 1):
        if ids[i] == im_start_id:
            if system_token_id is None or ids[i + 1] == system_token_id:
                sys_idx = i
                break
    if sys_idx < 0:
        return boundaries

    # Walk forward from sys_idx: every time we see <|im_end|> followed
    # (within 2 tokens) by <|im_start|>, record the position just after
    # that <|im_start|> as a cache cut-point.
    i = sys_idx + 1
    while i < len(ids):
        if ids[i] == im_end_id:
            found_start = False
            for j in range(i + 1, min(i + 3, len(ids))):
                if ids[j] == im_start_id:
                    boundaries.append(j + 1)
                    i = j + 1
                    found_start = True
                    break
            if not found_start:
                break
        else:
            i += 1
    return boundaries


def hash_prefix(prefix_ids, kv_k_type, fa_window):
    """Stable SHA-1 (truncated 16 B) of (token ids, kv type, fa window)."""
    h = hashlib.sha1()
    h.update(struct.pack("<I", len(prefix_ids)))
    h.update(struct.pack(f"<{len(prefix_ids)}i", *prefix_ids))
    h.update(str(kv_k_type).encode())
    h.update(b"\x00")
    h.update(struct.pack("<I", fa_window or 0))
    return h.digest()[:16]


# ---------------------------------------------------------------------------
# PrefixCache
# ---------------------------------------------------------------------------

class PrefixCache:
    """LRU prefix cache.  Daemon owns the GPU slots; Python tracks hash→slot.

    Parameters
    ----------
    daemon_stdin:
        The ``stdin`` pipe of the daemon subprocess (``subprocess.Popen.stdin``).
    await_reply:
        Async callable ``(prefix: str, timeout: float) -> str`` — provided by
        ``DaemonStdoutBus.await_reply``.
    daemon_lock:
        ``asyncio.Lock`` that serialises all stdin writes + stdout reads.
        Callers must acquire it before calling ``lookup`` and hold it through
        any subsequent ``RESTORE`` / ``SNAPSHOT`` IPC.
    tokenizer:
        HuggingFace tokenizer (used only to resolve Qwen chat marker ids).
    cap:
        Maximum number of snapshot slots.  0 disables the cache entirely.
    log_prefix:
        String prepended to cache-hit/miss log lines.
    """

    # Daemon-side hard cap (PREFIX_CACHE_SLOTS in test_dflash.cpp). Any
    # configured cap > this is silently clamped down — exceeding it would
    # cause silent SNAPSHOT failures on slots ≥ 8.
    DAEMON_MAX_SLOTS = 8

    def __init__(self, *, daemon_stdin, await_reply, daemon_lock,
                 tokenizer, kv_k_type: str, fa_window: int,
                 cap: int = 4, log_prefix: str = "[pc]"):
        self.stdin = daemon_stdin
        self._await_reply = await_reply
        self.lock = daemon_lock
        self.log_prefix = log_prefix
        # Cache key fields — fixed at daemon spawn (env vars passed through).
        # Mismatched values across turns are not possible within one server
        # process, but they're still part of the hash so a daemon restart
        # with different flags doesn't return stale state.
        self.kv_k_type = kv_k_type
        self.fa_window = fa_window

        if cap > self.DAEMON_MAX_SLOTS:
            print(f"{log_prefix} cap={cap} exceeds daemon limit "
                  f"({self.DAEMON_MAX_SLOTS}); clamping", flush=True)
            cap = self.DAEMON_MAX_SLOTS
        self.cap = cap

        if cap <= 0:
            self.disabled = True
            return
        self.disabled = False

        self.entries: OrderedDict[bytes, int] = OrderedDict()  # hash → slot_id
        self.next_slot = 0
        self.im_end, self.im_start, self.system_t = _qwen_marker_ids(tokenizer)
        # Pending eviction: set by prepare_inline_snap when at cap; the old
        # entry is NOT removed until confirm_inline_snap succeeds.  This ensures
        # that if the request aborts before confirm runs, the old entry survives
        # and the daemon slot count stays consistent.
        self._pending_evict_key: bytes | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def boundary(self, ids: list[int]) -> int:
        """Return first boundary (system-prompt end), or -1. Legacy helper."""
        if self.disabled:
            return -1
        return find_prefix_boundary(ids, self.im_end, self.im_start, self.system_t)

    def _all_boundaries(self, ids: list[int]) -> list[int]:
        """Return all candidate cache cut-points in ascending order."""
        return find_all_boundaries(ids, self.im_end, self.im_start, self.system_t)

    def lookup(self, prompt_ids: list[int]) -> tuple[int, int] | None:
        """Return ``(slot_id, prefix_len)`` for the LONGEST cached prefix, or ``None``.

        Iterates all block-aligned turn boundaries in ``prompt_ids``, checks
        each against the LRU index, and returns the deepest (longest) match.

        The caller must already hold ``daemon_lock`` before inspecting the
        returned slot, since the slot id may be evicted by a concurrent
        request otherwise.
        """
        if self.disabled:
            return None
        candidates = self._all_boundaries(prompt_ids)
        best: tuple[int, int] | None = None   # (slot_id, prefix_len)
        for cut in candidates:
            key = hash_prefix(prompt_ids[:cut], self.kv_k_type, self.fa_window)
            if key in self.entries:
                if best is None or cut > best[1]:
                    best = (self.entries[key], cut)
                self.entries.move_to_end(key)   # mark fresh
        if best is not None:
            print(f"{self.log_prefix} lookup hit slot={best[0]} prefix_len={best[1]} "
                  f"(of {len(prompt_ids)} total)", flush=True)
        return best

    def prepare_inline_snap(self, prompt_ids: list[int]) -> tuple[int, int] | None:
        """Pick a target boundary + slot for inline snapshot during the next
        request. Returns ``(slot_id, target_cut)`` or ``None`` if no
        snapshot is needed (e.g. boundary already cached).

        Caller must:
          1. Append ``snap=<target_cut>:<slot_id>`` to the daemon command
             that runs the actual response (bare prompt OR ``RESTORE``).
          2. After the daemon emits ``[snap] inline slot=N cur_pos=M``
             during prefill, call ``confirm_inline_snap(slot_id, target_cut,
             prompt_ids)`` to register the entry in the LRU.

        For an agent loop that monotonically grows conversation history, the
        most valuable cache point is "end of the most recent completed
        assistant message" — i.e., the second-to-last `<|im_start|>`
        boundary. The LAST boundary is the current turn's opening, whose
        content hasn't been generated yet.
        """
        if self.disabled:
            return None
        candidates = self._all_boundaries(prompt_ids)
        if not candidates:
            return None
        target_cut = candidates[-2] if len(candidates) >= 2 else candidates[-1]

        target_key = hash_prefix(prompt_ids[:target_cut],
                                  self.kv_k_type, self.fa_window)
        if target_key in self.entries:
            self.entries.move_to_end(target_key)
            return None   # already cached

        # Pick slot: when at cap, reserve the LRU slot WITHOUT evicting yet.
        # The actual eviction is deferred to confirm_inline_snap so that if the
        # request aborts before confirm runs, the old entry survives and the
        # daemon slot count stays consistent.
        if len(self.entries) >= self.cap:
            # Peek at LRU without removing.
            old_key = next(iter(self.entries))
            slot = self.entries[old_key]
            self._pending_evict_key = old_key
        else:
            slot = self.next_slot
            self.next_slot = (self.next_slot + 1) % self.cap
            self._pending_evict_key = None

        return (slot, target_cut)

    def confirm_inline_snap(self, slot: int, target_cut: int,
                             prompt_ids: list[int]) -> None:
        """Register an inline snapshot in the LRU after the daemon has
        successfully fired ``[snap] inline``. Called from the caller after
        the actual response stream completes.

        If prepare_inline_snap reserved a slot by displacing an LRU entry,
        the eviction happens HERE (atomically with the insert), so an aborted
        request that never reaches confirm leaves the old entry intact.
        """
        if self.disabled:
            return
        # Atomically evict the reserved old entry (if any) and insert the new one.
        if self._pending_evict_key is not None:
            self.entries.pop(self._pending_evict_key, None)
            self._pending_evict_key = None
        key = hash_prefix(prompt_ids[:target_cut],
                          self.kv_k_type, self.fa_window)
        self.entries[key] = slot
        print(f"{self.log_prefix} inline-snap committed slot={slot} "
              f"prefix_len={target_cut}", flush=True)

    def abort_inline_snap(self, slot: int) -> None:
        """Release the reservation made by prepare_inline_snap.

        At-cap case: prepare_inline_snap peeked at the LRU (old_key -> slot)
        and stashed old_key in _pending_evict_key WITHOUT removing it. We
        cannot tell from here whether the daemon already committed the
        snapshot to ``slot`` before the failure was observed:
          - If it didn't: old_key -> slot is still semantically valid and
            we should keep it.
          - If it did:    slot now holds the NEW prompt's KV, so old_key
            -> slot is stale and a future lookup would return data that
            doesn't match the key.
        Without daemon-side query we conservatively assume the worst and
        drop old_key from the LRU. We accept losing one valid cache entry
        in exchange for never returning a wrong-KV restore. Callers that
        know the daemon did NOT process the snap (e.g. early validation
        failure before any send) should evict only the pending key — but
        in practice, every failure path that calls this happens AFTER the
        daemon command was issued, so the conservative drop is correct.
        """
        if self.disabled:
            return
        if self._pending_evict_key is not None:
            self.entries.pop(self._pending_evict_key, None)
            self._pending_evict_key = None

    # ------------------------------------------------------------------
    # Option 3: full-compress-result cache
    # ------------------------------------------------------------------
    # When pFlash compression is enabled the existing prefix-cache path above
    # silently no-ops (compressed tokens lack Qwen chat-template markers so
    # find_all_boundaries returns []).  The full-cache path solves this by
    # caching the compressed cur_bin keyed on the ORIGINAL raw prompt_ids so
    # that an identical long prompt sent a second time skips BOTH the drafter
    # dance AND the target prefill.
    #
    # Slot allocation: prefix-cache uses slots [0, cap); full-cache uses slots
    # [cap, cap + full_cap).  Both are initialised at PrefixCache construction
    # time; the daemon cap (8) is shared, so prefix_cap + full_cap <= 8.
    # ------------------------------------------------------------------

    def init_full_cache(self, full_cap: int,
                        cache_dir: str | None = None) -> None:
        """Initialise the full-cache pool.  Must be called once after __init__
        if you want Option 3 to be active.  Idempotent if called again with
        the same parameters.

        Parameters
        ----------
        full_cap:
            Number of daemon slots reserved for full-cache entries.
            prefix_cap (self.cap) + full_cap must not exceed DAEMON_MAX_SLOTS.
        cache_dir:
            Directory to persist cur_bin files across requests.
            Defaults to /tmp/dflash-pflash-cache/.
        """
        if self.disabled or full_cap <= 0:
            self._full_cap = 0
            self._full_disabled = True
            return

        # Idempotency guard: a second call would otherwise reset full_entries +
        # slot allocator and orphan any cur_bin files already on disk.
        if not getattr(self, "_full_disabled", True):
            return

        remaining = self.DAEMON_MAX_SLOTS - self.cap
        if full_cap > remaining:
            print(f"{self.log_prefix} full-cache cap={full_cap} would exceed "
                  f"daemon limit (prefix uses {self.cap}); clamping to {remaining}",
                  flush=True)
            full_cap = remaining
        if full_cap <= 0:
            self._full_cap = 0
            self._full_disabled = True
            return

        self._full_cap = full_cap
        self._full_disabled = False
        # Slots used by the full-cache start AFTER the prefix-cache slots.
        self._full_slot_base = self.cap
        self._full_next_slot = 0  # relative; absolute = _full_slot_base + _full_next_slot
        # LRU map: prompt_ids_hash -> (absolute_slot, cached_cur_bin_path, cur_ids_len)
        self.full_entries: OrderedDict[bytes, tuple[int, str, int]] = OrderedDict()
        # Pending eviction: the LRU entry reserved for the next confirm.
        self._full_pending_evict_key: bytes | None = None
        self._full_pending_evict_path: str | None = None

        cache_dir_path = Path(cache_dir) if cache_dir else Path("/tmp/dflash-pflash-cache")
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        self._full_cache_dir = cache_dir_path
        print(f"{self.log_prefix} full-cache enabled: cap={full_cap} "
              f"slots=[{self._full_slot_base},{self._full_slot_base + full_cap}) "
              f"dir={cache_dir_path}", flush=True)

    def lookup_full(self, prompt_ids: list[int]) -> tuple[int, str, int] | None:
        """Exact-match on full prompt_ids hash (keyed on raw, pre-compression ids).

        Returns ``(slot, cached_cur_bin_path, cur_ids_len)`` on hit, else None.
        The cur_bin_path points to a file in the persistent cache dir that the
        caller passes directly to the daemon as a RESTORE command's second arg.

        Caller must hold daemon_lock before inspecting the returned slot.
        """
        if getattr(self, "_full_disabled", True):
            return None
        key = hash_prefix(prompt_ids, self.kv_k_type, self.fa_window)
        entry = self.full_entries.get(key)
        if entry is None:
            return None
        slot, cur_bin_path, cur_ids_len = entry
        # Verify the cached file still exists (could have been deleted externally).
        if not Path(cur_bin_path).exists():
            self.full_entries.pop(key, None)
            return None
        self.full_entries.move_to_end(key)  # mark fresh in LRU
        print(f"{self.log_prefix} full-cache hit slot={slot} "
              f"cur_ids_len={cur_ids_len} key={key.hex()[:8]}", flush=True)
        return slot, cur_bin_path, cur_ids_len

    def prepare_full_snap(self, prompt_ids: list[int]) -> tuple[int, int] | None:
        """Reserve a daemon slot for the full-prefill snapshot.

        Returns ``(absolute_slot, 0)`` — the second element is a placeholder;
        the real target_pos (== len(cur_ids)) is supplied by the caller to
        ``confirm_full_snap``.  Returns None if full-cache is disabled or the
        prompt is already cached.
        """
        if getattr(self, "_full_disabled", True):
            return None
        key = hash_prefix(prompt_ids, self.kv_k_type, self.fa_window)
        if key in self.full_entries:
            self.full_entries.move_to_end(key)
            return None  # already cached

        # Pick a slot, deferring eviction until confirm succeeds.
        if len(self.full_entries) >= self._full_cap:
            old_key = next(iter(self.full_entries))
            old_slot, old_path, _ = self.full_entries[old_key]
            self._full_pending_evict_key = old_key
            self._full_pending_evict_path = old_path
            abs_slot = old_slot
        else:
            abs_slot = self._full_slot_base + self._full_next_slot
            self._full_next_slot = (self._full_next_slot + 1) % self._full_cap
            self._full_pending_evict_key = None
            self._full_pending_evict_path = None

        return abs_slot, 0  # 0 is a placeholder; real pos passed to confirm

    def confirm_full_snap(self, slot: int, prompt_ids: list[int],
                          cur_bin_src: str | Path, cur_ids_len: int) -> None:
        """Persist cur_bin_src into the cache dir and register the entry.

        ``cur_bin_src`` is the path to the tempfile written by _maybe_compress;
        its content is copied (not moved, to keep the original available for the
        daemon) into the persistent cache dir before registering.

        Atomically evicts the LRU entry (and its on-disk file) if one was
        reserved by prepare_full_snap.
        """
        if getattr(self, "_full_disabled", True):
            return

        key = hash_prefix(prompt_ids, self.kv_k_type, self.fa_window)
        dest = self._full_cache_dir / (key.hex() + ".bin")

        try:
            shutil.copy2(str(cur_bin_src), str(dest))
        except OSError as exc:
            print(f"{self.log_prefix} full-cache: failed to copy cur_bin "
                  f"({cur_bin_src} -> {dest}): {exc}", flush=True)
            # Don't evict the old entry — leave cache consistent.
            self._full_pending_evict_key = None
            self._full_pending_evict_path = None
            return

        # Atomically evict the reserved entry (if any) and insert new one.
        if self._full_pending_evict_key is not None:
            evicted_path = self._full_pending_evict_path
            self.full_entries.pop(self._full_pending_evict_key, None)
            if evicted_path:
                Path(evicted_path).unlink(missing_ok=True)
            self._full_pending_evict_key = None
            self._full_pending_evict_path = None

        self.full_entries[key] = (slot, str(dest), cur_ids_len)
        print(f"{self.log_prefix} full-cache committed slot={slot} "
              f"cur_ids_len={cur_ids_len} key={key.hex()[:8]}", flush=True)

    def abort_full_snap(self, slot: int) -> None:
        """Cancel a prepare_full_snap reservation without registering anything.

        Clears the pending eviction so the old LRU entry is not evicted.
        """
        if getattr(self, "_full_disabled", True):
            return
        self._full_pending_evict_key = None
        self._full_pending_evict_path = None

    # Legacy out-of-band snapshot (kept for backward-compatibility tests
    # that call it directly; new code uses prepare_inline_snap +
    # confirm_inline_snap so the snapshot rides on the actual response).
    async def maybe_snapshot(self, prompt_ids: list[int],
                              token_stream_consumer=None) -> None:
        if self.disabled:
            return
        prep = self.prepare_inline_snap(prompt_ids)
        if prep is None:
            return
        slot, cut = prep

        import os, struct, tempfile
        fd, tmp_path = tempfile.mkstemp(suffix="_prefix.bin")
        with os.fdopen(fd, "wb") as f:
            for t in prompt_ids[:cut]:
                f.write(struct.pack("<i", int(t)))
        confirmed = False
        try:
            self._send(f"{tmp_path} 0\n")
            if token_stream_consumer is not None:
                await token_stream_consumer()
            self._send(f"SNAPSHOT {slot}\n")
            await self._await_reply("[snap] slot=")
            self.confirm_inline_snap(slot, cut, prompt_ids)
            confirmed = True
        finally:
            try: os.unlink(tmp_path)
            except OSError: pass
            if not confirmed:
                # The SNAPSHOT command may already have been processed on the
                # daemon (the slot now holds new-prompt KV) even though we
                # didn't observe the ack. To keep daemon and Python state
                # consistent: free the slot daemon-side (best effort), then
                # drop the at-cap LRU mapping in abort_inline_snap.
                try:
                    self._send(f"FREE_SNAPSHOT {slot}\n")
                    await self._await_reply("[snap] freed slot=", timeout=2.0)
                except Exception:
                    # If the daemon doesn't ack the free either, the slot may
                    # leak in the daemon until next startup_sync — but that's
                    # bounded and recoverable, while a stale hash->slot
                    # mapping in Python would cause silent KV corruption on
                    # the next lookup.
                    pass
                self.abort_inline_snap(slot)

    async def startup_sync(self) -> None:
        """Query the daemon for existing slots and free them all.

        Called once at server startup to ensure Python's hash table is
        consistent with the daemon's slot state (both empty after this).
        """
        if self.disabled:
            return
        async with self.lock:
            self._send("LIST_SLOTS\n")
            reply = await self._await_reply("[snap] slots=")
            slots_str = reply.split("[snap] slots=", 1)[1].strip()
            if not slots_str:
                return
            orphans = [s.strip() for s in slots_str.split(",") if s.strip()]
            for s in orphans:
                self._send(f"FREE_SNAPSHOT {s}\n")
                await self._await_reply("[snap] freed slot=")
            print(f"{self.log_prefix} freed {len(orphans)} orphaned daemon slots",
                  flush=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send(self, line: str) -> None:
        self.stdin.write(line.encode("utf-8"))
        self.stdin.flush()
