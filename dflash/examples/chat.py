"""
Multi-turn chat REPL with streaming output.

    python3 examples/chat.py

Tokens print as they are committed. Model reloads once per turn;
a daemon-mode binary that keeps the model resident is a planned follow-up.
"""
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "models" / "Qwen3.5-27B-Q4_K_M.gguf"
DRAFT_ROOT = ROOT / "models" / "draft"
BIN = ROOT / "build" / ("test_dflash.exe" if sys.platform == "win32" else "test_dflash")

BUDGET = 22
N_GEN  = 512
SYSTEM = "You are a concise, helpful assistant."


def resolve_draft() -> Path:
    if DRAFT_ROOT.is_file():
        return DRAFT_ROOT
    if DRAFT_ROOT.is_dir():
        for st in DRAFT_ROOT.rglob("model.safetensors"):
            return st
    sys.exit(
        f"draft weights not found under {DRAFT_ROOT}. Download them as documented in the README."
    )


def tokenize(tok, text: str, path: Path) -> int:
    ids = tok.encode(text, add_special_tokens=False)
    with open(path, "wb") as f:
        for t in ids:
            f.write(struct.pack("<i", int(t)))
    return len(ids)


def stream_generate(tok, bin_path: Path, target: Path, draft: Path,
                    prompt_bin: Path, n_gen: int, budget: int,
                    stop_ids: set[int]) -> str:
    """Spawn test_dflash, read tokens as they arrive, print + accumulate."""
    with tempfile.TemporaryDirectory() as d:
        out_bin = Path(d) / "out.bin"
        r, w = os.pipe()
        cmd = [str(bin_path), str(target), str(draft), str(prompt_bin),
               str(n_gen), str(out_bin),
               "--fast-rollback", "--ddtree", f"--ddtree-budget={budget}",
               f"--stream-fd={w}"]
        proc = subprocess.Popen(cmd, pass_fds=(w,),
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE)
        os.close(w)

        text = ""
        try:
            while True:
                b = os.read(r, 4)
                if not b or len(b) < 4:
                    break
                tok_id = struct.unpack("<i", b)[0]
                if tok_id in stop_ids:
                    break
                s = tok.decode([tok_id])
                text += s
                sys.stdout.write(s); sys.stdout.flush()
        except KeyboardInterrupt:
            proc.terminate()
            raise
        finally:
            proc.wait()
        return text


def main():
    if not BIN.is_file():
        sys.exit(f"binary not found at {BIN}. Build: "
                 "cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && "
                 "cmake --build build --target test_dflash -j")
    if not TARGET.is_file():
        sys.exit(f"target GGUF not found at {TARGET}")

    draft = resolve_draft()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B",
                                        trust_remote_code=True)
    stop_ids = set()
    for s in ("<|im_end|>", "<|endoftext|>"):
        ids = tok.encode(s, add_special_tokens=False)
        if ids: stop_ids.add(ids[0])

    messages = [{"role": "system", "content": SYSTEM}]
    print("Luce DFlash chat. Ctrl+C to quit a reply, Ctrl+D to exit.\n",
          flush=True)

    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user:
            continue

        messages.append({"role": "user", "content": user})
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)

        with tempfile.TemporaryDirectory() as d:
            in_bin = Path(d) / "in.bin"
            n = tokenize(tok, prompt, in_bin)
            sys.stdout.write("bot> "); sys.stdout.flush()
            try:
                reply = stream_generate(tok, BIN, TARGET, draft, in_bin,
                                        N_GEN, BUDGET, stop_ids)
            except KeyboardInterrupt:
                sys.stdout.write("\n[interrupted]\n")
                messages.pop()
                continue

        messages.append({"role": "assistant", "content": reply.strip()})


if __name__ == "__main__":
    main()
