"""
Streaming one-shot generation.

    python3 scripts/run.py --prompt "def fibonacci(n):"
    echo "Write a haiku about GPUs" | python3 scripts/run.py

Tokens print live as they are committed by the spec-decode loop.
Auto-applies Qwen3.5 chat template unless --raw is passed.
"""
import argparse
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def default_paths():
    return {
        "target": "models/Qwen3.5-27B-Q4_K_M.gguf",
        "draft":  "models/draft",
        "bin":    "build/test_dflash" + (".exe" if sys.platform == "win32" else ""),
    }


def resolve_draft(draft_dir: str) -> str:
    if draft_dir.endswith(".safetensors"):
        p = Path(draft_dir)
        if p.is_file():
            return str(p)
        raise FileNotFoundError(f"draft safetensors not found: {draft_dir}")

    p = Path(draft_dir)
    if p.is_file():
        return str(p)
    if p.is_dir():
        for st in p.rglob("model.safetensors"):
            return str(st)

    raise FileNotFoundError(
        f"no model.safetensors under {draft_dir}. Download it as documented in the README, or pass --draft explicitly."
    )


def tokenize(tokenizer, text: str, out_path: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    with open(out_path, "wb") as f:
        for t in ids:
            f.write(struct.pack("<i", int(t)))
    return len(ids)


def main():
    d = default_paths()
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--n-gen", type=int, default=256)
    ap.add_argument("--target", type=str, default=d["target"])
    ap.add_argument("--draft",  type=str, default=d["draft"])
    ap.add_argument("--bin",    type=str, default=d["bin"])
    ap.add_argument("--budget", type=int, default=22)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--system", type=str, default=None)
    ap.add_argument("--kv-q4", action="store_true",
                    help="Q4_0 KV cache (required for max_ctx=131072)")
    ap.add_argument("--kv-tq3", action="store_true",
                    help="TQ3_0 KV cache (3.5 bpv, near-lossless)")
    ap.add_argument("--max-ctx", type=int, default=0,
                    help="Override max KV context (default: auto-fit "
                         "prompt+n_gen+block, aligned to 256). Passing a "
                         "value much larger than needed (e.g. 131072 on a "
                         "16K prompt) massively degrades attention speed "
                         "because the kernel strides over unused KV.")
    args = ap.parse_args()

    prompt_text = args.prompt if args.prompt else sys.stdin.read().strip()
    if not prompt_text:
        sys.exit("no prompt")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B",
                                              trust_remote_code=True)

    if args.raw:
        text = prompt_text
    else:
        msgs = []
        if args.system:
            msgs.append({"role": "system", "content": args.system})
        msgs.append({"role": "user", "content": prompt_text})
        text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)

    draft_path = resolve_draft(args.draft)
    im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    im_end_id = im_end_id[0] if im_end_id else -1

    args.bin = str(Path(args.bin).resolve())
    bin_dir = str(Path(args.bin).parent)
    dll_dir = str(Path(args.bin).parent / "bin")
    env = {**os.environ}
    if sys.platform == "win32":
        env["PATH"] = dll_dir + os.pathsep + bin_dir + os.pathsep + env.get("PATH", "")
    if args.kv_q4:
        env["DFLASH27B_KV_Q4"] = "1"
    if args.kv_tq3:
        env["DFLASH27B_KV_TQ3"] = "1"

    with tempfile.TemporaryDirectory() as tmp:
        in_bin  = os.path.join(tmp, "prompt.bin")
        out_bin = os.path.join(tmp, "out.bin")
        n_tok = tokenize(tokenizer, text, in_bin)
        # Auto-fit max_ctx: prompt + gen + a small verify pad, aligned to
        # FATTN_KQ_STRIDE=256. Oversizing this is a performance trap —
        # attention compute scales with the allocated max_ctx, not the
        # actual filled kv_len, so a 131072 max_ctx on a 16K prompt runs
        # attention 8× slower than necessary (verified: 32K prompt with
        # max_ctx=131072 + --kv-q4 → 1035s prefill vs 38s at max_ctx=32768).
        if args.max_ctx > 0:
            max_ctx = args.max_ctx
        else:
            pad = 64  # covers q_len=16 + ddtree budget up to 22 with margin
            max_ctx = ((n_tok + args.n_gen + pad + 255) // 256) * 256
        print(f"[run] prompt {n_tok} tokens, streaming up to {args.n_gen} tokens, max_ctx={max_ctx}",
              file=sys.stderr, flush=True)

        r, w = os.pipe()
        if sys.platform == "win32":
            import msvcrt
            os.set_inheritable(w, True)
            stream_fd_val = int(msvcrt.get_osfhandle(w))
        else:
            stream_fd_val = w
        cmd = [args.bin, args.target, draft_path, in_bin,
               str(args.n_gen), out_bin,
               "--fast-rollback", "--ddtree", f"--ddtree-budget={args.budget}",
               f"--max-ctx={max_ctx}",
               f"--stream-fd={stream_fd_val}"]
        if sys.platform == "win32":
            proc = subprocess.Popen(cmd, env=env, close_fds=False,
                                    stdout=sys.stderr,
                                    stderr=subprocess.PIPE)
        else:
            proc = subprocess.Popen(cmd, pass_fds=(w,), env=env,
                                    stdout=sys.stderr,
                                    stderr=subprocess.PIPE)
        os.close(w)

        generated = 0
        buffer = b""
        try:
            while True:
                b = os.read(r, 4)
                if not b or len(b) < 4:
                    break
                tok_id = struct.unpack("<i", b)[0]
                generated += 1
                if tok_id == im_end_id:
                    break
                sys.stdout.write(tokenizer.decode([tok_id]))
                sys.stdout.flush()
        except KeyboardInterrupt:
            proc.terminate()
            sys.stderr.write("\n[run] interrupted\n")
            return
        finally:
            proc.wait()
            err = proc.stderr.read()
            if err:
                sys.stderr.buffer.write(err)
                sys.stderr.flush()
        print(file=sys.stderr, flush=True)
        print(f"[run] generated {generated} tokens", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
