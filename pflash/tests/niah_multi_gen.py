"""Generate harder multi-needle retrieval cases at long context."""

from __future__ import annotations

import argparse
import json
import random

from transformers import AutoTokenizer


FILLER = (
    "The meadow report says the grass is green. "
    "The weather note says the sky is blue. "
    "The archive note says the sun is yellow. "
    "A traveler went there and back again. "
)
NEEDLE_TMPL = "The special magic {key} number is: {value}."


def gen_one(seed: int, target_tokens: int, n_needles: int) -> dict:
    rng = random.Random(seed)
    keys: list[str] = []
    values: list[str] = []
    needles: list[str] = []
    for _ in range(n_needles):
        key = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=8))
        value = "".join(rng.choices("0123456789", k=7))
        keys.append(key)
        values.append(value)
        needles.append(NEEDLE_TMPL.format(key=key, value=value))

    target_chars = int(target_tokens * 4.0)
    filler = (FILLER * (target_chars // len(FILLER) + 1))[:target_chars]
    spans: list[str] = []
    prev = 0
    for i, needle in enumerate(needles):
        insert = int(target_chars * (i + 1) / (n_needles + 1))
        insert += rng.randint(-200, 200)
        insert = max(prev, min(len(filler), insert))
        spans.append(filler[prev:insert])
        spans.append(" " + needle + " ")
        prev = insert
    spans.append(filler[prev:])
    body = "".join(spans)
    key_list = ", ".join(keys)
    question = (
        f"What are the special magic numbers for these keys, in this exact order: {key_list}? "
        "Give only the comma-separated numbers and no explanation. /no_think"
    )
    prompt = (
        "Below is a long passage. Answer the question at the end based ONLY on information in the passage.\n\n"
        f"{body}\n\nQuestion: {question}\nAnswer:"
    )
    return {"prompt": prompt, "answer": values, "keys": keys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--ctx", type=int, default=60000)
    ap.add_argument("--needles", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.6-27B")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    with open(args.out, "w") as f:
        for i in range(args.n):
            ex = gen_one(seed=4200 + i, target_tokens=args.ctx, n_needles=args.needles)
            ex["n_tokens"] = len(tok.encode(ex["prompt"]))
            f.write(json.dumps(ex) + "\n")
            print(f"case {i}: ntok={ex['n_tokens']} needles={args.needles} answers={','.join(ex['answer'])}")
    print(f"saved {args.n} cases to {args.out}")


if __name__ == "__main__":
    main()
