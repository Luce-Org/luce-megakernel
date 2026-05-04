"""Generate common-words extraction cases where every list is relevant."""

from __future__ import annotations

import argparse
import json
import random

from transformers import AutoTokenizer


COMMON_POOL = [
    "amber", "basil", "cobalt", "dahlia", "ember", "fennel", "garnet", "hazel",
    "indigo", "juniper", "kelp", "lichen", "marble", "nectar", "onyx", "pebble",
]


def word_for(rng: random.Random, prefix: str) -> str:
    letters = "abcdefghijklmnopqrstuvwxyz"
    return prefix + "".join(rng.choices(letters, k=7))


def gen_one(seed: int, target_tokens: int, n_lists: int, list_size: int, n_common: int, n_decoys: int) -> dict:
    rng = random.Random(seed)
    common = rng.sample(COMMON_POOL, n_common)
    decoys = [word_for(rng, "decoy") for _ in range(n_decoys)]

    filler = (
        "This inventory paragraph is background text for the archive. "
        "Only the numbered word lists should be used for the answer. "
        "Do not infer from this sentence. "
    )
    target_chars = int(target_tokens * 4.0)
    filler_budget = max(0, target_chars - n_lists * list_size * 10)
    between = filler * (filler_budget // max(1, n_lists * len(filler)) + 1)
    between = between[: max(20, filler_budget // max(1, n_lists))]

    parts: list[str] = [
        "Find the words that appear in every numbered list. Each numbered list matters.\n\n"
    ]
    for i in range(n_lists):
        words = list(common)
        # Each decoy appears in most, but not all, lists. Missing-list positions
        # differ so every list is needed to reject at least one plausible decoy.
        for j, decoy in enumerate(decoys):
            missing = (seed + j * 7) % n_lists
            if i != missing:
                words.append(decoy)
        while len(words) < list_size:
            words.append(word_for(rng, "w"))
        rng.shuffle(words)
        parts.append(f"List {i + 1}: " + ", ".join(words) + ".\n")
        parts.append(between + "\n")

    question = (
        "Question: Which words appear in every numbered list? "
        "Return only those words as a comma-separated list, with no explanation. /no_think\nAnswer:"
    )
    prompt = "".join(parts) + "\n" + question
    return {"prompt": prompt, "answer": common, "forbidden": decoys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--ctx", type=int, default=64000)
    ap.add_argument("--lists", type=int, default=48)
    ap.add_argument("--list-size", type=int, default=48)
    ap.add_argument("--common", type=int, default=5)
    ap.add_argument("--decoys", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.6-27B")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    with open(args.out, "w") as f:
        for i in range(args.n):
            ex = gen_one(9000 + i, args.ctx, args.lists, args.list_size, args.common, args.decoys)
            ex["n_tokens"] = len(tok.encode(ex["prompt"]))
            f.write(json.dumps(ex) + "\n")
            print(f"case {i}: ntok={ex['n_tokens']} answer={','.join(ex['answer'])}")
    print(f"saved {args.n} cases to {args.out}")


if __name__ == "__main__":
    main()
