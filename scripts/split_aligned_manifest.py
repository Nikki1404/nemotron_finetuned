#!/usr/bin/env python3

import json
import random
from pathlib import Path

inp = Path("data/manifests/aligned_chunk_manifest.json")
out = Path("data/manifests")

rows = [
    json.loads(line)
    for line in inp.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

rows = [r for r in rows if r.get("text", "").strip()]

random.seed(42)
random.shuffle(rows)

n = len(rows)
train = rows[: int(n * 0.8)]
val = rows[int(n * 0.8): int(n * 0.9)]
test = rows[int(n * 0.9):]

for name, data in [
    ("train_aligned_manifest.json", train),
    ("val_aligned_manifest.json", val),
    ("test_aligned_manifest.json", test),
]:
    path = out / name
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in data) + "\n",
        encoding="utf-8",
    )
    print(name, len(data))
