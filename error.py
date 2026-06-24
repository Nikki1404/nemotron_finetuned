python3.11 - <<'PY'
import json
from pathlib import Path

for p in [
    "data/manifests/train_manifest.json",
    "data/manifests/val_manifest.json",
    "data/manifests/test_manifest.json",
]:
    path = Path(p)
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        obj["target_lang"] = "en-US"
        rows.append(json.dumps(obj, ensure_ascii=False))
    path.write_text("\n".join(rows) + "\n")
    print("updated", p)
PY


python3.11 scripts/evaluate_manifest.py \
  --model /srv/nemotron-3.5-asr-streaming-0.6b.nemo \
  --manifest data/manifests/test_manifest.json \
  --language en-US \
  --output-jsonl results_base.jsonl
