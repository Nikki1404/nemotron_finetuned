cd /home/CORP/re_nikitav/nemotron_finetuned && mkdir -p ft_models results/hparam_tuning

cd /home/CORP/re_nikitav/nemotron_finetuned && docker run --gpus all -it --rm -v $PWD:/workspace -v $PWD/ft_models:/srv/models nemotron_finetuned bash

cd /workspace && mkdir -p /srv/models results/hparam_tuning && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && export CUDA_HOME=/usr/local/cuda-12.4 && export LD_LIBRARY_PATH=/usr/local/cuda-12.4/nvvm/lib64:$LD_LIBRARY_PATH && unset NUMBA_CUDA_USE_NVIDIA_BINDING

cd /workspace && python3.11 scripts/augment_train_manifest.py --train-manifest data/manifests/train_aligned_manifest.json --out-manifest data/manifests/train_aligned_aug_manifest.json --out-audio-dir data/audio_aug --keep-original

cd /workspace && python3.11 scripts/finetune_nemotron.py --train-manifest data/manifests/train_aligned_aug_manifest.json --val-manifest data/manifests/val_aligned_manifest.json --base-model /srv/nemotron-3.5-asr-streaming-0.6b.nemo --output-nemo /srv/models/finetuned_nemotron_v1_lr3e6_ep2.nemo --freeze-mode decoder_only --max-epochs 2 --batch-size 1 --lr 3e-6 --language en-US --precision bf16-mixed

cp /srv/models/finetuned_nemotron_v1_lr3e6_ep2.nemo /srv/models/finetuned_nemotron_final.nemo && ls -lh /srv/models

cd /workspace && python3.11 scripts/evaluate_manifest.py --model /srv/models/finetuned_nemotron_final.nemo --manifest data/manifests/test_aligned_manifest.json --language en-US --output-jsonl results/hparam_tuning/final_eval.jsonl

cd /workspace && chmod +x scripts/run_hyparam_tuning.sh && bash scripts/run_hyparam_tuning.sh


cd /workspace && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && export CUDA_HOME=/usr/local/cuda-12.4 && export LD_LIBRARY_PATH=/usr/local/cuda-12.4/nvvm/lib64:$LD_LIBRARY_PATH && unset NUMBA_CUDA_USE_NVIDIA_BINDING && mkdir -p /srv/models results/hparam_tuning && chmod +x scripts/run_hyparam_tuning.sh && bash scripts/run_hyparam_tuning.shs


#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


def load_jsonl(path):
    rows = []
    p = Path(path)
    if not p.exists():
        print(f"[warn] missing: {path}")
        return rows

    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def avg(rows, key):
    vals = [float(r[key]) for r in rows if key in r and r[key] is not None]
    return sum(vals) / len(vals) if vals else None


def extract_entities(text):
    text = text.lower()
    numbers = re.findall(r"\b\d{2,}\b", text)
    tickets = re.findall(r"\bt\s*k\s*t\s*[a-z0-9\s]{4,20}\b", text)
    money = re.findall(r"\b\d+\s*(dollars?|usd)\b|\$\s*\d+", text)
    return {
        "numbers": set(numbers),
        "tickets": set(tickets),
        "money": set(money),
    }


def entity_score(rows):
    total = 0
    hit = 0

    for r in rows:
        ref = r.get("reference", "")
        pred = r.get("prediction", "")

        ref_e = extract_entities(ref)
        pred_e = extract_entities(pred)

        for k in ref_e:
            for item in ref_e[k]:
                total += 1
                if item in pred_e[k]:
                    hit += 1

    return (hit / total * 100) if total else None


def summarize(name, file):
    rows = load_jsonl(file)

    return {
        "model": name,
        "file": file,
        "rows": len(rows),
        "avg_wer": avg(rows, "wer"),
        "avg_cer": avg(rows, "cer"),
        "entity_score": entity_score(rows),
    }


def fmt(v):
    if v is None:
        return "NA"
    return f"{v:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--v1", required=True)
    parser.add_argument("--v2", required=True)
    parser.add_argument("--v3", required=True)
    parser.add_argument("--out", default="results/model_comparison_report.md")
    args = parser.parse_args()

    reports = [
        summarize("base", args.base),
        summarize("v1_lr3e6_ep2", args.v1),
        summarize("v2_lr2e6_ep3", args.v2),
        summarize("v3_lr1e6_ep3", args.v3),
    ]

    valid = [r for r in reports if r["avg_wer"] is not None]
    best = min(valid, key=lambda r: r["avg_wer"]) if valid else None

    lines = []
    lines.append("# Nemotron Fine-tuned Model Comparison\n")
    lines.append("| Model | Rows | Avg WER % | Avg CER % | Entity Score % |")
    lines.append("|---|---:|---:|---:|---:|")

    for r in reports:
        marker = " ⭐ Best" if best and r["model"] == best["model"] else ""
        lines.append(
            f"| {r['model']}{marker} | {r['rows']} | "
            f"{fmt(r['avg_wer'])} | {fmt(r['avg_cer'])} | {fmt(r['entity_score'])} |"
        )

    lines.append("\n## Recommendation\n")
    if best:
        lines.append(
            f"Best model by WER is **{best['model']}** "
            f"with WER **{fmt(best['avg_wer'])}%** and CER **{fmt(best['avg_cer'])}%**."
        )
    else:
        lines.append("No valid evaluation rows found.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nSaved report: {out}")


if __name__ == "__main__":
    main()
