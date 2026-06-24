#!/usr/bin/env python3
"""Evaluate a Nemotron/NeMo ASR .nemo model on a NeMo manifest."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def normalize_for_metric(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def edit_distance(a: List[str], b: List[str]) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            old = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
            prev = old
    return dp[-1]


def wer(ref: str, hyp: str) -> float:
    r = normalize_for_metric(ref).split()
    h = normalize_for_metric(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    return edit_distance(r, h) / len(r)


def cer(ref: str, hyp: str) -> float:
    r = list(normalize_for_metric(ref).replace(" ", ""))
    h = list(normalize_for_metric(hyp).replace(" ", ""))
    if not r:
        return 0.0 if not h else 1.0
    return edit_distance(r, h) / len(r)


def safe_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return safe_text(x[0]) if x else ""
    if hasattr(x, "text"):
        return str(x.text or "")
    return str(x)


def load_model(model_path: str, device: str, language: str):
    import torch
    import nemo.collections.asr as nemo_asr

    cls = nemo_asr.models.EncDecRNNTBPEModelWithPrompt
    if model_path.endswith(".nemo") or Path(model_path).exists():
        model = cls.restore_from(model_path, map_location="cpu")
    else:
        model = cls.from_pretrained(model_path, map_location="cpu")

    model = model.cuda() if device == "cuda" and torch.cuda.is_available() else model.cpu()
    try:
        model.set_inference_prompt(language)
    except Exception as e:
        print(f"[warn] set_inference_prompt({language}) failed: {e}")
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=".nemo path or HF model name")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--language", default="en-US")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="")
    ap.add_argument("--output-jsonl", default="", help="Alias for --output")
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    manifest = Path(args.manifest)
    rows: List[Dict] = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    audio_files = [r["audio_filepath"] for r in rows]

    model = load_model(args.model, args.device, args.language)
    print(f"[eval] Loaded model: {args.model}")
    print(f"[eval] Files: {len(audio_files)}")

    hyps = model.transcribe(audio_files, batch_size=args.batch_size)
    results = []
    wers, cers = [], []
    for row, hyp_obj in zip(rows, hyps):
        hyp = safe_text(hyp_obj)
        ref = row["text"]
        row_wer = wer(ref, hyp)
        row_cer = cer(ref, hyp)
        wers.append(row_wer)
        cers.append(row_cer)
        results.append({
            "audio_filepath": row["audio_filepath"],
            "use_case": row.get("use_case", ""),
            "reference": ref,
            "prediction": hyp,
            "wer": row_wer,
            "cer": row_cer,
        })
        print("\n---", row.get("use_case", Path(row["audio_filepath"]).name), "---")
        print("REF:", ref[:500])
        print("HYP:", hyp[:500])
        print(f"WER: {row_wer*100:.2f}% | CER: {row_cer*100:.2f}%")

    avg_wer = sum(wers) / max(1, len(wers))
    avg_cer = sum(cers) / max(1, len(cers))
    print("\n====================")
    print(f"Average WER: {avg_wer*100:.2f}%")
    print(f"Average CER: {avg_cer*100:.2f}%")
    print("====================")

    output_path = args.output_jsonl or args.output
    out = Path(output_path) if output_path else manifest.with_suffix(".predictions.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved predictions: {out}")


if __name__ == "__main__":
    main()
