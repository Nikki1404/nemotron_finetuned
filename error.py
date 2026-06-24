cat > scripts/evaluate_manifest.py <<'PY'
#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import torch
import nemo.collections.asr as nemo_asr


def norm_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9ñáéíóúü\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def edit_distance(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def wer(ref, hyp):
    r = norm_text(ref).split()
    h = norm_text(hyp).split()
    if not r:
        return 0.0 if not h else 100.0
    return 100.0 * edit_distance(r, h) / len(r)


def cer(ref, hyp):
    r = norm_text(ref).replace(" ", "")
    h = norm_text(hyp).replace(" ", "")
    if not r:
        return 0.0 if not h else 100.0
    return 100.0 * edit_distance(list(r), list(h)) / len(r)


def extract_text(x):
    if isinstance(x, str):
        return x
    if isinstance(x, list) and x:
        return extract_text(x[0])
    if hasattr(x, "text"):
        return x.text
    if isinstance(x, dict):
        return x.get("text") or x.get("pred_text") or str(x)
    return str(x)


def transcribe_one(model, audio_path, language):
    # Nemotron prompt model supports set_default_prompt / change_decoding_strategy in some NeMo builds.
    # Try all safe options, then call transcribe on one file.
    try:
        model.set_default_prompt(language)
    except Exception:
        pass

    try:
        model.set_inference_prompt(language)
    except Exception:
        pass

    with torch.no_grad():
        out = model.transcribe([audio_path], batch_size=1, verbose=False)

    return extract_text(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--language", default="en-US")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-jsonl", default="eval_predictions.jsonl")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]

    print(f"[eval] Loading model: {args.model}")
    model = nemo_asr.models.ASRModel.restore_from(args.model, map_location=args.device)
    model.eval()
    model = model.to(args.device)

    print(f"[eval] Files: {len(rows)}")
    total_wer = 0.0
    total_cer = 0.0
    outputs = []

    for idx, row in enumerate(rows, start=1):
        audio = row["audio_filepath"]
        ref = row.get("text", "")
        lang = row.get("target_lang") or row.get("language") or args.language

        print(f"[eval] {idx}/{len(rows)} {audio} lang={lang}")
        hyp = transcribe_one(model, audio, lang)

        row_wer = wer(ref, hyp)
        row_cer = cer(ref, hyp)

        total_wer += row_wer
        total_cer += row_cer

        outputs.append({
            "audio_filepath": audio,
            "language": lang,
            "reference": ref,
            "prediction": hyp,
            "wer": row_wer,
            "cer": row_cer,
        })

        print(f"  WER: {row_wer:.2f}% | CER: {row_cer:.2f}%")
        print(f"  PRED: {hyp[:300]}")

    avg_wer = total_wer / max(1, len(rows))
    avg_cer = total_cer / max(1, len(rows))

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for o in outputs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    print("\n========== SUMMARY ==========")
    print(f"Files: {len(rows)}")
    print(f"Average WER: {avg_wer:.2f}%")
    print(f"Average CER: {avg_cer:.2f}%")
    print(f"Saved predictions: {args.output_jsonl}")


if __name__ == "__main__":
    main()
PY




python3.11 scripts/evaluate_manifest.py --model /srv/nemotron-3.5-asr-streaming-0.6b.nemo --manifest data/manifests/test_manifest.json --language en-US --output-jsonl results_base.jsonl
