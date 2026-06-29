#!/usr/bin/env python3

import argparse
import json
import subprocess
import wave
from pathlib import Path


def duration_sec(path):
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def run(cmd):
    subprocess.run(cmd, check=True)


def augment_audio(src, out_dir):
    src = Path(src)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("speed095", ["-filter:a", "atempo=0.95"]),
        ("speed105", ["-filter:a", "atempo=1.05"]),
        ("volm3", ["-filter:a", "volume=-3dB"]),
        ("tel8k", ["-ar", "8000", "-ac", "1", "-af", "highpass=f=300,lowpass=f=3400"]),
    ]

    outputs = []

    for name, opts in configs:
        out = out_dir / f"{src.stem}_{name}.wav"
        cmd = ["ffmpeg", "-y", "-i", str(src)] + opts + ["-ar", "16000", "-ac", "1", str(out)]
        run(cmd)
        outputs.append((name, out))

    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--out-audio-dir", default="data/audio_aug")
    ap.add_argument("--keep-original", action="store_true")
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.train_manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    final_rows = []

    for i, row in enumerate(rows, 1):
        src = Path(row["audio_filepath"])

        if args.keep_original:
            final_rows.append(row)

        use_case = row.get("use_case", "unknown")
        safe_case = "".join(c if c.isalnum() else "_" for c in use_case.lower())
        out_dir = Path(args.out_audio_dir) / safe_case

        print(f"[{i}/{len(rows)}] augmenting {src}")

        for aug_name, aug_path in augment_audio(src, out_dir):
            new_row = dict(row)
            new_row["audio_filepath"] = str(aug_path.resolve())
            new_row["duration"] = round(duration_sec(aug_path), 3)
            new_row["augmentation"] = aug_name
            final_rows.append(new_row)

    out_path = Path(args.out_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in final_rows) + "\n",
        encoding="utf-8",
    )

    print("Original rows:", len(rows))
    print("Final rows:", len(final_rows))
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
