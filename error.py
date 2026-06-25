python3.11 - <<'PY'
import json, wave
from pathlib import Path

src = Path("data/audio_16k")
out_dir = Path("data/audio_short")
out_dir.mkdir(parents=True, exist_ok=True)

# Split into 8 sec chunks
import subprocess
for wav in src.glob("*.wav"):
    base = wav.stem
    subprocess.run([
        "ffmpeg", "-y", "-i", str(wav),
        "-f", "segment",
        "-segment_time", "8",
        "-c", "copy",
        str(out_dir / f"{base}_%03d.wav")
    ], check=True)

rows = []
for wav in sorted(out_dir.glob("*.wav")):
    with wave.open(str(wav), "rb") as w:
        dur = w.getnframes() / w.getframerate()
    if 0.5 <= dur <= 8.5:
        rows.append({
            "audio_filepath": str(wav.resolve()),
            "duration": dur,
            "text": "hi hello thank you for calling inspira financial",
            "target_lang": "en-US",
            "language": "en-US"
        })

Path("data/manifests/train_8sec_manifest.json").write_text(
    "\n".join(json.dumps(r) for r in rows) + "\n"
)

print("created rows:", len(rows))
PY



PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3.11 scripts/finetune_nemotron.py \
  --train-manifest data/manifests/train_8sec_manifest.json \
  --val-manifest data/manifests/val_manifest.json \
  --base-model /srv/nemotron-3.5-asr-streaming-0.6b.nemo \
  --output-nemo /srv/models/nemotron_inspira_decoder_ft.nemo \
  --freeze-mode decoder_only \
  --max-epochs 1 \
  --batch-size 1 \
  --lr 5e-6 \
  --language en-US \
  --precision 32-true
