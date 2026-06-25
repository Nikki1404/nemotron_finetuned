python3.11 - <<'PY'
import json, wave
from pathlib import Path

out = Path("data/manifests/train_short_manifest.json")
rows = []

for wav in Path("data/audio_chunks").glob("*/*.wav"):
    with wave.open(str(wav), "rb") as w:
        dur = w.getnframes() / w.getframerate()
    if dur <= 22:
        rows.append({
            "audio_filepath": str(wav.resolve()),
            "duration": dur,
            "text": "hi hello thank you for calling inspira financial",
            "target_lang": "en-US",
            "language": "en-US",
        })

out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print("wrote", out, "rows=", len(rows))
PY
