# Nemotron 3.5 ASR — Real-Time Streaming Server

Real-time multilingual ASR server powered by
**`nvidia/nemotron-3.5-asr-streaming-0.6b`** — a 600M-parameter
Cache-Aware FastConformer-RNNT model supporting **40 language-locales**
from a single checkpoint, with sub-100ms end-of-utterance latency.

---

## Project structure

```
nemotron_asr/
├── app/
│   ├── asr_engines/
│   │   ├── base.py            # ASREngine / ASRSession abstract interfaces
│   │   └── nemotron_asr.py    # NemotronStreamingASR + StreamingSession
│   ├── config.py              # Config dataclass + MODEL_MAP
│   ├── factory.py             # build_engine(cfg) factory
│   ├── main.py                # FastAPI + WebSocket endpoint
│   ├── streaming_session.py   # VAD + ASR orchestration per connection
│   └── vad.py                 # AdaptiveEnergyVAD
├── client.py                  # Test client (mic + WAV file modes)
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Quick start (local, no Docker)

```bash
pip install -r requirements.txt

# Start server (auto-detects language)
ASR_LANGUAGE=auto uvicorn app.main:app --host 0.0.0.0 --port 8002

# Or pin to English
ASR_LANGUAGE=en-US uvicorn app.main:app --host 0.0.0.0 --port 8002
```

---

## Docker

```bash
# Build (downloads ~2.4GB model into image)
docker build -t nemotron-asr .

# Run with GPU
docker compose up

# Override language at runtime
docker run --gpus all -p 8002:8002 \
  -e ASR_LANGUAGE=es-US \
  nemotron-asr
```

---

## Client usage

### Microphone (real-time)

```bash
# English (default)
python client.py --mic

# Spanish
python client.py --mic --language es-US

# Auto language detection
python client.py --mic --language auto
```

### WAV file

```bash
# Transcribe a file (as fast as possible)
python client.py --file audio.wav

# Spanish WAV file
python client.py --file audio_es.wav --language es-US

# Simulate real-time pacing
python client.py --file audio.wav --realtime

# Custom server URL
python client.py --mic --url ws://192.168.1.100:8002/asr/realtime-custom-vad
```

### Health check

```bash
python client.py --health
# or
curl http://localhost:8002/health
```

---

## WebSocket protocol

**Client → Server**

1. Send init JSON (text frame):

```json
{
  "backend":     "nemotron",
  "sample_rate": 16000,
  "language":    "en-US"
}
```

2. Stream raw **PCM16 mono** binary frames continuously.

**Server → Client**

```json
{ "type": "partial", "text": "Hello wor",      "t_start": 420 }
{ "type": "final",   "text": "Hello world.",    "t_start": 420 }
```

`t_start` is time-to-first-byte in milliseconds (ms from utterance start to first partial).

---

## Supported languages

| Locale | Language           | Locale | Language          |
|--------|--------------------|--------|-------------------|
| en-US  | English (US)       | es-US  | Spanish (US)      |
| en-GB  | English (UK)       | es-ES  | Spanish (Spain)   |
| fr-FR  | French             | fr-CA  | French (Canada)   |
| de-DE  | German             | it-IT  | Italian           |
| pt-BR  | Portuguese (BR)    | pt-PT  | Portuguese (PT)   |
| ru-RU  | Russian            | nl-NL  | Dutch             |
| pl-PL  | Polish             | cs-CZ  | Czech             |
| ar-AR  | Arabic             | hi-IN  | Hindi             |
| ja-JP  | Japanese           | ko-KR  | Korean            |
| vi-VN  | Vietnamese         | tr-TR  | Turkish           |
| nb-NO  | Norwegian Bokmål   | he-IL  | Hebrew            |
| da-DK  | Danish             | sv-SE  | Swedish           |
| bg-BG  | Bulgarian          | fi-FI  | Finnish           |
| hr-HR  | Croatian           | sk-SK  | Slovak            |
| uk-UA  | Ukrainian          | zh-CN  | Chinese Mandarin  |
| el-GR  | Greek              | hu-HU  | Hungarian         |
| ro-RO  | Romanian           | et-EE  | Estonian          |
| lt-LT  | Lithuanian         | lv-LV  | Latvian           |
| mt-MT  | Maltese            | sl-SI  | Slovenian         |
| th-TH  | Thai               | auto   | Auto-detect       |

---

## Environment variables

| Variable             | Default    | Description                                      |
|----------------------|------------|--------------------------------------------------|
| `ASR_BACKEND`        | `nemotron` | Engine backend                                   |
| `MODEL_NAME`         | HF repo ID | Path to .nemo file or HuggingFace model ID       |
| `DEVICE`             | `cuda`     | `cuda` or `cpu`                                  |
| `ASR_LANGUAGE`       | `auto`     | Server-default language (overridable per client) |
| `SAMPLE_RATE`        | `16000`    | Server audio sample rate (Hz)                    |
| `CONTEXT_RIGHT`      | `1`        | Encoder lookahead frames (0=80ms … 13=1120ms)    |
| `NEMO_END_SILENCE_MS`| `700`      | Silence duration to trigger end-of-utterance     |
| `NEMO_MIN_UTT_MS`    | `200`      | Min utterance length before endpointing          |
| `FINALIZE_PAD_MS`    | `500`      | Silence padding before final flush               |
| `LOG_LEVEL`          | `INFO`     | Python logging level                             |

---

## Concurrent multi-language (advanced)

`set_inference_prompt()` is model-level state — shared across concurrent
sessions on the same engine instance. For **simultaneous** en-US + es-US
sessions, add separate engine entries:

```python
# config.py
MODEL_MAP = {
    "nemotron-en": "nvidia/nemotron-3.5-asr-streaming-0.6b",
    "nemotron-es": "nvidia/nemotron-3.5-asr-streaming-0.6b",
}
```

Then in `factory.py`, map `nemotron-en` → `language="en-US"` and
`nemotron-es` → `language="es-US"`. Clients send `"backend": "nemotron-en"`
or `"backend": "nemotron-es"` in the init message.
