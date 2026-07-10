# Nemotron ASR Fine-Tuning — Code Review Explanation

## What This Project Does (Big Picture)

This is an **end-to-end system to fine-tune and serve NVIDIA's Nemotron 3.5 ASR (Automatic Speech Recognition) model** on custom domain audio — specifically banking/insurance customer-service call recordings. The output is a containerised real-time speech-to-text server that:

- Accepts live microphone or telephony audio over a **WebSocket**
- Also accepts recorded audio files via an **OpenAI-compatible REST API** (so tools like LiveKit can talk to it)
- Returns both partial (live) and final transcripts
- Uses a custom Voice Activity Detector (VAD) to decide when someone is speaking

---

## Project File Map

```
nemotron_finetuned_updated/
│
├── Dockerfile                          ← GPU container definition
├── requirements.txt                    ← Python dependencies for the server
│
├── 01_enter_training_container.sh      ← Step 1: build Docker image & open shell
├── 02_prepare_baseline_train_eval_inside_container.sh  ← Step 2: data prep + first fine-tune
├── 03_run_finetuned_server.sh          ← Step 3: run server with fine-tuned model
├── 04_run_base_server.sh               ← Run server with base (untuned) model
│
├── app/                                ← The ASR inference server
│   ├── config.py                       ← Centralised config via env vars
│   ├── factory.py                      ← Builds the right ASR engine from config
│   ├── main.py                         ← FastAPI app: WebSocket + REST endpoints
│   ├── streaming_session.py            ← VAD + ASR session orchestrator
│   ├── vad.py                          ← Adaptive Energy Voice Activity Detector
│   ├── asr_number_normalizer.py        ← Converts spoken numbers to digits
│   └── asr_engines/
│       ├── base.py                     ← Abstract interface for any ASR engine
│       └── nemotron_asr.py             ← Nemotron streaming RNNT implementation
│
├── scripts/                            ← Fine-tuning pipeline scripts
│   ├── prepare_dataset.py              ← CSV + WAV → NeMo JSONL manifests
│   ├── augment_train_manifest.py       ← Creates augmented variants (speed, noise, etc.)
│   ├── auto_align_chunks_with_base_asr.py  ← Chunk long WAVs + auto-label with base ASR
│   ├── finetune_nemotron.py            ← The actual fine-tuning script
│   ├── evaluate_manifest.py            ← WER/CER evaluation
│   ├── split_by_usecase_manifest.py    ← Train/val/test split by use-case
│   ├── compare_models_report.py        ← Markdown comparison table
│   └── run_hyparam_tuning.sh           ← Orchestrates 3 fine-tuning experiments
│
├── data/
│   ├── inspira_transcripts.csv         ← Ground-truth transcripts per use-case
│   ├── audio_16k/                      ← Re-sampled 16kHz WAV files (generated)
│   ├── audio_aug/                      ← Augmented WAVs (generated)
│   ├── audio_chunks/                   ← Chunked WAVs (generated)
│   └── manifests/                      ← JSONL manifests (generated)
│
├── raw_wavs/                           ← Original customer call recordings
├── ft_models/                          ← Output directory for .nemo fine-tuned models
├── results/                            ← WER/CER eval outputs + comparison reports
└── client.py                           ← Test client (WebSocket audio sender)
```

---

## 1. Infrastructure — `Dockerfile` + `requirements.txt`

### Why a Custom Docker Image?

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
```

**Why this base image:** NeMo and PyTorch need exact CUDA/cuDNN versions. Using NVIDIA's official image eliminates driver mismatch issues. `cudnn-runtime` means cuDNN (the GPU-accelerated deep learning library) is pre-installed.

### Python 3.11 Compiled from Source

```dockerfile
RUN wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz && \
    ./configure --enable-optimizations && make -j$(nproc)
```

**Why build from source:** The Ubuntu 22.04 apt repository ships Python 3.10. NeMo's current main branch requires 3.11. `--enable-optimizations` enables profile-guided optimisation for ~10% faster Python execution.

### Torch Installation Order Matters

```dockerfile
# Step 5: NeMo (pulls its own torch dependency)
RUN pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main"

# Step 6: FORCE-reinstall specific Torch AFTER NeMo
RUN pip install --force-reinstall torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

**Why:** NeMo's pip install would pull a newer/incompatible Torch. The `--force-reinstall` overwrites it with the exact CUDA 12.4 build. `torch.nn.Buffer` (required by NeMo's attention cache) was added in 2.6.0, so the version must be exactly 2.6.0.

### Build-time Model Bake-in

```dockerfile
RUN python3.11 - <<'EOF'
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/nemotron-3.5-asr-streaming-0.6b")
    model.save_to("/srv/nemotron-3.5-asr-streaming-0.6b.nemo")
EOF
```

**Why:** Downloads the 600M-parameter model from Hugging Face and saves it into the image layer. The container starts in ~3 seconds instead of downloading several GB every time.

### `requirements.txt` — Server Dependencies

| Package | Why it's needed |
|---|---|
| `fastapi` | Async HTTP + WebSocket framework |
| `uvicorn[standard]` | ASGI server to run FastAPI; `[standard]` adds WebSocket + HTTP/2 support |
| `websockets` | Low-level WebSocket handling |
| `numpy` | PCM audio arithmetic |
| `resampy` | High-quality audio resampling (client sends 8kHz, server needs 16kHz) |
| `librosa` | Audio utilities (loaded lazily) |
| `scipy` | Signal processing utilities |
| `soundfile` | Reading/writing WAV and FLAC files |
| `python-multipart` | Required by FastAPI for file upload parsing (`UploadFile`) |
| `huggingface_hub` | Model downloads from HuggingFace |
| `nemo-text-processing` | Inverse Text Normalisation (ITN) — converts "three hundred dollars" → "$300" |

---

## 2. App Configuration — `app/config.py`

```python
@dataclass(frozen=True)
class Config:
    asr_backend: str = os.getenv("ASR_BACKEND", "nemotron")
    vad_start_margin: float = float(os.getenv("VAD_START_MARGIN", "1.8"))
    vad_min_noise_rms: float = float(os.getenv("VAD_MIN_NOISE_RMS", "0.002"))
    pre_speech_ms: int = int(os.getenv("PRE_SPEECH_MS", "500"))
    max_utt_ms: int = int(os.getenv("MAX_UTT_MS", "30000"))
```

**Why `@dataclass(frozen=True)`:** Makes the config object immutable after creation — no accidental mutation. All values come from environment variables with sensible defaults, so the same Docker image can serve base or fine-tuned models by just changing env vars.

**Key tuning knobs and why:**
- `vad_start_margin=1.8` — Speech must be 1.8× louder than background noise to be considered speech (prevents false-starts on noisy telephony lines)
- `vad_min_noise_rms=0.002` — Minimum noise floor estimate (prevents division by zero in very quiet environments)
- `pre_speech_ms=500` — Capture 500ms before speech detection triggers, so the first syllable is never clipped
- `max_utt_ms=30000` — Force-flush utterance buffer at 30 seconds to prevent memory bloat on long silences

---

## 3. Engine Abstraction — `app/asr_engines/base.py`

```python
@dataclass(frozen=True)
class EngineCaps:
    streaming: bool       # true streaming (incremental state)
    partials: bool        # can emit partial transcripts during speech
    ttft_meaningful: bool # Time-To-First-Token is a meaningful latency metric

class ASRSession(Protocol):
    def accept_pcm16(self, pcm16: bytes) -> None: ...
    def step_if_ready(self) -> Optional[str]: ...
    def finalize(self, pad_ms: int) -> str: ...

class ASREngine(ABC):
    def load(self) -> float: ...
    def new_session(self, max_buffer_ms: int) -> ASRSession: ...
```

**Why this design:**
- `Protocol` (structural typing) means any class with these 3 methods qualifies as an `ASRSession` without needing to inherit. This is duck typing with IDE/type-checker support.
- `ASREngine` is an `ABC` (Abstract Base Class) — enforces that all engine implementations provide `load()` and `new_session()`.
- `EngineCaps` is a declarative capability manifest — the `StreamingSession` checks `engine.caps.partials` before trying to call `step_if_ready()`, so a non-streaming engine would work too.

---

## 4. Nemotron Engine — `app/asr_engines/nemotron_asr.py`

### Model Loading

```python
def load(self) -> float:
    self.model = nemo_asr.models.EncDecRNNTBPEModelWithPrompt.restore_from(
        self.model_name, map_location="cpu"
    )
    self.model = self.model.cuda()
    self.model.encoder.set_default_att_context_size([70, int(self.context_right)])
    self.model.change_decoding_strategy(OmegaConf.create({
        "strategy": "greedy",
        "greedy": {"max_symbols": 15, "loop_labels": False, "use_cuda_graph_decoder": False}
    }))
    self.model.eval()
    self.model.preprocessor.featurizer.dither = 0.0
```

**Key decisions explained:**

- **`EncDecRNNTBPEModelWithPrompt`** — This is NVIDIA's Nemotron model class. The architecture is: Conformer Encoder → RNN-T Decoder. "Prompt" means it accepts a language token (e.g., `en-US`, `es-US`) to guide recognition. "BPE" = Byte-Pair Encoding tokeniser.

- **`map_location="cpu"` then `.cuda()`** — Load to CPU first to avoid GPU OOM during model construction, then move to GPU. Standard practice for large models.

- **`set_default_att_context_size([70, 2])`** — Sets the Conformer's attention window: look back 70 frames (~700ms) and right-context 2 frames (~20ms). Smaller right context = lower latency, but slightly worse accuracy. This is the streaming constraint.

- **`strategy="greedy"`** — Greedy decoding (take the highest-probability token at each step) instead of beam search. Beam search is more accurate but 3-5× slower and incompatible with real-time streaming.

- **`dither=0.0`** — Disables random noise injection (used during training for regularisation). At inference, this would add variability to transcriptions — we want deterministic outputs.

- **`model.eval()`** — Switches off dropout and batch normalisation's training mode. Critical for consistent inference.

### Streaming Inference Loop

```python
@torch.inference_mode()
def stream_transcribe(self, audio_f32, cache, prev_hyp, prev_pred_out, emitted_frames, force_flush=False):
    mel, mel_len = self.model.preprocessor(input_signal=audio_tensor, length=audio_len)
    
    chunk_mel = mel[:, :, chunk_start:chunk_end]
    
    (prev_pred_out, texts, cache0, cache1, cache2, prev_hyp) = \
        self.model.conformer_stream_step(
            processed_signal=chunk_mel,
            cache_last_channel=cache[0],
            cache_last_time=cache[1],
            cache_last_channel_len=cache[2],
            keep_all_outputs=False,
            previous_hypotheses=prev_hyp,
            previous_pred_out=prev_pred_out,
            drop_extra_pre_encoded=drop_extra,
            return_transcription=True
        )
```

**Why `@torch.inference_mode()`:** More aggressive than `torch.no_grad()` — also disables view tracking and version counters. Reduces memory usage by ~20% and speeds up inference.

**The three-cache tuple `(cache0, cache1, cache2)`:**
- `cache_last_channel` — Key/Value cache for the Conformer's self-attention (what the model "remembers" from previous audio)
- `cache_last_time` — Convolutional context cache for the convolution layers
- `cache_last_channel_len` — Lengths of valid entries in the attention cache

These caches are the heart of streaming — they carry state between audio chunks so the model doesn't need to reprocess old audio. Resetting them starts a new utterance.

**Chunk calculation (`chunk_start`, `chunk_end`, `emitted_frames`):**

```python
if emitted_frames == 0:
    chunk_start = 0
    chunk_end = min(self.shift_frames, available)
else:
    chunk_start = max(0, emitted_frames - self.pre_cache_frames)
    chunk_end = min(emitted_frames + self.shift_frames, available)
```

The model processes audio in `shift_frames` (typically 8 frames = 80ms) steps. `pre_cache_frames` is a small overlap from the previous chunk — the Conformer's convolution layers need a few frames of context to avoid edge artifacts. `emitted_frames` tracks how far we've processed into the accumulating audio buffer.

### Warmup

```python
def _warmup(self):
    sess = self.new_session(max_buffer_ms=3000)
    silence = np.zeros(int(self.sr * 1.0), dtype=np.float32)
    sess.accept_pcm16(pcm16)
    sess.finalize(pad_ms=400)
```

**Why:** The first GPU inference call is always slow (CUDA kernel compilation, memory allocation). Running a dummy silent audio through the model after loading "warms up" all the CUDA kernels, so the first real user gets normal latency.

### `safe_text()` Helper

```python
def safe_text(h: Any) -> str:
    if h is None: return ""
    if isinstance(h, str): return h
    if isinstance(h, (list, tuple)) and len(h) > 0: return safe_text(h[0])
    if hasattr(h, "text"): return h.text or ""
    return str(h)
```

**Why:** NeMo's `transcribe()` and `conformer_stream_step()` return different output types depending on the version and config — sometimes a `str`, sometimes a `Hypothesis` object, sometimes a list of either. This recursive normaliser handles all cases without crashing.

---

## 5. Voice Activity Detection — `app/vad.py`

```python
class AdaptiveEnergyVAD:
    def push_frame(self, frame_pcm16: bytes):
        e = self._rms(frame_pcm16)
        if not self.in_speech:
            alpha = 0.97
            self.noise_rms = max(self.min_noise_rms, alpha * self.noise_rms + (1.0 - alpha) * e)
        
        threshold = max(self.min_noise_rms, self.noise_rms) * self.start_margin
        is_speech = e >= threshold
        self.ring.append(frame_pcm16)
        
        pre_roll = None
        if (not self.in_speech) and is_speech:
            self.in_speech = True
            pre_roll = b"".join(self.ring)
        
        return is_speech, pre_roll
```

**How it works:**
1. **RMS Energy** — Convert PCM16 bytes to float32, divide by 32768 (max int16) to normalize to [-1, 1], compute RMS (Root Mean Square). RMS approximates perceived loudness.
2. **Adaptive noise floor** — During silence, the noise estimate is updated with exponential moving average (`alpha=0.97` means 97% old value, 3% new). This adapts to different room/microphone noise levels.
3. **Threshold** — Speech is detected when energy is 1.8× above the noise floor (`start_margin`).
4. **Pre-roll ring buffer** — A circular buffer of the last 500ms of audio. When speech starts, we send this pre-roll to the ASR engine so the first syllable (which started before detection) is captured.

**Why a custom VAD instead of WebRTC VAD or Silero?**
- WebRTC VAD is aggressive for telephony but can cut early on quiet speakers
- Silero is more accurate but adds ~50ms latency per frame
- This custom energy VAD costs ~0.1ms per frame and is tunable via env vars

---

## 6. Streaming Session Orchestration — `app/streaming_session.py`

```python
class StreamingSession:
    def process_chunk(self, pcm: bytes) -> list:
        self.raw_buf.extend(pcm)
        while len(self.raw_buf) >= self.frame_bytes:
            frame = bytes(self.raw_buf[:self.frame_bytes])
            del self.raw_buf[:self.frame_bytes]
            is_speech, pre = self.vad.push_frame(frame)
            
            if pre and not self.utt_started:
                self.utt_started = True
                self.session.accept_pcm16(pre)   # send pre-roll to ASR
            
            if not self.utt_started:
                continue                          # discard silence
            
            self.session.accept_pcm16(frame)
            
            if self.engine.caps.partials:
                text = self.session.step_if_ready()
                if text:
                    events.append(("partial", text, ttfb_ms))
            
            # End of utterance detection
            if not is_speech and self.silence_ms >= self.engine.end_silence_ms:
                final = self.session.finalize(self.engine.finalize_pad_ms)
                events.append(("final", final, ttfb_ms))
                self.reset()
        
        return events
```

**The flow:**
1. Raw PCM bytes arrive → sliced into fixed 20ms frames
2. Each frame → VAD check
3. When speech starts → pre-roll + all subsequent frames go to the RNNT session
4. Every `shift_frames` (80ms), the RNNT may emit a partial transcript
5. When silence exceeds `end_silence_ms` (900ms) → `finalize()` forces the model to flush its buffer and emit the definitive final transcript
6. Session resets → ready for next utterance

**Why `finalize_pad_ms=800`:** The RNNT decoder is slightly behind the audio (it needs to see the last few tokens to confirm the final word). Padding 800ms of silence after the last speech frame gives the decoder time to "catch up" and emit all remaining tokens.

**Time-to-First-Byte (TTFB):** Measures latency from when the utterance started (`t_utt_start`) to when the first partial came back. This is logged per-session and useful for monitoring perceived responsiveness.

---

## 7. FastAPI Server — `app/main.py`

### Startup & Engine Preloading

```python
@app.on_event("startup")
async def startup_event():
    await preload_engines()
```

Loading the Nemotron model takes ~5-15 seconds. Doing it at server startup means the first real request is fast. `ENGINE_CACHE` is a module-level dict — safe for single-process uvicorn.

### WebSocket Endpoint `/asr/realtime-custom-vad`

```
Client → sends JSON init: {"backend": "nemotron", "sample_rate": 8000, "language": "en-US"}
Client → streams raw PCM16 binary frames (any size)
Server → emits: {"type": "partial", "text": "hello"} or {"type": "final", "text": "hello world"}
Client → sends {"type": "eof"} to signal end of audio
Server → emits remaining final + {"type": "done"}
```

**Resampling:** If the client sends 8kHz audio (telephony) but the server needs 16kHz, `resampy.resample()` converts on the fly. `resampy` uses band-limited sinc interpolation — much higher quality than simple linear interpolation.

**Thread offloading:**
```python
events = await loop.run_in_executor(None, session.process_chunk, data)
```
PyTorch inference blocks the GIL and is CPU-intensive during preprocessing. Running it in `run_in_executor` prevents it from blocking other WebSocket connections. `None` = use the default `ThreadPoolExecutor`.

### REST Endpoint `/v1/audio/transcriptions`

Mimics the OpenAI Whisper API signature. This lets LiveKit and other VoIP platforms point their ASR integration at this server without any code changes.

```python
pcm_bytes = convert_upload_to_pcm16_16k_mono(input_bytes=audio_bytes, suffix=suffix)
transcript = await run_pcm_through_session(pcm_bytes, engine, language)
return {"text": transcript}
```

**`convert_upload_to_pcm16_16k_mono`** uses ffmpeg to handle any audio format (WAV, MP3, M4A, OGG, WEBM) — much simpler than implementing multiple decoders.

### Session Logging

Every session (WebSocket or HTTP) gets its own timestamped directory under `/srv/audio_logs/`:
```
/srv/audio_logs/20260709_142530_192_168_1_5_54321/
  ├── audio.raw.pcm     ← raw PCM16 bytes captured in real-time
  ├── audio.wav         ← same audio as listenable WAV (written on session close)
  ├── metadata.json     ← session info, duration, final transcript
  └── events.jsonl      ← every partial/final event with timestamps
```

**Why log raw PCM:** WAV headers are written after the session ends (you need to know total byte count). Raw PCM is appended live so nothing is lost if the server crashes mid-session.

### Number Normalisation — `app/asr_number_normalizer.py`

```python
def normalize_asr_numbers(text: str, use_itn: bool = True) -> str:
    text = apply_custom_asr_rules(text)  # digit-by-digit: "one two three" → "123"
    if use_itn: text = apply_itn(text)   # NeMo ITN: "three hundred" → "300"
    text = apply_custom_asr_rules(text)  # clean up any remaining word-numbers
    text = normalize_ticket_ids(text)    # "T K T A B C 1 2" → "TKTABC12"
    return text
```

**Why two passes of `apply_custom_asr_rules`?** NeMo's ITN may emit digit strings that still need custom cleanup (e.g., ticket IDs). The custom rules are deterministic regex+dict lookups; ITN is a grammar-based normaliser that handles compound numbers like "twenty-three thousand four hundred fifty-six."

**`MULTIPLIERS` dict** (`double` → 2, `triple` → 3): Handles phone number dictation — callers often say "double zero" for "00" or "triple seven" for "777".

---

## 8. Fine-Tuning Pipeline

### Step 1: Data Preparation — `scripts/prepare_dataset.py`

```
Input:  data/inspira_transcripts.csv  (use_case, transcript)
        raw_wavs/Card_lost.wav, withdraw_money.wav, etc.

Output: data/audio_16k/<use_case>.wav   (16kHz mono, ffmpeg-converted)
        data/manifests/train_manifest.json   (NeMo JSONL format)
        data/manifests/val_manifest.json
        data/manifests/test_manifest.json
```

**NeMo Manifest format (one JSON per line):**
```json
{"audio_filepath": "/abs/path/card_lost.wav", "duration": 45.3, "text": "i lost my card can you help me", "use_case": "Card lost"}
```

**Transcript normalisation** — converts to lowercase, removes punctuation, fixes OCR/copy-paste artifacts (`"Social SecurityNumber"` → `"Social Security Number"`). ASR models are trained on unpunctuated, lowercase text — forcing the fine-tuning labels to match this distribution is critical.

**Train/val/test split:** With only 7 recordings, the code uses `5 train / 1 val / 1 test` (shuffled). Very small by ML standards, but this is intentional — the goal is domain adaptation (teaching the model domain vocabulary), not building from scratch.

### Step 2: Audio Augmentation — `scripts/augment_train_manifest.py`

With only 5 training recordings, augmentation is essential to prevent overfitting.

| Augmentation | ffmpeg filter | Purpose |
|---|---|---|
| `speed095` | `atempo=0.95` | Simulates slow talkers |
| `speed105` | `atempo=1.05` | Simulates fast talkers |
| `volm3` | `volume=-3dB` | Quieter speech |
| `volp3` | `volume=3dB` | Louder speech |
| `tel8k` | `highpass=300, lowpass=3400, ar=8000` | Telephony simulation (bandlimited audio, typical of phone calls) |
| `soft_noise` | `anoisesrc=pink:amplitude=0.002` | Adds light pink background noise |

5 recordings × 6 augmentations = **30 training samples** (or 35 with `--keep-original`).

**Why `atempo` not `asetrate`?** `atempo` changes speed without changing pitch — the audio sounds natural. `asetrate` would make voices sound chipmunk-like.

### Step 3: Long-Audio Chunking — `scripts/auto_align_chunks_with_base_asr.py`

Used when you have long recordings (entire call recordings, not pre-segmented). It:
1. Splits the recording into ~30-second chunks with ffmpeg's `-segment` filter
2. Transcribes each chunk with the **base model** (no fine-tuning)
3. Uses `SequenceMatcher` (fuzzy string matching) to align the base ASR output with the ground-truth CSV transcript
4. Assigns ground-truth text to chunks where alignment confidence exceeds a threshold

**Why not just use the base ASR transcript as ground truth?** The base model may have errors on domain terminology (e.g., it might transcribe "COBRA coverage" as "cobra coverage faq" or miss insurance jargon). The CSV has human-verified transcripts.

### Step 4: Fine-Tuning — `scripts/finetune_nemotron.py`

#### Architecture

```python
model = nemo_asr.models.EncDecRNNTBPEModelWithPrompt.restore_from(model_path, map_location='cpu')
```

**EncDecRNNTBPE** breaks down as:
- `Enc` = Conformer Encoder (Convolutional + Self-Attention per layer)
- `Dec` = RNNT Decoder (RNN Transducer — joint network of acoustic encoder + prediction network)
- `RNNT` = Recurrent Neural Network Transducer (streaming-compatible, no CTC alignment needed)
- `BPE` = Byte-Pair Encoding (subword tokeniser — handles "nemotron", "COBRA", etc. as sub-tokens)
- `WithPrompt` = the model accepts a language prompt token (e.g., `<en-US>`) prepended to the decoder

#### Freeze Modes

```python
def set_freeze_mode(model, freeze_mode):
    if freeze_mode == 'decoder_only':
        for p in model.parameters(): p.requires_grad = False
        for name, module in model.named_modules():
            if any(k in name.lower() for k in ('decoder', 'joint', 'prompt_kernel')):
                for p in module.parameters(): p.requires_grad = True
```

| Mode | What trains | Use case |
|---|---|---|
| `decoder_only` | Only decoder, joint network, prompt embeddings | Best for small datasets; encoder already knows acoustics |
| `last_encoder` | Decoder + last 5% of encoder layers | When domain has unusual pronunciation |
| `none` | Everything | Full fine-tuning; needs large dataset |

**Why `decoder_only` is the default:** The Conformer encoder learns general acoustic features (phonemes, prosody). Domain adaptation mainly requires the decoder to learn new vocabulary probabilities. Training only ~15% of parameters also reduces overfitting risk on tiny datasets.

#### `patch_batch_prompt_indices()`

```python
def patch_batch_prompt_indices(model, prompt_index):
    def add_prompt(batch):
        if isinstance(batch, (tuple, list)) and len(batch) == 4:
            signal, signal_len, transcript, transcript_len = batch
            prompt_indices = torch.full((signal.shape[0],), prompt_index, ...)
            return signal, signal_len, transcript, transcript_len, prompt_indices
        return batch
    model.training_step = types.MethodType(new_training_step, model)
```

**Why this monkey-patch:** NeMo's standard data loader for fine-tuning emits 4-element batches (audio, lengths, text, text_lengths). The Nemotron `WithPrompt` model's training step expects a 5th element (prompt_index). This patch intercepts each batch and injects the language prompt index without modifying NeMo's source code.

#### Optimiser & Schedule

```python
model.cfg.optim = OmegaConf.create({
    'name': 'adamw',
    'lr': args.lr,
    'betas': [0.9, 0.98],
    'weight_decay': 0.001,
    'sched': {'name': 'CosineAnnealing', 'warmup_steps': 10, 'min_lr': lr/30.0}
})
```

**Why AdamW:** Adam with decoupled weight decay. The original Adam applies weight decay as L2 regularisation, which interacts incorrectly with the adaptive learning rate. AdamW fixes this and is the standard for transformer fine-tuning.

**Why `betas=[0.9, 0.98]`:** `beta2=0.98` (vs. default 0.999) makes the second-moment estimate react faster to gradient changes. This works better with Conformers — same as the original "Attention is All You Need" paper.

**Why CosineAnnealing with warmup:** Learning rate starts at 0, ramps up for 10 steps (prevents large gradient updates on random initialisation), then decays following a cosine curve to `lr/30`. This avoids catastrophic forgetting of the base model's knowledge.

**Why `gradient_clip_val=1.0`:** Clips the gradient L2 norm to 1.0. Without this, a bad batch (e.g., very long audio with many tokens) can cause "gradient explosion" and completely destroy the model weights.

**Why `accumulate_grad_batches=8`:** Effective batch size = 1 × 8 = 8. Can't fit 8 audio samples in GPU memory simultaneously (each can be 30+ seconds of mel spectrogram), so we process 1 at a time and accumulate gradients before each parameter update. Produces identical training to batch size 8.

**Why `precision='bf16-mixed'`:** bfloat16 uses the same exponent range as float32 but fewer mantissa bits. Much more numerically stable than float16 for large models. `mixed` means weights stay float32 but forward/backward passes use bf16 — saves ~40% memory.

### Step 5: Evaluation — `scripts/evaluate_manifest.py`

```python
def wer(ref, hyp):
    r, h = norm(ref).split(), norm(hyp).split()
    return 100.0 * edit_distance(r, h) / len(r)

def cer(ref, hyp):
    r, h = norm(ref).replace(" ", ""), norm(hyp).replace(" ", "")
    return 100.0 * edit_distance(list(r), list(h)) / len(r)
```

**WER (Word Error Rate):** Edit distance at the word level. WER = 10% means 1 in 10 words is wrong.
**CER (Character Error Rate):** Edit distance at the character level. More sensitive to partial errors (e.g., "eight" vs "ate" is 1 word error / 4 character errors).

**Edit distance** (Levenshtein) = minimum insertions + deletions + substitutions to transform hypothesis into reference. Implemented here with classic dynamic programming O(n×m) — fine for short utterances.

### Step 6: Hyperparameter Search — `scripts/run_hyparam_tuning.sh`

Trains and evaluates 3 fine-tuned variants:

| Version | Learning Rate | Epochs | Freeze Mode |
|---|---|---|---|
| v1 | 3e-6 | 2 | decoder_only |
| v2 | 2e-6 | 3 | decoder_only |
| v3 | 5e-7 | 1 | last_encoder |

Then generates `results/hparam_tuning/model_comparison_report.md` comparing WER, CER, and an **entity score** (did the model correctly transcribe numbers, dollar amounts, and ticket IDs).

### Step 7: Comparison Report — `scripts/compare_models_report.py`

```python
def extract_entities(text):
    numbers = re.findall(r"\b\d{2,}\b", text)
    tickets = re.findall(r"\bt\s*k\s*t\s*[a-z0-9\s]{4,20}\b", text)
    money = re.findall(r"\b\d+\s*(dollars?|usd)\b|\$\s*\d+", text)
    return {"numbers": set(numbers), "tickets": set(tickets), "money": set(money)}
```

**Why entity score in addition to WER:** WER treats all words equally. A 15% WER model that always gets account numbers wrong is worse than a 20% WER model that always gets them right. Entity score directly measures the business-critical extractions.

---

## 9. Deployment Shell Scripts

| Script | What it does |
|---|---|
| `01_enter_training_container.sh` | `docker build` + `docker run --gpus all -v $PWD:/workspace ... bash` |
| `02_prepare_baseline_train_eval_inside_container.sh` | Inside the container: prepare data → baseline eval → fine-tune → eval again |
| `03_run_finetuned_server.sh` | `docker run -e MODEL_NAME=/srv/models/finetuned_nemotron_final.nemo ...` |
| `04_run_base_server.sh` | Same image, base model (no env override needed, it's baked in) |

**Why mount `$PWD` as `/workspace`:** Scripts and data live on the host. The container provides the GPU environment. Mounting means code changes don't require rebuilding the image.

**Why mount `ft_models` as `/srv/models`:** Fine-tuned `.nemo` files are large (>1GB). Mounting saves them to the host so they survive container deletion.

---

## Data Flow Summary

```
raw_wavs/*.wav
    │
    ▼ ffmpeg (16kHz mono)
data/audio_16k/*.wav
    │
    ├── augment_train_manifest.py
    │       │
    │       ▼ 6 augmentations each
    │   data/audio_aug/**/*.wav
    │
    ▼ prepare_dataset.py / auto_align_chunks_with_base_asr.py
data/manifests/*.json  (NeMo JSONL: audio_filepath, duration, text)
    │
    ▼ finetune_nemotron.py (PyTorch Lightning + NeMo)
ft_models/finetuned_nemotron_*.nemo
    │
    ▼ evaluate_manifest.py (WER, CER)
results/hparam_tuning/*_eval.jsonl
    │
    ▼ compare_models_report.py
results/hparam_tuning/model_comparison_report.md
    │
    ▼ 03_run_finetuned_server.sh
FastAPI WebSocket + REST server (port 8003)
    │
    ├── /asr/realtime-custom-vad  (WebSocket, real-time microphone)
    └── /v1/audio/transcriptions  (REST, LiveKit / batch files)
```

---

## Key Design Decisions Summary

| Decision | Rationale |
|---|---|
| RNNT over CTC | RNNT supports true streaming without CTC's conditional-independence assumption |
| Conformer encoder | Better than pure transformer for audio: local+global attention, conv captures fine-grained patterns |
| Freeze decoder only | Prevents overfitting on tiny domain dataset; encoder acoustics transfer well |
| AdamW + cosine schedule | Standard for transformer fine-tuning; prevents catastrophic forgetting |
| bf16-mixed precision | Memory-efficient, stable (vs. fp16 which can underflow) |
| Custom energy VAD | Ultra-low latency (<0.2ms/frame), fully tunable, no ML model required |
| Pre-roll buffer | Ensures first syllable never clipped even when VAD triggers slightly late |
| ffmpeg for format conversion | Handles 20+ audio formats without format-specific decoders |
| Build-time model bake-in | Container cold-start in seconds vs. minutes |
| `run_in_executor` for inference | Keeps event loop responsive for multiple concurrent WebSocket clients |
```
