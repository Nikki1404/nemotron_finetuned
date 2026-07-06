cd /home/CORP/re_nikitav/nemotron_finetuned && mkdir -p ft_models results/hparam_tuning

cd /home/CORP/re_nikitav/nemotron_finetuned && docker run --gpus all -it --rm -v $PWD:/workspace -v $PWD/ft_models:/srv/models nemotron_finetuned bash

cd /workspace && mkdir -p /srv/models results/hparam_tuning && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && export CUDA_HOME=/usr/local/cuda-12.4 && export LD_LIBRARY_PATH=/usr/local/cuda-12.4/nvvm/lib64:$LD_LIBRARY_PATH && unset NUMBA_CUDA_USE_NVIDIA_BINDING

cd /workspace && python3.11 scripts/augment_train_manifest.py --train-manifest data/manifests/train_aligned_manifest.json --out-manifest data/manifests/train_aligned_aug_manifest.json --out-audio-dir data/audio_aug --keep-original

cd /workspace && python3.11 scripts/finetune_nemotron.py --train-manifest data/manifests/train_aligned_aug_manifest.json --val-manifest data/manifests/val_aligned_manifest.json --base-model /srv/nemotron-3.5-asr-streaming-0.6b.nemo --output-nemo /srv/models/finetuned_nemotron_v1_lr3e6_ep2.nemo --freeze-mode decoder_only --max-epochs 2 --batch-size 1 --lr 3e-6 --language en-US --precision bf16-mixed

cp /srv/models/finetuned_nemotron_v1_lr3e6_ep2.nemo /srv/models/finetuned_nemotron_final.nemo && ls -lh /srv/models

cd /workspace && python3.11 scripts/evaluate_manifest.py --model /srv/models/finetuned_nemotron_final.nemo --manifest data/manifests/test_aligned_manifest.json --language en-US --output-jsonl results/hparam_tuning/final_eval.jsonl

cd /workspace && chmod +x scripts/run_hyparam_tuning.sh && bash scripts/run_hyparam_tuning.sh


cd /workspace && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && export CUDA_HOME=/usr/local/cuda-12.4 && export LD_LIBRARY_PATH=/usr/local/cuda-12.4/nvvm/lib64:$LD_LIBRARY_PATH && unset NUMBA_CUDA_USE_NVIDIA_BINDING && mkdir -p /srv/models results/hparam_tuning && chmod +x scripts/run_hyparam_tuning.sh && bash scripts/run_hyparam_tuning.shs


cd /workspace && python3.11 scripts/compare_models_report.py --base results/hparam_tuning/base_eval.jsonl --v1 results/hparam_tuning/v1_eval.jsonl --v2 results/hparam_tuning/v2_eval.jsonl --v3 results/hparam_tuning/v3_eval.jsonl --out results/hparam_tuning/model_comparison_report.md

cp /srv/models/finetuned_nemotron_v1_lr3e6_ep2.nemo /srv/models/finetuned_nemotron_final.nemo && ls -lh /srv/models/finetuned_nemotron_final.nemo

cd /home/CORP/re_nikitav/nemotron_finetuned && docker run --gpus all -it --rm -p 8003:8003 -v $PWD:/workspace -v $PWD/ft_models:/srv/models -e MODEL_NAME=/srv/models/finetuned_nemotron_final.nemo nemotron_finetuned uvicorn app.main:app --host 0.0.0.0 --port 8003

ps -ef | grep -E "uvicorn|app.main" | grep -v grep
python3.11 -c "import app.main,inspect; print(inspect.getfile(app.main))"
grep -n "v1/audio/transcriptions\|realtime-custom-vad\|AUDIO_LOG_DIR" /workspace/app/main.py
curl -s http://localhost:8003/


root@bcc6c2abcf6c:/srv# diff /srv/app/main.py  /workspace/app/main.py
3a4,5
> import os
> import subprocess
4a7,11
> import tempfile
> import wave
> from datetime import datetime, timezone
> from pathlib import Path
> from typing import Optional
8c15,16
< from fastapi import FastAPI, WebSocket
---
> from fastapi import FastAPI, WebSocket, UploadFile, File, Form
> from fastapi.responses import JSONResponse, PlainTextResponse
29a38,126
> # ---------------------------------------------------------------------
> # Additional logic: server-side audio/session logging
> # ---------------------------------------------------------------------
> AUDIO_LOG_DIR = Path(os.getenv("AUDIO_LOG_DIR", "/srv/audio_logs"))
> AUDIO_LOG_DIR.mkdir(parents=True, exist_ok=True)
>
>
> def utc_now_iso() -> str:
>     return datetime.now(timezone.utc).isoformat()
>
>
> def make_session_id(client) -> str:
>     ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
>
>     client_host = "unknown"
>     client_port = "unknown"
>
>     try:
>         client_host = str(client.host)
>         client_port = str(client.port)
>     except Exception:
>         pass
>
>     safe_host = client_host.replace(".", "_").replace(":", "_")
>     return f"{ts}_{safe_host}_{client_port}"
>
>
> def save_pcm_as_wav(pcm_bytes: bytes, wav_path: Path, sample_rate: int):
>     """
>     Save raw PCM16 mono audio bytes as WAV.
>     """
>     if not pcm_bytes:
>         return
>
>     with wave.open(str(wav_path), "wb") as wf:
>         wf.setnchannels(1)
>         wf.setsampwidth(2)
>         wf.setframerate(sample_rate)
>         wf.writeframes(pcm_bytes)
>
>
> def write_json(path: Path, data: dict):
>     with open(path, "w", encoding="utf-8") as f:
>         json.dump(data, f, indent=2, ensure_ascii=False)
>
>
> def append_jsonl(path: Path, data: dict):
>     with open(path, "a", encoding="utf-8") as f:
>         f.write(json.dumps(data, ensure_ascii=False) + "\n")
>
>
> def normalize_language(language: Optional[str]) -> str:
>     if not language:
>         return "auto"
>
>     lang = language.strip()
>
>     if not lang:
>         return "auto"
>
>     low = lang.lower()
>
>     if low == "auto":
>         return "auto"
>
>     if low == "en":
>         return "en-US"
>
>     if low == "es":
>         return "es-US"
>
>     return lang
>
>
> def clean_model_leaked_tags(text: str) -> str:
>     """
>     Removes leaked prompt tags like <en-US>, <es-US>.
>     """
>     if not text:
>         return ""
>
>     import re
>
>     return re.sub(r"<[a-z]{2}-[A-Z]{2}>\s*", "", text).strip()
>
>
> # ---------------------------------------------------------------------
> # Existing logic: preload engines
> # ---------------------------------------------------------------------
63c160,162
<         raise ValueError(f"Engine '{backend}' not loaded. Available: {list(ENGINE_CACHE.keys())}")
---
>         raise ValueError(
>             f"Engine '{backend}' not loaded. Available: {list(ENGINE_CACHE.keys())}"
>         )
66a166,168
> # ---------------------------------------------------------------------
> # Existing + additional health route
> # ---------------------------------------------------------------------
72a175,427
>         "sample_rate": cfg.sample_rate,
>         "audio_log_dir": str(AUDIO_LOG_DIR),
>     }
>
>
> @app.get("/")
> async def root():
>     return {
>         "status": "ok",
>         "service": "Nemotron 3.5 ASR Server",
>         "websocket_endpoint": "/asr/realtime-custom-vad",
>         "openai_transcription_endpoint": "/v1/audio/transcriptions",
>         "audio_log_dir": str(AUDIO_LOG_DIR),
>     }
>
>
> # ---------------------------------------------------------------------
> # Additional logic: OpenAI-compatible endpoint helpers
> # ---------------------------------------------------------------------
> def convert_upload_to_pcm16_16k_mono(
>     input_bytes: bytes,
>     suffix: str = ".wav",
> ) -> bytes:
>     """
>     Convert uploaded audio into raw PCM16 mono audio at cfg.sample_rate.
>     Supports WAV, MP3, M4A, FLAC, OGG, WEBM, etc. through ffmpeg.
>     """
>
>     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as in_file:
>         in_file.write(input_bytes)
>         input_path = in_file.name
>
>     output_path = input_path + ".pcm"
>
>     try:
>         cmd = [
>             "ffmpeg",
>             "-y",
>             "-i",
>             input_path,
>             "-ac",
>             "1",
>             "-ar",
>             str(cfg.sample_rate),
>             "-f",
>             "s16le",
>             "-acodec",
>             "pcm_s16le",
>             output_path,
>         ]
>
>         result = subprocess.run(
>             cmd,
>             stdout=subprocess.PIPE,
>             stderr=subprocess.PIPE,
>             text=True,
>         )
>
>         if result.returncode != 0:
>             raise RuntimeError(f"ffmpeg failed: {result.stderr}")
>
>         with open(output_path, "rb") as f:
>             return f.read()
>
>     finally:
>         try:
>             os.remove(input_path)
>         except FileNotFoundError:
>             pass
>
>         try:
>             os.remove(output_path)
>         except FileNotFoundError:
>             pass
>
>
> async def run_pcm_through_session(
>     pcm_bytes: bytes,
>     engine: ASREngine,
>     language: str,
>     source: str = "openai-http",
>     log_audio: bool = True,
> ) -> str:
>     """
>     Used by /v1/audio/transcriptions.
>     Runs raw PCM16 audio through the same StreamingSession as WebSocket mode.
>     """
>
>     session_id = f"openai_http_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
>     session_dir = AUDIO_LOG_DIR / session_id
>     session_dir.mkdir(parents=True, exist_ok=True)
>
>     raw_pcm_path = session_dir / "audio.raw.pcm"
>     wav_path = session_dir / "audio.wav"
>     metadata_path = session_dir / "metadata.json"
>     events_path = session_dir / "events.jsonl"
>
>     started_at = utc_now_iso()
>     start_perf = asyncio.get_running_loop().time()
>
>     session = StreamingSession(engine, cfg)
>
>     final_texts: list[str] = []
>     transcript_events: list[dict] = []
>
>     chunk_ms = 100
>     chunk_bytes = int(cfg.sample_rate * chunk_ms / 1000) * 2
>
>     if log_audio:
>         with open(raw_pcm_path, "wb") as f:
>             f.write(pcm_bytes)
>
>         try:
>             save_pcm_as_wav(
>                 pcm_bytes=pcm_bytes,
>                 wav_path=wav_path,
>                 sample_rate=cfg.sample_rate,
>             )
>         except Exception:
>             log.exception(f"Failed to save OpenAI HTTP WAV | session_id={session_id}")
>
>     write_json(
>         metadata_path,
>         {
>             "session_id": session_id,
>             "source": source,
>             "started_at_utc": started_at,
>             "language": language,
>             "server_sample_rate": cfg.sample_rate,
>             "raw_pcm_bytes": len(pcm_bytes),
>             "audio_duration_sec": round(len(pcm_bytes) / 2 / cfg.sample_rate, 3)
>             if pcm_bytes
>             else 0.0,
>             "audio_raw_pcm_path": str(raw_pcm_path),
>             "audio_wav_path": str(wav_path),
>             "events_path": str(events_path),
>             "status": "started",
>         },
>     )
>
>     loop = asyncio.get_running_loop()
>
>     for i in range(0, len(pcm_bytes), chunk_bytes):
>         chunk = pcm_bytes[i : i + chunk_bytes]
>
>         if not chunk:
>             continue
>
>         events = await loop.run_in_executor(
>             None,
>             session.process_chunk,
>             chunk,
>         )
>
>         for ev_type, text, ttfb in events:
>             text = clean_model_leaked_tags(text)
>             _log_transcript(ev_type, text, ttfb, language, source)
>
>             event_payload = {
>                 "timestamp_utc": utc_now_iso(),
>                 "session_id": session_id,
>                 "source": source,
>                 "type": ev_type,
>                 "text": text,
>                 "language": language,
>                 "ttfb_ms": ttfb,
>             }
>
>             transcript_events.append(event_payload)
>             append_jsonl(events_path, event_payload)
>
>             if ev_type == "final" and text:
>                 final_texts.append(text.strip())
>
>     flush_events = await loop.run_in_executor(None, session.flush)
>
>     for ev_type, text, ttfb in flush_events:
>         text = clean_model_leaked_tags(text)
>         _log_transcript(ev_type, text, ttfb, language, source)
>
>         event_payload = {
>             "timestamp_utc": utc_now_iso(),
>             "session_id": session_id,
>             "source": source,
>             "type": ev_type,
>             "text": text,
>             "language": language,
>             "ttfb_ms": ttfb,
>         }
>
>         transcript_events.append(event_payload)
>         append_jsonl(events_path, event_payload)
>
>         if ev_type == "final" and text:
>             final_texts.append(text.strip())
>
>     transcript = " ".join(final_texts).strip()
>
>     ended_at = utc_now_iso()
>     end_perf = asyncio.get_running_loop().time()
>     wall_duration_sec = end_perf - start_perf
>
>     write_json(
>         metadata_path,
>         {
>             "session_id": session_id,
>             "source": source,
>             "started_at_utc": started_at,
>             "ended_at_utc": ended_at,
>             "wall_duration_sec": round(wall_duration_sec, 3),
>             "audio_duration_sec": round(len(pcm_bytes) / 2 / cfg.sample_rate, 3)
>             if pcm_bytes
>             else 0.0,
>             "language": language,
>             "server_sample_rate": cfg.sample_rate,
>             "raw_pcm_bytes": len(pcm_bytes),
>             "audio_raw_pcm_path": str(raw_pcm_path),
>             "audio_wav_path": str(wav_path),
>             "events_path": str(events_path),
>             "final_transcript": transcript,
>             "final_transcripts": [
>                 ev["text"]
>                 for ev in transcript_events
>                 if ev.get("type") == "final" and ev.get("text")
>             ],
>             "status": "completed",
>         },
>     )
>
>     log.info(
>         f"OPENAI_HTTP_AUDIO_LOG_END | session_id={session_id} "
>         f"wall_duration_sec={wall_duration_sec:.3f} "
>         f"audio_duration_sec={len(pcm_bytes) / 2 / cfg.sample_rate:.3f} "
>         f"wav={wav_path}"
>     )
>
>     return transcript
>
>
> # ---------------------------------------------------------------------
> # Additional logic: OpenAI-compatible endpoints for LiveKit
> # ---------------------------------------------------------------------
> @app.get("/v1/models")
> async def list_openai_models():
>     return {
>         "object": "list",
>         "data": [
>             {
>                 "id": "nemotron-3.5-asr-streaming-0.6b",
>                 "object": "model",
>                 "owned_by": "local",
>             }
>         ],
75a431,532
> @app.post("/v1/audio/transcriptions")
> async def openai_audio_transcriptions(
>     file: UploadFile = File(...),
>     model: str = Form("nemotron-3.5-asr-streaming-0.6b"),
>     language: Optional[str] = Form("auto"),
>     response_format: Optional[str] = Form("json"),
> ):
>     """
>     OpenAI-compatible transcription endpoint.
>
>     LiveKit should use:
>         base_url="http://HOST:8003/v1"
>     """
>
>     lang = normalize_language(language)
>
>     log.info(
>         f"OpenAI-compatible STT request | model={model} language={lang} "
>         f"response_format={response_format} filename={file.filename}"
>     )
>
>     if model != "nemotron-3.5-asr-streaming-0.6b":
>         return JSONResponse(
>             status_code=400,
>             content={
>                 "error": {
>                     "message": f"Unsupported model: {model}",
>                     "type": "invalid_request_error",
>                 }
>             },
>         )
>
>     try:
>         engine = get_engine("nemotron")
>     except ValueError as e:
>         log.exception("Nemotron engine not available")
>         return JSONResponse(
>             status_code=500,
>             content={
>                 "error": {
>                     "message": str(e),
>                     "type": "server_error",
>                 }
>             },
>         )
>
>     try:
>         engine.set_language(lang)
>
>         audio_bytes = await file.read()
>
>         suffix = ".wav"
>         if file.filename and "." in file.filename:
>             suffix = "." + file.filename.rsplit(".", 1)[-1]
>
>         pcm_bytes = convert_upload_to_pcm16_16k_mono(
>             input_bytes=audio_bytes,
>             suffix=suffix,
>         )
>
>         transcript = await run_pcm_through_session(
>             pcm_bytes=pcm_bytes,
>             engine=engine,
>             language=lang,
>             source="openai-http",
>             log_audio=True,
>         )
>
>         if response_format == "text":
>             return PlainTextResponse(transcript)
>
>         if response_format == "verbose_json":
>             return {
>                 "task": "transcribe",
>                 "language": lang,
>                 "duration": round(len(pcm_bytes) / 2 / cfg.sample_rate, 3)
>                 if pcm_bytes
>                 else 0.0,
>                 "text": transcript,
>                 "segments": [],
>             }
>
>         return {
>             "text": transcript,
>         }
>
>     except Exception as e:
>         log.exception("OpenAI-compatible transcription failed")
>         return JSONResponse(
>             status_code=500,
>             content={
>                 "error": {
>                     "message": str(e),
>                     "type": "server_error",
>                 }
>             },
>         )
>
>
> # ---------------------------------------------------------------------
> # Existing WebSocket endpoint with added logging logic
> # ---------------------------------------------------------------------
80a538,562
>     # Additional logic: create per-session logging folder
>     session_id = make_session_id(ws.client)
>     session_dir = AUDIO_LOG_DIR / session_id
>     session_dir.mkdir(parents=True, exist_ok=True)
>
>     raw_pcm_path = session_dir / "audio.raw.pcm"
>     wav_path = session_dir / "audio.wav"
>     metadata_path = session_dir / "metadata.json"
>     events_path = session_dir / "events.jsonl"
>
>     session_started_at = utc_now_iso()
>     session_start_perf = asyncio.get_running_loop().time()
>
>     captured_pcm = bytearray()
>     transcript_events: list[dict] = []
>
>     log.info(
>         f"AUDIO_LOG_START | session_id={session_id} client={ws.client} "
>         f"dir={session_dir} started_at={session_started_at}"
>     )
>
>     backend = None
>     client_sample_rate = None
>     language = None
>
85a568,579
>
>         write_json(
>             metadata_path,
>             {
>                 "session_id": session_id,
>                 "client": str(ws.client),
>                 "started_at_utc": session_started_at,
>                 "status": "bad_init_message",
>                 "error": str(e),
>             },
>         )
>
96a591,609
>     language = normalize_language(language)
>
>     write_json(
>         metadata_path,
>         {
>             "session_id": session_id,
>             "client": str(ws.client),
>             "started_at_utc": session_started_at,
>             "backend": backend,
>             "language": language,
>             "client_sample_rate": client_sample_rate,
>             "server_sample_rate": cfg.sample_rate,
>             "audio_raw_pcm_path": str(raw_pcm_path),
>             "audio_wav_path": str(wav_path),
>             "events_path": str(events_path),
>             "status": "started",
>         },
>     )
>
98a612,622
>
>         error_payload = {
>             "timestamp_utc": utc_now_iso(),
>             "session_id": session_id,
>             "type": "error",
>             "text": f"Unknown backend '{backend}'",
>             "language": language,
>         }
>
>         append_jsonl(events_path, error_payload)
>
106a631,641
>
>         error_payload = {
>             "timestamp_utc": utc_now_iso(),
>             "session_id": session_id,
>             "type": "error",
>             "text": str(e),
>             "language": language,
>         }
>
>         append_jsonl(events_path, error_payload)
>
116c651
<         f"client={ws.client}"
---
>         f"client={ws.client} session_id={session_id}"
142c677,681
<                         log.info(f"EOF from {ws.client} — flushing last utterance")
---
>                         log.info(
>                             f"EOF from {ws.client} — flushing last utterance | "
>                             f"session_id={session_id}"
>                         )
>
144a684
>
145a686,687
>                             text = clean_model_leaked_tags(text)
>
146a689,701
>
>                             event_payload = {
>                                 "timestamp_utc": utc_now_iso(),
>                                 "session_id": session_id,
>                                 "type": ev_type,
>                                 "text": text,
>                                 "language": language,
>                                 "ttfb_ms": ttfb,
>                             }
>
>                             transcript_events.append(event_payload)
>                             append_jsonl(events_path, event_payload)
>
148c703,709
<                                 json.dumps({"type": ev_type, "text": text, "t_start": ttfb})
---
>                                 json.dumps(
>                                     {
>                                         "type": ev_type,
>                                         "text": text,
>                                         "t_start": ttfb,
>                                     }
>                                 )
149a711,721
>
>                         done_payload = {
>                             "timestamp_utc": utc_now_iso(),
>                             "session_id": session_id,
>                             "type": "done",
>                             "text": "",
>                             "language": language,
>                         }
>
>                         append_jsonl(events_path, done_payload)
>
150a723
>
160a734,739
>             # Additional logic: save audio server-side
>             captured_pcm.extend(data)
>
>             with open(raw_pcm_path, "ab") as f:
>                 f.write(data)
>
164a744,745
>                 text = clean_model_leaked_tags(text)
>
165a747,759
>
>                 event_payload = {
>                     "timestamp_utc": utc_now_iso(),
>                     "session_id": session_id,
>                     "type": ev_type,
>                     "text": text,
>                     "language": language,
>                     "ttfb_ms": ttfb,
>                 }
>
>                 transcript_events.append(event_payload)
>                 append_jsonl(events_path, event_payload)
>
167c761,767
<                     json.dumps({"type": ev_type, "text": text, "t_start": ttfb})
---
>                     json.dumps(
>                         {
>                             "type": ev_type,
>                             "text": text,
>                             "t_start": ttfb,
>                         }
>                     )
171a772
>
172a774,823
>         # Additional logic: finalize session metadata and WAV
>         session_ended_at = utc_now_iso()
>         session_end_perf = asyncio.get_running_loop().time()
>         wall_duration_sec = session_end_perf - session_start_perf
>
>         audio_duration_sec = 0.0
>         if captured_pcm:
>             audio_duration_sec = len(captured_pcm) / 2 / cfg.sample_rate
>
>         try:
>             save_pcm_as_wav(
>                 pcm_bytes=bytes(captured_pcm),
>                 wav_path=wav_path,
>                 sample_rate=cfg.sample_rate,
>             )
>         except Exception:
>             log.exception(f"Failed to save WAV for session_id={session_id}")
>
>         final_metadata = {
>             "session_id": session_id,
>             "client": str(ws.client),
>             "started_at_utc": session_started_at,
>             "ended_at_utc": session_ended_at,
>             "wall_duration_sec": round(wall_duration_sec, 3),
>             "audio_duration_sec": round(audio_duration_sec, 3),
>             "backend": backend,
>             "language": language,
>             "client_sample_rate": client_sample_rate,
>             "server_sample_rate": cfg.sample_rate,
>             "raw_pcm_bytes": len(captured_pcm),
>             "audio_raw_pcm_path": str(raw_pcm_path),
>             "audio_wav_path": str(wav_path),
>             "events_path": str(events_path),
>             "final_transcripts": [
>                 ev["text"]
>                 for ev in transcript_events
>                 if ev.get("type") == "final" and ev.get("text")
>             ],
>             "status": "completed",
>         }
>
>         write_json(metadata_path, final_metadata)
>
>         log.info(
>             f"AUDIO_LOG_END | session_id={session_id} client={ws.client} "
>             f"wall_duration_sec={wall_duration_sec:.3f} "
>             f"audio_duration_sec={audio_duration_sec:.3f} "
>             f"wav={wav_path}"
>         )
