import asyncio
import json
import logging
import sys

import numpy as np
import resampy
from fastapi import FastAPI, WebSocket

from app.config import load_config, Config, MODEL_MAP
from app.factory import build_engine
from app.streaming_session import StreamingSession
from app.asr_engines.base import ASREngine


cfg = load_config()

logging.basicConfig(
    level=cfg.log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("asr_server")

app = FastAPI(title="Nemotron 3.5 ASR Server", version="1.0.0")

ENGINE_CACHE: dict[str, ASREngine] = {}


async def preload_engines():
    log.info("Preloading ASR engines...")

    for backend, model_name in MODEL_MAP.items():
        try:
            log.info(f"Initializing engine: {backend} ({model_name})")

            tmp_cfg = Config()
            object.__setattr__(tmp_cfg, "asr_backend", backend)
            object.__setattr__(tmp_cfg, "model_name", model_name)
            object.__setattr__(tmp_cfg, "device", cfg.device)
            object.__setattr__(tmp_cfg, "sample_rate", cfg.sample_rate)

            engine = build_engine(tmp_cfg)
            load_sec = engine.load()

            ENGINE_CACHE[backend] = engine
            log.info(f"✅ Preloaded '{backend}' in {load_sec:.2f}s")

        except Exception:
            log.exception(f"Failed to preload '{backend}'")

    log.info(f"All engines preloaded. Available: {list(ENGINE_CACHE.keys())}")


@app.on_event("startup")
async def startup_event():
    log.info("Server startup initiated")
    await preload_engines()


def get_engine(backend: str) -> ASREngine:
    if backend not in ENGINE_CACHE:
        raise ValueError(f"Engine '{backend}' not loaded. Available: {list(ENGINE_CACHE.keys())}")
    return ENGINE_CACHE[backend]


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engines": list(ENGINE_CACHE.keys()),
        "device": cfg.device,
    }


@app.websocket("/asr/realtime-custom-vad")
async def ws_asr(ws: WebSocket):
    log.info(f"WS connection request from {ws.client}")
    await ws.accept()

    try:
        raw_init = await ws.receive_text()
        init = json.loads(raw_init)
    except Exception as e:
        log.warning(f"Bad init message: {e}")
        await ws.close(code=4001)
        return

    backend = init.get("backend", "nemotron")
    client_sample_rate = int(init.get("sample_rate", cfg.sample_rate))

    language = init.get("language")
    if not language:
        log.warning(f"Client {ws.client} did not send 'language' — defaulting to 'auto'")
        language = "auto"

    if backend not in MODEL_MAP:
        log.warning(f"Invalid backend requested: '{backend}'")
        await ws.send_text(json.dumps({"error": f"Unknown backend '{backend}'"}))
        await ws.close(code=4000)
        return

    try:
        engine = get_engine(backend)
    except ValueError as e:
        log.error(str(e))
        await ws.send_text(json.dumps({"error": str(e)}))
        await ws.close(code=4000)
        return

    engine.set_language(language)

    log.info(
        f"WS connected | backend={backend} language={language} "
        f"client_sr={client_sample_rate} server_sr={cfg.sample_rate} "
        f"client={ws.client}"
    )

    def upsample_if_needed(pcm: bytes) -> bytes:
        if not pcm or client_sample_rate == cfg.sample_rate:
            return pcm
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        y = resampy.resample(x, client_sample_rate, cfg.sample_rate)
        y = np.clip(y, -1.0, 1.0)
        return (y * 32767.0).astype(np.int16).tobytes()

    session = StreamingSession(engine, cfg)

    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                log.info(f"Client disconnected: {ws.client}")
                break

            # Text frame: EOF signal {"type": "eof"}
            if msg.get("text"):
                try:
                    ctrl = json.loads(msg["text"])
                    if ctrl.get("type") == "eof":
                        log.info(f"EOF from {ws.client} — flushing last utterance")
                        loop = asyncio.get_running_loop()
                        events = await loop.run_in_executor(None, session.flush)
                        for ev_type, text, ttfb in events:
                            _log_transcript(ev_type, text, ttfb, language, ws.client)
                            await ws.send_text(
                                json.dumps({"type": ev_type, "text": text, "t_start": ttfb})
                            )
                        await ws.send_text(json.dumps({"type": "done"}))
                except (json.JSONDecodeError, AttributeError):
                    pass
                continue

            data = msg.get("bytes")
            if data is None:
                continue

            data = upsample_if_needed(data)

            loop = asyncio.get_running_loop()
            events = await loop.run_in_executor(None, session.process_chunk, data)

            for ev_type, text, ttfb in events:
                _log_transcript(ev_type, text, ttfb, language, ws.client)
                await ws.send_text(
                    json.dumps({"type": ev_type, "text": text, "t_start": ttfb})
                )

    except Exception:
        log.exception(f"Error during WebSocket session for {ws.client}")
    finally:
        log.info(f"WS session closed for {ws.client}")


def _log_transcript(ev_type: str, text: str, ttfb_ms, language: str, client):
    if ev_type == "partial":
        log.debug(f"PARTIAL | lang={language} client={client} | {text}")
    elif ev_type == "final":
        ttfb_str = f" ttfb={ttfb_ms}ms" if ttfb_ms is not None else ""
        log.info(f"FINAL   | lang={language} client={client}{ttfb_str} | {text}")
