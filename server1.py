import asyncio
import json
import logging
import time
from contextlib import suppress

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps

app = FastAPI()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('whisper-live')

SAMPLE_RATE = 16000
LANGUAGE = 'lt'  # change to 'en' for English TV
MODEL_NAME = 'large-v3'  # or 'turbo'
CHECK_EVERY = 1.2
PARTIAL_EVERY = 2.0
FINAL_SILENCE = 0.95
MAX_UTTERANCE = 30.0
MIN_AUDIO = 0.8

model = WhisperModel(MODEL_NAME, device='cuda', compute_type='float16')
vad_options = VadOptions(threshold=0.5, min_speech_duration_ms=200,
                         min_silence_duration_ms=800, speech_pad_ms=150)

# A single 8-GB GPU: serialize inference across connections.
inference_lock = asyncio.Lock()


def speech_ranges(samples):
    return get_speech_timestamps(samples, vad_options)


def run_whisper(samples, is_final):
    segments, _ = model.transcribe(
        samples.astype(np.float32, copy=False), language=LANGUAGE,
        beam_size=5 if is_final else 1,
        vad_filter=False, condition_on_previous_text=False,
        temperature=0.0,
    )
    return ' '.join(s.text.strip() for s in segments).strip()


@app.websocket('/ws/transcribe')
async def ws_transcribe(ws: WebSocket):
    await ws.accept()
    blocks = []
    block_len = 0
    last_check = 0.0
    last_partial = 0.0

    async def process(force=False):
        nonlocal blocks, block_len, last_check, last_partial
        if block_len < int(MIN_AUDIO * SAMPLE_RATE):
            return
        samples = np.concatenate(blocks)
        # VAD on the buffered utterance; keep silence to detect phrase end.
        ranges = await asyncio.to_thread(speech_ranges, samples)
        if not ranges:
            if force or len(samples) >= SAMPLE_RATE * 5:
                blocks, block_len = [], 0
            return
        last_end = ranges[-1]['end'] / SAMPLE_RATE
        trailing = len(samples) / SAMPLE_RATE - last_end
        is_final = force or trailing >= FINAL_SILENCE or len(samples) >= MAX_UTTERANCE * SAMPLE_RATE
        now = time.monotonic()
        if not is_final and (now - last_partial) < PARTIAL_EVERY:
            return
        # Bound recognition to speech and keep padding around boundaries.
        start = max(0, ranges[0]['start'] - int(0.2 * SAMPLE_RATE))
        end = min(len(samples), ranges[-1]['end'] + int(0.2 * SAMPLE_RATE))
        audio = samples[start:end]
        async with inference_lock:
            text = await asyncio.to_thread(run_whisper, audio, is_final)
        if text:
            await ws.send_text(json.dumps({
                'type': 'final' if is_final else 'partial', 'text': text,
            }, ensure_ascii=False))
        if is_final:
            blocks, block_len = [], 0
            last_partial = 0.0
        else:
            last_partial = time.monotonic()

    try:
        while True:
            message = await ws.receive()
            if message['type'] == 'websocket.disconnect':
                break
            if message.get('text') == 'stop':
                await process(force=True)
                await ws.send_json({'type': 'stopped'})
                break
            data = message.get('bytes')
            if not data:
                continue
            if len(data) % 4:
                continue
            arr = np.frombuffer(data, dtype='<f4')
            # Protect memory from untrusted/unbounded uploads.
            if len(arr) > SAMPLE_RATE * 5:
                await ws.close(code=1009)
                break
            blocks.append(arr.copy())
            block_len += len(arr)
            now = time.monotonic()
            if now - last_check >= CHECK_EVERY:
                last_check = now
                await process()
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception('WebSocket transcription failure')
        with suppress(Exception):
            await ws.send_json({'type': 'error', 'message': 'Transkripcijos klaida serveryje'})
    finally:
        with suppress(Exception):
            await ws.close()
