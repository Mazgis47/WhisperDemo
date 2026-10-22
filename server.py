import asyncio
import json
import logging
import time
from contextlib import suppress

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps
from punctuation import Punctuator

app = FastAPI()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('whisper-live')

SAMPLE_RATE = 16000
LANGUAGE = 'lt'
MODEL_NAME = '/home/mp/whisper-models/azuolas-ct2'
CHECK_EVERY = 1.2
PARTIAL_EVERY = 2.0
FINAL_SILENCE = 0.95
MAX_UTTERANCE = 30.0
MIN_AUDIO = 0.8

model = WhisperModel(MODEL_NAME, device='cuda', compute_type='float16')
vad_options = VadOptions(threshold=0.5, min_speech_duration_ms=200,
                         min_silence_duration_ms=500, speech_pad_ms=150)

# The punctuation model runs on CPU, not in RTX 2070 VRAM.
log.info('Loading punctuation model on CPU...')
punctuator = Punctuator()
inference_lock = asyncio.Lock()


def speech_ranges(samples):
    return get_speech_timestamps(samples, vad_options)


def run_whisper(samples, is_final):
    segments, _ = model.transcribe(
        samples.astype(np.float32, copy=False), language=LANGUAGE,
        beam_size=5 if is_final else 1, vad_filter=False,
        condition_on_previous_text=False, temperature=0.0,
    )
    return ' '.join(s.text.strip() for s in segments).strip()


@app.websocket('/ws/transcribe')
async def ws_transcribe(ws: WebSocket):
    await ws.accept()
    blocks, block_len = [], 0
    last_check = last_partial = 0.0
    seq = 0
    punctuation_enabled = False
    punctuation_jobs = set()

    async def punctuate_and_send(sequence, text):
        try:
            formatted = await asyncio.to_thread(punctuator, text)
            await ws.send_text(json.dumps({
                'type': 'punctuation', 'seq': sequence, 'text': formatted
            }, ensure_ascii=False))
        except Exception:
            log.exception('Punctuation restoration error')
            with suppress(Exception):
                await ws.send_json({'type': 'punctuation_error', 'seq': sequence})

    async def process(force=False):
        nonlocal blocks, block_len, last_check, last_partial, seq
        if block_len < int(MIN_AUDIO * SAMPLE_RATE):
            return
        samples = np.concatenate(blocks)
        ranges = await asyncio.to_thread(speech_ranges, samples)
        if not ranges:
            if force or len(samples) >= SAMPLE_RATE * 5:
                blocks, block_len = [], 0
            return
        last_end = ranges[-1]['end'] / SAMPLE_RATE
        trailing = len(samples) / SAMPLE_RATE - last_end
        is_final = (force or trailing >= FINAL_SILENCE
                    or len(samples) >= MAX_UTTERANCE * SAMPLE_RATE)
        now = time.monotonic()
        if not is_final and (now - last_partial) < PARTIAL_EVERY:
            return
        start = max(0, ranges[0]['start'] - int(0.2 * SAMPLE_RATE))
        end = min(len(samples), ranges[-1]['end'] + int(0.2 * SAMPLE_RATE))
        audio = samples[start:end]
        async with inference_lock:
            text = await asyncio.to_thread(run_whisper, audio, is_final)
        if text:
            if is_final:
                seq += 1
                await ws.send_text(json.dumps({
                    'type': 'final', 'seq': seq, 'text': text
                }, ensure_ascii=False))
                if punctuation_enabled:
                    job = asyncio.create_task(punctuate_and_send(seq, text))
                    punctuation_jobs.add(job)
                    job.add_done_callback(punctuation_jobs.discard)
            else:
                await ws.send_text(json.dumps({
                    'type': 'partial', 'text': text
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
            if message.get('text') == 'punctuation_on':
                punctuation_enabled = True
                continue
            if message.get('text') == 'punctuation_off':
                punctuation_enabled = False
                continue
            if message.get('text') == 'stop':
                await process(force=True)
                if punctuation_jobs:
                    await asyncio.gather(*punctuation_jobs, return_exceptions=True)
                await ws.send_json({'type': 'stopped'})
                break
            data = message.get('bytes')
            if not data or len(data) % 4:
                continue
            arr = np.frombuffer(data, dtype='<f4')
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
        for job in punctuation_jobs:
            job.cancel()
        with suppress(Exception):
            await ws.close()
