"""Speech recognition backends, chosen with the ASR_MODEL environment variable.

    ASR_MODEL=azuolas   Ąžuolas (faster-whisper, numatytasis)
    ASR_MODEL=large-v3  Whisper large-v3 (faster-whisper)
    ASR_MODEL=turbo     Whisper large-v3-turbo (faster-whisper)
    ASR_MODEL=canary    NVIDIA Canary-1B-v2 (transformers)

Canary options: CANARY_PNC=0 disables its own punctuation/capitalization (then
punctuation.py is used as for Whisper), CANARY_DTYPE=float32 if float16 misbehaves.
"""
import logging
import os

import numpy as np

log = logging.getLogger('asr')

SAMPLE_RATE = 16000
LANGUAGE = 'lt'
DEFAULT_MODEL = 'azuolas'
# name -> (label, model id or path, outputs its own punctuation/capitalization)
MODELS = {
    'azuolas': ('Ąžuolas', '/home/mp/whisper-models/azuolas-ct2', False),
    'large-v3': ('Whisper large-v3', 'large-v3', True),
    'turbo': ('Whisper large-v3-turbo', 'large-v3-turbo', True),
    'canary': ('Canary-1B-v2', 'nvidia/canary-1b-v2', True),
}
# Offline files are cut into speech chunks for Canary (no long-form in transformers).
MAX_CHUNK_SECONDS = 30


class WhisperASR:
    def __init__(self, label, path, native_punctuation):
        from faster_whisper import WhisperModel
        self.label = label
        # Ąžuolas writes lowercase without punctuation; stock Whisper punctuates.
        self.native_punctuation = native_punctuation
        self.model = WhisperModel(path, device='cuda', compute_type='float16')

    def transcribe(self, samples, is_final):
        segments, _ = self.model.transcribe(
            samples.astype(np.float32, copy=False), language=LANGUAGE,
            beam_size=5 if is_final else 1, vad_filter=False,
            condition_on_previous_text=False, temperature=0.0,
        )
        return ' '.join(s.text.strip() for s in segments).strip()

    def transcribe_file(self, path):
        """Yields (start, end, text) per segment."""
        segments, info = self.model.transcribe(path, language=LANGUAGE, beam_size=5,
                                               vad_filter=True, condition_on_previous_text=False)
        log.info('Trukmė: %.1f s', info.duration)
        for s in segments:
            yield s.start, s.end, s.text.strip()


class CanaryASR:
    def __init__(self, label, path, native_punctuation):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        self.label = label
        self.torch = torch
        # RTX 2070 (Turing) has no bfloat16; float16 halves VRAM vs the float32 checkpoint.
        self.dtype = getattr(torch, os.environ.get('CANARY_DTYPE', 'float16'))
        self.punctuation = native_punctuation and os.environ.get('CANARY_PNC', '1') != '0'
        self.native_punctuation = self.punctuation
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(path, dtype=self.dtype).to('cuda').eval()

    def transcribe(self, samples, is_final):
        inputs = self.processor.apply_transcription_request(
            audio=samples.astype(np.float32, copy=False), source_language=LANGUAGE,
            punctuation=self.punctuation)
        inputs = inputs.to(self.model.device, dtype=self.dtype)
        # Greedy decoding sometimes drops most of a segment when the topic changes
        # near its end; beam search keeps it (final only, partials stay fast).
        with self.torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=512,
                                      num_beams=4 if is_final else 1)
        return self.processor.decode(ids, skip_special_tokens=True)[0].strip()

    def transcribe_file(self, path):
        """Yields (start, end, text) per speech chunk (VAD, at most MAX_CHUNK_SECONDS)."""
        from faster_whisper import decode_audio
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        audio = decode_audio(path, sampling_rate=SAMPLE_RATE)
        log.info('Trukmė: %.1f s', len(audio) / SAMPLE_RATE)
        chunks = get_speech_timestamps(audio, VadOptions(
            min_silence_duration_ms=500, speech_pad_ms=200,
            max_speech_duration_s=MAX_CHUNK_SECONDS))
        for c in chunks:
            text = self.transcribe(audio[c['start']:c['end']], True)
            if text:
                yield c['start'] / SAMPLE_RATE, c['end'] / SAMPLE_RATE, text


def load(name=None):
    """Loads the backend named by `name`, the ASR_MODEL env var, or the default."""
    name = (name or os.environ.get('ASR_MODEL') or DEFAULT_MODEL).strip().lower()
    if name not in MODELS:
        raise ValueError('Nežinomas ASR_MODEL=%r; galimi: %s' % (name, ', '.join(MODELS)))
    label, path, native_punctuation = MODELS[name]
    log.info('Loading ASR model: %s (%s)', label, path)
    backend = CanaryASR if name == 'canary' else WhisperASR
    return backend(label, path, native_punctuation)
