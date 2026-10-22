"""Offline transcription of an audio file, with the same optional steps as server.py.

    python transcribe.py audio.mp3 out.txt [--spell] [--punct] [--no-timestamps]

--spell  taisyti žodyne nerastus žodžius (LT-Checker, lt_spell.py)
--punct  atkurti skyrybą (punctuation.py, žodžių nekeičia)
"""
import argparse
import logging
import sys
import time

from faster_whisper import WhisperModel

MODEL_NAME = '/home/mp/whisper-models/azuolas-ct2'


def main():
    p = argparse.ArgumentParser(description='Garso failo transkripcija (lt).')
    p.add_argument('input', help='garso failas, pvz. audio.mp3')
    p.add_argument('output', help='rezultato tekstinis failas')
    p.add_argument('--spell', action='store_true',
                   help='taisyti žodyne nerastus žodžius (LT-Checker)')
    p.add_argument('--punct', action='store_true',
                   help='įjungti skyrybos atkūrimą (be žodžių taisymo)')
    p.add_argument('--no-timestamps', action='store_true',
                   help='nerašyti [pradžia - pabaiga] prie segmentų')
    p.add_argument('--model', default=MODEL_NAME, help='Whisper modelis (numatytasis: Ąžuolas)')
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    for noisy in ('httpx', 'httpx2', 'huggingface_hub', 'transformers'):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    log = logging.getLogger('transcribe')

    # Optional steps are loaded only when requested (CPU/RAM only, not VRAM).
    corrector = punctuator = None
    if args.spell:
        from lt_spell import Corrector
        corrector = Corrector()
    if args.punct:
        from punctuation import Punctuator
        punctuator = Punctuator()

    t0 = time.monotonic()
    model = WhisperModel(args.model, device='cuda', compute_type='float16')
    segments, info = model.transcribe(args.input, language='lt', beam_size=5,
                                      vad_filter=True, condition_on_previous_text=False)
    log.info('Trukmė: %.1f s', info.duration)

    changes = 0
    with open(args.output, 'w', encoding='utf-8') as out:
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            if corrector:
                text, pieces = corrector.correct(text)
                for fixed, original in pieces:
                    if original is not None:
                        changes += 1
                        log.info('  %s -> %s', original, fixed)
            if punctuator:
                try:
                    text = punctuator(text)
                except Exception:
                    log.exception('Skyrybos klaida, segmentas paliekamas be skyrybos')
            line = text if args.no_timestamps else '[%.2f - %.2f] %s' % (seg.start, seg.end, text)
            out.write(line + '\n')
            out.flush()
            print(line, flush=True)

    log.info('Baigta per %.1f s -> %s%s', time.monotonic() - t0, args.output,
             ' (pataisyta žodžių: %d)' % changes if corrector else '')


if __name__ == '__main__':
    sys.exit(main())
