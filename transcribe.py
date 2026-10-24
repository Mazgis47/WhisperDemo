"""Offline transcription of an audio file, with the same optional steps as server.py.

    python transcribe.py audio.mp3 out.txt [--spell] [--punct] [--no-timestamps] [--model M]

--spell  taisyti žodyne nerastus žodžius (LT-Checker, lt_spell.py)
--punct  atkurti skyrybą (punctuation.py, žodžių nekeičia; Canary skyrybą sudeda pats)
--model  azuolas | large-v3 | turbo | canary (numatytasis – ASR_MODEL env arba azuolas)
"""
import argparse
import logging
import sys
import time

import asr


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
    p.add_argument('--model', choices=list(asr.MODELS),
                   help='ASR modelis (numatytasis: ASR_MODEL env arba azuolas)')
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
    model = asr.load(args.model)
    # Canary punctuates by itself; punctuation.py would only double it.
    if args.punct and not model.native_punctuation:
        from punctuation import Punctuator
        punctuator = Punctuator()

    t0 = time.monotonic()
    changes = 0
    with open(args.output, 'w', encoding='utf-8') as out:
        for start, end, text in model.transcribe_file(args.input):
            if not text:
                continue
            # Punctuation first (it never changes ASR words), then spelling on its result.
            if punctuator:
                try:
                    text = punctuator(text)
                except Exception:
                    log.exception('Skyrybos klaida, segmentas paliekamas be skyrybos')
            if corrector:
                text, pieces = corrector.correct(text)
                for fixed, original in pieces:
                    if original is not None:
                        changes += 1
                        log.info('  %s -> %s', original, fixed)
            line = text if args.no_timestamps else '[%.2f - %.2f] %s' % (start, end, text)
            out.write(line + '\n')
            out.flush()
            print(line, flush=True)

    log.info('Baigta (%s) per %.1f s -> %s%s', model.label, time.monotonic() - t0, args.output,
             ' (pataisyta žodžių: %d)' % changes if corrector else '')


if __name__ == '__main__':
    sys.exit(main())
