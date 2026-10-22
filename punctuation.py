"""CPU-only punctuation restoration; never edits an ASR word."""
import re
from transformers import pipeline

MODEL = 'Horizon-Labs/punctuation-restoration-small'
ALLOWED = {'.', ',', '?', ':', '-'}

class Punctuator:
    def __init__(self):
        self.tagger = pipeline('token-classification', model=MODEL,
                               aggregation_strategy='none', device=-1)

    def __call__(self, raw: str) -> str:
        words = raw.split()
        if not words:
            return ''
        rendered = []
        # Short independent windows fit the model context. Never regenerate words.
        for offset in range(0, len(words), 60):
            chunk = words[offset:offset + 60]
            text = ' '.join(chunk)
            positions = list(re.finditer(r'\S+', text))
            predictions = self.tagger(text)
            for match, word in zip(positions, chunk):
                # The punctuation label is assigned to the last token of each word.
                inside = [p for p in predictions
                          if p['end'] <= match.end() and p['end'] > match.start()]
                label = inside[-1]['entity'] if inside else '0'
                mark = label if label in ALLOWED else ''
                rendered.append(word + mark)
        result = ' '.join(rendered)
        # Uppercase first letter after sentence-ending punctuation. Proper nouns
        # and remaining spelling stay exactly as recognized by Whisper.
        result = re.sub(r'(^|[.!?]\s+)([a-ząčęėįšųūž])',
                        lambda m: m.group(1) + m.group(2).upper(), result)
        # Invariant: no ASR word can be removed, added, or respelled.
        def tokens(s):
            return re.sub(r'[.,?:-]', '', s).casefold().split()
        if tokens(result) != tokens(raw):
            raise ValueError('Punctuation output changed original words')
        return result
