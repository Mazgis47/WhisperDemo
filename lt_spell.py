"""Lithuanian word check and auto-correction for ASR output.

Ported from LT-Checker (C#): DictionaryBinaryReader + WordAnalyzer (morphology),
SuggestionIndex (suggestions), SpellCheckService.TrySplitGluedWord (glued words),
plus CheckWord exceptions (numbers, ordinals, ALL CAPS, abbreviations).
No sentence context is used: an unknown word is replaced by its first suggestion.
"""
import logging
import os
import re
import struct
import time
from functools import lru_cache

import numpy as np

log = logging.getLogger('lt-spell')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LT-Checker', 'data')
# Colloquial words with a fixed replacement ("nu na"); checked before any other rule.
COLLOQUIAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'colloquial.txt')
NO_PREFIX = 105
# Rank suggestions by lt_freq.bin corpus frequency (as in C# SuggestionIndex).
USE_FREQUENCY = True
# lt_freq.bin words seen fewer times are mostly noise ("žųsis 1") and are not loaded.
MIN_SUGGESTION_FREQUENCY = 5


class _BinReader:
    """.NET BinaryReader compatible reader (little-endian, 7-bit length strings)."""

    def __init__(self, path):
        with open(path, 'rb') as f:
            self.buf = f.read()
        self.pos = 0

    def i32(self):
        v = struct.unpack_from('<i', self.buf, self.pos)[0]
        self.pos += 4
        return v

    def boolean(self):
        v = self.buf[self.pos] != 0
        self.pos += 1
        return v

    def string(self):
        length = shift = 0
        while True:
            b = self.buf[self.pos]
            self.pos += 1
            length |= (b & 0x7F) << shift
            if b < 0x80:
                break
            shift += 7
        s = self.buf[self.pos:self.pos + length].decode('utf-8')
        self.pos += length
        return s

    def nullable_i32(self):
        return self.i32() if self.boolean() else None

    def nullable_string(self):
        return self.string() if self.boolean() else None


# ---------------------------------------------------------------- morphology

class Morphology:
    """dict.bin (DictionaryBinaryReader) + WordAnalyzer.AnalyzeSingleWord."""

    def __init__(self, path):
        r = _BinReader(path)
        version = r.i32()
        if version not in (1, 2):
            raise ValueError('Nepalaikoma dict.bin versija: %d' % version)

        # normalized root -> [(root, f_id, id)]
        self.roots = {}
        for _ in range(r.i32()):
            saknis = r.string()
            fid, rid = r.i32(), r.i32()
            r.nullable_i32(); r.nullable_i32(); r.nullable_i32(); r.nullable_i32()
            self.roots.setdefault(saknis.lower(), []).append((saknis, fid, rid))

        self.prefixes = []  # [(pr, pr_nr)] in file order
        for _ in range(r.i32()):
            pr = r.string()
            self.prefixes.append((pr, r.i32()))

        # ending text -> [(gal, f_id, fnr, sangr, dgs)]
        self.endings = {}
        for _ in range(r.i32()):
            gal = r.string()
            fid, fnr = r.i32(), r.i32()
            r.nullable_string(); r.nullable_string()
            sangr = r.i32()
            r.i32(); r.i32()                      # kl_nr, gim
            dgs = r.i32()
            for _ in range(8):                    # linksnis .. id
                r.i32()
            r.nullable_i32()                      # antrinis
            self.endings.setdefault(gal, []).append((gal, fid, fnr, sangr, dgs))

        self.daznis_zero = set()
        for _ in range(r.i32()):
            r.i32()
            zodis = r.string()
            pr_nr, saknis_id, fnr, svoris = r.i32(), r.i32(), r.i32(), r.i32()
            if svoris == 0:
                self.daznis_zero.add((zodis, pr_nr, saknis_id, fnr))

        self.bendratys_only = set()
        for _ in range(r.i32()):
            self.bendratys_only.add((r.i32(), r.i32()))
        # Klausimai and verb requirements are only needed for sentence context.

    def analyze(self, word):
        """Returns [(surface, id, dgs)] like WordAnalyzer.AnalyzeSingleWord."""
        word = word.strip()
        if not word:
            return []
        raw = self._matches(word, '', NO_PREFIX)
        for pr, pr_nr in self.prefixes:
            if pr_nr == NO_PREFIX or not pr or not word.startswith(pr):
                continue
            raw.extend(self._matches(word[len(pr):], pr, pr_nr))
        result = [(a['priesd'] + a['saknis'] + a['gal'], a['id'], a['dgs'])
                  for a in raw if self._is_allowed(a)]
        if not result:
            result = _synthesize_surname(word)
        return result

    def _matches(self, text, prefix, pr_nr):
        out = []
        for i in range(len(text) + 1):
            roots = self.roots.get(text[:i].lower())
            if not roots:
                continue
            endings = self.endings.get(text[i:].lower())
            if not endings:
                continue
            for saknis, fid, rid in roots:
                for gal, efid, fnr, sangr, dgs in endings:
                    if fid == efid:
                        out.append({'priesd': prefix, 'pr_nr': pr_nr, 'saknis': saknis,
                                    'fid': fid, 'id': rid, 'gal': gal, 'fnr': fnr,
                                    'sangr': sangr, 'dgs': dgs})
        return out

    def _is_allowed(self, r):
        pr, fid, rid, sangr = r['pr_nr'], r['fid'], r['id'], r['sangr']
        saknis, gal = r['saknis'], r['gal']
        if ((pr != 105 and fid > 200 and (fid < 800 or fid >= 900)) or
                (800 < fid < 900 and pr not in (53, 41, 222, 105))):
            return False
        if sangr == 1 and pr != 105:
            return False
        bp = saknis[:1] in ('b', 'p')
        if pr in (100, 106, 109) and not bp:
            return False
        if pr in (101, 107, 31, 91, 110, 108) and bp:
            return False
        if pr in (38, 50, 75, 102) and saknis[:1] == 't':
            return False
        if pr >= 112 and sangr != 0:
            return False
        if pr in (111, 221) and rid not in (29204, 87134):
            return False
        if (pr in (226, 235, 229, 230, 227) and rid != 2 and
                (fid != 122 or not (gal.startswith('e') or gal.startswith('ė')))):
            return False
        if pr == 105 and fid == 122 and gal == 'ėra':
            return False
        if pr in (232, 233, 234) and rid not in (16588, 17627):
            return False
        if pr == 236 and rid != 124173:
            return False
        if rid in (124262, 124794) and sangr != 1 and not r['priesd'].endswith('si'):
            return False
        if r['fnr'] == 449 and (pr, rid) in self.bendratys_only:
            return False
        if (r['priesd'] + saknis + gal, pr, rid, r['fnr']) in self.daznis_zero:
            return False
        return True


# Unknown capitalized words that inflect like Lithuanian surnames
# (WordAnalyzer.TrySynthesizeCapitalizedProperNounAnalyses).
_MASC_MARKERS = ('ausk', 'insk', 'eck', 'ūn', 'ėn', 'ait', 'evič', 'avič', 'en')
_FEM_ONLY_MARKERS = ('ien', 'yt', 'ut', 'iūt', 'uvien')


def _ends_with_any(stem, markers):
    s = stem.lower()
    return any(s.endswith(m) for m in markers)


def _masc_stem(stem):
    return _ends_with_any(stem, _MASC_MARKERS)


def _us_stem(stem):
    return stem[-1:] in ('k', 'K')


def _fem_stem(stem):
    return _masc_stem(stem) or _ends_with_any(stem, _FEM_ONLY_MARKERS) or _us_stem(stem)


_SURNAME_PARADIGMS = (
    (('as', 'o', 'ui', 'ą', 'u', 'e'), ('ai', 'ų', 'ams', 'us', 'ais', 'uose'), _masc_stem),
    (('is', 'io', 'iui', 'į', 'iu', 'yje'), ('iai', 'ių', 'iams', 'ius', 'iais', 'iuose'), _masc_stem),
    (('ys', 'io', 'iui', 'į', 'iu', 'yje'), ('iai', 'ių', 'iams', 'ius', 'iais', 'iuose'), None),
    (('ius', 'iaus', 'iui', 'ių', 'iumi', 'iuje'), ('iai', 'ių', 'iams', 'ius', 'iais', 'iuose'), _masc_stem),
    (('us', 'aus', 'ui', 'ų', 'umi', 'uje'), ('ūs', 'ų', 'ums', 'us', 'umis', 'uose'), _us_stem),
    (('ė', 'ės', 'ei', 'ę', 'e', 'ėje'), ('ės', 'ių', 'ėms', 'es', 'ėmis', 'ėse'), _fem_stem),
)
_MIN_SURNAME_STEM = 3


def _synthesize_surname(word):
    if len(word) < _MIN_SURNAME_STEM + 1 or not word[0].isupper():
        return []
    out = []
    low = word.lower()
    for singular, plural, gate in _SURNAME_PARADIGMS:
        for dgs, suffixes in ((1, singular), (2, plural)):
            for suffix in suffixes:
                if len(word) - len(suffix) < _MIN_SURNAME_STEM or not low.endswith(suffix):
                    continue
                if gate is not None and not gate(word[:len(word) - len(suffix)]):
                    continue
                out.append((word, -1, dgs))  # synthetic: negative id
    return out


# ---------------------------------------------------------------- suggestions

_DIACRITIC_BASE = str.maketrans('ąčęėįšųūžĄČĘĖĮŠŲŪŽ', 'aceeisuuzACEEISUUZ')


def _is_diacritic_variant(a, b):
    return a != b and a.translate(_DIACRITIC_BASE) == b.translate(_DIACRITIC_BASE)


def osa_distance_max2(s, t):
    """SuggestionIndex.LevenshteinDistanceMax2 (with adjacent transposition); >2 -> 3."""
    n, m = len(s), len(t)
    if abs(n - m) > 2:
        return 3
    if n == 0:
        return m if m <= 2 else 3
    if m == 0:
        return n if n <= 2 else 3
    prev2 = None
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        row_min = i
        si = s[i - 1]
        for j in range(1, m + 1):
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (si != t[j - 1]))
            if i > 1 and j > 1 and si == t[j - 2] and s[i - 2] == t[j - 1]:
                v = min(v, prev2[j - 2] + 1)
            cur[j] = v
            if v < row_min:
                row_min = v
        if row_min > 2:
            return 3
        prev2, prev = prev, cur
    return prev[m] if prev[m] <= 2 else 3


class SuggestionIndex:
    """lt_freq.bin (SuggestionWordBinaryReader) + SuggestionIndex.Suggest."""

    def __init__(self, path):
        r = _BinReader(path)
        version = r.i32()
        if version != 1:
            raise ValueError('Nepalaikoma lt_freq.bin versija: %d' % version)
        # Words below MIN_SUGGESTION_FREQUENCY are not loaded at all. Morphology
        # words are sorted by frequency (descending), so the first rare one ends
        # the load; lexicon-only words have reduced frequencies interleaved
        # (SuggestionLexiconBuilder), so rare ones are only skipped.
        words, norms, freq, morph = [], [], [], []
        for _ in range(r.i32()):
            r.i32()
            word = r.string()
            norm = r.string()
            f = r.i32()
            m = r.boolean()
            if f < MIN_SUGGESTION_FREQUENCY:
                if m:
                    break
                continue
            words.append(word)
            norms.append(norm if norm else word.strip().lower())
            freq.append(f)
            morph.append(m)
        count = len(words)
        self.words, self.norms = words, norms
        self.freq = np.asarray(freq, dtype=np.int32)
        self.morph = np.asarray(morph, dtype=bool)
        self.exact = set(words)
        self.lengths = np.fromiter((len(n) for n in norms), dtype=np.int16, count=count)

        postings = {}
        for i, norm in enumerate(norms):
            if len(norm) == 1:
                bigrams = (norm,)
            else:
                bigrams = {norm[k:k + 2] for k in range(len(norm) - 1)}
            for bg in bigrams:
                lst = postings.get(bg)
                if lst is None:
                    postings[bg] = [i]
                else:
                    lst.append(i)
        self.bigrams = {bg: np.asarray(ids, dtype=np.int32) for bg, ids in postings.items()}

    def __contains__(self, word):
        return word in self.exact

    def suggest(self, word, max_results=5):
        """Returns [(candidate, distance, morphology_match)] best first."""
        word = word.strip()
        if not word or word in self.exact:
            return []
        word_caps = _is_all_caps(word)
        norm = word.lower()
        n = len(norm)

        if n <= 3:
            ids = np.nonzero((self.lengths >= n - 1) & (self.lengths <= n + 1))[0]
            counts = np.zeros(len(ids), dtype=np.int32)
        else:
            lists = [self.bigrams[bg] for bg in {norm[k:k + 2] for k in range(n - 1)}
                     if bg in self.bigrams]
            if not lists:
                return []
            ids, counts = np.unique(np.concatenate(lists), return_counts=True)
            threshold = 1 if n <= 4 else 2 if n <= 7 else 3
            keep = counts >= threshold
            ids, counts = ids[keep], counts[keep]
        keep = np.abs(self.lengths[ids].astype(np.int32) - n) <= 2
        ids, counts = ids[keep], counts[keep]

        scored = []
        for wid, bg_matches in zip(ids.tolist(), counts.tolist()):
            cand_norm = self.norms[wid]
            if not _enough_common_chars(cand_norm, norm):
                continue
            dist = osa_distance_max2(norm, cand_norm)
            if dist > 2:
                continue
            cand = self.words[wid]
            # Auto-replacement (no user choice, no frequency to rank): never
            # introduce ALL CAPS, and lexicon-only words (no morphology) are
            # accepted only as a pure case fix ("larsas" -> "Larsas").
            if not word_caps and len(cand) > 1 and _is_all_caps(cand):
                continue
            if dist > 0 and not self.morph[wid]:
                continue
            score = _score(bool(self.morph[wid]), dist, int(self.freq[wid]), bg_matches,
                           cand_norm, norm, cand, word)
            tie_freq = -int(self.freq[wid]) if USE_FREQUENCY else 0
            scored.append((score, dist, tie_freq, cand, bool(self.morph[wid])))
        scored.sort()
        out, seen = [], set()
        for _, dist, _, cand, morph in scored:
            if cand in seen:
                continue
            seen.add(cand)
            out.append((cand, dist, morph))
            if len(out) >= max_results:
                break
        return out


def _enough_common_chars(a, b):
    min_len = min(len(a), len(b))
    matches = sum(1 for i in range(min_len) if a[i] == b[i] or _is_diacritic_variant(a[i], b[i]))
    if min_len <= 4:
        return matches >= 1
    if min_len <= 8:
        return matches >= 2
    return matches >= 3


def _score(morph, distance, frequency, bigram_matches, cand_norm, input_norm, cand, original):
    score = distance * 1000.0
    if USE_FREQUENCY:
        score -= max(12.0, np.log10(max(1, frequency)) * 200.0)
    score -= bigram_matches * 5.0
    len_diff = abs(len(cand_norm) - len(input_norm))
    score += len_diff * 15.0
    if len_diff == 0:
        score -= 8.0
    if cand_norm[:1] and cand_norm[:1] == input_norm[:1]:
        score -= 3.0
    score -= 12.0 * sum(1 for x, y in zip(cand_norm, input_norm) if _is_diacritic_variant(x, y))
    if original[0].isupper() and cand[0].isupper():
        score -= 10.0
    elif original[0].islower() and cand[0].islower():
        score -= 5.0
    if morph:
        score -= 200
    return score


# ---------------------------------------------------------------- exceptions

# SentenceTokenizer.Abbreviations (single-token entries; checked lowercased, no dot).
ABBREVIATIONS = {
    'a', 'angl', 'pr', 'm', 'e', 'adv', 'akad', 'aklg', 'akt', 'al', 'apyg', 'aps', 'apskr',
    'asist', 'asmv', 'avd', 'atsak', 'aut', 'biol', 'bkl', 'bot', 'bt', 'buv', 'chem', 'd',
    'dab', 'dail', 'dek', 'dėst', 'dir', 'dirig', 'doc', 'dr', 'drp', 'dš', 'egz', 'eil',
    'ekon', 'el', 'etc', 'ež', 'fak', 'faks', 'filol', 'filos', 'g', 'gyv', 'gen', 'geol',
    'gerb', 'gim', 'gyd', 'gv', 'įl', 'įn', 'insp', 'inž', 'pan', 'istor', 'k', 'em',
    'kand', 'kat', 'kg', 'kirč', 'kyš', 'kl', 'kln', 'kn', 'koresp', 'kpt', 'kr', 'kt',
    'kun', 'liet', 'ltn', 'mat', 'med', 'mėn', 'mgr', 'mgnt', 'min', 'mjr', 'mln', 'mlrd',
    'mok', 'mst', 'mstl', 'n', 'nkt', 'ntk', 'nr', 'ojo', 'ijo', 'osios', 'oji', 'p', 'pav',
    'pavad', 'pirm', 'pl', 'plg', 'plk', 'proc', 'prof', 'prok', 'prot', 'pss', 'pšt',
    'pvz', 'r', 'red', 'rš', 'raj', 'rus', 's', 'sąs', 'sav', 'saviv', 'sekr', 'sek', 'sen',
    'sk', 'skg', 'skyr', 'skv', 'sp', 'spec', 'sr', 'st', 'str', 'stud', 'šnek', 'šv', 't',
    'techn', 'tel', 'teol', 'tir', 'tūkst', 'up', 'upl', 'v', 'vad', 'val', 'ved', 'vet',
    'vnt', 'vrš', 'vyr', 'vyresn', 'vs', 'vt', 'vtv', 'vv', 'zool', 'žml', 'žr', 'šmt',
    'nuotr',
    # Parts of multi-part abbreviations ("t. y.", "p. m. e.", "ž. ū.") seen as one token.
    'y', 'ū', 'ž', 'š', 'l',
}
# Glued-word split parts must have 3+ letters, except these common short words.
_SHORT_SPLIT_WORDS = {'į', 'o', 'aš', 'tu', 'ir', 'be', 'ne', 'su', 'už', 'iš', 'po', 'ar',
                      'jo', 'ją', 'jį', 'jų', 'to', 'tą', 'ką', 'ta', 'ko', 'kur', 'nė'}
_WORD_RE = re.compile(r'[^\W\d_]+')


def _is_all_caps(word):
    return any(c.isalpha() for c in word) and all(c.isupper() for c in word if c.isalpha())


# ---------------------------------------------------------------- corrector

def load_colloquial(path=COLLOQUIAL_FILE):
    """'žodis pakeitimas' per line, '#' comments; a missing file means no entries."""
    table = {}
    if not os.path.exists(path):
        return table
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.split('#', 1)[0].split(None, 1)
            if len(parts) == 2:
                table[parts[0].lower()] = parts[1].strip()
    return table


class Corrector:
    def __init__(self, data_dir=DATA_DIR, colloquial_file=COLLOQUIAL_FILE):
        t0 = time.monotonic()
        self.morph = Morphology(os.path.join(data_dir, 'dict.bin'))
        t1 = time.monotonic()
        self.index = SuggestionIndex(os.path.join(data_dir, 'lt_freq.bin'))
        self.colloquial = load_colloquial(colloquial_file)
        log.info('LT spell data loaded: dict.bin %.1fs, lt_freq.bin %.1fs (%d words), '
                 'colloquial %d', t1 - t0, time.monotonic() - t1, len(self.index.words),
                 len(self.colloquial))

    @lru_cache(maxsize=200000)
    def _analyses(self, word):
        return tuple(self.morph.analyze(word))

    def _has_exact_case(self, word, allow_synthetic=True):
        # SpellCheckService.FilterExactCaseAnalyses: surface must equal the word.
        return any(s == word and (allow_synthetic or rid >= 0)
                   for s, rid, _ in self._analyses(word))

    def _morph_known(self, word, allow_synthetic=True):
        # SpellCheckService.ComputeInputAnalyses. CheckWord's "capitalized plural"
        # retry is left out on purpose: it accepted "lietuvos" ("Lietuvos" has a
        # plural analysis), but lowercase proper nouns from ASR must be fixed.
        if self._has_exact_case(word, allow_synthetic):
            return True
        lower = word.lower()
        return lower != word and self._has_exact_case(lower, allow_synthetic)

    def is_known(self, word):
        if word in self.index:
            return True
        # ASR may capitalize a sentence-initial word (CheckWord isFirstInSentence path).
        if word[0].isupper() and word.lower() in self.index:
            return True
        return self._morph_known(word)

    def _valid_part(self, part):
        # Stricter than C# (which only suggests): automatic split parts must be
        # dictionary words of 3+ letters (or a common short word) AND be loaded
        # from lt_freq.bin (frequency >= MIN_SUGGESTION_FREQUENCY): rare forms
        # like "ažo" made "uždarbiažo" -> "uždarbi ažo".
        if part not in self.index:
            return False
        if len(part) < 3:
            return part.lower() in _SHORT_SPLIT_WORDS
        return self._morph_known(part, allow_synthetic=False)

    def split_glued(self, word):
        """SpellCheckService.TrySplitGluedWord: 'manošuo' -> 'mano šuo'."""
        if not (5 <= len(word) <= 40) or not word.isalpha() or any(c.isupper() for c in word[1:]):
            return []
        lower = word.lower()
        capitalized = word[0].isupper()
        candidates = []
        for cut in range(1, len(lower)):
            left, right = lower[:cut], lower[cut:]
            if not self._valid_part(right):
                continue
            left_cap = left[0].upper() + left[1:]
            if self._valid_part(left):
                first = left_cap if capitalized else left
            elif len(left) >= 3 and self._valid_part(left_cap):
                first = left_cap
            else:
                continue
            score = min(len(left), len(right))
            if left in self.index or left_cap in self.index:
                score += 0.5
            if right in self.index:
                score += 0.5
            candidates.append((score, first + ' ' + right))
        candidates.sort(key=lambda c: -c[0])  # stable: equal scores keep cut order
        return [c[1] for c in candidates[:3]]

    @lru_cache(maxsize=50000)
    def best_replacement(self, word):
        """First suggestion for an unknown word, or None (word is kept)."""
        fixed = self.colloquial.get(word.lower())
        if fixed:
            return fixed
        if self.is_known(word):
            return None
        # Dictionary form differing only in case ("makronas" -> "Makronas"). Taken
        # from dict.bin directly, so new dictionary words work without lt_freq.bin.
        low = word.lower()
        case_fix = sorted({s for s, rid, _ in self._analyses(low) if rid >= 0 and s.lower() == low})
        if case_fix:
            return case_fix[0]
        regular = self.index.suggest(word, 5)
        # A case/diacritic-free identical form ("londoną" -> "Londoną") beats a split.
        if regular and regular[0][1] == 0:
            return regular[0][0]
        no_analyses = not self._morph_known(word)
        splits = self.split_glued(word) if no_analyses else []
        if splits:
            return splits[0]
        return regular[0][0] if regular else None

    def replacement_for(self, text, start, end):
        """Replacement for the word text[start:end], or None to keep it."""
        word = text[start:end]
        before = text[start - 1] if start > 0 else ''
        after = text[end] if end < len(text) else ''
        # Part of a code/number ("A320", "5-asis") or URL/e-mail: keep.
        if before.isdigit() or after.isdigit() or (before and before in '@/') or (after and after in '@/'):
            return None
        if _is_all_caps(word):                       # IsAllCapsAcronym
            return None
        if word.lower() in ABBREVIATIONS:            # IsKnownAbbreviation
            return None
        if any(c.isupper() for c in word[1:]):      # CamelCase / brand names
            return None
        replacement = self.best_replacement(word)
        if not replacement or replacement == word:
            return None
        if word[0].isupper() and replacement[0].islower():
            replacement = replacement[0].upper() + replacement[1:]
        return replacement

    def correct(self, text):
        """Returns (corrected_text, pieces); pieces = [[text, original_or_None], ...]."""
        pieces, pos = [], 0
        for m in _WORD_RE.finditer(text):
            replacement = self.replacement_for(text, m.start(), m.end())
            if replacement is None:
                continue
            if m.start() > pos:
                pieces.append([text[pos:m.start()], None])
            pieces.append([replacement, m.group(0)])
            pos = m.end()
        if pos < len(text):
            pieces.append([text[pos:], None])
        return ''.join(p[0] for p in pieces), pieces


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)
    c = Corrector()
    for line in (sys.argv[1:] or sys.stdin):
        t0 = time.monotonic()
        fixed, pieces = c.correct(line.strip())
        print(fixed)
        for r, o in pieces:
            if o is not None:
                print('  %s -> %s' % (o, r))
        print('  (%.2fs)' % (time.monotonic() - t0))
