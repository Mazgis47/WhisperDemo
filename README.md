# Whisper Live LT v2 – kalbos atpažinimas, žodžių taisymas ir skyryba

Lietuviška kalbos transkripcija iš mikrofono (gyvai) arba iš garso failo (offline):

1. **Kalbos atpažinimas** – Ąžuolas, Whisper large-v3 / turbo arba Canary-1B-v2 (GPU), žr. [ASR modelis](#asr-modelis).
2. **Skyrybos atkūrimas** – prideda `. , ? : -` ir sakinio pradžios didžiąją raidę, žodžių nekeičia (CPU).
3. **Žodžių taisymas (LT-Checker)** – žodyne nerastą žodį pakeičia pirmu pasiūlymu (CPU). Taikomas tekstui su skyryba, o jei skyryba išjungta – originaliam.

2 ir 3 žingsniai neprivalomi, įjungiami atskirai.

## Failai

| Failas | Paskirtis |
|---|---|
| `server.py` | Gyvos transkripcijos WebSocket serveris (`/ws/transcribe`) |
| `transcribe.py` | Offline garso failo transkripcija į tekstinį failą |
| `asr.py` | ASR modeliai (Ąžuolas, large-v3, turbo, Canary); pasirenkama `ASR_MODEL` |
| `lt_spell.py` | Žodžių tikrinimas ir taisymas (LT-Checker C# logika, perkelta į Python) |
| `colloquial.txt` | Šnekamosios kalbos žodžiai su fiksuotu pakeitimu (`nu na`) |
| `punctuation.py` | Skyrybos atkūrimas (`Horizon-Labs/punctuation-restoration-small`) |
| `index.html`, `pcm-worklet.js` | Web klientas (mikrofonas → serveris) |
| `LT-Checker/` | LT-Checker repozitorija; naudojami tik `data/dict.bin` ir `data/lt_freq.bin` (į git neįtraukta) |

## Ubuntu serveris (NVIDIA GPU)

Pavyzdžiuose: virtuali aplinka `~/whisper-env`, projekto katalogas `~/whisper/v2` – pasikeiskite į savo.

### Paruošimas

```bash
source ~/whisper-env/bin/activate
pip install -U transformers torch faster-whisper librosa   # Canary reikia transformers >= 5.20 ir librosa
cd ~/whisper/v2
git clone <LT-Checker repo> LT-Checker   # reikia LT-Checker/data/dict.bin ir lt_freq.bin
```

Skyrybos modelis ir pasirinktas ASR modelis atsisiunčiami per pirmą paleidimą.

### ASR modelis

Pasirenkamas aplinkos kintamuoju `ASR_MODEL` (galioja ir `server.py`, ir `transcribe.py`):

| `ASR_MODEL` | Modelis | VRAM | Skyryba | Pastabos |
|---|---|---|---|---|
| `azuolas` (numatytasis) | Ąžuolas (faster-whisper) | ~3,7 GB | `punctuation.py` | Lietuviškai tiksliausias; kelias – `MODELS` faile `asr.py` |
| `large-v3` | Whisper large-v3 | ~3,7 GB | pats | Atsisiunčiamas iš HF |
| `turbo` | Whisper large-v3-turbo | ~2,1 GB | pats | Greičiausias, mažiausiai tikslus |
| `canary` | NVIDIA Canary-1B-v2 (transformers) | ~2,0 GB (float16) | pats | Galutiniams segmentams beam search (4) |

```bash
ASR_MODEL=canary uvicorn server:app --host 0.0.0.0 --port 8001
```

Modeliams, kurie skyrybą ir didžiąsias raides sudeda patys (large-v3, turbo, Canary), `punctuation.py` nenaudojamas – jis ją tik padvigubintų; varnelė „Skyrybos atkūrimas“ tada tik rodo stulpelį.

Canary:
* savo skyrybą išjungti – `CANARY_PNC=0` (tada skyryba per `punctuation.py`, kaip Ąžuolui);
* greedy dekodavimas kartais praleidžia didžiąją segmento dalį, kai gale prasideda kita tema – todėl galutiniams segmentams naudojamas beam search;
* veikia `float16` (RTX 2070 neturi bfloat16); jei rezultatai keisti – `CANARY_DTYPE=float32` (~4 GB VRAM);
* failai (`transcribe.py`) karpomi pagal VAD į ≤ 30 s gabalus.

Web puslapio antraštėje rodomas serverio naudojamas modelis.

### Gyvas serveris

```bash
source ~/whisper-env/bin/activate
cd ~/whisper/v2
uvicorn server:app --host 0.0.0.0 --port 8001
```

Startas: ASR modelis į GPU (žr. lentelę), LT-Checker žodynas (~5 s, ~430 MB RAM), skyrybos modelis (CPU).
Pakeitus `server.py` arba `lt_spell.py`, serverį reikia perkrauti (Ctrl+C ir paleisti iš naujo).

### Offline transkripcija

```bash
source ~/whisper-env/bin/activate
cd ~/whisper/v2
python transcribe.py irasas.mp3 rezultatas.txt --spell --punct
```

| Parametras | Reikšmė |
|---|---|
| `input` | Garso failas (mp3, wav, …) |
| `output` | Rezultato tekstinis failas (UTF-8) |
| `--spell` | Taisyti žodyne nerastus žodžius (LT-Checker) |
| `--punct` | Įjungti skyrybos atkūrimą (be žodžių taisymo) |
| `--no-timestamps` | Nerašyti `[pradžia - pabaiga]` prie segmentų |
| `--model` | `azuolas`, `large-v3`, `turbo` arba `canary` (numatytasis – `ASR_MODEL` arba `azuolas`) |

Kiekvienas segmentas rašomas atskiroje eilutėje: `[0.08 - 23.78] tekstas`. Eilutės į failą rašomos iš karto, tad eigą matyti ekrane.
Su `--spell` ekrane rodomi visi pakeitimai (`lietuva -> Lietuva`) ir jų skaičius; į failą rašomas tik galutinis tekstas.
Jei tuo metu veikia ir gyvas serveris, GPU bus du ASR modeliai – gali pritrūkti VRAM (8 GB).

## Windows klientas

Nukopijuokite `index.html` ir `pcm-worklet.js` į bet kurį katalogą (pvz. `whisper_live_lt_v2`) ir jame paleiskite:

```powershell
python -m http.server 5500 --bind 127.0.0.1
```

Atidarykite http://localhost:5500/index.html (ne `file://`). Serverio adresą nustatykite `URL` faile `index.html`, pvz. `ws://<serverio-IP>:8001/ws/transcribe`.
Atnaujinus `index.html`, puslapį perkraukite su Ctrl+F5.

| Valdiklis | Veikimas |
|---|---|
| **Pradėti / Sustabdyti** | Mikrofono įrašymas |
| **Išvalyti** | Išvalo tekstą, įrašymas tęsiasi (puslapio perkrauti nereikia) |
| **Taisyti žodyne nerastus žodžius** | Įjungta pagal nutylėjimą; rodo stulpelį „Pataisyti žodžiai“ |
| **Įjungti skyrybos atkūrimą** | Rodo stulpelį „Su skyrybos ženklais“ |

* **Originali transkripcija** – visada tiksliai tai, ką atpažino Whisper.
* **Su skyrybos ženklais** – originalas su skyryba (žodžiai nekeisti).
* **Pataisyti žodžiai** – galutinis tekstas: taisoma tekste su skyryba, o jei skyryba išjungta – originale. Pakeisti žodžiai pažymėti geltonai; užvedus pelę rodoma „Buvo: …“.
* Varnelės veikia tik **naujiems** segmentams, ankstesni neperskaičiuojami.
* Puslapis automatiškai slenka prie naujausio teksto, kol esate apačioje; pakilus aukščiau – nejuda.

Viešam (WAN) naudojimui reikia TLS (HTTPS/WSS) ir autentifikacijos.

## Sakinio pabaigos laikas (`server.py`)

| Parametras | Reikšmė | Paaiškinimas |
|---|---|---|
| `CHECK_EVERY` | 0.4 s | Kaip dažnai tikrinama, ar baigėsi kalba |
| `FINAL_SILENCE` | 0.7 s | Tylos trukmė, po kurios segmentas laikomas baigtu |
| `PARTIAL_EVERY` | 2.0 s | Kaip dažnai rodomas tarpinis (pilkas) tekstas |
| `MAX_UTTERANCE` | 30 s | Ilgesnis segmentas baigiamas priverstinai |

Nuo kalbos pabaigos iki galutinio segmento – apie 0,85–1,25 s + Whisper apdorojimas.
`FINAL_SILENCE` nemažinkite žemiau ~0,5 s (VAD `min_silence_duration_ms=500`) – sakiniai bus karpomi per įprastas pauzes.
Trumpesni segmentai = mažiau konteksto Whisper ir skyrybai; jei karpo per dažnai, bandykite 0,8 s.

## Žodžių taisymas (`lt_spell.py`)

Perkelta iš LT-Checker C#: `DictionaryBinaryReader` + `WordAnalyzer` (morfologija, `IsAllowed`, pavardžių atpažinimas pagal galūnę), `SuggestionIndex` (pasiūlymai), `TrySplitGluedWord` (sulipę žodžiai). Sakinio kontekstas (derinimas su kitais žodžiais) nenaudojamas.

**Žodis laikomas teisingu**, jei jį atpažįsta morfologija (`dict.bin`) arba jis tiksliai yra `lt_freq.bin`.
**Nežinomas žodis** pakeičiamas pirmu pasiūlymu (atstumas ≤ 2 raidės); jei pasiūlymo nėra – paliekamas.

**Nekeičiami:**
* VISOS DIDŽIOSIOS raidės, žodžiai su skaitmenimis, skaičiai, eilės skaitvardžiai (`5-asis`);
* žinomos santrumpos (`SentenceTokenizer.Abbreviations` sąrašas iš C#);
* žodžiai su didžiąja raide viduryje (`McDonald`), URL / el. pašto dalys.

**Skirtumai nuo C#** (ten pasiūlymus renkasi žmogus, čia – automatiškai):
* žodžių dažnis rangavimui naudojamas kaip C# (`USE_FREQUENCY = True`; `False` – be dažnio);
* `lt_freq.bin` žodžiai su dažniu < 5 nekraunami (`MIN_SUGGESTION_FREQUENCY`) – greitesnis startas, mažiau RAM, mažiau šiukšlių;
* pakeitimas niekada nebūna VISOMIS DIDŽIOSIOMIS;
* žodis be morfologijos patvirtinimo siūlomas tik kaip raidžių dydžio pataisa (`rusija` → `Rusija`);
* skaidant sulipusį žodį abi dalys turi būti žinomos morfologijai, ≥ 3 raidžių (išskyrus aš, ir, be, su, …) ir įkeltos iš `lt_freq.bin` (dažnis ≥ 5) – retos formos neskaido (`uždarbiažo` ≠ `uždarbi ažo`);
* tik raidžių dydžiu ar diakritikais besiskiriantis variantas turi pirmenybę prieš skaidymą;
* išimta C# taisyklė, priimdavusi mažąja raide parašytą tikrinį daiktavardį su daugiskaitos analize (`lietuvos` dabar → `Lietuvos`).

**Šnekamosios kalbos žodžiai** – `colloquial.txt` (`žodis pakeitimas` eilutėje, pvz. `nu na`). Tikrinama prieš žodyną ir pasiūlymus, didžioji pirma raidė išlaikoma (`Nu` → `Na`). Skirta žodžiams, kurių pagal raides ir dažnį neišspręsi (`nu` pagal dažnį visada taptų `su`). Pakeitus failą, serverį reikia perkrauti.

Patikrinti atskirus žodžius ar tekstą:

```bash
python lt_spell.py "šiandien lietuva yra viena iš pirmųjų europos sąjungos šalių"
```

## Žinomi apribojimai

* **Žodyno spragos** – žodis, kurio nėra `dict.bin` (pvz. `retkarčiais`, `Velykos`), pakeičiamas artimiausiu žinomu (`renkančiais`, `vėlyvos`). Taisyti reikia LT-Checker žodyne.
* **Vardai be morfologijos** (`Larsas`, `Makronas`) turi mažą dažnį `lt_freq.bin` ir nebekraunami – keičiami į panašų žodį (`barsas`, `mikronas`).
* **Be konteksto** – pasiūlymas renkamas tik pagal raides ir dažnį, todėl dažnesnis žodis gali laimėti prieš tinkamą sakiniui.
* **Skyryba** – taikoma kiekvienam segmentui atskirai (60 žodžių langais), gali klysti ties ribomis; tikrinių vardų didžiųjų raidžių neatkuria (tai daro žodžių taisymas).
* `MAX_UTTERANCE` segmentai baigiami priverstinai, be suderinimo per ribą.
