# Whisper Live LT v2 – Žodžiai iš Ąžuolo, skyryba atskirai

## Ubuntu (192.168.68.110)

```bash
source ~/whisper-env/bin/activate
pip install -U 'transformers>=4.57,<5' torch
mkdir -p ~/whisper-live
# Extract ZIP; copy whisper_live_lt_v2/server.py and punctuation.py into ~/whisper-live
cd ~/whisper-live
python -c "from punctuation import Punctuator; p=Punctuator(); print(p('labas rytas ar tu mane girdi šiandien yra graži diena'))"
uvicorn server:app --host 0.0.0.0 --port 8001
```

`server.py` currently points to `/home/mp/whisper-models/azuolas-ct2`. Change `MODEL_NAME` if your model lives elsewhere.

The first launch downloads `Horizon-Labs/punctuation-restoration-small` (~140M parameters). It runs on CPU and requires additional system RAM. It does not run a language model to rewrite words.

## Windows

Copy `index.html` and `pcm-worklet.js` to `C:\Mazho\AI\Audio\whisper_live_lt_v2` and run:

```powershell
cd C:\Mazho\AI\Audio\whisper_live_lt_v2
python -m http.server 5500 --bind 127.0.0.1
```

Open http://localhost:5500/index.html (not file://). `index.html` uses `ws://192.168.68.110:8001/ws/transcribe`.

Check/uncheck "Įjungti skyrybos atkūrimą" to control processing for *newly finalized* segments. The original text is always visible. Prior finalized segments only get punctuation if the checkbox was checked when the segment was finalized.

For WAN/public access, add TLS (HTTPS/WSS) and authentication.

## Notes

* The punctuation model predicts `. , ? : -`, no grammatical correction, no restoration of proper-name capitalization.
* Sentence-initial capitalization is a separate deterministic step and does not alter word spelling beyond the first letter.
* Small-word-chunk tagging can make mistakes at chunk/phrase boundaries; compare against the raw pane.
* `MAX_UTTERANCE` segments are forcibly finalized without cross-boundary reconciliation, like the previous prototype.
* Dependencies were syntax-checked. Model inference requires downloading model files and must be checked on your machine.
