# LFM Speech-to-Type

> A local, privacy-first alternative to WhisperFlow — powered by Liquid AI's **LFM2.5-Audio-1.5B** model running entirely on your machine.

Press a hotkey → speak → text appears in whatever input field is focused. No cloud, no subscription, no data leaving your computer.

---

## Demo

| State | Tray icon |
|-------|-----------|
| Idle / ready | 🟢 Green mic |
| Recording | 🔴 Red mic |
| Transcribing | 🟡 Yellow mic |

---

## Requirements

| Requirement | Notes |
|-------------|-------|
| Windows 10/11 | Tested on Windows 11 |
| Python 3.11+ | [python.org](https://python.org) |
| Visual Studio 2022 | Community edition is free — needed to build the inference server |
| CMake | Install via `pip install cmake` |
| ~1 GB disk | For Q4_0 model files |
| HuggingFace account | Free — needed to download the model |

> **GPU optional.** A CUDA GPU speeds up inference but the Q4_0 models run fine on CPU.
> CUDA 12.6 + MSVC 14.39+ required for GPU builds (see [GPU build](#gpu-build-optional)).

---

## Setup

### 1 — Clone this repo

```bash
git clone https://github.com/YOUR_USERNAME/lfm-speech-to-type.git
cd lfm-speech-to-type
```

### 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3 — HuggingFace token

The model is gated — you need a free HuggingFace account and to accept the model license.

1. Create an account at [huggingface.co](https://huggingface.co)
2. Visit [LiquidAI/LFM2.5-Audio-1.5B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF) and accept the license
3. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Save it:

```bash
# Windows
python -c "from pathlib import Path; Path('~/.cache/huggingface').expanduser().mkdir(parents=True, exist_ok=True); Path('~/.cache/huggingface/token').expanduser().write_text('hf_YOUR_TOKEN_HERE')"
```

Or install the HF CLI and run `huggingface-cli login`.

### 4 — Download model files

```bash
pip install huggingface_hub

python -c "
from huggingface_hub import hf_hub_download
from pathlib import Path

repo = 'LiquidAI/LFM2.5-Audio-1.5B-GGUF'
dest = Path('models')
dest.mkdir(exist_ok=True)

files = [
    'LFM2.5-Audio-1.5B-Q4_0.gguf',
    'mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf',
    'vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf',
    'tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf',
]
for f in files:
    print(f'Downloading {f}...')
    hf_hub_download(repo, f, local_dir=dest)
print('Done.')
"
```

Total download: ~1 GB.

### 5 — Build the inference server

The app uses a custom fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) that adds LFM audio support ([PR #18641](https://github.com/ggml-org/llama.cpp/pull/18641)).

```bash
# Install cmake if you don't have it
pip install cmake

# Clone llama.cpp and check out the audio PR branch
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git fetch origin pull/18641/head:pr-18641
git checkout pr-18641

# Configure (CPU-only — works on any machine)
cmake -B build -G "Visual Studio 17 2022" -A x64 ^
    -DBUILD_SHARED_LIBS=OFF ^
    -DGGML_CUDA=OFF ^
    -DLLAMA_CURL=OFF

# Build only the audio server (the CLI target has a Windows build issue — not needed)
cmake --build build --config Release --target llama-liquid-audio-server -j 8

cd ..
```

The binary lands at:
```
llama.cpp\build\bin\Release\llama-liquid-audio-server.exe
```

#### GPU build (optional)

Requires MSVC 14.39+ (VS 17.9+) and CUDA 12.x. Upgrade VS first:

```
winget upgrade Microsoft.VisualStudio.2022.Community
```

Then replace `-DGGML_CUDA=OFF` with `-DGGML_CUDA=ON` in the cmake configure step.

---

## Running

```bash
python lfm_speech_to_type.py
```

Or double-click **`start.bat`**.

The server takes **30–60 seconds** to load on first launch. A tray icon appears in the system tray when ready.

---

## Hotkey

Default: **`Ctrl + Alt + Space`**

### Change it in `config.ini`

On first run a `config.ini` is created automatically:

```ini
[settings]
; Examples: ctrl+alt+space | ctrl+shift+r | f9
hotkey = ctrl+alt+space

port = 8142
threads = 4
sample_rate = 16000
```

Edit and save — takes effect on next launch.

### Or pass it as a CLI flag

```bash
python lfm_speech_to_type.py --hotkey ctrl+shift+r
python lfm_speech_to_type.py --hotkey f9
```

---

## All CLI options

```
usage: lfm_speech_to_type.py [-h] [--hotkey HOTKEY] [--port PORT]
                              [--host HOST] [--threads THREADS] [--no-server]

options:
  --hotkey    Toggle hotkey (default: ctrl+alt+space)
  --port      Audio server port (default: 8142)
  --host      Audio server host (default: 127.0.0.1)
  --threads   CPU threads for inference (default: 4)
  --no-server Skip launching the server (assume it is already running)
```

---

## Running the server manually

If you want to run the inference server separately (e.g. to keep it warm between app restarts):

```bash
llama.cpp\build\bin\Release\llama-liquid-audio-server.exe ^
  -m  models\LFM2.5-Audio-1.5B-Q4_0.gguf ^
  -mm models\mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf ^
  -mv models\vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf ^
  --tts-speaker-file models\tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf ^
  -t 4 --host 127.0.0.1 --port 8142
```

Then launch the app with `--no-server`:

```bash
python lfm_speech_to_type.py --no-server
```

---

## How it works

```
Hotkey pressed
    → Microphone recording starts (pyaudio, 16 kHz float32)
Hotkey pressed again
    → WAV bytes encoded as base64
    → POST /v1/chat/completions  (OpenAI-compatible streaming API)
        system: "Perform ASR."
        user:   { type: input_audio, data: <base64 wav> }
    → Streamed text collected
    → Text pasted into active input field via clipboard (Ctrl+V)
```

The model (`LFM2.5-Audio-1.5B`) runs locally via a custom llama.cpp server — no network calls, no API keys, no telemetry.

---

## Project structure

```
lfm-speech-to-type/
├── lfm_speech_to_type.py   # Main app
├── test_asr.py             # Quick smoke-test (server must be running)
├── requirements.txt
├── start.bat               # Windows launcher
├── config.ini              # Auto-created on first run (gitignored)
├── models/                 # GGUF files (gitignored — download manually)
└── llama.cpp/              # Cloned + built locally (gitignored)
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `[ERROR] Server binary not found` | Complete step 5 (Build) |
| `[ERROR] Model file not found` | Complete step 4 (Download) |
| Server starts but 404 on all routes | Normal — server only exposes `/v1/chat/completions` |
| Hotkey not working | Run as Administrator (Windows requires admin for some global hooks) |
| CUDA build: `cudafe++ ACCESS_VIOLATION` | Upgrade MSVC to 14.39+ via VS Installer or `winget upgrade` |
| Transcription is slow | Increase `--threads`, or upgrade to GPU build |

---

## Credits

- [Liquid AI](https://liquid.ai) — LFM2.5-Audio-1.5B model
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — inference engine
- [PR #18641](https://github.com/ggml-org/llama.cpp/pull/18641) — LFM audio support

---

## License

MIT
