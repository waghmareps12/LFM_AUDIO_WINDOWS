"""
LFM Speech-to-Type
A local, privacy-first alternative to WhisperFlow using Liquid AI's LFM2.5-Audio model.

Press your configured hotkey to start recording (tray turns red), press again to
transcribe and type the result into whatever input field is active.

Configuration: edit config.ini or pass CLI flags (see --help).
"""

import base64
import io
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import keyboard
import numpy as np
import pyaudio
import pyperclip
import soundfile as sf
from openai import OpenAI
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent.resolve()
MODELS_DIR = SCRIPT_DIR / "models"
SERVER_BIN = (
    SCRIPT_DIR / "llama.cpp" / "build" / "bin" / "Release" / "llama-liquid-audio-server.exe"
)
CONFIG_FILE = SCRIPT_DIR / "config.ini"

MODEL_MAIN      = MODELS_DIR / "LFM2.5-Audio-1.5B-Q4_0.gguf"
MODEL_MMPROJ    = MODELS_DIR / "mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf"
MODEL_VOCODER   = MODELS_DIR / "vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf"
MODEL_TOKENIZER = MODELS_DIR / "tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf"

# ---------------------------------------------------------------------------
# Config  (config.ini wins over these defaults; CLI flags win over config.ini)
# ---------------------------------------------------------------------------

DEFAULTS = {
    "hotkey": "ctrl+alt+space",
    "port": 8142,
    "host": "127.0.0.1",
    "threads": 4,
    "sample_rate": 16000,
}


def load_config() -> dict:
    """Read config.ini if it exists, falling back to DEFAULTS."""
    import configparser
    cfg = dict(DEFAULTS)
    if CONFIG_FILE.exists():
        parser = configparser.ConfigParser()
        parser.read(CONFIG_FILE)
        s = parser["settings"] if "settings" in parser else {}
        cfg["hotkey"]      = s.get("hotkey",      cfg["hotkey"])
        cfg["port"]        = int(s.get("port",     cfg["port"]))
        cfg["host"]        = s.get("host",         cfg["host"])
        cfg["threads"]     = int(s.get("threads",  cfg["threads"]))
        cfg["sample_rate"] = int(s.get("sample_rate", cfg["sample_rate"]))
    return cfg


def write_default_config():
    """Write config.ini with defaults if it doesn't exist yet."""
    if CONFIG_FILE.exists():
        return
    CONFIG_FILE.write_text(
        "[settings]\n"
        "# Hotkey to toggle recording on/off.\n"
        "# Examples: ctrl+alt+space  |  ctrl+shift+r  |  f9\n"
        f"hotkey = {DEFAULTS['hotkey']}\n\n"
        "# Audio server port\n"
        f"port = {DEFAULTS['port']}\n\n"
        "# CPU threads used by the inference server\n"
        f"threads = {DEFAULTS['threads']}\n\n"
        "# Microphone sample rate (Hz)\n"
        f"sample_rate = {DEFAULTS['sample_rate']}\n"
    )
    print(f"[CONFIG] Created default config at: {CONFIG_FILE}")


# ---------------------------------------------------------------------------
# Tray icon
# ---------------------------------------------------------------------------

def _make_icon(color: str) -> Image.Image:
    """Colored microphone icon for the system tray."""
    palette = {
        "green":  (0, 200, 80, 255),
        "red":    (220, 40, 40, 255),
        "yellow": (240, 200, 0, 255),
        "gray":   (120, 120, 120, 255),
    }
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([4, 4, 60, 60], fill=palette.get(color, palette["gray"]))
    d.rectangle([26, 14, 38, 36], fill=(255, 255, 255, 255))   # mic body
    d.arc([22, 24, 42, 48], 0, 180, fill=(255, 255, 255, 255), width=3)  # arc
    d.line([32, 48, 32, 54], fill=(255, 255, 255, 255), width=3)  # stand
    return img


# ---------------------------------------------------------------------------
# Audio recorder
# ---------------------------------------------------------------------------

class AudioRecorder:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.recording = False
        self._frames: list[bytes] = []
        self._stream = None
        self._pa = None
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            if self.recording:
                return
            self._frames = []
            self.recording = True
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024,
            stream_callback=self._cb,
        )
        self._stream.start_stream()

    def _cb(self, in_data, *_):
        with self._lock:
            if self.recording:
                self._frames.append(in_data)
        return (None, pyaudio.paContinue)

    def stop(self) -> bytes | None:
        """Stop and return WAV bytes, or None if recording was too short."""
        with self._lock:
            self.recording = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa:
            self._pa.terminate()
            self._pa = None
        with self._lock:
            frames, self._frames = self._frames, []

        if not frames:
            return None
        audio = np.frombuffer(b"".join(frames), dtype=np.float32)
        if len(audio) / self.sample_rate < 0.3:
            return None
        buf = io.BytesIO()
        sf.write(buf, audio, self.sample_rate, format="WAV")
        buf.seek(0)
        return buf.read()


# ---------------------------------------------------------------------------
# ASR client
# ---------------------------------------------------------------------------

class ASRClient:
    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self.client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key="dummy")

    def transcribe(self, wav_bytes: bytes) -> str:
        encoded = base64.b64encode(wav_bytes).decode()
        stream = self.client.chat.completions.create(
            model="",
            messages=[
                {"role": "system", "content": "Perform ASR."},
                {"role": "user", "content": [
                    {"type": "input_audio",
                     "input_audio": {"data": encoded, "format": "wav"}}
                ]},
            ],
            stream=True,
            max_tokens=512,
        )
        parts = []
        for chunk in stream:
            if chunk.choices[0].finish_reason == "stop":
                break
            if chunk.choices[0].delta.content:
                parts.append(chunk.choices[0].delta.content)
        return "".join(parts).strip()

    def is_alive(self) -> bool:
        """Probe the server — it has no /health, so a 400 from chat/completions means up."""
        try:
            r = httpx.post(
                f"http://{self._host}:{self._port}/v1/chat/completions",
                json={"model": "", "messages": [], "max_tokens": 1},
                timeout=3.0,
            )
            return r.status_code in (200, 400)
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Server manager
# ---------------------------------------------------------------------------

class ServerManager:
    def __init__(self, host: str, port: int, threads: int):
        self._host = host
        self._port = port
        self._threads = threads
        self.process: subprocess.Popen | None = None
        self._log_file = None

    def start(self) -> bool:
        if self.process and self.process.poll() is None:
            return True

        if not SERVER_BIN.exists():
            print(f"[ERROR] Server binary not found: {SERVER_BIN}")
            print("        See README.md — Build section.")
            return False

        missing = [f for f in (MODEL_MAIN, MODEL_MMPROJ, MODEL_VOCODER, MODEL_TOKENIZER)
                   if not f.exists()]
        if missing:
            for f in missing:
                print(f"[ERROR] Model file not found: {f}")
            print("        See README.md — Download Models section.")
            return False

        cmd = [
            str(SERVER_BIN),
            "-m",  str(MODEL_MAIN),
            "-mm", str(MODEL_MMPROJ),
            "-mv", str(MODEL_VOCODER),
            "--tts-speaker-file", str(MODEL_TOKENIZER),
            "-t", str(self._threads),
            "--host", self._host,
            "--port", str(self._port),
        ]

        # Log to file — avoids pipe-buffer deadlock during model loading
        log_path = SCRIPT_DIR / "server.log"
        self._log_file = open(log_path, "w")
        print(f"[SERVER] Starting on {self._host}:{self._port} (log → server.log)")

        self.process = subprocess.Popen(
            cmd,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )

        # Poll until the server accepts connections (up to 90 s)
        for i in range(180):
            if self.process.poll() is not None:
                self._log_file.close()
                tail = (SCRIPT_DIR / "server.log").read_text(errors="ignore")[-2000:]
                print(f"[SERVER] Crashed during startup.\n{tail}")
                return False
            try:
                r = httpx.post(
                    f"http://{self._host}:{self._port}/v1/chat/completions",
                    json={"model": "", "messages": [], "max_tokens": 1},
                    timeout=2.0,
                )
                if r.status_code in (200, 400):
                    print(f"[SERVER] Ready (PID {self.process.pid})")
                    return True
            except Exception:
                pass
            if i > 0 and i % 20 == 0:
                print(f"[SERVER] Still loading… ({i // 2}s elapsed)")
            time.sleep(0.5)

        print("[SERVER] Timed out waiting for startup.")
        self.stop()
        return False

    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None
        if self._log_file and not self._log_file.closed:
            self._log_file.close()
        print("[SERVER] Stopped.")


# ---------------------------------------------------------------------------
# Type into active field via clipboard paste
# ---------------------------------------------------------------------------

def type_text(text: str):
    if not text:
        return
    try:
        prev = pyperclip.paste()
    except Exception:
        prev = ""
    pyperclip.copy(text)
    time.sleep(0.05)
    keyboard.send("ctrl+v")
    time.sleep(0.1)
    try:
        pyperclip.copy(prev)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class LFMSpeechToType:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.server = ServerManager(cfg["host"], cfg["port"], cfg["threads"])
        self.recorder = AudioRecorder(cfg["sample_rate"])
        self.asr = ASRClient(cfg["host"], cfg["port"])
        self.hotkey: str = cfg["hotkey"]
        self.is_recording = False
        self.tray = None
        self._stop = threading.Event()

    # -- tray -----------------------------------------------------------------

    def _set_icon(self, color: str):
        if self.tray:
            self.tray.icon = _make_icon(color)

    def _run_tray(self):
        import pystray

        def on_quit(icon, _):
            self._stop.set()
            icon.stop()

        menu = pystray.Menu(
            pystray.MenuItem(f"LFM Speech-to-Type  [{self.hotkey}]", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", on_quit),
        )
        self.tray = pystray.Icon("lfm-stt", _make_icon("gray"), "LFM Speech-to-Type", menu)
        self.tray.run()

    # -- recording ------------------------------------------------------------

    def _toggle(self):
        if not self.is_recording:
            self.is_recording = True
            self._set_icon("red")
            print("[REC] Recording… (press hotkey again to stop)")
            self.recorder.start()
            return

        self.is_recording = False
        self._set_icon("yellow")
        print("[REC] Stopped. Transcribing…")

        wav = self.recorder.stop()
        if wav is None:
            print("[REC] Too short — ignored.")
            self._set_icon("green")
            return

        def _transcribe():
            try:
                text = self.asr.transcribe(wav)
                if text:
                    print(f"[ASR] → {text}")
                    type_text(text)
                else:
                    print("[ASR] Empty result.")
            except Exception as e:
                print(f"[ASR] Error: {e}")
            finally:
                self._set_icon("green")

        threading.Thread(target=_transcribe, daemon=True).start()

    # -- run ------------------------------------------------------------------

    def run(self):
        print("=" * 55)
        print("  LFM Speech-to-Type  (Liquid AI LFM2.5-Audio)")
        print("=" * 55)
        print(f"  Hotkey  : {self.hotkey}  (toggle record/transcribe)")
        print(f"  Server  : {self.cfg['host']}:{self.cfg['port']}")
        print(f"  Models  : {MODELS_DIR}")
        print(f"  Config  : {CONFIG_FILE}")
        print("=" * 55)

        if not self.server.start():
            print("\n[FATAL] Could not start audio server. See README.md.")
            sys.exit(1)

        self._set_icon("green")
        keyboard.add_hotkey(self.hotkey, self._toggle, suppress=False)

        print(f"\n[READY] Press {self.hotkey} to start recording.")
        print("[READY] Right-click tray icon to quit.\n")

        threading.Thread(target=self._run_tray, daemon=True).start()

        try:
            while not self._stop.is_set():
                self._stop.wait(timeout=1.0)
        except KeyboardInterrupt:
            pass
        finally:
            print("\n[QUIT] Shutting down…")
            keyboard.unhook_all()
            self.server.stop()
            if self.tray:
                self.tray.stop()
            print("[QUIT] Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    write_default_config()
    cfg = load_config()

    p = argparse.ArgumentParser(
        description="LFM Speech-to-Type — local WhisperFlow alternative using LFM2.5-Audio"
    )
    p.add_argument("--hotkey",  default=cfg["hotkey"],
                   help=f"Toggle hotkey (default: {cfg['hotkey']})")
    p.add_argument("--port",    type=int, default=cfg["port"],
                   help=f"Audio server port (default: {cfg['port']})")
    p.add_argument("--host",    default=cfg["host"],
                   help=f"Audio server host (default: {cfg['host']})")
    p.add_argument("--threads", type=int, default=cfg["threads"],
                   help=f"CPU threads for inference (default: {cfg['threads']})")
    p.add_argument("--no-server", action="store_true",
                   help="Skip launching the server (assume it is already running)")
    args = p.parse_args()

    cfg.update(hotkey=args.hotkey, port=args.port, host=args.host, threads=args.threads)

    app = LFMSpeechToType(cfg)

    if args.no_server:
        if not app.asr.is_alive():
            print(f"[WARN] No server detected at {args.host}:{args.port}")
        app.server = ServerManager(args.host, args.port, args.threads)  # no-op
        app._set_icon("green")
        keyboard.add_hotkey(app.hotkey, app._toggle, suppress=False)
        print(f"[READY] Press {app.hotkey} to start recording.")
        threading.Thread(target=app._run_tray, daemon=True).start()
        try:
            while not app._stop.is_set():
                app._stop.wait(timeout=1.0)
        except KeyboardInterrupt:
            pass
        finally:
            keyboard.unhook_all()
            if app.tray:
                app.tray.stop()
    else:
        app.run()


if __name__ == "__main__":
    main()
