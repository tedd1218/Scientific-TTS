#!/usr/bin/env python3
"""
IPA to Speech - speaks plain text and /IPA/ segments from any input file.

Supported line formats (can be mixed freely):
  plain text sentence
  /ɪpə/
  The drug /əˌsiːtəˈmɪnəfən/ is common.

IPA backend  : edge-tts (Microsoft Edge neural TTS, requires internet + pip install edge-tts)
Text backend : pyttsx3 (Windows SAPI, no install needed on Windows)  |  pip install pyttsx3
Audio output : Windows MCI (built-in, plays MP3, no install needed)

Quick start:
  pip install edge-tts pyttsx3
  python ipa_speaker.py input.txt --all

Usage:
  python ipa_speaker.py                             # interactive, reads sample_100_ipa.txt
  python ipa_speaker.py input.txt                   # speak any mixed file interactively
  python ipa_speaker.py input.txt --all             # speak entire file non-stop
  python ipa_speaker.py input.txt --output out.wav  # save to WAV file (requires espeak-ng)
  python ipa_speaker.py input.txt --output out.mp3  # save to MP3 file (requires espeak-ng + ffmpeg)
"""

import asyncio
import ctypes
import os
import re
import sys
import subprocess
import tempfile
import wave
from pathlib import Path

DEFAULT_FILE = Path(__file__).parent / "sample_100_ipa.txt"
IPA_PATTERN = re.compile(r"/([^/]+)/")


# ---------------------------------------------------------------------------
# Line parsing
# ---------------------------------------------------------------------------

def split_line(line: str, ipa_only: bool = False) -> list[tuple[str, bool]]:
    """Split a line into (segment, is_ipa) pairs.

    ipa_only=True: drop plain-text segments that sit immediately before an IPA
    segment (e.g. the redundant word in "acetaminophen /ˌæs.ɪ.tə.ˈmɪn.ə.fən/").
    Plain text with no adjacent IPA (sentences, suffixes) is still kept.
    """
    segments = []
    last = 0
    matches = list(IPA_PATTERN.finditer(line))

    for i, m in enumerate(matches):
        before = line[last : m.start()].strip()
        if before:
            # In ipa_only mode, drop bare words that serve only as IPA labels
            is_label = ipa_only and not any(c in before for c in " ,.:;?!")
            if not is_label:
                segments.append((before, False))
        segments.append((m.group(1).strip(), True))
        last = m.end()

    remainder = line[last:].strip()
    if remainder:
        segments.append((remainder, False))
    return segments


def load_lines(path: Path) -> list[str]:
    return [l.rstrip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Audio playback (Windows MCI — built-in, plays MP3)
# ---------------------------------------------------------------------------

def _play_mp3_windows(path: str) -> None:
    """Blocking MP3 playback using Windows MCI (no install needed)."""
    winmm = ctypes.windll.winmm
    alias = "ipa_tts"
    winmm.mciSendStringW(f'open "{path}" type mpegvideo alias {alias}', None, 0, None)
    winmm.mciSendStringW(f"play {alias} wait", None, 0, None)
    winmm.mciSendStringW(f"close {alias}", None, 0, None)


def _play_audio(path: str) -> None:
    if sys.platform == "win32":
        _play_mp3_windows(path)
    else:
        # Linux/Mac: try common MP3 players first
        for player in (
            ["mpg123", "-q", path],
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
            ["afplay", path],  # macOS
        ):
            try:
                subprocess.run(player, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except OSError:
                continue  # not installed (FileNotFoundError or PermissionError)

        # Fallback: convert MP3 → WAV via ffmpeg then play with paplay (common in WSL)
        wav = path.replace(".mp3", ".wav")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", path, wav],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            subprocess.run(["paplay", wav], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except OSError:
            pass
        finally:
            if os.path.exists(wav):
                os.unlink(wav)

        raise RuntimeError(
            "No audio player found. Install one:\n"
            "  sudo apt install mpg123        (recommended)\n"
            "  sudo apt install ffmpeg        (also enables ffplay)"
        )


# ---------------------------------------------------------------------------
# IPA speech via edge-tts (Microsoft Edge neural TTS)
# ---------------------------------------------------------------------------

async def _edge_synthesise(text: str, *, ipa: bool, tmp_path: str) -> None:
    import edge_tts  # type: ignore

    if ipa:
        escaped = text.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
        # Pass only the <phoneme> tag — edge-tts wraps it in <speak> internally.
        # Passing a full <speak> document causes it to be read as literal text.
        content = f'<phoneme alphabet="ipa" ph="{escaped}"> </phoneme>'
        communicate = edge_tts.Communicate(content, "en-US-JennyNeural")
    else:
        communicate = edge_tts.Communicate(text, "en-US-JennyNeural")

    await communicate.save(tmp_path)


def _speak_edge(text: str, *, ipa: bool) -> bool:
    """Speak text or IPA via edge-tts → MP3 → playback."""
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp = f.name
        asyncio.run(_edge_synthesise(text, ipa=ipa, tmp_path=tmp))
        _play_audio(tmp)
        return True
    except ImportError:
        return False  # edge_tts not installed
    except Exception:
        return False
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Plain-text fallback: pyttsx3
# ---------------------------------------------------------------------------

_pyttsx3_engine = None


def _speak_pyttsx3(text: str) -> bool:
    global _pyttsx3_engine
    try:
        if _pyttsx3_engine is None:
            import pyttsx3  # type: ignore
            _pyttsx3_engine = pyttsx3.init()
            _pyttsx3_engine.setProperty("rate", 140)
        _pyttsx3_engine.say(text)
        _pyttsx3_engine.runAndWait()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# IPA → espeak-ng phoneme notation converter
# ---------------------------------------------------------------------------

# Longer sequences must come before their prefixes (e.g. "iː" before "i").
_IPA_PAIRS: list[tuple[str, str]] = [
    # Long vowels / diphthongs
    ("iː", "i:"), ("uː", "u:"), ("ɑː", "A:"), ("ɔː", "O:"), ("ɜː", "3:"),
    ("eɪ", "eI"), ("aɪ", "aI"), ("ɔɪ", "OI"), ("aʊ", "aU"), ("oʊ", "oU"),
    ("əʊ", "@U"), ("ɪə", "I@"), ("eə", "E@"), ("ʊə", "U@"),
    # Affricates
    ("tʃ", "tS"), ("dʒ", "dZ"),
    # Single vowels
    ("ɪ", "I"), ("ɛ", "E"), ("æ", "{"), ("ʌ", "V"), ("ɒ", "Q"),
    ("ʊ", "U"), ("ə", "@"), ("ɑ", "A"), ("ɔ", "O"),
    # Consonants
    ("ŋ", "N"), ("ʃ", "S"), ("ʒ", "Z"), ("θ", "T"), ("ð", "D"), ("ɹ", "r"),
    # Stress markers
    ("ˈ", "'"), ("ˌ", ","),
]

# Build lookup: 2-char sequences first, then 1-char
_IPA_2 = {src: dst for src, dst in _IPA_PAIRS if len(src) == 2}
_IPA_1 = {src: dst for src, dst in _IPA_PAIRS if len(src) == 1}


def _ipa_to_espeak(ipa: str) -> str:
    """Convert IPA Unicode string to espeak-ng phoneme notation."""
    tokens = []
    i = 0
    while i < len(ipa):
        two = ipa[i: i + 2]
        if two in _IPA_2:
            tokens.append(_IPA_2[two])
            i += 2
        else:
            ch = ipa[i]
            tokens.append(_IPA_1.get(ch, ch))  # keep unmapped chars as-is
            i += 1
    return "".join(tokens)


# ---------------------------------------------------------------------------
# Audio file output
# ---------------------------------------------------------------------------

def _generate_segment_mp3(text: str, mp3_path: str, *, is_ipa: bool) -> bool:
    """Render one segment to MP3 using edge-tts (neural voice, requires internet)."""
    try:
        import edge_tts  # type: ignore
        if is_ipa:
            escaped = text.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
            content = f'<phoneme alphabet="ipa" ph="{escaped}"> </phoneme>'
        else:
            content = text
        communicate = edge_tts.Communicate(content, "en-US-JennyNeural")
        asyncio.run(communicate.save(mp3_path))
        return True
    except Exception:
        return False


def _generate_segment_wav(text: str, wav_path: str, *, is_ipa: bool) -> bool:
    """Render one segment to WAV using espeak-ng (offline, IPA-accurate)."""
    content = f"[[{_ipa_to_espeak(text)}]]" if is_ipa else text
    try:
        subprocess.run(
            ["espeak-ng", "-w", wav_path, content],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _concat_wavs(wav_files: list[str], output_path: str) -> None:
    """Concatenate WAV files into one using the built-in wave module."""
    with wave.open(output_path, "wb") as out_wav:
        for i, wav_file in enumerate(wav_files):
            with wave.open(wav_file, "rb") as in_wav:
                if i == 0:
                    out_wav.setparams(in_wav.getparams())
                out_wav.writeframes(in_wav.readframes(in_wav.getnframes()))



def render_to_file(lines: list[str], output_path: str, ipa_only: bool = False) -> None:
    """Render all lines to a WAV or MP3 file.

    Tries edge-tts first (neural voice, needs internet).
    Falls back to espeak-ng (offline, IPA-accurate).
    MP3 output requires ffmpeg for concatenation.
    WAV output uses Python's built-in wave module (no extra tools needed).
    """
    want_mp3 = output_path.endswith(".mp3")
    if not output_path.endswith((".wav", ".mp3")):
        output_path += ".wav"
        want_mp3 = False

    segments = [
        (text, is_ipa)
        for line in lines
        for text, is_ipa in split_line(line, ipa_only=ipa_only)
    ]

    wav_tmps: list[str] = []

    try:
        # Render each segment to a temporary WAV:
        #   plain text → edge-tts MP3 → convert to WAV  (neural voice)
        #                fallback: espeak-ng WAV         (offline)
        #   IPA        → espeak-ng WAV                  (accurate phonemes)
        for text, is_ipa in segments:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                seg_wav = f.name
            wav_tmps.append(seg_wav)

            if is_ipa:
                _generate_segment_wav(text, seg_wav, is_ipa=True)
            else:
                # Try edge-tts → convert MP3 to WAV
                mp3_ok = False
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    tmp_mp3 = f.name
                try:
                    if _generate_segment_mp3(text, tmp_mp3, is_ipa=False):
                        result = subprocess.run(
                            ["ffmpeg", "-y", "-i", tmp_mp3, seg_wav],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        )
                        mp3_ok = result.returncode == 0
                finally:
                    try: os.unlink(tmp_mp3)
                    except OSError: pass

                if not mp3_ok:
                    _generate_segment_wav(text, seg_wav, is_ipa=False)

        if not any(os.path.getsize(f) > 44 for f in wav_tmps):
            print("  [!] No audio generated.")
            print("      Install espeak-ng:  sudo apt install espeak-ng")
            return

        # Concatenate all WAVs, then convert to final format if needed
        if want_mp3:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                concat_wav = f.name
            try:
                _concat_wavs(wav_tmps, concat_wav)
                subprocess.run(
                    ["ffmpeg", "-y", "-i", concat_wav, output_path],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            finally:
                try: os.unlink(concat_wav)
                except OSError: pass
        else:
            _concat_wavs(wav_tmps, output_path)

        print(f"  Saved: {output_path}")

    finally:
        for tmp in wav_tmps:
            try:
                os.unlink(tmp)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# IPA speech via espeak-ng (Linux/WSL)
# ---------------------------------------------------------------------------

def _speak_espeak_ipa(ipa: str) -> bool:
    """Convert IPA to espeak phoneme notation then speak via espeak-ng."""
    phonemes = _ipa_to_espeak(ipa)
    try:
        subprocess.run(
            ["espeak-ng", f"[[{phonemes}]]"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError:
        return False


# ---------------------------------------------------------------------------
# Public speak functions
# ---------------------------------------------------------------------------

def speak_ipa(ipa: str) -> None:
    """Speak IPA using espeak-ng (accurate phoneme rendering)."""
    if _speak_espeak_ipa(ipa):
        return
    print("  [!] IPA speech requires espeak-ng:  sudo apt install espeak-ng")


def speak_text(text: str) -> None:
    """Speak plain text."""
    if _speak_edge(text, ipa=False):
        return
    if _speak_pyttsx3(text):
        return
    print("  [!] No TTS backend found.  pip install edge-tts pyttsx3")


def speak_line(line: str, ipa_only: bool = False) -> None:
    """Speak one line, routing IPA and plain-text segments to the right backend."""
    for text, is_ipa in split_line(line, ipa_only=ipa_only):
        if is_ipa:
            speak_ipa(text)
        else:
            speak_text(text)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_line(i: int, line: str, ipa_only: bool = False) -> None:
    segments = split_line(line, ipa_only=ipa_only)
    parts = []
    for text, is_ipa in segments:
        parts.append(f"[IPA: /{text}/]" if is_ipa else text)
    print(f"  [{i:>3}]  {'  '.join(parts)}")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def mode_all(lines: list[str], ipa_only: bool = False) -> None:
    print(f"Speaking all {len(lines)} lines. Ctrl+C to stop.\n")
    for i, line in enumerate(lines, 1):
        display_line(i, line, ipa_only)
        speak_line(line, ipa_only)
    print()


def mode_interactive(lines: list[str], ipa_only: bool = False) -> None:
    print("IPA to Speech  —  interactive mode")
    print("Commands: <number>  |  next/n  |  prev/p  |  all  |  quit/q\n")

    pos = 0

    def show_and_speak():
        display_line(pos + 1, lines[pos], ipa_only)
        speak_line(lines[pos], ipa_only)

    show_and_speak()

    while True:
        try:
            cmd = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if cmd in ("q", "quit", "exit"):
            break
        elif cmd in ("n", "next", ""):
            pos = min(pos + 1, len(lines) - 1)  # type: ignore[assignment]
            show_and_speak()
        elif cmd in ("p", "prev", "back"):
            pos = max(pos - 1, 0)  # type: ignore[assignment]
            show_and_speak()
        elif cmd == "all":
            mode_all(lines, ipa_only)
            pos = len(lines) - 1  # type: ignore[assignment]
        elif cmd.isdigit():
            n = int(cmd) - 1
            if 0 <= n < len(lines):
                pos = n  # type: ignore[assignment]
                show_and_speak()
            else:
                print(f"  No line #{cmd}.")
        else:
            print("  Unknown command.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]

    file_path = DEFAULT_FILE
    speak_all = False
    output_path = None

    i = 0
    while i < len(args):
        if args[i] == "--all":
            speak_all = True
        elif args[i] == "--output" and i + 1 < len(args):
            output_path = args[i + 1]
            i += 1
        else:
            file_path = Path(args[i])
        i += 1

    if not file_path.exists():
        sys.exit(f"File not found: {file_path}")

    lines = load_lines(file_path)
    if not lines:
        sys.exit("No content found in file.")

    # For the default sample file (word /ipa/ format), skip speaking the redundant
    # word label so only the IPA phonemes are voiced — avoids the robotic text-TTS
    # pronunciation being heard alongside the natural IPA pronunciation.
    ipa_only = (file_path.resolve() == DEFAULT_FILE.resolve())

    print(f"Loaded {len(lines)} lines from {file_path.name}\n")

    if output_path:
        render_to_file(lines, output_path, ipa_only)
    elif speak_all:
        mode_all(lines, ipa_only)
    else:
        mode_interactive(lines, ipa_only)


if __name__ == "__main__":
    main()
