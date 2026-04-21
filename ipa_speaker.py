#!/usr/bin/env python3
"""
IPA to Speech - speaks plain text and /IPA/ segments from any input file.

Supported line formats (can be mixed freely):
  plain text sentence
  /ɪpə/
  The drug /əˌsiːtəˈmɪnəfən/ is common.

Backend : Amazon Polly (neural TTS, requires AWS credentials)
          Install: pip install boto3
          Credentials: set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
          or configure via: aws configure

Usage:
  python ipa_speaker.py                             # interactive, reads input.txt
  python ipa_speaker.py input.txt                   # speak any mixed file interactively
  python ipa_speaker.py input.txt --all             # speak entire file non-stop
  python ipa_speaker.py input.txt --output out.wav  # save to WAV file
"""

import csv
import os
import re
import sys
import subprocess
import tempfile
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3

# Load AWS credentials from CSV if not already set in the environment
_CSV = Path(__file__).parent / "docspeaker_accessKeys.csv"
if _CSV.exists() and not os.environ.get("AWS_ACCESS_KEY_ID"):
    with _CSV.open(encoding="utf-8-sig") as _f:
        _row = list(csv.DictReader(_f))[0]
    os.environ["AWS_ACCESS_KEY_ID"]     = _row["Access key ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = _row["Secret access key"]
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

DEFAULT_FILE = Path(__file__).parent / "input.txt"
IPA_PATTERN = re.compile(r"/([^/]+)/")

# Text segments that are pure punctuation/hyphens — skip them to avoid
# espeak reading "hyphen", "apostrophe", etc. between IPA tokens.
_SKIP_TEXT = re.compile(r"^[\s\-–—''\".,;:!?()\[\]{}]+$")


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

    for m in matches:
        before = line[last : m.start()].strip()
        if before and not _SKIP_TEXT.match(before):
            is_label = ipa_only and not any(c in before for c in " ,.:;?!")
            if not is_label:
                segments.append((before, False))
        segments.append((m.group(1).strip(), True))
        last = m.end()

    remainder = line[last:].strip()
    if remainder and not _SKIP_TEXT.match(remainder):
        segments.append((remainder, False))
    return segments


def load_lines(path: Path) -> list[str]:
    raw = [l.rstrip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    total = len(raw)
    lines = []
    for i, line in enumerate(raw, 1):
        lines.append(line)
        print(f"\r  Loading lines... {i}/{total}", end="", flush=True)
    print()
    return lines


# ---------------------------------------------------------------------------
# Amazon Polly helpers
# ---------------------------------------------------------------------------

POLLY_VOICE   = "Joanna"   # neural en-US voice; supports IPA <phoneme>
POLLY_ENGINE  = "neural"
POLLY_RATE    = "16000"    # Hz — PCM sample rate returned by Polly
POLLY_REGION  = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

_polly = None  # created on first use


def _get_polly():
    global _polly
    if _polly is None:
        _polly = boto3.client("polly", region_name=POLLY_REGION)
    return _polly


def _xml_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def _make_ssml(text: str, *, is_ipa: bool) -> str:
    if is_ipa:
        return f'<speak><phoneme alphabet="ipa" ph="{_xml_escape(text)}"> </phoneme></speak>'
    return f"<speak>{_xml_escape(text)}</speak>"


def _line_to_ssml(line: str, ipa_only: bool = False) -> str:
    """Build a single SSML string for a whole line, with IPA segments inline."""
    parts = ["<speak>"]
    for text, is_ipa in split_line(line, ipa_only=ipa_only):
        if is_ipa:
            parts.append(f'<phoneme alphabet="ipa" ph="{_xml_escape(text)}"> </phoneme>')
        else:
            parts.append(_xml_escape(text))
    parts.append("</speak>")
    return "".join(parts)


def check_polly() -> bool:
    """Verify Polly credentials and synthesize a short test phrase."""
    try:
        resp = _get_polly().synthesize_speech(
            Text="<speak>test</speak>",
            TextType="ssml",
            OutputFormat="pcm",
            SampleRate=POLLY_RATE,
            VoiceId=POLLY_VOICE,
            Engine=POLLY_ENGINE,
        )
        size = len(resp["AudioStream"].read())
        print(f"  Amazon Polly: ok  (voice={POLLY_VOICE}, engine={POLLY_ENGINE}, pcm_bytes={size})")
        return True
    except Exception as e:
        print(f"  [!] Amazon Polly error: {e}")
        print("      Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION")
        return False


def _polly_wav_ssml(ssml: str, wav_path: str) -> bool:
    """Send a pre-built SSML string to Polly and write the result as a WAV."""
    delay = 2
    while True:
        try:
            resp = _get_polly().synthesize_speech(
                Text=ssml,
                TextType="ssml",
                OutputFormat="pcm",
                SampleRate=POLLY_RATE,
                VoiceId=POLLY_VOICE,
                Engine=POLLY_ENGINE,
            )
            pcm = resp["AudioStream"].read()
            with wave.open(wav_path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(int(POLLY_RATE))
                w.writeframes(pcm)
            return True
        except Exception as e:
            if "ThrottlingException" in str(e) or "Rate exceeded" in str(e):
                print(f"\n  [Polly throttled] retrying in {delay}s...", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 30)
            else:
                print(f"\n  [Polly FAIL] {e}", flush=True)
                return False


def _polly_wav(text: str, wav_path: str, *, is_ipa: bool) -> bool:
    return _polly_wav_ssml(_make_ssml(text, is_ipa=is_ipa), wav_path)


def _polly_speak(text: str, *, is_ipa: bool) -> bool:
    """Speak a segment by synthesizing to a temp WAV and playing it."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        ok = _polly_wav(text, tmp, is_ipa=is_ipa)
        if ok:
            if sys.platform == "win32":
                subprocess.run(
                    ["powershell", "-c", f'(New-Object Media.SoundPlayer "{tmp}").PlaySync()'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            elif sys.platform == "darwin":
                subprocess.run(["afplay", tmp], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.run(["aplay", tmp], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return ok
    finally:
        try: os.unlink(tmp)
        except OSError: pass


# ---------------------------------------------------------------------------
# Audio file output
# ---------------------------------------------------------------------------


def _concat_wavs(wav_files: list[str], output_path: str) -> None:
    """Concatenate WAV files into one using the built-in wave module."""
    with wave.open(output_path, "wb") as out_wav:
        params_set = False
        for wav_file in wav_files:
            if not os.path.exists(wav_file) or os.path.getsize(wav_file) <= 44:
                continue
            try:
                with wave.open(wav_file, "rb") as in_wav:
                    if not params_set:
                        out_wav.setparams(in_wav.getparams())
                        params_set = True
                    out_wav.writeframes(in_wav.readframes(in_wav.getnframes()))
            except EOFError:
                continue


_segment_log_path: str = "segment_log.txt"
_in_progress: dict[int, str] = {}   # idx -> log line
_log_lock = threading.Lock()


def _write_log() -> None:
    """Rewrite the log file with only the currently in-progress segments."""
    with open(_segment_log_path, "w", encoding="utf-8") as f:
        for line in sorted(_in_progress.values()):
            f.write(line + "\n")


def _render_one_line(args: tuple) -> tuple[int, str]:
    """Render a full line as a single SSML Polly call. Returns (index, wav_path)."""
    idx, line, ipa_only = args
    label = f"[{idx:>4}] {line[:80]!r}"
    entry = f"[{time.strftime('%H:%M:%S')}] START {label}"

    with _log_lock:
        _in_progress[idx] = entry
        _write_log()

    ssml = _line_to_ssml(line, ipa_only=ipa_only)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        seg_wav = f.name

    _polly_wav_ssml(ssml, seg_wav)
    size = os.path.getsize(seg_wav) if os.path.exists(seg_wav) else 0

    with _log_lock:
        _in_progress.pop(idx, None)
        _write_log()

    status = "ok" if size > 44 else "SILENT"
    print(f"\n  [{idx:>4}] {status} {size:>7}b  {label}", flush=True)

    return idx, seg_wav


def render_to_file(lines: list[str], output_path: str, ipa_only: bool = False) -> None:
    """Render all lines to a WAV file using Amazon Polly."""
    if not output_path.endswith(".wav"):
        output_path += ".wav"

    check_polly()

    with _log_lock:
        _in_progress.clear()
        _write_log()
    print(f"  Segment log: {_segment_log_path}  (only in-progress segments shown — hangers stay visible)\n")

    total_lines = len(lines)
    wav_tmps: list[str | None] = [None] * total_lines

    try:
        completed = 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(_render_one_line, (idx, line, ipa_only)): idx
                for idx, line in enumerate(lines)
            }
            for future in as_completed(futures):
                try:
                    idx, wav_path = future.result(timeout=60)
                except Exception:
                    idx = futures[future]
                    wav_path = None
                wav_tmps[idx] = wav_path
                completed += 1
                print(f"\r  Rendering lines {completed}/{total_lines} ({100 * completed // total_lines}%)", end="", flush=True)
        print()

        valid = [f for f in wav_tmps if f is not None]
        if not valid or not any(os.path.getsize(f) > 44 for f in valid):
            print("  [!] No audio generated. Check AWS credentials and region.")
            return

        _concat_wavs(valid, output_path)
        print(f"  Saved: {output_path}")

    finally:
        for tmp in wav_tmps:
            if tmp is None:
                continue
            try:
                os.unlink(tmp)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Interactive / all modes
# ---------------------------------------------------------------------------

def speak_segment(text: str, is_ipa: bool) -> None:
    if not _polly_speak(text, is_ipa=is_ipa):
        print("  [!] Polly failed. Check AWS credentials and region.")


def speak_line(line: str, ipa_only: bool = False) -> None:
    ssml = _line_to_ssml(line, ipa_only=ipa_only)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        if _polly_wav_ssml(ssml, tmp):
            if sys.platform == "win32":
                subprocess.run(
                    ["powershell", "-c", f'(New-Object Media.SoundPlayer "{tmp}").PlaySync()'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            elif sys.platform == "darwin":
                subprocess.run(["afplay", tmp], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.run(["aplay", tmp], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            print("  [!] Polly failed. Check AWS credentials and region.")
    finally:
        try: os.unlink(tmp)
        except OSError: pass


def display_line(i: int, line: str, ipa_only: bool = False) -> None:
    parts = []
    for text, is_ipa in split_line(line, ipa_only=ipa_only):
        parts.append(f"[IPA: /{text}/]" if is_ipa else text)
    print(f"  [{i:>3}]  {'  '.join(parts)}")


def mode_all(lines: list[str], ipa_only: bool = False) -> None:
    total = len(lines)
    print(f"Speaking all {total} lines. Ctrl+C to stop.\n")
    for i, line in enumerate(lines, 1):
        print(f"  Progress: {i}/{total} ({100 * i // total}%)")
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
            pos = min(pos + 1, len(lines) - 1)
            show_and_speak()
        elif cmd in ("p", "prev", "back"):
            pos = max(pos - 1, 0)
            show_and_speak()
        elif cmd == "all":
            mode_all(lines, ipa_only)
            pos = len(lines) - 1
        elif cmd.isdigit():
            n = int(cmd) - 1
            if 0 <= n < len(lines):
                pos = n
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

    file_path = Path("input.txt")
    speak_all = False
    output_path = "out.wav"

    max_words = None
    i = 0
    while i < len(args):
        if args[i] == "--all":
            speak_all = True
        elif args[i] == "--output" and i + 1 < len(args):
            output_path = args[i + 1]
            i += 1
        elif args[i] == "--words" and i + 1 < len(args):
            max_words = int(args[i + 1])
            i += 1
        else:
            file_path = Path(args[i])
        i += 1

    if not file_path.exists():
        sys.exit(f"File not found: {file_path}")

    lines = load_lines(file_path)
    if not lines:
        sys.exit("No content found in file.")

    if max_words is not None:
        truncated, total = [], 0
        for line in lines:
            words = len(re.findall(r'\S+', line))
            if total + words > max_words:
                # include a partial line up to the limit
                tokens = re.findall(r'\S+', line)
                truncated.append(" ".join(tokens[:max_words - total]))
                break
            truncated.append(line)
            total += words
        lines = truncated
        print(f"  (limited to first {max_words} words: {len(lines)} lines)")

    ipa_only = (file_path.resolve() == DEFAULT_FILE.resolve()) if args else False

    print(f"Loaded {len(lines)} lines from {file_path.name}\n")

    if output_path:
        render_to_file(lines, output_path, ipa_only)
    elif speak_all:
        mode_all(lines, ipa_only)
    else:
        mode_interactive(lines, ipa_only)


if __name__ == "__main__":
    main()
