#!/usr/bin/env python3
"""
IPA to Speech - speaks plain text and /IPA/ segments from any input file.

Supported line formats (can be mixed freely):
  plain text sentence
  /ɪpə/
  The drug /əˌsiːtəˈmɪnəfən/ is common.

Backend : espeak-ng (offline, no internet required)
          Install: https://espeak-ng.org  or  choco install espeak-ng  (Windows)

Usage:
  python ipa_speaker.py                             # interactive, reads input.txt
  python ipa_speaker.py input.txt                   # speak any mixed file interactively
  python ipa_speaker.py input.txt --all             # speak entire file non-stop
  python ipa_speaker.py input.txt --output out.wav  # save to WAV file
"""

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
# espeak-ng helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# IPA → espeak-ng phoneme notation
# ---------------------------------------------------------------------------

# Longer sequences must come before their prefixes (e.g. "iː" before "i").
_IPA_PAIRS: list[tuple[str, str]] = [
    # Long vowels
    ("iː", "i:"),   # fleece
    ("uː", "u:"),   # goose
    ("ɑː", "A:"),   # palm / father
    ("ɔː", "O:"),   # thought / law
    ("ɜː", "3:"),   # nurse (r-coloured in en-us)
    ("oː", "O:"),   # importance-type /oːr/
    # American English diphthongs
    ("eɪ", "eI"),   # face
    ("aɪ", "aI"),   # price
    ("ɔɪ", "OI"),   # choice
    ("aʊ", "aU"),   # mouth
    ("oʊ", "oU"),   # goat (American)
    ("əʊ", "oU"),   # goat (British notation → American rendering)
    # American English rhotic diphthongs (replaces British I@/E@/U@)
    ("ɪə", "Ir"),   # near  → "here"
    ("eə", "Er"),   # square → "there"
    ("ʊə", "Ur"),   # cure  → "tour"
    # Affricates
    ("tʃ", "tS"),   # church
    ("dʒ", "dZ"),   # judge
    # R-coloured vowels — pairs absorb any trailing ɹ/ɾ to prevent double-r
    ("ɚɹ", "3"),    # inter-, butter + explicit ɹ
    ("ɚɾ", "3"),    # properties: ɚ + flap
    ("ɝɹ", "3:"),   # bird + explicit ɹ
    ("ɝɾ", "3:"),   # bird + flap
    ("ɚ",  "3"),    # r-coloured schwa (butter, letter, importance)
    ("ɝ",  "3:"),   # r-coloured open-mid (bird, nurse)
    # Monophthong vowels
    ("ɪ", "I"),     # kit
    ("ɛ", "E"),     # dress
    ("æ", "a"),     # trap  ('{' causes file-parse issues in espeak-ng; 'a' is close)
    ("ʌ", "V"),     # strut
    ("ʊ", "U"),     # foot
    ("ə", "@"),     # schwa
    ("ɐ", "@"),     # near-open central → schwa
    ("ɑ", "A:"),    # palm
    ("ɒ", "A:"),    # lot  (en-us cot/caught merger → A:)
    ("ɔ", "A:"),    # thought (en-us cot/caught merger → A:)
    ("ᵻ", "I"),     # near-close central reduced vowel (e.g. dˈɛnsᵻɾi)
    # Consonants
    ("ŋ", "N"),     # sing
    ("ʃ", "S"),     # ship
    ("ʒ", "Z"),     # measure
    ("θ", "T"),     # thin
    ("ð", "D"),     # this
    ("ɹ", "r"),     # red (IPA r vs ASCII r)
    ("ɾ", "d"),     # alveolar flap → brief /d/ (butter, quantitative)
    ("ɡ", "g"),     # IPA g vs ASCII g
    # Stress / syllable markers
    ("ˈ", "'"),     # primary stress
    ("ˌ", ","),     # secondary stress
    (".", "-"),     # syllable boundary
    # Length mark — consumed by vowel pairs above; drop strays
    ("ː", ""),
]

_IPA_2 = {src: dst for src, dst in _IPA_PAIRS if len(src) == 2}
_IPA_1 = {src: dst for src, dst in _IPA_PAIRS if len(src) == 1}


def _ipa_to_espeak(ipa: str) -> str:
    """Convert IPA Unicode string to espeak-ng [[...]] phoneme notation."""
    tokens = []
    i = 0
    while i < len(ipa):
        two = ipa[i: i + 2]
        if two in _IPA_2:
            tokens.append(_IPA_2[two])
            i += 2
        else:
            ch = ipa[i]
            tokens.append(_IPA_1.get(ch, ch))
            i += 1
    return "".join(tokens)


def _xml_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def _make_ssml(text: str, *, is_ipa: bool) -> str:
    if is_ipa:
        return f'<speak><phoneme alphabet="ipa" ph="{_xml_escape(text)}"> </phoneme></speak>'
    return f"<speak>{_xml_escape(text)}</speak>"


def _espeak_run(args: list[str], text: str) -> tuple[bool, str]:
    """Run espeak-ng with text via a temp file. Returns (success, stderr)."""
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False,
                                         mode="w", encoding="utf-8") as f:
            f.write(text)
            tmp = f.name
        result = subprocess.run(
            ["espeak-ng", "-v", "en-us", "-f", tmp] + args,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        err = result.stderr.decode("utf-8", errors="replace").strip()
        return result.returncode == 0, err
    except FileNotFoundError:
        return False, "espeak-ng not found"
    finally:
        if tmp:
            try: os.unlink(tmp)
            except OSError: pass


_SSML_FLAG: str | None = None  # set by check_espeak(): flag string if supported, None to use [[]] notation


def check_espeak() -> bool:
    """Run a quick sanity check and print what espeak-ng version is available."""
    try:
        r = subprocess.run(["espeak-ng", "--version"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ver = (r.stdout + r.stderr).decode("utf-8", errors="replace").strip().splitlines()[0]
        print(f"  espeak-ng: {ver}")
    except FileNotFoundError:
        print("  [!] espeak-ng not found in PATH.")
        print("      Windows: choco install espeak-ng")
        return False

    # Test plain text → WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        test_wav = f.name
    ok, err = _espeak_run(["-w", test_wav], "test voice check")
    size = os.path.getsize(test_wav) if os.path.exists(test_wav) else 0
    try: os.unlink(test_wav)
    except OSError: pass
    print(f"  plain-text test: rc={'ok' if ok else 'FAIL'}  wav_size={size}  err={err!r}")

    # Probe SSML flag: try --ssml then -m
    global _SSML_FLAG
    ssml_test = '<speak><phoneme alphabet="ipa" ph="hɛloʊ"> </phoneme></speak>'
    for flag in ("--ssml", "-m"):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            test_wav = f.name
        ok2, err2 = _espeak_run([flag, "-w", test_wav], ssml_test)
        size2 = os.path.getsize(test_wav) if os.path.exists(test_wav) else 0
        try: os.unlink(test_wav)
        except OSError: pass
        print(f"  ssml/ipa  test ({flag}): rc={'ok' if ok2 else 'FAIL'}  wav_size={size2}  err={err2!r}")
        if ok2 and size2 > 5000:
            _SSML_FLAG = flag
            print(f"  Using SSML flag: {flag}")
            break
    else:
        print("  SSML not supported — falling back to [[phoneme]] notation.")

    # Test [[phoneme]] notation
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        test_wav = f.name
    ok3, err3 = _espeak_run(["-w", test_wav], "[[h@'l@U]]")
    size3 = os.path.getsize(test_wav) if os.path.exists(test_wav) else 0
    try: os.unlink(test_wav)
    except OSError: pass
    print(f"  [[phoneme]] test: rc={'ok' if ok3 else 'FAIL'}  wav_size={size3}  err={err3!r}")

    # Test individual phonemes that are known to be tricky
    phoneme_tests = [
        ("[[a]]",        "/æ/ (a)"),
        ("[[S]]",        "/ʃ/ (cap-S)"),
        ("[['akS@n]]",   "action suffix"),
        ("[[fr'akS@n@l]]",  "fractional"),
        ("[[,Int@r'akS@n]]", "interaction"),
    ]
    for content, desc in phoneme_tests:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tw = f.name
        tok, terr = _espeak_run(["-w", tw], content)
        tsz = os.path.getsize(tw) if os.path.exists(tw) else 0
        try: os.unlink(tw)
        except OSError: pass
        print(f"  phoneme test {desc:<25} rc={'ok' if tok else 'FAIL'}  wav={tsz}b  err={terr!r}")

    # Show conversions for known-tricky IPA sequences
    for sample in ("dˈɛnsᵻɾi", "ˌɪntɚɹˈækʃən", "fɹˈækʃənəl", "kwˈɔntᵻɾɪtˌɪv"):
        print(f"  IPA '{sample}' → [[{_ipa_to_espeak(sample)}]]")
    return ok


def _ipa_content(text: str) -> tuple[str, list[str]]:
    """Return (text_to_pass, extra_flags) for an IPA segment.

    Uses SSML <phoneme> if supported, otherwise falls back to [[...]] notation.
    """
    if _SSML_FLAG:
        return _make_ssml(text, is_ipa=True), [_SSML_FLAG]
    return f"[[{_ipa_to_espeak(text)}]]", []


def _espeak_speak(text: str, *, is_ipa: bool) -> bool:
    """Speak a segment directly via espeak-ng (no file output)."""
    if is_ipa:
        content, flags = _ipa_content(text)
    else:
        content, flags = text, []
    ok, _ = _espeak_run(flags, content)
    return ok


def _espeak_wav(text: str, wav_path: str, *, is_ipa: bool) -> bool:
    """Render a segment to WAV via espeak-ng."""
    if is_ipa:
        content, flags = _ipa_content(text)
    else:
        content, flags = text, []
    ok, err = _espeak_run(flags + ["-w", wav_path], content)
    if not ok:
        print(f"\n  [espeak-ng FAIL] {err[:120]}", flush=True)
    return ok


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


def _render_one_segment(args: tuple) -> tuple[int, str]:
    """Render a single (text, is_ipa) segment to a temp WAV. Returns (index, wav_path)."""
    idx, text, is_ipa = args
    if is_ipa and not _SSML_FLAG:
        espeak_str = _ipa_to_espeak(text)
        label = f"[{idx:>4}] IPA: {text[:60]!r}  →  [[{espeak_str}]]"
    else:
        label = f"[{idx:>4}] {'IPA' if is_ipa else 'TXT'}: {text[:80]!r}"
    entry = f"[{time.strftime('%H:%M:%S')}] START {label}"

    with _log_lock:
        _in_progress[idx] = entry
        _write_log()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        seg_wav = f.name

    ok = _espeak_wav(text, seg_wav, is_ipa=is_ipa)
    size = os.path.getsize(seg_wav) if os.path.exists(seg_wav) else 0

    with _log_lock:
        _in_progress.pop(idx, None)
        _write_log()

    if is_ipa:
        status = "ok" if size > 44 else "SILENT"
        print(f"\n  [{idx:>4}] {status} {size:>7}b  {label}", flush=True)

    return idx, seg_wav


def render_to_file(lines: list[str], output_path: str, ipa_only: bool = False) -> None:
    """Render all lines to a WAV file using espeak-ng."""
    if not output_path.endswith(".wav"):
        output_path += ".wav"

    check_espeak()

    # Clear state from any previous run
    with _log_lock:
        _in_progress.clear()
        _write_log()
    print(f"  Segment log: {_segment_log_path}  (only in-progress segments shown — hangers stay visible)\n")

    segments = [
        (text, is_ipa)
        for line in lines
        for text, is_ipa in split_line(line, ipa_only=ipa_only)
    ]

    total_segments = len(segments)
    wav_tmps: list[str | None] = [None] * total_segments

    try:
        completed = 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(_render_one_segment, (idx, text, is_ipa)): idx
                for idx, (text, is_ipa) in enumerate(segments)
            }
            for future in as_completed(futures):
                try:
                    idx, wav_path = future.result(timeout=60)
                except Exception:
                    idx = futures[future]
                    wav_path = None
                wav_tmps[idx] = wav_path
                completed += 1
                print(f"\r  Rendering segments {completed}/{total_segments} ({100 * completed // total_segments}%)", end="", flush=True)
        print()

        valid = [f for f in wav_tmps if f is not None]
        if not valid or not any(os.path.getsize(f) > 44 for f in valid):
            print("  [!] No audio generated. Is espeak-ng installed?")
            print("      Windows: choco install espeak-ng")
            print("      Linux:   sudo apt install espeak-ng")
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
    if not _espeak_speak(text, is_ipa=is_ipa):
        print("  [!] espeak-ng not found.")
        print("      Windows: choco install espeak-ng")
        print("      Linux:   sudo apt install espeak-ng")


def speak_line(line: str, ipa_only: bool = False) -> None:
    for text, is_ipa in split_line(line, ipa_only=ipa_only):
        speak_segment(text, is_ipa)


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
