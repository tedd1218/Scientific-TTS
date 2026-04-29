#!/usr/bin/env python3
"""
pipeline.py — runs the three scripts in sequence: PDF → WAV

  Step 1: pdfparser/parse_pdf.py    PDF       → plain text
  Step 2: g2p/scientific_g2p.py    plain text → IPA-annotated text
  Step 3: ipa_speaker.py            IPA text   → WAV

Usage:
  python pipeline.py paper.pdf
  python pipeline.py paper.pdf --output paper.wav
  python pipeline.py paper.pdf --output paper.wav --words 500
  python pipeline.py paper.pdf --output paper.wav --no-keep-txt
"""

import os
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# MathPix credentials
# ---------------------------------------------------------------------------

MATHPIX_APP_ID  = "scientifictts_0534cf_df0e60"
MATHPIX_APP_KEY = "d52e3e73eaa3c82c84aa6b79664519ef2e1b3080a30ce03b64b0c30c35dd7f03"

# ---------------------------------------------------------------------------

HERE = Path(__file__).parent
PYTHON = sys.executable  # same interpreter that is running this script


def run(description: str, cmd: list[str]) -> None:
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"\n[ERROR] Step failed (exit code {result.returncode}). Stopping.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PDF → speech pipeline"
    )
    parser.add_argument("pdf", help="Input PDF file")
    parser.add_argument("--output", default=None,
                        help="Output WAV file (default: <pdf_stem>.wav)")
    parser.add_argument("--words", type=int, default=None, metavar="N",
                        help="Limit to first N words (for quick testing)")
    parser.add_argument("--no-keep-txt", action="store_true",
                        help="Delete the intermediate .txt and .ipa.txt files after conversion")
    args = parser.parse_args()

    pdf_path   = Path(args.pdf).resolve()
    output_wav = Path(args.output).resolve() if args.output else pdf_path.with_suffix(".wav")
    stem       = pdf_path.stem

    # Intermediate file paths (placed next to the output WAV)
    out_dir  = output_wav.parent
    plain_txt = out_dir / f"{stem}.txt"
    ipa_txt   = out_dir / f"{stem}_ipa.txt"

    print(f"Input PDF : {pdf_path}")
    print(f"Output WAV: {output_wav}")

    # ------------------------------------------------------------------
    # Step 1: PDF → plain text
    # ------------------------------------------------------------------
    run(
        "Step 1/3 — PDF → plain text  (pdfparser/parse_pdf.py)",
        [
            PYTHON, str(HERE / "pdfparser" / "parse_pdf.py"),
            str(pdf_path),
            "--output", str(plain_txt),
            "--mathpix-id",  MATHPIX_APP_ID,
            "--mathpix-key", MATHPIX_APP_KEY,
        ],
    )

    # ------------------------------------------------------------------
    # Step 2: plain text → IPA-annotated text
    # ------------------------------------------------------------------
    run(
        "Step 2/3 — plain text → IPA-annotated text  (g2p/scientific_g2p.py)",
        [
            PYTHON, str(HERE / "g2p" / "scientific_g2p.py"),
            "--input",  str(plain_txt),
            "--output", str(ipa_txt),
        ],
    )

    # ------------------------------------------------------------------
    # Step 3: IPA-annotated text → WAV
    # ------------------------------------------------------------------
    words_args = ["--words", str(args.words)] if args.words else []
    run(
        "Step 3/3 — IPA text → WAV  (ipa_speaker.py)",
        [
            PYTHON, str(HERE / "ipa_speaker.py"),
            str(ipa_txt),
            "--output", str(output_wav),
        ] + words_args,
    )

    # ------------------------------------------------------------------
    # Clean up intermediate files unless --keep-txt
    # ------------------------------------------------------------------
    if args.no_keep_txt:
        for f in (plain_txt, ipa_txt):
            try:
                f.unlink()
            except OSError:
                pass

    print(f"\nDone.  Output: {output_wav}")


if __name__ == "__main__":
    main()
