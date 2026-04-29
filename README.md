# Scientific PDF-to-Speech with G2P

End-to-end pipeline for converting scientific PDFs into speech:

1. Parse PDF content into cleaned, TTS-friendly text
2. Add IPA phoneme annotations for scientific terms using a transfer-learned G2P model
3. Synthesize speech with Amazon Polly

This repository was built for CMU ECE 18-786 (Introduction to Deep Learning) and focuses on improving pronunciation of technical/scientific vocabulary.

## Features

- PDF to text conversion using MathPix API (`pdfparser/parse_pdf.py`)
- LaTeX-aware text cleanup (equations, symbols, sectioning, citation cleanup)
- Scientific G2P model with pretraining + finetuning workflow (`g2p/scientific_g2p.py`)
- IPA-annotated intermediate text generation for downstream TTS
- Speech synthesis from mixed plain text + IPA via Amazon Polly (`ipa_speaker.py`)
- One-command end-to-end pipeline (`pipeline.py`)
- Optional Gradio web UI (`pipeline_ui.py`)
- Basic evaluation artifacts (PER evaluation flow, training curves, human eval CSV)

## Repository Structure

- `pipeline.py` - main CLI orchestrator (`PDF -> TXT -> IPA TXT -> WAV`)
- `pipeline_ui.py` - Gradio interface for running the pipeline from a browser
- `pdfparser/parse_pdf.py` - MathPix parsing + markdown/LaTeX normalization
- `g2p/scientific_g2p.py` - scientific G2P training, evaluation, and text annotation
- `ipa_speaker.py` - Amazon Polly synthesis for text and `/IPA/` segments
- `checkpoints/` - pretrained and finetuned G2P checkpoints
- `data/` - dictionaries, vocab files, and corpora used by G2P training

## Installation

### 1) Clone and create environment

```bash
git clone <repo-url>
cd 18786-DL-project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install Python dependencies

```bash
pip install \
  boto3 \
  gradio \
  torch \
  numpy \
  pandas \
  matplotlib \
  phonemizer \
  datasets \
  arxiv \
  nltk \
  jiwer \
  tqdm
```

Notes:
- `g2p/scientific_g2p.py` contains notebook-derived training/evaluation code and needs the ML stack above.
- `pipeline.py` and `pdfparser/parse_pdf.py` rely mostly on Python standard library + credentials.

### 3) Install system dependencies

`phonemizer` uses eSpeak-NG backend for IPA generation.

- macOS (Homebrew):

```bash
brew install espeak-ng
```

- Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y espeak-ng espeak-ng-data libespeak-ng-dev
```

### 4) Configure external credentials

#### MathPix (required for PDF parsing)

Set credentials as environment variables (recommended):

```bash
export MATHPIX_APP_ID="your_mathpix_app_id"
export MATHPIX_APP_KEY="your_mathpix_app_key"
```

You can also pass them directly to `parse_pdf.py` via flags.

#### AWS Polly (required for speech synthesis)

Set AWS credentials/region:

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"
```

Or configure via AWS CLI:

```bash
aws configure
```

## Usage

### 1) Run full PDF -> WAV pipeline

```bash
python pipeline.py paper.pdf
python pipeline.py paper.pdf --output paper.wav
python pipeline.py paper.pdf --output paper.wav --words 500
python pipeline.py paper.pdf --output paper.wav --no-keep-txt
```

Outputs:
- `<paper>.txt` - cleaned text from PDF parser
- `<paper>_ipa.txt` - IPA-annotated text
- `<paper>.wav` - synthesized speech

### 2) Parse PDF only

```bash
python pdfparser/parse_pdf.py paper.pdf --output result.txt
python pdfparser/parse_pdf.py paper.pdf --output result.txt --save-json intermediate.json
python pdfparser/parse_pdf.py paper.pdf --mathpix-id ID --mathpix-key KEY
```

### 3) Run G2P processing/training script

```bash
python g2p/scientific_g2p.py --input input.txt --output output_ipa.txt
```

This script also contains full pretraining/finetuning/evaluation logic and can generate:
- `checkpoints/pretrained_cmudict.pt`
- `checkpoints/finetuned_scientific.pt`
- `pretrain_curve.png`, `finetune_curve.png`
- `human_eval.csv`

### 4) Synthesize from IPA text

```bash
python ipa_speaker.py input_ipa.txt --output out.wav
python ipa_speaker.py input_ipa.txt --output out.wav --words 500
python ipa_speaker.py input_ipa.txt --all
```

Expected line formats in input text (mixed is supported):
- plain text sentence
- `/ipa/`
- `The drug /əˌsiːtəˈmɪnəfən/ is common.`

### 5) Launch web UI

```bash
python pipeline_ui.py
```

Then open the Gradio URL, upload a PDF, and run the pipeline interactively.

## How It Works

### Stage A: PDF Parsing (`pdfparser/parse_pdf.py`)

- Sends PDF to MathPix and retrieves markdown output
- Normalizes unusual Unicode artifacts from OCR/math extraction
- Converts LaTeX math/equations into spoken-friendly text
- Removes non-speech content (figures, captions, structural noise, references)
- Writes clean plain text suitable for TTS/G2P

### Stage B: Scientific G2P (`g2p/scientific_g2p.py`)

- Builds/loads a scientific pronunciation dictionary (`data/scientific_dictionary.json`)
- Builds/loads CMUDict-derived data and IPA mappings
- Defines a seq2seq Transformer G2P model
- Pretrains on general-English pronunciations, then finetunes on scientific vocabulary
- Produces IPA predictions used to annotate technical words in text

### Stage C: TTS Synthesis (`ipa_speaker.py`)

- Parses each line into plain-text and `/IPA/` segments
- Generates SSML with `<phoneme alphabet="ipa" ...>` for IPA segments
- Calls Amazon Polly neural voice to synthesize speech
- Concatenates per-line WAV chunks into final output

### End-to-End Orchestration (`pipeline.py`)

- Runs Stage A, then Stage B annotation, then Stage C synthesis
- Handles intermediate file paths and optional cleanup

## Data and Model Artifacts

- `data/scientific_dictionary.json` - scientific term -> IPA dictionary
- `data/cmudict.dict`, `data/cmudict_ipa.json` - source lexicon and IPA conversion
- `data/char_vocab.json`, `data/phoneme_vocab.json` - token vocabularies
- `checkpoints/pretrained_cmudict.pt` - pretrained G2P checkpoint
- `checkpoints/finetuned_scientific.pt` - scientific-domain finetuned checkpoint

## Troubleshooting

- **MathPix auth errors**: verify `MATHPIX_APP_ID` and `MATHPIX_APP_KEY`
- **Polly auth/region errors**: verify AWS credentials and `AWS_DEFAULT_REGION`
- **`phonemizer` backend errors**: ensure `espeak-ng` is installed and available in `PATH`
- **No audio output**: confirm AWS quotas/permissions and inspect `segment_log.txt`

## Notes

- `g2p/scientific_g2p.py` is notebook-derived and includes both research/training and CLI-style processing code in one file.
- For reproducible environments, consider creating a `requirements.txt` from the dependency list above.
