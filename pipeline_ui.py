#!/usr/bin/env python3
"""
pipeline_ui.py — Gradio web UI for pipeline.py

Install:
  pip install gradio

Run:
  python pipeline_ui.py
  -> Opens in your browser automatically
"""

import base64
import subprocess
import sys
import tempfile
from pathlib import Path

import gradio as gr

HERE = Path(__file__).parent
PYTHON = sys.executable


def run_pipeline(pdf_file, word_limit: int, keep_txt: bool):
    """Generator — yields (log_text, wav_or_None) as each line arrives."""
    if pdf_file is None:
        yield "Please upload a PDF file.", None
        return

    pdf_path = Path(pdf_file.name)
    out_dir = Path(tempfile.mkdtemp())
    wav_out = out_dir / (pdf_path.stem + ".wav")

    cmd = [
        PYTHON, str(HERE / "pipeline.py"),
        str(pdf_path),
        "--output", str(wav_out),
    ]
    if word_limit and word_limit > 0:
        cmd += ["--words", str(word_limit)]
    if not keep_txt:
        cmd.append("--no-keep-txt")

    log = f"Running pipeline on: {pdf_path.name}\n{'='*60}\n"
    yield log, None

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            log += line
            yield log, None
        proc.wait()

        if proc.returncode == 0 and wav_out.exists():
            log += f"\n✓ Done!  Output: {wav_out.name}"
            yield log, str(wav_out)
        else:
            log += f"\n✗ Pipeline failed (exit code {proc.returncode})."
            yield log, None

    except Exception as exc:
        yield log + f"\nError: {exc}", None


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="PDF to Speech") as demo:
    gr.Markdown("# PDF to Speech Pipeline")
    gr.Markdown("Upload a PDF → plain text → IPA annotation → WAV audio.")

    with gr.Row():
        with gr.Column():
            pdf_input   = gr.File(label="Input PDF", file_types=[".pdf"])
            word_limit  = gr.Number(label="Word limit (0 = full document)",
                                    value=0, precision=0, minimum=0)
            keep_txt    = gr.Checkbox(label="Keep intermediate .txt files",
                                      value=True)
            run_btn     = gr.Button("Run Pipeline", variant="primary")
            pdf_viewer  = gr.HTML(value="<p style='color:gray'>Upload a PDF above to preview it here.</p>")

        with gr.Column():
            log_out = gr.Textbox(label="Log", lines=24, interactive=False, elem_id="log_out")
            wav_out = gr.Audio(label="Output WAV", type="filepath")

    def show_pdf(f):
        if f is None:
            return "<p style='color:gray'>Upload a PDF above to preview it here.</p>"
        b64 = base64.b64encode(Path(f.name).read_bytes()).decode()
        return f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="600px" style="border:1px solid #ccc; border-radius:4px;"></iframe>'

    pdf_input.change(fn=show_pdf, inputs=[pdf_input], outputs=[pdf_viewer])

    # Auto-scroll the log textarea to the bottom whenever its value changes
    log_out.change(
        fn=None,
        js="() => { const el = document.querySelector('#log_out textarea'); if (el) el.scrollTop = el.scrollHeight; }",
    )

    run_btn.click(
        fn=run_pipeline,
        inputs=[pdf_input, word_limit, keep_txt],
        outputs=[log_out, wav_out],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=False, share=True)
