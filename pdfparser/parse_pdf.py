"""
Scientific TTS -- PDF Parsing Pipeline
=======================================
MathPix -> Markdown -> TTS-ready plain text

Setup
-----
    pip install requests
    export MATHPIX_APP_ID=your_id
    export MATHPIX_APP_KEY=your_key

Usage
-----
    python parse_pdf.py paper.pdf
    python parse_pdf.py paper.pdf --output result.txt
    python parse_pdf.py paper.pdf --save-json intermediate.json
    python parse_pdf.py paper.pdf --mathpix-id ID --mathpix-key KEY
"""

import re
import json
import time
import argparse
import unicodedata
from pathlib import Path


# ---------------------------------------------------------------------------
# Unicode normalization
# ---------------------------------------------------------------------------

UNICODE_FIXES = {
    '\u0484': '(',  '\u0485': ')',
    '\u0488': '[',  '\u0489': ']',
    '\u035d': '(',  '\u035e': ')',
    '\u0361': '-',
    '\u2010': '-',  '\u2011': '-',  '\u2012': '-',
    '\u2013': '-',  '\u2014': '-',  '\u2015': '-',
    '\u2018': "'",  '\u2019': "'",
    '\u201c': '"',  '\u201d': '"',
    '\u2032': "'",  '\u2033': "''",
    '\u2400': 'epsilon', '\u2401': 'delta', '\u2402': 'pi',
    '\u2403': 'Pi', '\u2404': 'sigma', '\u2405': 'theta',
    '⑀': 'epsilon', '⌸': 'Pi', 'Ј': "'",
    '͑': '(', '͒': ')', '͓': '[', '͔': ']',
    '␣': 'alpha', '␤': 'beta', '␥': 'gamma', '␦': 'delta',
    'Ͻ': '<', 'Ͼ': '>', 'Ϯ': '±', 'Ն': '≥', 'Յ': '≤',
    '͵': ',',
    '∼': '~', '≃': '≈', '⩽': '≤', '⩾': '≥',
    '∫': 'integral',
    'ˆ': '^',
}


def normalize_unicode(text: str) -> str:
    for bad, good in UNICODE_FIXES.items():
        text = text.replace(bad, good)
    text = ''.join(
        c if unicodedata.category(c) != 'Mn' else ''
        for c in unicodedata.normalize('NFD', text)
    )
    return unicodedata.normalize('NFC', text)


# ---------------------------------------------------------------------------
# LaTeX -> spoken form
# ---------------------------------------------------------------------------

LATEX_RULES: list[tuple[str, str]] = [
    # Equation tags and environments
    (r"\\tag\{[^}]*\}", ""),
    (r"\\begin\{cases\}", ""),
    (r"\\end\{cases\}", ""),
    (r"\\\\",            "; or "),
    (r"&",               ", where "),
    # Spacing
    (r"\\,", " "), (r"\\!", ""), (r"\\;", " "), (r"\\:", " "),
    (r"\\quad", " "), (r"\\qquad", " "),
    # Font/formatting commands (expand early so frac/sqrt see plain atoms)
    (r"\\mathcal\{([A-Za-z]+)\}", r"\1"),
    (r"\\mathbf\{([^}]+)\}", r"\1"),
    (r"\\mathit\{([^}]+)\}", r"\1"),
    (r"\\mathrm\{([^}]+)\}", r"\1"),
    (r"\\text\{([^}]+)\}", r"\1"),
    (r"\\rm\{([^}]+)\}", r"\1"),
    (r"\\rm\s+", ""),
    # Greek letters
    (r"\\alpha", "alpha"),   (r"\\beta", "beta"),    (r"\\gamma", "gamma"),
    (r"\\Gamma", "Gamma"),   (r"\\delta", "delta"),  (r"\\Delta", "Delta"),
    (r"\\epsilon", "epsilon"), (r"\\varepsilon", "epsilon"),
    (r"\\zeta", "zeta"),     (r"\\eta", "eta"),
    (r"\\theta", "theta"),   (r"\\Theta", "Theta"),  (r"\\vartheta", "theta"),
    (r"\\iota", "iota"),     (r"\\kappa", "kappa"),
    (r"\\lambda", "lambda"), (r"\\Lambda", "Lambda"),
    (r"\\chi", "chi"),       (r"\\Chi", "Chi"),
    (r"\\mu", "mu"),         (r"\\nu", "nu"),
    (r"\\xi", "xi"),         (r"\\Xi", "Xi"),
    (r"\\pi", "pi"),         (r"\\Pi", "Pi"),
    (r"\\rho", "rho"),       (r"\\sigma", "sigma"),   (r"\\Sigma", "Sigma"),
    (r"\\tau", "tau"),       (r"\\phi", "phi"),        (r"\\Phi", "Phi"),
    (r"\\varphi", "phi"),    (r"\\psi", "psi"),        (r"\\Psi", "Psi"),
    (r"\\omega", "omega"),   (r"\\Omega", "Omega"),
    (r"\\nabla", "nabla"),   (r"\\partial", "partial"),
    # Operators and relations
    (r"\\times", "times"),   (r"\\cdot", "dot"),       (r"\\div", "divided by"),
    (r"\\pm", "plus or minus"),                         (r"\\mp", "minus or plus"),
    (r"\\leq\b", "less than or equal to"),              (r"\\le\b", "less than or equal to"),
    (r"\\geq\b", "greater than or equal to"),           (r"\\ge\b", "greater than or equal to"),
    (r"\\neq", "not equal to"),                         (r"\\ne", "not equal to"),
    (r"\\approx", "approximately equals"),              (r"\\equiv", "is equivalent to"),
    (r"\\sim", "is similar to"),                        (r"\\simeq", "is approximately equal to"),
    (r"\\propto", "is proportional to"),
    (r"\\notin", "not in"),  (r"\\in", "in"),
    (r"\\subset", "subset of"),                         (r"\\subseteq", "subset of or equal to"),
    (r"\\supset", "superset of"),                       (r"\\supseteq", "superset of or equal to"),
    (r"\\cup\b", "union"),                              (r"\\cap\b", "intersection"),
    (r"\\emptyset", "empty set"),                       (r"\\varnothing", "empty set"),
    # Calculus and sums
    (r"\\sum_\{([^}]+)\}\^\{([^}]+)\}", r"the sum from \1 to \2 of"),
    (r"\\sum", "the sum of"),
    (r"\\prod_\{([^}]+)\}\^\{([^}]+)\}", r"the product from \1 to \2 of"),
    (r"\\prod", "the product of"),
    (r"\\int_\{([^}]+)\}\^\{([^}]+)\}", r"the integral from \1 to \2 of"),
    (r"\\int", "the integral of"),
    (r"\\iint", "the double integral of"),
    (r"\\iiint", "the triple integral of"),
    (r"\\oint", "the contour integral of"),
    (r"\\lim_\{([^}]+)\}", r"the limit as \1 of"),
    (r"\\lim", "the limit of"),
    (r"\\infty", "infinity"),
    (r"\\to", "approaches"),     (r"\\rightarrow", "approaches"),
    (r"\\leftarrow", "from"),    (r"\\Rightarrow", "implies"),
    (r"\\Leftarrow", "is implied by"), (r"\\Leftrightarrow", "if and only if"),
    # Trig / log / exp
    (r"\\log", "log"), (r"\\ln", "natural log"), (r"\\exp", "exponential"),
    (r"\\sin", "sine"), (r"\\cos", "cosine"), (r"\\tan", "tangent"),
    (r"\\cot", "cotangent"), (r"\\sec", "secant"), (r"\\csc", "cosecant"),
    (r"\\sinh", "hyperbolic sine"), (r"\\cosh", "hyperbolic cosine"),
    (r"\\tanh", "hyperbolic tangent"),
    (r"\\arcsin", "arcsine"), (r"\\arccos", "arccosine"), (r"\\arctan", "arctangent"),
    # Common functions
    (r"\\min", "minimum"), (r"\\max", "maximum"),
    (r"\\sup", "supremum"), (r"\\inf", "infimum"),
    (r"\\arg", "argument"), (r"\\det", "determinant"), (r"\\dim", "dimension"),
    (r"\\ker", "kernel"), (r"\\Im", "imaginary part"), (r"\\Re", "real part"),
    # Parentheses
    (r"\\left\(", "("), (r"\\right\)", ")"),
    (r"\\left\[", "["), (r"\\right\]", "]"),
    (r"\\left\{", "{"), (r"\\right\}", "}"),
    (r"\\left<", "<"), (r"\\right>", ">"),
    (r"\\langle", "left angle bracket"), (r"\\rangle", "right angle bracket"),
    # Sub/superscripts and fracs (after greek/formatting so atoms are plain)
    (r"([A-Za-z0-9]+)_\{([^}]+)\}\^\{([^}]+)\}", r"\1 sub \2 to the power of \3"),
    (r"([A-Za-z0-9]+)_\{([^}]+)\}",              r"\1 sub \2"),
    (r"([A-Za-z0-9])_([A-Za-z0-9]+)",            r"\1 sub \2"),
    (r"\{([^}]+)\}\^\{([^}]+)\}",                r"\1 to the power of \2"),
    (r"([A-Za-z0-9_{}]+)\^\{([^}]+)\}",          r"\1 to the power of \2"),
    (r"([A-Za-z0-9])\^([0-9]+)",                 r"\1 to the power of \2"),
    (r"([A-Za-z0-9])\^\{-([0-9]+)\}",            r"\1 to the power of negative \2"),
    (r"\\frac\{([^{}]+)\}\{([^{}]+)\}",          r"\1 over \2"),
    (r"\\dfrac\{([^{}]+)\}\{([^{}]+)\}",         r"\1 over \2"),
    (r"\\tfrac\{([^{}]+)\}\{([^{}]+)\}",         r"\1 over \2"),
    (r"\\sqrt\{([^}]+)\}",                       r"the square root of \1"),
    (r"\\sqrt\[([^]]+)\]\{([^}]+)\}",            r"the \1-th root of \2"),
    (r"\\left\\\|([^\\]+)\\right\\\|",           r"the norm of \1"),
    (r"\\left\|([^\\]+)\\right\|",               r"the absolute value of \1"),
    # Accent commands
    (r"\\hat\{([^}]+)\}", r"\1 hat"),
    (r"\\widehat\{([^}]+)\}", r"\1 hat"),
    (r"\\tilde\{([^}]+)\}", r"\1 tilde"),
    (r"\\bar\{([^}]+)\}", r"\1 bar"),
    (r"\\vec\{([^}]+)\}", r"\1 vector"),
    (r"\\dot\{([^}]+)\}", r"\1 dot"),
    (r"\\ddot\{([^}]+)\}", r"\1 double dot"),
    # Cleanup
    (r"\{\}", ""),
    (r"\{([^{}]+)\}",        r"\1"),
    (r"\\[a-zA-Z]+\*?",      ""),
    (r"\\left\.?|\\right\.?", ""),
    (r"\s{2,}",              " "),
]

LATEX_DISPLAY_RE = re.compile(r"\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]")
LATEX_INLINE_RE  = re.compile(r"\$[^$\n]+\$")


def latex_to_spoken(latex: str) -> str:
    text = latex.strip()
    text = re.sub(r"^\\\(|\\\)$", "", text)
    text = re.sub(r"^\$\$|\$\$$", "", text)
    text = re.sub(r"^\$|\$$", "", text)
    text = re.sub(r"^\\\[|\\\]$", "", text)
    text = text.replace(r"\[", "").replace(r"\]", "")
    text = text.strip()
    for _ in range(3):
        prev = text
        for pattern, repl in LATEX_RULES:
            text = re.sub(pattern, repl, text)
        if text == prev:
            break
    text = re.sub(r"\\(?=[^a-zA-Z])", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def convert_inline_latex(text: str) -> str:
    max_iterations = 10
    iteration = 0
    while r'\(' in text and iteration < max_iterations:
        text = re.sub(r'\\\((.+?)\\\)', lambda m: latex_to_spoken(m.group(0)), text)
        iteration += 1
    text = re.sub(r'(?<!\$)\$([^\$\n]+)\$(?!\$)',
                  lambda m: latex_to_spoken(m.group(0)), text)
    text = re.sub(r'\$\$(.+?)\$\$',
                  lambda m: latex_to_spoken(m.group(0)), text, flags=re.DOTALL)
    text = re.sub(r'\\\[(.+?)\\\]',
                  lambda m: latex_to_spoken(m.group(0)), text, flags=re.DOTALL)
    return text


# ---------------------------------------------------------------------------
# MathPix: PDF -> Markdown
# ---------------------------------------------------------------------------

def run_mathpix(pdf_path: Path, app_id: str, app_key: str) -> str:
    """Submit PDF to MathPix API, poll until done, return markdown text."""
    import urllib.request
    import urllib.error

    boundary = "----MathPixBoundary"
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    options = json.dumps({
        "math_inline_delimiters": ["$", "$"],
        "math_display_delimiters": ["$$", "$$"],
        "rm_spaces": True,
    }).encode()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="options_json"\r\n\r\n'
    ).encode() + options + b"\r\n" + (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{pdf_path.name}"\r\n'
        f"Content-Type: application/pdf\r\n\r\n"
    ).encode() + pdf_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        "https://api.mathpix.com/v3/pdf",
        data=body,
        headers={
            "app_id": app_id,
            "app_key": app_key,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        raise RuntimeError(f"MathPix upload failed: {e}")

    if "pdf_id" not in result:
        raise RuntimeError(f"MathPix upload error: {result}")

    pdf_id = result["pdf_id"]
    print(f"      -> MathPix PDF ID: {pdf_id}")

    # Poll for completion
    status_req = urllib.request.Request(
        f"https://api.mathpix.com/v3/pdf/{pdf_id}",
        headers={"app_id": app_id, "app_key": app_key},
    )
    for _ in range(60):
        time.sleep(5)
        try:
            with urllib.request.urlopen(status_req, timeout=30) as resp:
                status = json.loads(resp.read().decode())
        except urllib.error.URLError:
            continue
        print(f"      -> MathPix progress: {status.get('percent_done', 0):.0f}%", end="\r")
        if status.get("status") == "completed":
            print()
            break
        if status.get("status") == "error":
            raise RuntimeError(f"MathPix conversion failed: {status}")
    else:
        raise RuntimeError("MathPix conversion timed out.")

    # Download .mmd
    mmd_req = urllib.request.Request(
        f"https://api.mathpix.com/v3/pdf/{pdf_id}.mmd",
        headers={"app_id": app_id, "app_key": app_key},
    )
    with urllib.request.urlopen(mmd_req, timeout=60) as resp:
        return resp.read().decode("utf-8")


# ---------------------------------------------------------------------------
# Preprocess MathPix markdown
# ---------------------------------------------------------------------------

def preprocess_markdown(mmd_text: str) -> str:
    """Normalize MathPix markdown: convert LaTeX structure tags to markdown,
    remove figures, captions, and anything with no spoken form."""
    # Remove figures and images
    mmd_text = re.sub(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}', '', mmd_text, flags=re.DOTALL)
    mmd_text = re.sub(r'\\includegraphics[^\n]*', '', mmd_text)
    mmd_text = re.sub(r'https?://cdn\.mathpix\.com\S*', '', mmd_text)

    # Remove inline citation superscripts before they get converted to "to the power of N"
    # Math-delimited forms: ${}^{1}$, $^{1,2}$, \({}^{1}\), \(^{4-8}\)
    mmd_text = re.sub(r'\$\{\}\s*\^[\d,\s\-]+\$', '', mmd_text)
    mmd_text = re.sub(r'\$\^[\d,\s\-]+\$', '', mmd_text)
    mmd_text = re.sub(r'\\\(\{\}\s*\^\{?[\d,\s\-]+\}?\\\)', '', mmd_text)
    mmd_text = re.sub(r'\\\(\^\{?[\d,\s\-]+\}?\\\)', '', mmd_text)
    # Raw superscript attached to word or punctuation: word^4, word^{4,5}, .^4
    mmd_text = re.sub(r'(?<=[A-Za-z.,!?;])\s*\^\{?[\d,\-]+\}?', '', mmd_text)

    # Remove captions
    mmd_text = re.sub(r'\\captionsetup[^\n]*', '', mmd_text)
    mmd_text = re.sub(r'\\caption\*?\{[^}]*\}', '', mmd_text)

    # \title{...} -> # heading
    mmd_text = re.sub(
        r'\\title\{([^}]+)\}',
        lambda m: f'# {m.group(1).strip()}',
        mmd_text, flags=re.DOTALL
    )

    # Strip abstract tags, keep content
    mmd_text = re.sub(r'\\begin\{abstract\}', '', mmd_text)
    mmd_text = re.sub(r'\\end\{abstract\}',   '', mmd_text)

    # \section*{...} -> ## heading
    mmd_text = re.sub(
        r'\\section\*?\{([^}]+)\}',
        lambda m: f'\n\n## {m.group(1).strip()}\n\n',
        mmd_text
    )
    # \subsection*{...} -> ### heading
    mmd_text = re.sub(
        r'\\subsection\*?\{([^}]+)\}',
        lambda m: f'\n\n### {m.group(1).strip()}\n\n',
        mmd_text
    )

    # Standalone figure/table labels
    mmd_text = re.sub(r'^(FIG\.|Fig\.|TABLE|Table)\s+[\w\.]+\s*$', '', mmd_text, flags=re.MULTILINE)

    # Other structure-only commands
    for cmd in [r'\\maketitle', r'\\tableofcontents', r'\\newpage', r'\\clearpage',
                r'\\noindent', r'\\medskip', r'\\bigskip', r'\\smallskip',
                r'\\vspace\{[^}]*\}', r'\\hspace\{[^}]*\}', r'\\label\{[^}]*\}',
                r'\\centering', r'\\raggedright', r'\\columnbreak']:
        mmd_text = re.sub(cmd, '', mmd_text)

    mmd_text = re.sub(r'\n{3,}', '\n\n', mmd_text)
    return mmd_text


# ---------------------------------------------------------------------------
# Parse markdown into blocks
# ---------------------------------------------------------------------------

def parse_markdown(mmd_text: str, source_path: str = "") -> dict:
    """Convert markdown text into a document dict with title and blocks."""
    blocks = []
    title = ""
    page = 0
    mmd_text = normalize_unicode(mmd_text)

    for para in re.split(r"\n{2,}", mmd_text):
        para = para.strip()
        if not para:
            continue

        # Headings
        if para.startswith("#"):
            text = normalize_unicode(para.lstrip("#").strip())
            if not title:
                title = text  # first heading is the title
                continue
            # Detect reference section — tag everything after as references
            if re.match(r"^references?$", text, re.IGNORECASE):
                blocks.append({"kind": "reference", "text": text, "page": page, "spoken_form": None})
                continue
            blocks.append({"kind": "heading", "text": text, "page": page, "spoken_form": None})
            continue

        # Display equation block
        if LATEX_DISPLAY_RE.fullmatch(para.strip()):
            spoken = normalize_unicode(latex_to_spoken(para.strip()))
            blocks.append({"kind": "equation", "text": para.strip(), "page": page, "spoken_form": spoken})
            continue

        # Body paragraph (may contain inline equations)
        para = normalize_unicode(para)
        blocks.append({"kind": "body", "text": para, "page": page, "spoken_form": None})

    return {
        "source_path": source_path,
        "title": title,
        "authors": "",
        "abstract": "",
        "blocks": blocks,
    }


# ---------------------------------------------------------------------------
# Fragment merging
# ---------------------------------------------------------------------------

def is_incomplete_sentence(text: str) -> bool:
    text = text.strip()
    if len(text) < 20:
        return True
    if text and text[0].islower() and text[0] not in ['e', 'i']:
        return True
    if re.search(r'\b(and|or|the|of|to|from|in|with|where|which|that|is|are|was|were)\s*$',
                 text, re.IGNORECASE):
        return True
    if re.match(r'^(and|or|where|which|that|with|as|for|but)\s+', text, re.IGNORECASE):
        return True
    return False


def merge_fragments(blocks: list) -> list:
    merged = []
    buffer = None
    for block in blocks:
        if block['kind'] in ['reference', 'heading']:
            if buffer:
                merged.append(buffer)
                buffer = None
            merged.append(block)
            continue
        text = block['text'].strip()
        if is_incomplete_sentence(text):
            if buffer:
                buffer['text'] += ' ' + text
                buffer['spoken_form'] = (
                    (buffer.get('spoken_form') or '') + ' ' + (block.get('spoken_form') or '')
                ).strip() or None
            else:
                buffer = block.copy()
        else:
            if buffer:
                merged.append(buffer)
            buffer = block.copy()
    if buffer:
        merged.append(buffer)
    return merged


# ---------------------------------------------------------------------------
# Convert to TTS text
# ---------------------------------------------------------------------------

def is_junk_block(block: dict) -> bool:
    text = block["text"].strip()
    text_norm = normalize_unicode(text)

    if not text:
        return True
    if re.search(r'\\includegraphics|\\captionsetup|\\caption\*?\{', text):
        return True
    if re.search(r'https?://cdn\.mathpix\.com', text):
        return True
    if re.search(r'\\begin\{table\}|\\begin\{tabular\}|\\end\{table\}|\\end\{tabular\}', text):
        return True
    if re.match(r"^(FIG\.|Fig\.|TABLE|Table)\s+[\w\.]+\.?\s*$", text_norm):
        return True
    if re.match(r"^(Figure|Fig\.?)\s*\d+[\.:)]", text_norm, re.IGNORECASE):
        return True
    if re.match(r"^[\(](Color|Colour)\s+(online|print)[\)]", text_norm, re.IGNORECASE):
        return True
    if re.match(r"^[·\s]*[\(\[]?\d+[\)\]]?[·\s]*$", text_norm):
        return True
    if re.match(r"^(and|or|the|of|in|to|from|for|at|by)$", text_norm, re.IGNORECASE):
        return True
    if re.match(r"^[·\s\.,;:\-\(\)\[\]]+$", text_norm):
        return True
    if re.match(r"^(\[\d+\])+$", text_norm):
        return True
    if re.match(r"^[\*†‡§¶]+$", text_norm):
        return True
    if len(text_norm) < 3 and not text_norm.isalnum():
        return True
    if re.match(r'^intersectiontion', text_norm):
        return True
    # DOI spam
    if re.search(r'(\b\d{5,}\b\s*){8,}', text_norm):
        return True
    # Axis label / repeated garbage (very high word repetition)
    words = text_norm.split()
    if len(words) >= 8 and len(set(words)) / len(words) < 0.6:
        return True
    # Author affiliation address blocks
    has_affil = bool(re.search(r'\\\({}?\^?\{?[0-9]', text) or re.search(r'\^\{?[0-9]', text_norm))
    has_inst  = bool(re.search(r'University|Institute|Department|Laboratory|CERMICS|School of', text_norm, re.IGNORECASE))
    if has_affil and has_inst:
        return True
    return False


def convert_implicit_subscripts(text: str) -> str:
    text = re.sub(r'\bE\s+(xc|Hxc|hxc|corr|ex|int|kin|tot|pot|elec)\b', r'E sub \1', text)
    text = re.sub(r'\b([A-Z])\s+(LSD|PZ|NK|HF|GGA|DFT|SCF)\b', r'\1 sub \2', text)
    text = re.sub(r'\bmu\s+([A-Z])\b', r'mu sub \1', text)
    text = re.sub(r'\b(rho|psi|phi|chi)\s+([a-z]|[A-Z]|\d+[a-z]?)\b', r'\1 sub \2', text)
    text = re.sub(r'\bepsilon\s+(\d+[a-z]?)\b', r'epsilon sub \1', text)
    text = re.sub(r'\bI\s+N\b', r'I sub N', text)
    text = re.sub(r'\bE\s+N\b', r'E sub N', text)
    return text


def remove_citation_numbers(text: str) -> str:
    protections = {}
    counter = [0]

    def protect(match):
        ph = f"PROTECTED_{counter[0]}_"
        protections[ph] = match.group(0)
        counter[0] += 1
        return ph

    text = re.sub(r'\b\d+\.\d+\b', protect, text)
    text = re.sub(r'\b(over|sub|power of)\s+\d+', protect, text)
    text = re.sub(r'\d+\s+(over|sub)\b', protect, text)
    text = re.sub(r'\b\d{4,}\b', protect, text)
    text = re.sub(r'(\[\d+\]){2,}', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[[\d,\s]+\]', '', text)
    text = re.sub(r',\s*\d{1,3}(?=\s|$|[.,;:])', '', text)
    text = re.sub(r'\.\s+\d{1,3}\s+(?=[A-Z(])', '. ', text)
    text = re.sub(r'\s+\d{1,3}\s*\.$', '.', text, flags=re.MULTILINE)
    text = re.sub(r'\s{2,}', ' ', text)
    for ph, orig in protections.items():
        text = text.replace(ph, orig)
    return text


def fix_word_spacing(text: str) -> str:
    text = re.sub(r'\bthe([bcd fghjklmnpqrstvwxyz][a-z]{3,})', r'the \1', text)
    text = re.sub(r'\bthe\s+reby\b', 'thereby', text)
    return text


def strip_affiliation_superscripts(text: str) -> str:
    text = re.sub(r'\\\({}?\^?\{?[^)]+\\\)', '', text)
    text = re.sub(r'\^\{[0-9,\*†‡]+\}', '', text)
    return re.sub(r'\s{2,}', ' ', text).strip()


def clean_text_block(text: str) -> str:
    # Strip remaining LaTeX structure commands
    text = re.sub(r'\\(title|author|abstract|maketitle|label|ref|cite|footnote|textbf|textit|emph|underline|texttt)\{([^}]*)\}', r'\2', text)
    text = re.sub(r'\\begin\{[^}]+\}|\\end\{[^}]+\}', '', text)
    text = re.sub(r'\\includegraphics[^\s]*', '', text)
    text = re.sub(r'https?://\S+', '', text)
    # Figure/table references
    text = re.sub(r"\b(Figure|Fig\.?)\s+\d+[\(a-z\)]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(Table|Tab\.?)\s+[IVX0-9]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(Figs?\.?)\s+\d+(\s+and\s*\d+)?[\(a-z\)]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[?Eq\.\s*\(?\d+\)?\]?", "the equation", text, flags=re.IGNORECASE)
    text = re.sub(r"\[?Eqs\.\s*\(?\d+\)?\s*and\s*\(?\d+\)?\]?", "the equations", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(Sec\.|Section)\s+[IVX]+\.?[A-Z0-9\.]*", "the section", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(Ref\.|Reference)\s+\d+", "the reference", text, flags=re.IGNORECASE)
    # Remove "to the power of N" citation superscripts that survived latex conversion
    # (citation ranges like "3", "4-8", "11,26-38")
    _cite = r'\d{1,3}(?:[,\-]\d{1,3})*'
    text = re.sub(r'(?<=[.!?,;])\s*to the power of ' + _cite + r'\b', '', text)
    text = re.sub(r'\bto the power of ' + _cite + r'(?=\s+[A-Z])', '', text)
    text = re.sub(r'\bto the power of ' + _cite + r'(?=\s+(?:which|that|where|in\b|and\b|but\b|however|while|here|further|this|these|we|the|as\b|by\b|of\b|it\b|a\b|an\b|such|also|notably|indeed|although|since))', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+to the power of ' + _cite + r'\s*$', '', text, flags=re.MULTILINE)

    # Punctuation cleanup
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\s+([.,])\s+', r'\1 ', text)
    text = re.sub(r'([.,])\1+', r'\1', text)
    text = re.sub(r'^\s*[,\.]\s*', '', text)
    text = re.sub(r'\s*[,\.]\s*$', '', text)
    return text.strip()


def detect_incomplete_sentence(text: str) -> bool:
    patterns = [
        r'\b(namely|i\.e\.|e\.g\.|such as|including|like|as follows?)\s*$',
        r'\b(is|are|was|were|be|been|being)\s+(defined as|depicted in|shown in|given by|expressed as)\s*$',
        r'\b(the|a|an)\s+\w+\s+(of|in|for|to|from|by|at)\s*$',
        r'\brespectively\s*$',
    ]
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    if len(text.split()) < 5 and not re.search(r'[.!?]$', text):
        return True
    return False


def write_txt(doc: dict, txt_path: str) -> None:
    with open(txt_path, "w", encoding="utf-8") as f:
        if doc.get("title"):
            f.write(normalize_unicode(doc["title"]) + "\n")
        if doc.get("authors"):
            authors = normalize_unicode(doc["authors"])
            authors = re.sub(r'\^\{[^}]*\}', '', authors).strip()
            f.write(authors + "\n\n")

        for block in doc["blocks"]:
            if block["kind"] == "reference":
                continue
            if is_junk_block(block):
                continue

            raw = block.get("text", "")
            if block["kind"] == "equation":
                text = latex_to_spoken(raw)
            elif re.search(r'\\\({}?\^', raw):
                text = strip_affiliation_superscripts(raw)
            else:
                text = raw

            text = normalize_unicode(text)
            text = fix_word_spacing(text)
            text = convert_inline_latex(text)
            text = convert_implicit_subscripts(text)
            text = remove_citation_numbers(text)
            text = clean_text_block(text)

            if not text or text.isspace():
                continue

            is_connector = re.match(r"^(where|with)$", text.strip(), re.IGNORECASE)
            if (not is_connector and block["kind"] != "heading"
                    and detect_incomplete_sentence(text) and len(text.split()) < 8):
                continue

            if block["kind"] == "heading":
                f.write("\n" + text.upper() + "\n\n")
            else:
                f.write(text + "\n\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(
        description="Scientific TTS PDF Parser -- MathPix -> TTS-ready plain text"
    )
    parser.add_argument("pdf", help="Path to input PDF")
    parser.add_argument("--mathpix-id",  default=None,
                        help="MathPix app_id (or set MATHPIX_APP_ID env var)")
    parser.add_argument("--mathpix-key", default=None,
                        help="MathPix app_key (or set MATHPIX_APP_KEY env var)")
    parser.add_argument("--output",    default=None,
                        help="Output .txt path (default: <pdf_stem>.txt)")
    parser.add_argument("--save-json", default=None,
                        help="Also save intermediate parsed JSON to this path")
    args = parser.parse_args()

    mx_id  = args.mathpix_id  or os.environ.get("MATHPIX_APP_ID",  "")
    mx_key = args.mathpix_key or os.environ.get("MATHPIX_APP_KEY", "")
    if not mx_id or not mx_key:
        parser.error(
            "MathPix credentials required.\n"
            "Set MATHPIX_APP_ID and MATHPIX_APP_KEY env vars, "
            "or pass --mathpix-id / --mathpix-key."
        )

    pdf_path = Path(args.pdf)
    txt_out  = args.output or f"{pdf_path.stem}.txt"

    print(f"[1/3] Running MathPix on {pdf_path.name}...")
    mmd_text = run_mathpix(pdf_path, mx_id, mx_key)

    print("[2/3] Parsing markdown...")
    mmd_text = preprocess_markdown(mmd_text)
    doc = parse_markdown(mmd_text, source_path=str(pdf_path))
    print(f"      -> {len(doc['blocks'])} blocks extracted")
    print(f"      -> Title: {doc['title'] or '(none detected)'}")

    print("[3/3] Merging fragments and writing output...")
    doc["blocks"] = merge_fragments(doc["blocks"])

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"      -> JSON saved to: {args.save_json}")

    write_txt(doc, txt_out)
    print(f"\nDone. Output written to: {txt_out}")
