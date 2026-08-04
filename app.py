"""
Handwriting-to-Text — Private & Auto-Deleted
----------------------------------------------
The smallest possible working version of the idea:

1. Someone uploads a photo of handwriting.
2. We send it to an AI model that reads it and hands back clean text.
3. We NEVER save the photo to disk, NEVER log it, and NEVER keep it after
   the request finishes. It only ever exists in the computer's memory for
   the few seconds it takes to process, then it's gone.

That's the whole privacy promise, actually implemented — not just written
in a policy nobody reads.

v2 changes (after first real test came back low quality):
- Switched to Opus 5 — Anthropic's most capable current model, better at
  reading messy/cursive handwriting than the model used in v1.
- Rewrote the prompt to be much more explicit about accuracy, uncertain
  words, and preserving structure instead of "cleaning it up."
"""

import base64
import io
import os
import re

from flask import Flask, render_template_string, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from markupsafe import Markup, escape
from PIL import Image, ImageOps
from werkzeug.middleware.proxy_fix import ProxyFix

import anthropic

app = Flask(__name__)

# Render (like most hosts) sits your app behind a reverse proxy, so without
# this, Flask sees the proxy's own IP for every single visitor instead of
# each person's real IP — which would make the rate limit below either
# useless (everyone shares one bucket) or wrong. This tells Flask to trust
# the standard X-Forwarded-For header the proxy sets.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1)

# Every hit to /transcribe costs real money (an Opus API call). Without a
# limit, someone impatiently spam-clicking — or a bot — could burn through
# the whole monthly budget in seconds and take the tool down for everyone
# else. This caps how many transcriptions any single visitor (identified by
# IP address) can request per minute and per day.
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    storage_uri="memory://",
    default_limits=[],
)

# Reject uploads bigger than 25 MB outright — generous enough for a real
# phone photo (which we then shrink below), while still blocking anything
# absurd.
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

# Handwriting doesn't need a 12-megapixel photo to be read accurately —
# anything beyond this on the longest edge is wasted size and wasted money.
MAX_IMAGE_DIMENSION = 2000  # pixels, longest edge

MAX_CONTEXT_LENGTH = 300  # characters — keeps the optional hint from ballooning cost

# Where feedback, bug reports, and ideas go. Shown on every page so testers
# have an easy way to reach a real person instead of just giving up silently.
CONTACT_EMAIL = "deividunas11@gmail.com"

# Optional, no-pressure way for people to help cover API/hosting costs.
# Never blocks or limits anything — purely there for whoever wants to use it.
KOFI_URL = "https://ko-fi.com/realarthoar"


def shrink_image_if_needed(image_bytes: bytes) -> tuple[bytes, str]:
    """Resize large photos down to a sensible size and re-encode as JPEG.

    Modern phone photos are often 10-20+ MB, way more detail than needed to
    read handwriting. Shrinking them means faster uploads, lower API cost,
    and it's why the size limit above can stay generous instead of
    rejecting people's real photos.

    If anything about the image can't be read (corrupt file, odd format),
    we fall back to sending the original bytes untouched rather than
    failing the whole request.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # Phone photos often store pixels sideways/upside-down and rely on an
        # invisible "please rotate this" tag to display correctly. Since we
        # re-save the image below (which drops that tag), we have to bake
        # the correct rotation into the actual pixels now, or it'll come out
        # sideways everywhere from here on.
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        width, height = image.size
        longest_edge = max(width, height)
        if longest_edge > MAX_IMAGE_DIMENSION:
            scale = MAX_IMAGE_DIMENSION / longest_edge
            image = image.resize(
                (int(width * scale), int(height * scale)), Image.LANCZOS
            )
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=85)
        return output.getvalue(), "image/jpeg"
    except Exception:  # noqa: BLE001 — any decode issue, just use the original
        return image_bytes, "image/jpeg"

# Your API key. Get one at https://console.anthropic.com — never put the
# actual key in this file when you share/deploy it. Set it as an
# environment variable instead (see README.md).
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def build_prompt(context_hint: str, previous_guess: str = "") -> str:
    hint_line = ""
    if context_hint.strip():
        hint_line = (
            f"\nContext from the person who uploaded this: {context_hint.strip()}\n"
            "Use this to help resolve ambiguous words, names, or language — "
            "but still transcribe only what is actually written.\n"
        )

    rethink_line = ""
    if previous_guess.strip():
        rethink_line = f"""
This is a close-up crop of a small piece of handwriting that was hard to read.
A previous automated attempt read it as: "{previous_guess.strip()}"
That reading may be WRONG — do not simply repeat it. Look at the image fresh,
as if you had never seen that guess. Only keep the same reading if, after
genuinely careful re-examination of the letter shapes, you are confident it's
correct. If a different reading now seems more likely, use that instead. If
you're still not sure, use word[?] rather than defaulting back to the
previous guess out of habit.
"""

    return f"""You are transcribing a photo of handwritten text as accurately as possible.
{hint_line}{rethink_line}
Rules:
- Transcribe EXACTLY what is written, preserving original spelling, punctuation, and line breaks.
- Do not summarize, paraphrase, correct grammar, or "clean up" the writing.
- If a word is genuinely illegible, write [illegible] in its place rather than guessing.
- If you are unsure about a specific word but have a reasonable guess, write it followed by a question mark in brackets, like this: word[?]
- If any text is crossed out or struck through in the photo, wrap just that word or phrase in double tildes, like this: ~~word~~. Crossed-out text is often written faster and messier than the rest — take the same care reading it as anything else, and use word[?] instead of guessing if you're not genuinely confident.
- If the handwriting contains a table, grid, or calendar-like structure with clear rows and columns, represent it as a markdown table: a header row and each data row wrapped in | pipes |, with a |---|---| separator row directly under the header. Keep the number of columns consistent with what's actually in the photo.
- Preserve the layout as closely as possible (line breaks, bullet points, indentation).
- Return ONLY the transcription. No commentary, no "Here is the transcription:", nothing else.
"""

UPLOAD_FORM = """
<!doctype html>
<html>
<head>
  <title>Handwriting to Text — Private & Auto-Deleted</title>
  <style>
    body { font-family: sans-serif; max-width: 480px; margin: 80px auto; text-align: center; }
    h1 { font-size: 1.4em; }
    .promise { color: #2a6f2a; font-size: 0.95em; margin-bottom: 30px; }
    input[type=file] { margin: 20px 0; }
    button { padding: 10px 24px; font-size: 1em; cursor: pointer; }
    .tips { text-align: left; background: #f4f4f4; border-radius: 8px; padding: 14px 18px; margin: 20px 0; font-size: 0.9em; color: #333; }
    .tips p.tips-title { font-weight: bold; margin: 0 0 8px; text-align: center; }
    .tips ul { margin: 0; padding-left: 20px; }
    .tips li { margin-bottom: 6px; }
    .notice { font-size: 0.8em; color: #777; margin-bottom: 4px; }
    .dropzone {
      border: 2px dashed #aaa;
      border-radius: 10px;
      padding: 30px 16px;
      margin: 16px 0 4px;
      cursor: pointer;
      color: #555;
      transition: background 0.15s, border-color 0.15s;
    }
    .dropzone.dragover { background: #eef6ff; border-color: #3a7bd5; color: #3a7bd5; }
    .dropzone .filename { font-weight: bold; margin-top: 8px; color: #2a6f2a; }
    .camera-btn {
      display: inline-block; margin-top: 8px; font-size: 0.85em;
      color: #3a7bd5; text-decoration: underline; cursor: pointer; background: none; border: none;
    }
    .contact-footer { margin-top: 30px; font-size: 0.85em; color: #999; }
    .contact-footer a { color: #3a7bd5; }
  </style>
</head>
<body>
  <h1>Turn your handwriting into clean text</h1>
  <p class="notice">This is an early test, not a finished product — thanks for trying it. First load (or after a few idle minutes) can take 10-30 seconds to wake up; that's normal.</p>
  <p class="promise">Your photo is read once, transcribed, and never saved on our end.
  Our API provider briefly processes it to read the handwriting, never uses it
  to train any model or shares it — and automatically deletes their own copy
  within 30 days.</p>
  <div class="tips">
    <p class="tips-title">For the best result</p>
    <ul>
      <li>Good, even lighting — avoid shadows falling across the page (near a window works well)</li>
      <li>Hold your phone flat and straight above the page, not at an angle</li>
      <li>Let the page fill most of the frame, and make sure it's in focus before you shoot</li>
      <li>Avoid direct flash on glossy or shiny paper — it causes glare that blocks text</li>
      <li>Flatten out folded or curled pages as much as you can first</li>
      <li>If a page is very large or dense, consider photographing it in two closer shots instead of one distant one</li>
      <li>Clarity matters more than megapixels — a sharp, well-lit ordinary photo beats a blurry high-res one</li>
    </ul>
  </div>
  <form method="post" enctype="multipart/form-data" action="/transcribe" id="uploadForm">
    <div class="dropzone" id="dropzone">
      <div id="dropzoneText">Drag &amp; drop a photo here, or click to browse</div>
      <div class="filename" id="filenameLabel"></div>
    </div>
    <input type="file" name="photo" id="photoInput" accept="image/*" required style="display:none;">
    <button type="button" class="camera-btn" id="cameraBtn">Or take a photo now</button>
    <input type="file" id="cameraInput" accept="image/*" capture="environment" style="display:none;">
    <br>
    <input type="text" name="context" placeholder="Optional: language, topic, or hint (e.g. 'Lithuanian names and shift times')" style="width: 90%; padding: 8px; margin: 10px 0;"><br>
    <button type="submit" id="submitBtn">Transcribe it</button>
  </form>
  <script>
    const dropzone = document.getElementById('dropzone');
    const photoInput = document.getElementById('photoInput');
    const cameraInput = document.getElementById('cameraInput');
    const cameraBtn = document.getElementById('cameraBtn');
    const filenameLabel = document.getElementById('filenameLabel');
    const dropzoneText = document.getElementById('dropzoneText');
    const uploadForm = document.getElementById('uploadForm');

    function showChosenFile(file) {
      filenameLabel.textContent = file ? file.name : '';
      dropzoneText.textContent = file ? 'Photo selected — click to change' : 'Drag & drop a photo here, or click to browse';
    }

    // Click the dropzone to open the normal file picker.
    dropzone.addEventListener('click', () => photoInput.click());

    photoInput.addEventListener('change', () => {
      if (photoInput.files.length) showChosenFile(photoInput.files[0]);
    });

    // Drag-and-drop support.
    ['dragenter', 'dragover'].forEach(evt =>
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
      })
    );
    ['dragleave', 'drop'].forEach(evt =>
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
      })
    );
    dropzone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files.length) {
        photoInput.files = files;
        showChosenFile(files[0]);
      }
    });

    // "Take a photo now" opens the camera directly (mobile), then copies
    // the result into the same field the form actually submits.
    cameraBtn.addEventListener('click', () => cameraInput.click());
    cameraInput.addEventListener('change', () => {
      if (cameraInput.files.length) {
        photoInput.files = cameraInput.files;
        showChosenFile(cameraInput.files[0]);
      }
    });

    const submitBtn = document.getElementById('submitBtn');
    uploadForm.addEventListener('submit', (e) => {
      if (!photoInput.files.length) {
        e.preventDefault();
        alert('Please choose or drop a photo first.');
        return;
      }
      // Stop double/triple-clicking from firing multiple paid API calls
      // while people wait for the (sometimes slow) first response.
      submitBtn.disabled = true;
      submitBtn.textContent = 'Reading your handwriting... (can take up to 30s)';
    });
  </script>
  <p class="contact-footer">Found a bug, or have an idea? Email: <a href="mailto:{{ contact_email }}?subject=Handwriting%20app%20feedback">{{ contact_email }}</a><br>
  Free tool, running on my own budget — if you want to help out: <a href="{{ kofi_url }}" target="_blank" rel="noopener">{{ kofi_url }}</a></p>
</body>
</html>
"""

RESULT_PAGE = """
<!doctype html>
<html>
<head><title>Your Transcription</title>
  <style>
    body { font-family: sans-serif; max-width: 1000px; margin: 40px auto; }
    .columns { display: flex; gap: 24px; flex-wrap: wrap; align-items: flex-start; }
    .columns img { max-width: 460px; max-height: 80vh; border-radius: 8px; border: 1px solid #ddd; }
    .result-content { background: #f4f4f4; padding: 16px; border-radius: 8px; flex: 1; min-width: 300px; overflow-x: auto; }
    .promise { color: #2a6f2a; font-size: 0.9em; }
    .legend { font-size: 0.8em; color: #666; margin-top: 6px; }
    .legend span { color:#c0392b; font-weight:bold; background:#fdecea; padding:0 2px; border-radius:3px; }
    .edit-box { margin-top: 24px; }
    .edit-box textarea { width: 100%; min-height: 160px; font-family: monospace; font-size: 0.95em; padding: 10px; border-radius: 8px; border: 1px solid #ccc; box-sizing: border-box; }
    .edit-box button { margin-top: 8px; padding: 6px 14px; cursor: pointer; }
    .fix-word-box { margin-top: 30px; padding: 14px 18px; background: #f9f9f9; border-radius: 8px; border: 1px solid #eee; }
    .fix-word-box h3 { margin: 0 0 8px; font-size: 1em; }
    .fix-word-box input[type=text] { width: 90%; padding: 6px; margin: 6px 0; }
  </style>
</head>
<body>
  <h1>Here's your text</h1>
  <p>Compare side by side to check the AI matched the right words to the right part of the page:</p>
  <div class="columns">
    <img src="data:{{ media_type }};base64,{{ image_b64 }}" alt="Your uploaded photo">
    <div class="result-content">{{ text }}</div>
  </div>
  <p class="legend"><span>Highlighted text</span> means the AI wasn't fully confident — worth double-checking against the photo.</p>

  <div class="edit-box">
    <p style="font-size:0.9em; color:#555; margin-bottom:4px;">Edit or copy the plain text below — no need to leave this page:</p>
    <textarea id="editableText">{{ raw_text }}</textarea><br>
    <button type="button" id="copyBtn">Copy text</button>
    <span id="copyStatus" style="font-size:0.85em; color:#2a6f2a; margin-left:8px;"></span>
  </div>

  <div class="fix-word-box">
    <h3>Got a specific word wrong?</h3>
    <p style="font-size:0.85em; color:#666; margin:0 0 6px;">Take or upload a close-up photo of just that part, tell us what we guessed, and we'll take a fresh, careful look at just that piece.</p>
    <form method="post" enctype="multipart/form-data" action="/transcribe" id="fixWordForm">
      <input type="file" name="photo" accept="image/*" required><br>
      <input type="text" name="previous_guess" placeholder="What did we get wrong? (e.g. 'Datos keistos')" required>
      <input type="text" name="context" placeholder="Optional: what it should probably say, or language hint">
      <button type="submit" id="fixWordBtn">Re-read this part</button>
    </form>
  </div>

  <p class="promise" style="margin-top:24px;">This photo was never saved to disk — it's only shown here, in your own browser, for this one page.
  Once you leave or refresh this page, it's gone for good.</p>
  <a href="/">Try another page</a>
  <p style="margin-top: 20px; font-size: 0.85em; color: #999;">This is a free early test, running on my own budget for API and hosting costs. Never required, just appreciated: <a href="{{ kofi_url }}" target="_blank" rel="noopener" style="color: #3a7bd5;">{{ kofi_url }}</a></p>
  <p style="font-size: 0.85em; color: #999;">Was this transcription wrong or weird somewhere? Email: <a href="mailto:{{ contact_email }}?subject=Handwriting%20app%20-%20transcription%20issue" style="color: #3a7bd5;">{{ contact_email }}</a></p>
  <script>
    document.getElementById('copyBtn').addEventListener('click', () => {
      const textarea = document.getElementById('editableText');
      navigator.clipboard.writeText(textarea.value).then(() => {
        const status = document.getElementById('copyStatus');
        status.textContent = 'Copied!';
        setTimeout(() => { status.textContent = ''; }, 2000);
      });
    });

    // Same double-submit protection as the main upload form.
    const fixWordForm = document.getElementById('fixWordForm');
    const fixWordBtn = document.getElementById('fixWordBtn');
    fixWordForm.addEventListener('submit', () => {
      fixWordBtn.disabled = true;
      fixWordBtn.textContent = 'Re-reading... (can take up to 30s)';
    });
  </script>
</body>
</html>
"""

ERROR_PAGE = """
<!doctype html>
<html><body style="font-family: sans-serif; max-width: 480px; margin: 60px auto;">
  <h1>Something went wrong</h1>
  <p>{{ error }}</p>
  <a href="/">Go back</a>
  <p style="margin-top: 30px; font-size: 0.85em; color: #999;">If this keeps happening, email: <a href="mailto:{{ contact_email }}?subject=Handwriting%20app%20-%20error" style="color: #3a7bd5;">{{ contact_email }}</a></p>
</body></html>
"""


def friendly_error_message(exc: Exception) -> str:
    """Translate technical/API errors into something a stranger can understand.

    We never want a random tester to see a raw API error dump — it looks
    broken even when it's actually something simple like "the monthly
    budget cap was reached."
    """
    text = str(exc).lower()
    if "credit" in text or "balance" in text:
        return (
            "This early test has hit its monthly usage budget. "
            "Please check back later, or let the person who shared this link know!"
        )
    if "rate limit" in text or "429" in text:
        return "This tool is getting a lot of use right now — please wait a minute and try again."
    return (
        "Something went wrong reading that photo. Please try again, "
        "or try a different photo if this keeps happening."
    )


UNCERTAIN_STYLE = 'color:#c0392b; font-weight:bold; background:#fdecea; padding:0 2px; border-radius:3px;'


def _inline_format(segment: str) -> str:
    """Escape a piece of text safely, then apply our own formatting markers.

    Order matters here:
    1. Highlight [illegible] and word[?] in red first, while the text is
       still plain — so the AI's uncertainty is impossible to miss.
    2. Apply ~~word~~ strikethrough last, since its regex can safely wrap
       around the <span> tags already inserted by step 1.
    """
    escaped = str(escape(segment))
    escaped = re.sub(
        r"\[illegible\]",
        f'<span style="{UNCERTAIN_STYLE}" title="The AI could not read this — check the photo">[illegible]</span>',
        escaped,
    )
    escaped = re.sub(
        r"(\S+?\[\?\])",
        rf'<span style="{UNCERTAIN_STYLE}" title="The AI was not fully sure about this — double check it">\1</span>',
        escaped,
    )
    return re.sub(r"~~(.+?)~~", r"<s>\1</s>", escaped)


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def _is_separator_row(line: str) -> bool:
    if not _is_table_row(line):
        return False
    cells = line.strip().strip("|").split("|")
    return all(re.fullmatch(r"\s*:?-+:?\s*", cell) for cell in cells)


def _parse_blocks(raw_text: str) -> list[tuple]:
    """Split the AI's raw output into a sequence of blocks.

    Each block is either:
    - ("table", header_cells, rows)  — a detected markdown table
    - ("text", lines)                — everything else, as-is

    Both format_transcription() (HTML) and clean_text_for_copy() (plain
    text) are built on top of this shared parser, so table-detection logic
    only lives in one place.
    """
    lines = raw_text.split("\n")
    blocks: list[tuple] = []
    plain_buffer: list[str] = []
    i = 0
    n = len(lines)

    def flush_plain() -> None:
        if plain_buffer:
            blocks.append(("text", list(plain_buffer)))
            plain_buffer.clear()

    while i < n:
        line = lines[i]
        if _is_table_row(line) and i + 1 < n and _is_separator_row(lines[i + 1]):
            flush_plain()
            header_cells = [c.strip() for c in line.strip().strip("|").split("|")]
            i += 2  # skip header row + separator row
            rows = []
            while i < n and _is_table_row(lines[i]):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            blocks.append(("table", header_cells, rows))
        else:
            plain_buffer.append(line)
            i += 1

    flush_plain()
    return blocks


def format_transcription(raw_text: str) -> Markup:
    """Turn the AI's transcription into readable HTML.

    Two things handled here:
    - ~~word~~ becomes real strikethrough, for crossed-out text.
    - A markdown-style table (| col | col |, with a --- separator row) gets
      converted into an actual HTML <table> instead of showing raw pipe
      characters as plain text — much closer to what a real grid or
      calendar on the original page actually looks like.
    """
    html_parts: list[str] = []

    for block in _parse_blocks(raw_text):
        if block[0] == "table":
            _, header_cells, rows = block
            table_html = ['<table style="border-collapse: collapse; margin: 10px 0;">', "<tr>"]
            table_html += [
                f'<th style="border:1px solid #ccc; padding:5px 10px; background:#f0f0f0;">{_inline_format(c)}</th>'
                for c in header_cells
            ]
            table_html.append("</tr>")
            for row in rows:
                table_html.append("<tr>")
                table_html += [
                    f'<td style="border:1px solid #ccc; padding:5px 10px;">{_inline_format(c)}</td>'
                    for c in row
                ]
                table_html.append("</tr>")
            table_html.append("</table>")
            html_parts.append("".join(table_html))
        else:
            _, lines = block
            joined = "\n".join(lines)
            html_parts.append(
                f'<div style="white-space: pre-wrap; margin: 4px 0;">{_inline_format(joined)}</div>'
            )

    return Markup("".join(html_parts))


TABLE_COLUMN_GAP = "   "  # 3 spaces of breathing room between copied table columns


def clean_text_for_copy(raw_text: str) -> str:
    """Build the plain-text version used in the editable/copyable box.

    The AI's raw output uses markdown syntax (| pipes | for tables, ~~word~~
    for crossed-out text) that's meant to be *parsed*, not read directly —
    pasted straight into Notepad or a note app it looks like broken code.

    So instead of showing that raw syntax, we convert it into something
    that actually looks clean when pasted elsewhere:
    - Table columns are padded with spaces to a fixed width per column, so
      they line up cleanly in ANY plain text editor. (Tabs looked tidy in
      the browser textarea but only jump to an editor's next fixed
      tab-stop — a longer word in one row pushes everything after it
      further right than a shorter word in the row above, which is exactly
      the jagged, hard-to-read mess this was producing elsewhere.)
    - ~~word~~ markers are stripped, since the strikethrough styling only
      makes sense visually on the result page itself.
    - [illegible] and word[?] are left as-is — those are meaningful notes
      a reader should still see even outside this page.
    """
    out_lines: list[str] = []

    for block in _parse_blocks(raw_text):
        if block[0] == "table":
            _, header_cells, rows = block
            all_rows = [header_cells] + rows
            col_count = len(header_cells)
            col_widths = [0] * col_count
            for row in all_rows:
                for idx in range(col_count):
                    cell = row[idx] if idx < len(row) else ""
                    col_widths[idx] = max(col_widths[idx], len(cell))

            for row in all_rows:
                padded_cells = []
                for idx in range(col_count):
                    cell = row[idx] if idx < len(row) else ""
                    # Don't pad the last column — no point trailing spaces
                    # after the final word on a line.
                    if idx == col_count - 1:
                        padded_cells.append(cell)
                    else:
                        padded_cells.append(cell.ljust(col_widths[idx]))
                out_lines.append(TABLE_COLUMN_GAP.join(padded_cells))
        else:
            _, lines = block
            out_lines.extend(line.replace("~~", "") for line in lines)

    return "\n".join(out_lines)


@app.errorhandler(413)
def file_too_large(_error):
    return render_template_string(
        ERROR_PAGE,
        error="That file is too large (25 MB limit). Try a different photo.",
        contact_email=CONTACT_EMAIL,
    ), 413


@app.errorhandler(429)
def rate_limited(_error):
    return render_template_string(
        ERROR_PAGE,
        error=(
            "You've sent a lot of requests in a short time, so this early "
            "test is pausing you briefly to keep it available for everyone. "
            "Please wait a minute (or try again tomorrow) and try again."
        ),
        contact_email=CONTACT_EMAIL,
    ), 429


@app.route("/")
def index():
    return render_template_string(
        UPLOAD_FORM, contact_email=CONTACT_EMAIL, kofi_url=KOFI_URL
    )


@app.route("/transcribe", methods=["POST"])
@limiter.limit("5 per minute; 30 per day")
def transcribe():
    uploaded_file = request.files.get("photo")
    if not uploaded_file or uploaded_file.filename == "":
        return render_template_string(
            ERROR_PAGE, error="No photo was uploaded.", contact_email=CONTACT_EMAIL
        )

    media_type = uploaded_file.mimetype or ""
    if not media_type.startswith("image/"):
        return render_template_string(
            ERROR_PAGE,
            error=f"That doesn't look like an image file ({media_type or 'unknown type'}). Please upload a photo.",
            contact_email=CONTACT_EMAIL,
        )

    # Read the image straight into memory. We never call uploaded_file.save(),
    # so it never touches the disk at all.
    original_bytes = uploaded_file.read()
    image_bytes, media_type = shrink_image_if_needed(original_bytes)
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    context_hint = request.form.get("context", "")[:MAX_CONTEXT_LENGTH]
    previous_guess = request.form.get("previous_guess", "")[:MAX_CONTEXT_LENGTH]

    try:
        message = client.messages.create(
            model="claude-opus-5",
            # Generous headroom: Opus spends some of this budget "thinking"
            # before it writes the actual transcription, so a dense full
            # page of handwriting could get cut off with a smaller limit.
            max_tokens=8192,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": build_prompt(context_hint, previous_guess),
                        },
                    ],
                }
            ],
        )
        # Opus sometimes returns a "thinking" block before the actual answer,
        # so we look for the first block that's actually text instead of
        # assuming it's content[0].
        text = next(
            (block.text for block in message.content if block.type == "text"),
            None,
        )
        if text is None:
            raise ValueError("The model didn't return any readable text.")
    except Exception as exc:  # noqa: BLE001 — never show a raw technical error to a stranger
        return render_template_string(
            ERROR_PAGE, error=friendly_error_message(exc), contact_email=CONTACT_EMAIL
        )

    # Rendered once, straight back into this response — never written to a
    # file, database, or log. It exists only in this one page load.
    response = render_template_string(
        RESULT_PAGE,
        text=format_transcription(text),
        raw_text=clean_text_for_copy(text),
        image_b64=image_b64,
        media_type=media_type,
        contact_email=CONTACT_EMAIL,
        kofi_url=KOFI_URL,
    )
    del original_bytes
    del image_bytes
    del image_b64
    return response


if __name__ == "__main__":
    app.run(debug=True)
