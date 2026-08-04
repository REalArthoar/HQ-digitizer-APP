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
from markupsafe import Markup, escape
from PIL import Image, ImageOps

import anthropic

app = Flask(__name__)

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

def build_prompt(context_hint: str) -> str:
    hint_line = ""
    if context_hint.strip():
        hint_line = (
            f"\nContext from the person who uploaded this: {context_hint.strip()}\n"
            "Use this to help resolve ambiguous words, names, or language — "
            "but still transcribe only what is actually written.\n"
        )

    return f"""You are transcribing a photo of handwritten text as accurately as possible.
{hint_line}
Rules:
- Transcribe EXACTLY what is written, preserving original spelling, punctuation, and line breaks.
- Do not summarize, paraphrase, correct grammar, or "clean up" the writing.
- If a word is genuinely illegible, write [illegible] in its place rather than guessing.
- If you are unsure about a specific word but have a reasonable guess, write it followed by a question mark in brackets, like this: word[?]
- If any text is crossed out or struck through in the photo, wrap just that word or phrase in double tildes, like this: ~~word~~
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
  <p class="promise">Your photo is read once, transcribed, and deleted immediately.
  It is never stored, never logged, and never used to train any model.</p>
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
    <button type="submit">Transcribe it</button>
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

    uploadForm.addEventListener('submit', (e) => {
      if (!photoInput.files.length) {
        e.preventDefault();
        alert('Please choose or drop a photo first.');
      }
    });
  </script>
  <p class="contact-footer">Found a bug, or have an idea? <a href="mailto:{{ contact_email }}?subject=Handwriting%20app%20feedback">Email me</a> — or just copy: <strong>{{ contact_email }}</strong><br>
  Free tool, running on my own budget — <a href="{{ kofi_url }}" target="_blank" rel="noopener">Ko-fi tip jar</a> if you ever want to help out.</p>
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
    pre { white-space: pre-wrap; background: #f4f4f4; padding: 16px; border-radius: 8px; flex: 1; min-width: 300px; }
    .promise { color: #2a6f2a; font-size: 0.9em; }
  </style>
</head>
<body>
  <h1>Here's your text</h1>
  <p>Compare side by side to check the AI matched the right words to the right part of the page:</p>
  <div class="columns">
    <img src="data:{{ media_type }};base64,{{ image_b64 }}" alt="Your uploaded photo">
    <pre>{{ text }}</pre>
  </div>
  <p class="promise">This photo was never saved to disk — it's only shown here, in your own browser, for this one page.
  Once you leave or refresh this page, it's gone for good.</p>
  <a href="/">Try another page</a>
  <p style="margin-top: 20px; font-size: 0.85em; color: #999;">This is a free early test, running on my own budget for API and hosting costs. If it saved you some time, you're welcome to <a href="{{ kofi_url }}" target="_blank" rel="noopener" style="color: #3a7bd5;">chip in on Ko-fi</a> — never required, just appreciated.</p>
  <p style="font-size: 0.85em; color: #999;">Was this transcription wrong or weird somewhere? <a href="mailto:{{ contact_email }}?subject=Handwriting%20app%20-%20transcription%20issue" style="color: #3a7bd5;">Let me know</a> — or copy: <strong>{{ contact_email }}</strong></p>
</body>
</html>
"""

ERROR_PAGE = """
<!doctype html>
<html><body style="font-family: sans-serif; max-width: 480px; margin: 60px auto;">
  <h1>Something went wrong</h1>
  <p>{{ error }}</p>
  <a href="/">Go back</a>
  <p style="margin-top: 30px; font-size: 0.85em; color: #999;">If this keeps happening, <a href="mailto:{{ contact_email }}?subject=Handwriting%20app%20-%20error" style="color: #3a7bd5;">email me</a> — or copy: <strong>{{ contact_email }}</strong></p>
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


def format_transcription(raw_text: str) -> Markup:
    """Turn the AI's ~~word~~ markers into real strikethrough, safely.

    We escape the raw text first (so any stray < > & the AI transcribed
    can't break the page or inject anything), then convert our own
    ~~word~~ marker into an actual <s> tag afterwards — the marker uses
    only plain characters, so escaping doesn't touch it.
    """
    escaped = str(escape(raw_text))
    with_strikethrough = re.sub(r"~~(.+?)~~", r"<s>\1</s>", escaped)
    return Markup(with_strikethrough)


@app.errorhandler(413)
def file_too_large(_error):
    return render_template_string(
        ERROR_PAGE,
        error="That file is too large (25 MB limit). Try a different photo.",
        contact_email=CONTACT_EMAIL,
    ), 413


@app.route("/")
def index():
    return render_template_string(
        UPLOAD_FORM, contact_email=CONTACT_EMAIL, kofi_url=KOFI_URL
    )


@app.route("/transcribe", methods=["POST"])
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
                            "text": build_prompt(context_hint),
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
