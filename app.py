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

v3 changes (October 2026 check-up):
- Upgraded to Opus 5.5 / Sonnet 5.5, the current model generation.
- iPhone HEIC photos are now read properly (via pillow-heif) instead of
  being sent to the AI mislabeled as JPEG and failing.
- If a photo truly can't be opened, the visitor gets a clear "try a JPG or
  PNG" message instead of a vague error after a wasted API call.
- Package versions pinned in requirements.txt so a surprise upstream
  release can't break a redeploy.

v4 changes (November 2026):
- Uploads no longer get lost when Render's free server is asleep: the page
  checks /health first, shows a "waking up" message, and sends the photo
  once the server is ready. Reloading /transcribe goes back to the start.
- Warm "paper" background with the content on a card, plus mobile sizing.
"""

import base64
import io
import os
import re
import time
from urllib.parse import quote

from flask import Flask, redirect, render_template_string, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from markupsafe import Markup, escape
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pillow_heif import register_heif_opener
from werkzeug.middleware.proxy_fix import ProxyFix

import anthropic

# iPhones save photos as HEIC by default. Pillow can't open those on its
# own — this one line teaches it how, so iPhone photos get shrunk and
# converted to JPEG like everything else.
register_heif_opener()

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

# Our best model, used for every request by default.
PRIMARY_MODEL = "claude-opus-5-5"
# If Opus itself is overloaded even after retries, fall back to this one.
# Different models are usually served with separate capacity, so Sonnet can
# often still take a request when Opus can't. It's a real step down in
# transcription precision from Opus, but a solid middle ground — good
# accuracy for a fraction of the cost, and far better than Haiku for
# reading messy handwriting carefully. Only ever used as a last resort.
FALLBACK_MODEL = "claude-sonnet-5-5"


class UnreadableImageError(Exception):
    """The uploaded file couldn't be opened as an image at all."""


def shrink_image_if_needed(image_bytes: bytes) -> tuple[bytes, str]:
    """Resize large photos down to a sensible size and re-encode as JPEG.

    Modern phone photos are often 10-20+ MB, way more detail than needed to
    read handwriting. Shrinking them means faster uploads, lower API cost,
    and it's why the size limit above can stay generous instead of
    rejecting people's real photos.

    If the image can't be opened at all (corrupt file, unsupported format),
    this raises UnreadableImageError. It used to send the original bytes
    anyway, labeled as JPEG — but the AI service rejects mislabeled or
    oversized files, so that only ever produced a vague error after a
    wasted request. Failing early lets us tell the person exactly what to do.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # Modern phones can shoot 40+ megapixel photos. Without this line,
        # PIL would decode that huge original at full size into memory
        # FIRST, and only shrink it afterward — on a real test photo, that
        # single decode step used more memory than the server had (512MB),
        # crashing the whole app with an out-of-memory error before the
        # photo even got as far as being read. draft() tells the JPEG
        # decoder to do its own fast downscaling *during* decoding instead,
        # so the huge full-size version never gets fully loaded into memory
        # in the first place. It only ever hands back something >= the size
        # we ask for, so the precise resize below still produces the same
        # final quality — this only touches memory use, not image quality.
        # (No effect on non-JPEG formats; harmless no-op for those.)
        image.draft("RGB", (MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))
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
    except Exception as exc:  # noqa: BLE001 — any decode issue
        # Log the real reason, so a pattern (e.g. one phone/format
        # consistently failing) is visible in Render's logs.
        app.logger.warning("Could not open uploaded image: %s", exc)
        raise UnreadableImageError(str(exc)) from exc

# Small, cheap model used only to work out which way is up. Picking the
# upright picture out of four is an easy question, so the smallest model
# is plenty — it costs a tiny fraction of a cent per photo.
ORIENT_MODEL = "claude-haiku-5-5"
ORIENT_TILE = 512  # pixels per picture in the A/B/C/D grid
ORIENT_ANGLES = (0, 90, 180, 270)  # clockwise turn shown as A, B, C, D


def _orientation_grid(image: Image.Image) -> bytes:
    """One small picture with the photo shown four ways, labeled A-D."""
    gap = 30
    grid = Image.new("RGB", (ORIENT_TILE * 2 + gap, ORIENT_TILE * 2 + gap), "white")
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default(size=44)
    for i, (angle, label) in enumerate(zip(ORIENT_ANGLES, "ABCD")):
        tile = image.rotate(-angle, expand=True)
        tile.thumbnail((ORIENT_TILE, ORIENT_TILE))
        x = (i % 2) * (ORIENT_TILE + gap)
        y = (i // 2) * (ORIENT_TILE + gap)
        grid.paste(tile, (x + (ORIENT_TILE - tile.width) // 2, y + (ORIENT_TILE - tile.height) // 2))
        draw.rectangle([x, y, x + 56, y + 56], fill="black")
        draw.text((x + 14, y + 2), label, fill="white", font=font)
    out = io.BytesIO()
    grid.save(out, format="JPEG", quality=80)
    return out.getvalue()


def auto_orient(image_bytes: bytes) -> bytes:
    """Turn the photo so the handwriting is upright, without asking the visitor.

    People photograph notebooks sideways or upside down all the time, and
    reading symbols like - | < > V correctly depends on knowing which way is
    up. Asking text recognition to "turn them in its head" proved unreliable,
    so instead we fix the photo itself before it's read.

    If anything goes wrong here (unclear answer, network error, budget), we
    just carry on with the photo as it was — this step should never be the
    reason a transcription fails.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        grid_b64 = base64.b64encode(_orientation_grid(image)).decode("utf-8")
        message = client.messages.create(
            model=ORIENT_MODEL,
            max_tokens=1024,
            timeout=30,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": grid_b64}},
                    {"type": "text", "text": (
                        "This picture shows the same photo of handwriting four times, each turned "
                        "a different way, labeled A, B, C and D. In which one is the handwriting "
                        "upright, so the words read normally from left to right and the lines go "
                        "from top to bottom? If the page is a bit tilted, pick the one closest to "
                        "upright. Answer with just the letter."
                    )},
                ],
            }],
        )
        answer = next((b.text for b in message.content if b.type == "text"), "")
        match = re.fullmatch(r"\W*([ABCD])\W*", answer.strip())
        if not match:
            app.logger.warning("Auto-orient: unclear answer %r, leaving photo as is", answer[:40])
            return image_bytes
        angle = ORIENT_ANGLES["ABCD".index(match.group(1))]
        if angle == 0:
            return image_bytes
        app.logger.info("Auto-orient: turning photo %s degrees clockwise", angle)
        out = io.BytesIO()
        image.rotate(-angle, expand=True).save(out, format="JPEG", quality=85)
        return out.getvalue()
    except Exception as exc:  # noqa: BLE001 — never let this step break a transcription
        app.logger.warning("Auto-orient skipped: %s", exc)
        return image_bytes


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
- Transcribe EVERY handwritten mark on the page, not just words: every symbol of ANY kind, with no exceptions. The examples here are only examples, not a complete list — punctuation (. , : ; ! ? ' " ( ) [ ] / \\ | _ ~), math and comparison signs (+ − = × ÷ < > ≤ ≥ ≈ % ^), currency and special characters ($ € £ & @ # *), dashes and lines, tick marks and checkmarks (✓), crosses (✗ x), arrows (→ ← ↑ ↓ ↔ ⇒), letters used as marks (V, X, O), bullets and dots (• ·), boxes and circles (☐ ☑ ○ ●), stars (★ ☆), hearts (♥), numbers, circled or underlined markers, and anything else the writer drew. Include them even when they stand alone on an otherwise empty line, sit in a margin, or form a column next to the text. These marks often carry meaning for the writer (a tracker, a rating, a status), so never skip any of them as decoration.
- Write each symbol with the character (keyboard or Unicode) that best matches its shape as drawn — e.g. a hand-drawn arrow becomes →, not "->" or nothing. Only use ✓ or ✗ when it is clearly a checkmark or a cross. Keep each mark on the same line, in the same position, as it appears on the page.
- The photo has already been turned so the writing is upright. Read every symbol exactly as it is drawn in this image — do NOT mentally rotate or flip any symbol. A mark that looks like - is "-", one that looks like | is "|", one that looks like V is "V", one that looks like < is "<", and so on.
- Go down the page ONE WRITTEN LINE AT A TIME, and give each written line its own output line, in the same order. This matters most for lists, columns, or rows of marks: never gather similar-looking marks from different lines into a single row. For example, if three lines each have a dash and then a V, the output is three lines, each like "-   V" — not "- - -" followed by "V V V".
- Follow the lines of the paper exactly. On lined (ruled) paper, every printed line from the first written line to the last becomes exactly one output line, and a printed line with nothing written on it becomes an empty output line — so the empty lines in your output match the empty lines on the page, one for one. If someone's writing is taller than one printed line, it still counts as one line. Don't add empty lines after the last written line. On unlined paper, do the same with the gaps between lines of writing: a gap the height of one line of writing becomes one empty line.
- Keep the horizontal gaps too: reproduce indentation and the empty space between words or marks on a line using spaces, sized to the gap as it looks on the page (about one space for each letter-width of empty paper). For example, a dash at the start of a line and a V a few centimetres further along becomes something like "-          V". Never squeeze marks together, and never space them out evenly if they aren't evenly spaced on the page.
- If a mark has no matching character (a small doodle or drawing), describe it briefly in square brackets, like [drawing: smiley face]. If you can see there is a mark but can't tell which symbol it is, write [mark?].
- Scan the WHOLE page, edge to edge, including margins, corners, and lines below the main text, before you finish.
- The photo may be rotated or taken at an angle. Read the writing in its natural direction, and keep the lines in the order the writer wrote them.
- Preserve the layout as closely as possible (line breaks, bullet points, indentation).
- Return ONLY the transcription. No commentary, no "Here is the transcription:", nothing else.
"""


def call_model_with_retries(model: str, media_type: str, image_b64: str, prompt: str, retries: int = 3):
    """Ask a specific model to read the image, retrying through brief 529
    "Overloaded" errors before giving up on that model.

    Shared by both the primary (Opus) and fallback (Sonnet) attempts in the
    /transcribe route below, so the retry/backoff logic only lives once.
    Raises the last OverloadedError if every attempt for this model fails.
    """
    last_overload_error: anthropic.OverloadedError | None = None
    for attempt in range(retries):
        try:
            return client.messages.create(
                model=model,
                # Generous headroom: the model spends some of this budget
                # "thinking" before it writes the actual transcription, so a
                # dense full page of handwriting could get cut off with a
                # smaller limit.
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
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
        except anthropic.OverloadedError as exc:
            last_overload_error = exc
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))  # 2s, then 4s, ...
    raise last_overload_error


# Shared look for every page: a soft warm "paper" background with the
# content sitting on a white card, instead of plain white everywhere.
PAGE_STYLE = """
    html {
      min-height: 100%;
      background-color: #efe7da;
      background-image:
        radial-gradient(rgba(120, 90, 50, 0.07) 1px, transparent 1px),
        linear-gradient(180deg, #f6f1e8 0%, #ebe1d1 100%);
      background-size: 18px 18px, 100% 100%;
    }
    body {
      margin: 0; padding: 40px 16px; box-sizing: border-box;
      font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
      color: #2b2724;
    }
    .card {
      background: #fffdf9; margin: 0 auto; padding: 28px 30px;
      border-radius: 16px; border: 1px solid #e6dccb;
      box-shadow: 0 10px 30px rgba(70, 50, 25, 0.10), 0 1px 3px rgba(70, 50, 25, 0.06);
    }
    @media (max-width: 560px) { body { padding: 16px 10px; } .card { padding: 20px 16px; } }
    .wake-status { font-size: 0.85em; color: #8a5a00; background: #fff6e0; border-radius: 8px; padding: 8px 12px; margin-top: 10px; display: none; }
"""

# Render's free plan puts the app to sleep after ~15 idle minutes. If an old
# tab is still open and someone uploads a photo while it's asleep, Render
# answers with its own "waking up" screen and the upload is lost. So before
# sending a photo, the page first checks /health. If the server isn't awake
# yet, it shows a friendly "waking up" message on OUR page, keeps checking,
# and only sends the photo once the server is really ready.
WAKE_SCRIPT = """
    async function serverIsAwake() {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 10000);
      try {
        const r = await fetch('/health?t=' + Date.now(), { cache: 'no-store', signal: controller.signal });
        if (!r.ok) return false;
        const data = await r.json();  // Render's wake-up page isn't JSON, so this fails until we're really up
        return data && data.ok === true;
      } catch (e) {
        return false;
      } finally {
        clearTimeout(timer);
      }
    }

    async function waitForServer(statusEl, maxSeconds) {
      if (await serverIsAwake()) return true;
      statusEl.style.display = 'block';
      statusEl.textContent = 'Waking up the server (it naps when nobody is using it)... this can take up to a minute. Your photo will be sent automatically, no need to reload.';
      const start = Date.now();
      while (Date.now() - start < maxSeconds * 1000) {
        await new Promise(res => setTimeout(res, 3000));
        if (await serverIsAwake()) { statusEl.style.display = 'none'; return true; }
      }
      statusEl.textContent = "The server is taking unusually long to wake up. Please wait a moment and press the button again.";
      return false;
    }

    // Hand a form over to the "wait until awake, then send" flow.
    function protectForm(form, button, busyText, statusEl, beforeSend) {
      const idleText = button.textContent;
      let sending = false;
      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (sending) return;
        if (beforeSend && !beforeSend()) return;
        sending = true;
        // Stop double/triple-clicking from firing multiple paid API calls.
        button.disabled = true;
        button.textContent = 'Checking the server...';
        if (await waitForServer(statusEl, 120)) {
          button.textContent = busyText;
          form.submit();  // the real upload, now that the server is ready
        } else {
          sending = false;
          button.disabled = false;
          button.textContent = idleText;
        }
      });
      // If the person comes back with the browser's Back button, reset it.
      window.addEventListener('pageshow', () => {
        sending = false; button.disabled = false; button.textContent = idleText;
      });
    }

    // Start waking the server as soon as the page opens, so it's usually
    // ready by the time someone has picked their photo.
    serverIsAwake();
"""

UPLOAD_FORM = """
<!doctype html>
<html>
<head>
  <title>Handwriting to Text — Private & Auto-Deleted</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    __PAGE_STYLE__
    .card { max-width: 520px; text-align: center; }
    h1 { font-size: 1.4em; }
    .promise { color: #2a6f2a; font-size: 0.95em; margin-bottom: 30px; }
    input[type=file] { margin: 20px 0; }
    button { padding: 10px 24px; font-size: 1em; cursor: pointer; }
    .tips { text-align: left; background: #f7f2ea; border-radius: 8px; padding: 14px 18px; margin: 20px 0; font-size: 0.9em; color: #333; }
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
<div class="card">
  <h1>Turn your handwriting into clean text</h1>
  <p class="notice">This is an early test, not a finished product — thanks for trying it. First load (or after a few idle minutes) can take 10-30 seconds to wake up; that's normal.</p>
  <p class="promise">Your photo is read once, transcribed, and never saved on our end.
  Our API provider briefly processes it to read the handwriting, never uses it
  to train any model or shares it — and normally deletes their own copy
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
    <div class="wake-status" id="wakeStatus"></div>
  </form>
  <script>
    __WAKE_SCRIPT__
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

    protectForm(
      uploadForm,
      document.getElementById('submitBtn'),
      'Reading your handwriting... (can take up to 30s)',
      document.getElementById('wakeStatus'),
      () => {
        if (!photoInput.files.length) {
          alert('Please choose or drop a photo first.');
          return false;
        }
        return true;
      }
    );
  </script>
  <p class="contact-footer">Found a bug, or have an idea? Email: <a href="mailto:{{ contact_email }}?subject=Handwriting%20app%20feedback">{{ contact_email }}</a><br>
  Free tool, running on my own budget — if you want to help out: <a href="{{ kofi_url }}" target="_blank" rel="noopener">{{ kofi_url }}</a></p>
</div>
</body>
</html>
"""

RESULT_PAGE = """
<!doctype html>
<html>
<head><title>Your Transcription</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    __PAGE_STYLE__
    .card { max-width: 1040px; }
    .columns { display: flex; gap: 24px; flex-wrap: wrap; align-items: flex-start; }
    .columns img { max-width: min(460px, 100%); max-height: 80vh; border-radius: 8px; border: 1px solid #ddd; }
    .result-content { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 0.92em; line-height: 1.55; background: #f7f2ea; padding: 16px; border-radius: 8px; flex: 1; min-width: 300px; overflow-x: auto; }
    .promise { color: #2a6f2a; font-size: 0.9em; }
    .legend { font-size: 0.8em; color: #666; margin-top: 6px; }
    .legend span { color:#c0392b; font-weight:bold; background:#fdecea; padding:0 2px; border-radius:3px; }
    .edit-box { margin-top: 24px; }
    .edit-box textarea { width: 100%; min-height: 160px; font-family: monospace; font-size: 0.95em; padding: 10px; border-radius: 8px; border: 1px solid #ccc; box-sizing: border-box; }
    .edit-box button { margin-top: 8px; padding: 6px 14px; cursor: pointer; }
    .fix-word-box { margin-top: 30px; padding: 14px 18px; background: #faf6ef; border-radius: 8px; border: 1px solid #ece3d4; }
    .fix-word-box h3 { margin: 0 0 8px; font-size: 1em; }
    .fix-word-box input[type=text] { width: 90%; padding: 6px; margin: 6px 0; }
  </style>
</head>
<body>
<div class="card">
  <h1>Here's your text</h1>
  <p>Compare side by side to check the right words were matched to the right part of the page:</p>
  <div class="columns">
    <img src="data:{{ media_type }};base64,{{ image_b64 }}" alt="Your uploaded photo">
    <div class="result-content">{{ text }}</div>
  </div>
  <p class="legend"><span>Highlighted text</span> means the tool wasn't fully sure — worth double-checking against the photo.</p>
  {% if used_fallback_model %}
  <p style="font-size:0.85em; color:#a06a00; background:#fff8e6; padding:8px 12px; border-radius:6px; margin-top:10px;">
    Text recognition was briefly overloaded with demand, so the tool read this one in backup mode instead.
    It may be very slightly less precise than usual — worth a quick double-check, and feel free to try again in a bit for full precision.
  </p>
  {% endif %}

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
      <div class="wake-status" id="wakeStatus"></div>
    </form>
  </div>

  <p class="promise" style="margin-top:24px;">This photo was never saved to disk — it's only shown here, in your own browser, for this one page.
  Once you leave or refresh this page, it's gone for good.</p>
  <a href="/">Try another page</a>
  <p style="margin-top: 20px; font-size: 0.85em; color: #999;">This is a free early test, running on my own budget for API and hosting costs. Never required, just appreciated: <a href="{{ kofi_url }}" target="_blank" rel="noopener" style="color: #3a7bd5;">{{ kofi_url }}</a></p>
  <p style="font-size: 0.85em; color: #999;">Was this transcription wrong or weird somewhere? Email: <a href="mailto:{{ contact_email }}?subject=Handwriting%20app%20-%20transcription%20issue" style="color: #3a7bd5;">{{ contact_email }}</a></p>
  <script>
    __WAKE_SCRIPT__
    document.getElementById('copyBtn').addEventListener('click', () => {
      const textarea = document.getElementById('editableText');
      navigator.clipboard.writeText(textarea.value).then(() => {
        const status = document.getElementById('copyStatus');
        status.textContent = 'Copied!';
        setTimeout(() => { status.textContent = ''; }, 2000);
      });
    });

    // Same wake-up check and double-submit protection as the main form —
    // someone could sit on this page long enough for the server to nap.
    protectForm(
      document.getElementById('fixWordForm'),
      document.getElementById('fixWordBtn'),
      'Re-reading... (can take up to 30s)',
      document.getElementById('wakeStatus')
    );
  </script>
</div>
</body>
</html>
"""

ERROR_PAGE = """
<!doctype html>
<html>
<head>
  <title>{{ title or "Something went wrong" }}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    __PAGE_STYLE__
    .card { max-width: 520px; }
    a { color: #3a7bd5; }
    .whose { display: inline-block; font-size: 0.85em; font-weight: 600; border-radius: 999px; padding: 4px 12px; margin: 0 0 4px; }
    .whose-server { background: #e8f1fb; color: #2b5d9a; }
    .whose-photo  { background: #fff3dc; color: #8a5a00; }
    .whose-wait   { background: #eef6ea; color: #2a6f2a; }
    .todo { background: #f7f2ea; border-radius: 8px; padding: 10px 14px; }
    .err-footer { margin-top: 28px; font-size: 0.85em; color: #8a8178; }
    .err-code { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 0.95em; color: #6b5d4d; }
  </style>
</head>
<body>
<div class="card">
  <h1>{{ title or "Something went wrong" }}</h1>
  {% if budget %}
  <p>{{ error }} <a href="mailto:{{ contact_email }}?subject=Handwriting%20app%20-%20out%20of%20budget" style="color: #3a7bd5; font-weight: bold;">{{ contact_email }}</a></p>
  {% if kofi_url %}
  <p style="background: #f7f2ea; border-radius: 8px; padding: 10px 14px;">Want it back sooner? Any donation, even a tiny one, gets it topped up faster, and I'll add more than you gave: <a href="{{ kofi_url }}" target="_blank" rel="noopener" style="font-weight: bold;">{{ kofi_url }}</a></p>
  {% endif %}
  <a href="/">Go back</a>
  {% else %}
  {% if whose %}<p class="whose whose-{{ whose }}">{{ whose_text }}</p>{% endif %}
  <p>{{ error }}</p>
  {% if todo %}<p class="todo"><strong>What you can do:</strong> {{ todo }}</p>{% endif %}
  <a href="/">Go back and try again</a>
  <p class="err-footer">Still stuck? Email <a href="mailto:{{ contact_email }}?subject={{ mail_subject }}">{{ contact_email }}</a>{% if code %} and mention the error code below{% endif %}.
  {% if code %}<br><span class="err-code">Error code: {{ code }}</span>{% endif %}</p>
  {% endif %}
</div>
</body></html>
"""

# Drop the shared style and wake-up script into each page.
UPLOAD_FORM, RESULT_PAGE, ERROR_PAGE = (
    page.replace("__PAGE_STYLE__", PAGE_STYLE).replace("__WAKE_SCRIPT__", WAKE_SCRIPT)
    for page in (UPLOAD_FORM, RESULT_PAGE, ERROR_PAGE)
)


BUDGET_ERROR_KEYWORDS = ("credit", "balance", "billing", "usage limit", "spend limit")


def is_budget_error(exc: Exception) -> bool:
    """True when the API refused because the money ran out.

    Covers both cases: prepaid credits hitting zero ("credit balance is too
    low") and a monthly spending cap set in the Anthropic Console being
    reached (worded as a usage/spend limit instead).
    """
    text = str(exc).lower()
    return any(keyword in text for keyword in BUDGET_ERROR_KEYWORDS)


BUDGET_ERROR_TITLE = "Out of budget"
BUDGET_ERROR_MESSAGE = "Please contact the admin to top it up:"


class EmptyTranscriptionError(Exception):
    """The AI answered, but with no text in it."""


# Small label at the top of each error page, so people instantly know
# whether to fix their photo, just wait, or tell the admin.
WHOSE_TEXT = {
    "server": "Your photo is fine — this problem is on our side.",
    "photo": "This is about the photo itself.",
    "wait": "Nothing is broken — it just needs a short wait.",
}

# Every error the visitor can see: code -> (whose, heading, what happened, what to do).
ERRORS = {
    "OVERLOADED": ("wait", "Very busy right now",
        "Text recognition is overloaded with demand from everyone using it at the moment. "
        "The tool tried several times, and also tried backup mode, but both were busy.",
        "Wait 30 seconds to a minute, then try again. Your photo wasn't saved, so you'll need to choose it again."),
    "BUSY": ("wait", "Too many photos at once",
        "Lots of people are using this tool at the same moment, and the tool has reached the text recognition limit "
        "on how many photos it can read per minute.",
        "Wait a minute, then try again."),
    "TIMEOUT": ("server", "That took too long",
        "Text recognition didn't answer in time. This sometimes happens with very large or very dense pages.",
        "Try again. If it keeps happening, photograph the page in two halves instead of one."),
    "CONNECTION": ("server", "Couldn't reach text recognition",
        "The tool couldn't connect to text recognition. This is usually a brief network hiccup.",
        "Try again in a minute."),
    "SERVICE_DOWN": ("server", "Text recognition had a problem",
        "Text recognition had a temporary error on its end.",
        "Try again in a minute or two."),
    "SETUP": ("server", "The tool needs fixing",
        "The tool's connection to text recognition isn't set up correctly right now. "
        "The admin has to fix this, so trying again won't help yet.",
        "Please email the admin (below) with the error code, so it gets fixed quickly."),
    "REQUEST": ("server", "The tool made a mistake",
        "Text recognition rejected the way the tool asked it to read your photo. That's a bug on our side.",
        "Try once more. If it happens again, please email the admin with the error code."),
    "PHOTO_REJECTED": ("photo", "This photo couldn't be accepted",
        "Text recognition refused this image. It may be too large, in an unusual format, or damaged.",
        "Try a JPG or PNG photo, or take a screenshot of your photo and upload that instead."),
    "NO_TEXT": ("photo", "No text came back",
        "The tool read your photo but couldn't find any text. This can happen if there's no readable "
        "writing in it, or the photo is very dark, blurry, or taken from far away.",
        "Try a closer, sharper photo in good light, with the page filling most of the frame."),
    "UNKNOWN": ("server", "Something unexpected went wrong",
        "We hit an error we didn't expect. It's most likely on our side, not a problem with your photo.",
        "Try once more. If it happens again, please email the admin with the error code so it can be fixed."),
    "NO_PHOTO": ("photo", "No photo was uploaded",
        "We didn't receive a photo with your request.",
        "Go back, choose or take a photo, then press \"Transcribe it\"."),
    "NOT_IMAGE": ("photo", "That's not an image file",
        "The file you chose doesn't look like a photo or picture.",
        "Choose a photo instead, such as a JPG, PNG, or iPhone HEIC."),
    "UNREADABLE": ("photo", "We couldn't open that photo",
        "The file may be damaged or in a format we can't read.",
        "Try again with a JPG or PNG. Taking a screenshot of the photo and uploading that usually works too."),
    "TOO_LARGE": ("photo", "That file is too large",
        "Photos can be up to 25 MB, and this one is bigger.",
        "Choose a smaller photo, or take a screenshot of it and upload that."),
    "LIMIT": ("wait", "Please slow down a little",
        "You've sent a lot of photos in a short time, so this free tool is pausing you briefly "
        "to keep it available for everyone. The limit is 5 photos per minute and 30 per day.",
        "Wait a minute and try again. If you've reached the daily limit, try again tomorrow."),
}


def classify_error(exc: Exception) -> str:
    """Pick which ERRORS entry fits a failed AI request."""
    if isinstance(exc, anthropic.OverloadedError):
        return "OVERLOADED"
    if isinstance(exc, anthropic.RateLimitError):
        return "BUSY"
    if isinstance(exc, anthropic.APITimeoutError):  # must come before APIConnectionError
        return "TIMEOUT"
    if isinstance(exc, anthropic.APIConnectionError):
        return "CONNECTION"
    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError)):
        return "SETUP"
    if isinstance(exc, anthropic.RequestTooLargeError):
        return "PHOTO_REJECTED"
    if isinstance(exc, anthropic.BadRequestError):
        return "PHOTO_REJECTED" if "image" in str(exc).lower() else "REQUEST"
    if isinstance(exc, anthropic.InternalServerError):
        return "SERVICE_DOWN"
    if isinstance(exc, EmptyTranscriptionError):
        return "NO_TEXT"
    return "UNKNOWN"


def error_page(code: str, status: int = 200):
    """Render the error page for one of the ERRORS codes.

    The error code shown on the page includes the time (UTC), so when
    someone emails it in, you can find the exact matching entry in
    Render's logs.
    """
    whose, title, what, todo = ERRORS[code]
    stamped_code = f"{code} · {time.strftime('%Y-%m-%d %H:%M', time.gmtime())} UTC"
    return render_template_string(
        ERROR_PAGE,
        title=title,
        error=what,
        todo=todo,
        whose=whose,
        whose_text=WHOSE_TEXT[whose],
        code=stamped_code,
        mail_subject=quote(f"Handwriting app error: {stamped_code}"),
        contact_email=CONTACT_EMAIL,
    ), status


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
        f'<span style="{UNCERTAIN_STYLE}" title="This part could not be read — check the photo">[illegible]</span>',
        escaped,
    )
    escaped = re.sub(
        r"\[mark\?\]",
        f'<span style="{UNCERTAIN_STYLE}" title="There is a symbol or mark here that could not be identified — check the photo">[mark?]</span>',
        escaped,
    )
    escaped = re.sub(
        r"(\S+?\[\?\])",
        rf'<span style="{UNCERTAIN_STYLE}" title="The tool was not fully sure about this — double check it">\1</span>',
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
    return error_page("TOO_LARGE", 413)


@app.errorhandler(429)
def rate_limited(_error):
    return error_page("LIMIT", 429)


@app.route("/")
def index():
    return render_template_string(
        UPLOAD_FORM, contact_email=CONTACT_EMAIL, kofi_url=KOFI_URL
    )


@app.route("/health")
def health():
    """Tiny "are you awake?" check used by the pages before uploading.
    Never calls the AI, so it costs nothing."""
    return {"ok": True}


@app.route("/transcribe", methods=["GET"])
def transcribe_reload():
    """Someone reloaded the result/loading page, or landed here after
    Render's wake-up screen. There's no photo to read, so just send them
    back to the start instead of showing a "Method Not Allowed" error."""
    return redirect("/")


@app.route("/transcribe", methods=["POST"])
@limiter.limit("5 per minute; 30 per day")
def transcribe():
    uploaded_file = request.files.get("photo")
    if not uploaded_file or uploaded_file.filename == "":
        return error_page("NO_PHOTO")

    media_type = uploaded_file.mimetype or ""
    if not media_type.startswith("image/"):
        return error_page("NOT_IMAGE")

    # Read the image straight into memory. We never call uploaded_file.save(),
    # so it never touches the disk at all.
    original_bytes = uploaded_file.read()
    try:
        image_bytes, media_type = shrink_image_if_needed(original_bytes)
    except UnreadableImageError:
        return error_page("UNREADABLE")
    # Photos of notebooks are often taken sideways or upside down. Turn the
    # photo so the writing is upright before it gets read — no buttons for
    # the visitor to press.
    image_bytes = auto_orient(image_bytes)
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    context_hint = request.form.get("context", "")[:MAX_CONTEXT_LENGTH]
    previous_guess = request.form.get("previous_guess", "")[:MAX_CONTEXT_LENGTH]

    prompt = build_prompt(context_hint, previous_guess)
    used_fallback_model = False

    try:
        try:
            message = call_model_with_retries(PRIMARY_MODEL, media_type, image_b64, prompt)
        except anthropic.OverloadedError:
            # Our best model is still overloaded even after its own
            # retries — rather than fail the request outright, try once
            # more on a different model. It's usually served with separate
            # capacity, so it can often succeed even when Opus can't.
            app.logger.warning(
                "Opus overloaded after retries, falling back to %s", FALLBACK_MODEL
            )
            message = call_model_with_retries(FALLBACK_MODEL, media_type, image_b64, prompt, retries=2)
            used_fallback_model = True

        # Opus sometimes returns a "thinking" block before the actual answer,
        # so we look for the first block that's actually text instead of
        # assuming it's content[0].
        text = next(
            (block.text for block in message.content if block.type == "text"),
            None,
        )
        if text is None:
            raise EmptyTranscriptionError("The model didn't return any readable text.")
    except Exception as exc:  # noqa: BLE001 — never show a raw technical error to a stranger
        # The person sees a friendly message below, but we still need the
        # REAL error somewhere so we can actually debug it — this prints
        # the full technical error and traceback to Render's logs only,
        # never to the visitor's screen.
        if is_budget_error(exc):
            app.logger.exception("Transcription failed [BUDGET]: %s", exc)
            return render_template_string(
                ERROR_PAGE,
                budget=True,
                title=BUDGET_ERROR_TITLE,
                error=BUDGET_ERROR_MESSAGE,
                contact_email=CONTACT_EMAIL,
                kofi_url=KOFI_URL,
            )
        code = classify_error(exc)
        app.logger.exception("Transcription failed [%s]: %s", code, exc)
        return error_page(code)

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
        used_fallback_model=used_fallback_model,
    )
    del original_bytes
    del image_bytes
    del image_b64
    return response


if __name__ == "__main__":
    app.run(debug=True)
