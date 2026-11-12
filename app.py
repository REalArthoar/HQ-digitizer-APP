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
import hashlib
import hmac
import secrets
import io
import json
import urllib.request
import os
import re
import time
from urllib.parse import quote

from flask import Flask, Request, make_response, redirect, render_template_string, request
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

class InMemoryRequest(Request):
    """Keep every upload in memory, never in a temporary file on disk.

    By default Flask quietly writes any upload bigger than 500 KB (so almost
    every phone photo) to a temporary file on the server's disk while the
    request is processed. It's deleted afterwards, but it still breaks the
    "your photo never touches the disk" promise. This keeps it in RAM only.
    """

    def _get_file_stream(self, total_content_length, content_type, filename=None, content_length=None):
        return io.BytesIO()


app = Flask(__name__)
app.request_class = InMemoryRequest

# Render (like most hosts) sits your app behind a reverse proxy, so without
# this, Flask sees the proxy's own IP for every single visitor instead of
# each person's real IP — which would make the rate limit below either
# useless (everyone shares one bucket) or wrong. This tells Flask to trust
# the standard X-Forwarded-For header the proxy sets.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)  # x_proto: know when visitors use https

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
app.config["MAX_CONTENT_LENGTH"] = 90 * 1024 * 1024  # room for several pages in one Premium upload

# Premium "many pages at once": most pages in one go.
MAX_PAGES = 15

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


# ---------------------------------------------------------------- Premium
# Premium is unlocked with a signed key, so no database is needed: each key
# carries its own expiry date plus a "seal" made with PREMIUM_KEY_SECRET.
# Nobody can make a valid key without that secret, and nothing about the
# visitor has to be stored anywhere.
#
# Settings (in Render -> Environment):
#   PREMIUM_KEY_SECRET  a long random password. Without it, Premium is off.
#   ADMIN_PASSWORD      opens /admin, where you can make keys by hand.
#   PREMIUM_PUBLIC      set to 1 to show the "unlock with a key" box to
#                       everyone. Until then visitors only see "Coming soon",
#                       but an unlock link still works, for testing.
PREMIUM_KEY_SECRET = os.environ.get("PREMIUM_KEY_SECRET", "")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
PREMIUM_PUBLIC = os.environ.get("PREMIUM_PUBLIC", "") == "1"
PREMIUM_COOKIE = "hq_premium"
# The three passes. "1 month" is 31 days so nobody loses a day.
PASSES = {"1 day": 1, "7 days": 7, "1 month": 31}


def _key_seal(payload: bytes) -> bytes:
    return hmac.new(PREMIUM_KEY_SECRET.encode(), b"hq-premium-v1" + payload, hashlib.sha256).digest()[:10]


def make_premium_key(days: int) -> str:
    """Create a key that unlocks Premium for `days` days from now."""
    expires = int(time.time()) + days * 86400
    payload = expires.to_bytes(5, "big") + secrets.token_bytes(3)  # random part: every key is different
    code = base64.b32encode(payload + _key_seal(payload)).decode().rstrip("=")
    return "HQ-" + "-".join(code[i:i + 4] for i in range(0, len(code), 4))


def premium_key_expiry(key: str) -> int | None:
    """When a key runs out (unix time), or None if the key isn't genuine.

    Doesn't check whether it has already expired — see premium_until().
    """
    if not PREMIUM_KEY_SECRET or not key:
        return None
    code = re.sub(r"[^A-Z2-7]", "", key.upper().removeprefix("HQ-").removeprefix("HQ"))
    try:
        raw = base64.b32decode(code + "=" * (-len(code) % 8))
    except ValueError:
        return None
    if len(raw) != 18:
        return None
    payload, seal = raw[:8], raw[8:]
    if not hmac.compare_digest(seal, _key_seal(payload)):
        return None
    return int.from_bytes(payload[:5], "big")


def premium_until() -> int | None:
    """Expiry time of this visitor's Premium pass, or None if they have none."""
    expires = premium_key_expiry(request.cookies.get(PREMIUM_COOKIE, ""))
    if expires and expires > time.time():
        return expires
    return None


def nice_date(unix_time: int) -> str:
    return time.strftime("%d %B %Y", time.gmtime(unix_time)).lstrip("0")


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

DEFAULT_INTRO = "You are transcribing a photo of handwritten text as accurately as possible."
DEFAULT_OUTRO = '- Return ONLY the transcription. No commentary, no "Here is the transcription:", nothing else.'


def build_prompt(context_hint: str, previous_guess: str = "", intro: str = DEFAULT_INTRO, outro: str = DEFAULT_OUTRO) -> str:
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

    return f"""{intro}
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
{outro}
"""


def build_pages_prompt(page_count: int, context_hint: str, follow_story: bool) -> str:
    """Prompt for several pages read together in one go (Premium)."""
    intro = (
        f"You are transcribing {page_count} photos of handwritten pages as accurately as possible. "
        "They are given in order: Page 1 first, then Page 2, and so on. The person arranged them "
        "in this order on purpose, so treat it as the order they were written in."
    )
    if follow_story:
        intro += """

Before you transcribe, look across ALL the pages and work out whether they belong together as one continuous piece: for example a diary or journal with dates, a schedule or log with times, a story or letter, or notes and lists that carry on from one page to the next. If they do, use that connection to read unclear handwriting more accurately:
- Dates, days and times should make sense in sequence. For example, if page 1 ends on "Mon 3 March", a smudged date at the start of page 2 is more likely "Tue 4 March" than "Tue 14 March".
- Names, places and repeated words should be read the same way on every page.
- A sentence cut off at the bottom of one page usually continues at the top of the next.
Use this context ONLY to choose between possible readings of unclear handwriting. Never add, change or "fix" anything that is clearly written, and never move text from one page to another. If the pages are not connected, simply transcribe each one on its own."""
    outro = (
        "- Output format (follow it exactly): start each page with a line \"=== PAGE n ===\" (n is the page number), "
        "followed by that page's transcription, using every rule above. After the last page, write a line "
        "\"=== NOTES ===\" followed by one or two short, plain sentences for the person"
        + (
            ": what kind of document this seems to be (for example \"Diary entries from 3 to 9 March.\" or "
            "\"These pages don't seem to be connected.\"), and, only if the pages look out of order, which order "
            "seems right and why (for example \"Page 3 may come before page 2: its date is earlier.\")."
            if follow_story else
            ", only if something important needs pointing out (otherwise leave the notes empty)."
        )
        + " No other commentary anywhere."
    )
    return build_prompt(context_hint, intro=intro, outro=outro)


def split_pages(text: str, page_count: int) -> tuple[list[str] | None, str]:
    """Split a several-page answer into one text per page, plus the notes."""
    notes = ""
    notes_match = re.search(r"^[ \t*#_]*=+[ \t]*NOTES[ \t]*=+[ \t*#_:]*$", text, re.M | re.I)
    if notes_match:
        notes = text[notes_match.end():].strip()
        text = text[:notes_match.start()]
    parts = re.split(r"^[ \t*#_]*=+[ \t]*PAGE[ \t]+(\d+)[ \t]*=+[ \t*#_:]*$", text, flags=re.M | re.I)
    found = {}
    for i in range(1, len(parts) - 1, 2):
        found[int(parts[i])] = parts[i + 1].strip("\n")
    if not found:
        return None, notes
    return [found.get(n, "") for n in range(1, page_count + 1)], notes


def image_block(media_type: str, image_b64: str) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}}


def read_with_fallback(content: list, max_tokens: int = 8192):
    """Ask the best model, retrying through brief overloads, then the backup
    model if it's still overloaded. Returns (message, used_fallback)."""
    try:
        return call_model_with_retries(PRIMARY_MODEL, content, max_tokens=max_tokens), False
    except anthropic.OverloadedError:
        # Different models are usually served with separate capacity, so the
        # backup can often succeed even when the best one can't.
        app.logger.warning("Opus overloaded after retries, falling back to %s", FALLBACK_MODEL)
        return call_model_with_retries(FALLBACK_MODEL, content, retries=2, max_tokens=max_tokens), True


def message_text(message) -> str:
    """The answer's text. The model sometimes returns a "thinking" block
    first, so look for the first block that's actually text."""
    text = next((block.text for block in message.content if block.type == "text"), None)
    if text is None:
        raise EmptyTranscriptionError("The model didn't return any readable text.")
    return text


def call_model_with_retries(model: str, content: list, retries: int = 3, max_tokens: int = 8192):
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
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": content}],
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
    .premium { margin-top: 28px; text-align: left; border: 1px dashed #d8c7a8; border-radius: 12px; padding: 16px 18px; background: #fdf8ef; }
    .premium-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 10px; }
    .premium-head h2 { font-size: 1.05em; margin: 0; }
    .soon-badge { flex-shrink: 0; font-size: 0.72em; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: #8a5a00; background: #ffeec7; border-radius: 999px; padding: 4px 10px; }
    .premium ul { list-style: none; margin: 0; padding: 0; }
    .premium li { display: flex; gap: 10px; padding: 8px 0; border-top: 1px solid #f0e6d4; opacity: 0.85; }
    .premium li:first-child { border-top: none; }
    .premium .feat-icon { font-size: 1.2em; line-height: 1.3; }
    .premium .feat-title { font-weight: 600; }
    .premium .feat-desc { font-size: 0.85em; color: #6b5d4d; }
    .premium-foot { font-size: 0.8em; color: #8a8178; margin: 10px 0 0; }
    .premium-on { border-style: solid; border-color: #cfe3c6; background: #fbfdf8; }
    .active-badge { flex-shrink: 0; font-size: 0.72em; font-weight: 700; color: #2a6f2a; background: #e3f2dc; border-radius: 999px; padding: 4px 10px; }
    .unlocked-msg { background: #e9f6e4; color: #2a6f2a; border-radius: 8px; padding: 8px 12px; font-size: 0.9em; margin: 0 0 10px; }
    .premium-on form { text-align: center; }
    .order-hint { font-size: 0.85em; color: #6b5d4d; margin: 10px 0 4px; }
    .page-list { list-style: none; margin: 8px 0; padding: 0; text-align: left; }
    .page-list li { display: flex; align-items: center; gap: 10px; background: #fff; border: 1px solid #ece3d4; border-radius: 8px; padding: 6px 8px; margin: 6px 0; cursor: grab; }
    .page-list li.dragging { opacity: 0.4; }
    .page-list li.drop-target { border-color: #3a7bd5; box-shadow: 0 0 0 2px #cfe0f7; }
    .page-list .num { font-weight: 700; color: #8a5a00; min-width: 1.6em; text-align: center; }
    .page-list .thumb { width: 52px; height: 52px; object-fit: cover; border-radius: 4px; background: #f0e9dd; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 0.7em; color: #8a8178; }
    .page-list .name { flex: 1; font-size: 0.85em; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .page-list .ctrls button { padding: 4px 9px; font-size: 0.9em; margin-left: 2px; }
    .story-opt { display: flex; gap: 10px; align-items: flex-start; text-align: left; font-size: 0.9em; margin: 12px 0 4px; cursor: pointer; }
    .story-opt input { margin-top: 4px; }
    .story-opt small { color: #6b5d4d; }
    .key-form { display: flex; gap: 8px; margin-top: 12px; }
    .key-form input { flex: 1; min-width: 0; padding: 8px; font-family: ui-monospace, Menlo, Consolas, monospace; }
    .key-form button { padding: 8px 14px; }
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

  {% if premium_until %}
  <section class="premium premium-on" id="premium" aria-label="Premium">
    <div class="premium-head">
      <h2>Premium: many pages</h2>
      <span class="active-badge">Active until {{ premium_until }}</span>
    </div>
    {% if just_unlocked %}<p class="unlocked-msg">Premium unlocked. Welcome, and thank you for your support!</p>{% endif %}
    <form method="post" enctype="multipart/form-data" action="/transcribe-pages" id="pagesForm">
      <div class="dropzone" id="pagesDrop">
        <div>Drag &amp; drop up to {{ max_pages }} page photos here, or click to choose</div>
      </div>
      <input type="file" name="photos" id="pagesInput" accept="image/*" multiple style="display:none;">
      <p class="order-hint" id="orderHint" hidden>Put the pages in the order they were written: drag them, or use ↑ ↓.</p>
      <ol class="page-list" id="pageList"></ol>
      <label class="story-opt">
        <input type="checkbox" name="follow_story" value="1" checked>
        <span><strong>Follow the story across pages</strong><br>
        <small>Uses dates, times and the flow of the writing to read unclear words. Turn off if the pages aren't connected.</small></span>
      </label>
      <input type="text" name="context" placeholder="Optional: language, topic, or hint" style="width: 90%; padding: 8px; margin: 10px 0;"><br>
      <button type="submit" id="pagesBtn">Transcribe pages</button>
      <div class="wake-status" id="pagesWake"></div>
    </form>
  </section>
  {% else %}
  <section class="premium" id="premium" aria-label="Premium features coming soon">
    <div class="premium-head">
      <h2>Premium</h2>
      <span class="soon-badge">Coming soon</span>
    </div>
    <ul>
      <li><span class="feat-icon" aria-hidden="true">🗂️</span><div>
        <div class="feat-title">Many pages at once</div>
        <div class="feat-desc">Upload a whole notebook or diary in one go and get all the text back together.</div></div></li>
      <li><span class="feat-icon" aria-hidden="true">📖</span><div>
        <div class="feat-title">Follows the story</div>
        <div class="feat-desc">Reads all pages together and uses dates, times and the flow of the writing to read unclear words more accurately.</div></div></li>
      <li><span class="feat-icon" aria-hidden="true">↕️</span><div>
        <div class="feat-title">Put pages in order</div>
        <div class="feat-desc">Drag your photos into the right order before reading, for the most accurate result.</div></div></li>
    </ul>
    {% if premium_public %}
    <form method="post" action="/unlock" class="key-form">
      <input type="text" name="key" placeholder="Have a key? HQ-XXXX-XXXX-…" autocomplete="off" required>
      <button type="submit">Unlock</button>
    </form>
    <p class="premium-foot">Passes for 1 day, 7 days or 1 month on <a href="{{ kofi_url }}" target="_blank" rel="noopener">Ko-fi</a>. Your key arrives by email.</p>
    {% else %}
    <p class="premium-foot">Not available yet. Supporters on Ko-fi will get access first.</p>
    {% endif %}
  </section>
  {% endif %}
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

    // ---------------- Premium: many pages, in the visitor's chosen order
    const pagesForm = document.getElementById('pagesForm');
    if (pagesForm) {
      const MAX_PAGES = {{ max_pages }};
      const pagesInput = document.getElementById('pagesInput');
      const pagesDrop = document.getElementById('pagesDrop');
      const pageList = document.getElementById('pageList');
      const orderHint = document.getElementById('orderHint');
      const pagesBtn = document.getElementById('pagesBtn');
      const pages = [];  // {file, url} in reading order
      let dragFrom = null;

      function addPages(files) {
        for (const file of files) {
          const isImage = file.type.startsWith('image/') || /\\.(heic|heif)$/i.test(file.name);
          if (!isImage) continue;
          if (pages.length >= MAX_PAGES) { alert('You can read up to ' + MAX_PAGES + ' pages at once.'); break; }
          pages.push({ file: file, url: URL.createObjectURL(file) });
        }
        renderPages();
      }

      function move(from, to) {
        if (to < 0 || to >= pages.length || from === to) return;
        const [item] = pages.splice(from, 1);
        pages.splice(to, 0, item);
        renderPages();
      }

      function renderPages() {
        pageList.innerHTML = '';
        pages.forEach((page, i) => {
          const li = document.createElement('li');
          li.draggable = true;
          const num = document.createElement('span'); num.className = 'num'; num.textContent = i + 1;
          const thumb = document.createElement('img'); thumb.className = 'thumb'; thumb.alt = '';
          thumb.src = page.url;
          thumb.onerror = () => { const ph = document.createElement('span'); ph.className = 'thumb'; ph.textContent = 'photo'; thumb.replaceWith(ph); };
          const name = document.createElement('span'); name.className = 'name'; name.textContent = page.file.name;
          const ctrls = document.createElement('span'); ctrls.className = 'ctrls';
          [['↑', 'Move up', () => move(i, i - 1)], ['↓', 'Move down', () => move(i, i + 1)],
           ['✕', 'Remove', () => { URL.revokeObjectURL(page.url); pages.splice(i, 1); renderPages(); }]
          ].forEach(([label, title, action]) => {
            const b = document.createElement('button'); b.type = 'button'; b.textContent = label; b.title = title;
            b.setAttribute('aria-label', title + ' (page ' + (i + 1) + ')');
            b.addEventListener('click', action); ctrls.appendChild(b);
          });
          li.append(num, thumb, name, ctrls);
          li.addEventListener('dragstart', () => { dragFrom = i; li.classList.add('dragging'); });
          li.addEventListener('dragend', () => li.classList.remove('dragging'));
          li.addEventListener('dragover', (e) => { e.preventDefault(); li.classList.add('drop-target'); });
          li.addEventListener('dragleave', () => li.classList.remove('drop-target'));
          li.addEventListener('drop', (e) => { e.preventDefault(); e.stopPropagation(); li.classList.remove('drop-target'); if (dragFrom !== null) move(dragFrom, i); dragFrom = null; });
          pageList.appendChild(li);
        });
        orderHint.hidden = pages.length < 2;
        pagesBtn.textContent = pages.length > 1 ? 'Transcribe ' + pages.length + ' pages' : 'Transcribe pages';
      }

      pagesDrop.addEventListener('click', () => pagesInput.click());
      pagesInput.addEventListener('change', () => { addPages(pagesInput.files); pagesInput.value = ''; });
      ['dragenter', 'dragover'].forEach(evt => pagesDrop.addEventListener(evt, (e) => { e.preventDefault(); pagesDrop.classList.add('dragover'); }));
      ['dragleave', 'drop'].forEach(evt => pagesDrop.addEventListener(evt, (e) => { e.preventDefault(); pagesDrop.classList.remove('dragover'); }));
      pagesDrop.addEventListener('drop', (e) => addPages(e.dataTransfer.files));

      protectForm(pagesForm, pagesBtn, 'Reading your pages... (about 20 seconds per page, so a few minutes for a big batch)', document.getElementById('pagesWake'), () => {
        if (!pages.length) { alert('Please choose your page photos first.'); return false; }
        // Put the files into the form in exactly the order shown.
        const ordered = new DataTransfer();
        pages.forEach(page => ordered.items.add(page.file));
        pagesInput.files = ordered.files;
        return true;
      });
    }

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
  <h1>All done! Your text is ready</h1>
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

PAGES_RESULT = """
<!doctype html>
<html>
<head><title>Your Pages</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    __PAGE_STYLE__
    .card { max-width: 1040px; }
    .columns { display: flex; gap: 24px; flex-wrap: wrap; align-items: flex-start; }
    .columns img { max-width: min(420px, 100%); max-height: 70vh; border-radius: 8px; border: 1px solid #ddd; }
    .result-content { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 0.92em; line-height: 1.55; background: #f7f2ea; padding: 16px; border-radius: 8px; flex: 1; min-width: 280px; overflow-x: auto; }
    .page-block { margin-top: 28px; padding-top: 18px; border-top: 1px solid #ece3d4; }
    .page-block h2 { font-size: 1.1em; margin: 0 0 12px; color: #8a5a00; }
    .notes { background: #eef4fb; border-radius: 10px; padding: 12px 16px; color: #2b4a6f; }
    .notes strong { display: block; margin-bottom: 4px; }
    .warn { font-size: 0.85em; color: #a06a00; background: #fff8e6; padding: 8px 12px; border-radius: 6px; }
    .legend { font-size: 0.8em; color: #666; margin-top: 6px; }
    .legend span { color:#c0392b; font-weight:bold; background:#fdecea; padding:0 2px; border-radius:3px; }
    .edit-box { margin-top: 30px; }
    .edit-box textarea { width: 100%; min-height: 220px; font-family: monospace; font-size: 0.95em; padding: 10px; border-radius: 8px; border: 1px solid #ccc; box-sizing: border-box; }
    .edit-box button { margin-top: 8px; padding: 6px 14px; cursor: pointer; }
    .promise { color: #2a6f2a; font-size: 0.9em; }
    .intro { color: #5a5048; margin-top: -6px; }
  </style>
</head>
<body>
<div class="card">
  <h1>All done! Your {{ pages|length }} pages are ready</h1>
  <p class="intro">Have a look below. Anything highlighted is worth a quick check against the photo.</p>
  {% if notes %}<div class="notes"><strong>What the tool noticed</strong>{{ notes }}</div>{% endif %}
  {% if truncated %}<p class="warn">This was a lot of writing for one go, so the end may be cut off. If the last page looks incomplete, read it again on its own.</p>{% endif %}
  {% if used_fallback_model %}<p class="warn">Text recognition was briefly overloaded with demand, so the tool read these in backup mode. Worth a quick double-check.</p>{% endif %}
  {% for page in pages %}
  <div class="page-block">
    <h2>Page {{ loop.index }}</h2>
    <div class="columns">
      <img src="data:{{ page.media_type }};base64,{{ page.image_b64 }}" alt="Photo of page {{ loop.index }}">
      <div class="result-content">{{ page.html }}</div>
    </div>
  </div>
  {% endfor %}
  <p class="legend"><span>Highlighted text</span> means the tool wasn't fully sure — worth double-checking against the photo.</p>
  <div class="edit-box">
    <p style="font-size:0.9em; color:#555; margin-bottom:4px;">All pages as plain text, ready to edit or copy:</p>
    <textarea id="editableText">{{ all_text }}</textarea><br>
    <button type="button" id="copyBtn">Copy all pages</button>
    <span id="copyStatus" style="font-size:0.85em; color:#2a6f2a; margin-left:8px;"></span>
  </div>
  <p class="promise" style="margin-top:24px;">These photos were never saved — they're only shown here, in your own browser, for this one page. Once you leave or refresh, they're gone for good.</p>
  <a href="/#premium">Read more pages →</a>
  <script>
    document.getElementById('copyBtn').addEventListener('click', () => {
      navigator.clipboard.writeText(document.getElementById('editableText').value).then(() => {
        const status = document.getElementById('copyStatus');
        status.textContent = 'Copied!';
        setTimeout(() => { status.textContent = ''; }, 2000);
      });
    });
  </script>
</div>
</body>
</html>
"""

ADMIN_PAGE = """
<!doctype html>
<html>
<head><title>Admin — Premium keys</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="robots" content="noindex">
  <style>
    __PAGE_STYLE__
    .card { max-width: 560px; }
    label { display: block; margin: 12px 0 4px; font-weight: 600; }
    input, select { padding: 8px; font-size: 1em; width: 100%; box-sizing: border-box; }
    button { margin-top: 14px; padding: 10px 20px; font-size: 1em; cursor: pointer; }
    .key { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 1.1em; background: #f7f2ea; padding: 12px; border-radius: 8px; word-break: break-all; }
    .bad { color: #a33; }
  </style>
</head>
<body>
<div class="card">
  <h1>Make a Premium key</h1>
  {% if error %}<p class="bad">{{ error }}</p>{% endif %}
  {% if key %}
  <p>Key for <strong>{{ pass_name }}</strong>, valid until {{ until }}:</p>
  <p class="key">{{ key }}</p>
  <p>One-click unlock link (opening it unlocks Premium in that browser):</p>
  <p class="key">{{ link }}</p>
  <hr>
  {% endif %}
  <form method="post" action="/admin">
    <label for="pw">Admin password</label>
    <input type="password" id="pw" name="password" required autocomplete="current-password">
    <label for="pass">Pass</label>
    <select id="pass" name="pass">{% for name in passes %}<option>{{ name }}</option>{% endfor %}</select>
    <button type="submit">Make key</button>
  </form>
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
    .whose-key    { background: #f3ecfb; color: #6a3d9a; }
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
UPLOAD_FORM, RESULT_PAGE, ERROR_PAGE, PAGES_RESULT, ADMIN_PAGE = (
    page.replace("__PAGE_STYLE__", PAGE_STYLE).replace("__WAKE_SCRIPT__", WAKE_SCRIPT)
    for page in (UPLOAD_FORM, RESULT_PAGE, ERROR_PAGE, PAGES_RESULT, ADMIN_PAGE)
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
    "key": "This is about your Premium pass.",
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
    "TOO_MANY_PAGES": ("photo", "Too many pages at once",
        f"You can read up to {MAX_PAGES} pages in one go.",
        f"Go back and remove some pages, then read the rest in a second batch."),
    "PREMIUM_NEEDED": ("key", "This needs Premium",
        "Reading many pages at once is a Premium feature, and there's no active Premium pass in this browser.",
        "If you have a key, use the unlock link from your email again, in the browser you want to use."),
    "KEY_INVALID": ("key", "That key doesn't work",
        "The key wasn't recognised. It may be incomplete, or have a typo in it.",
        "Copy the key again from your email, including the HQ- at the start, or use the unlock button in it."),
    "KEY_EXPIRED": ("key", "That Premium pass has ended",
        "This key was genuine, but its time is up.",
        "Get a new pass to keep using Premium. Single photos are always free."),
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
    "TOO_LARGE": ("photo", "That upload is too large",
        "Uploads can be up to 90 MB in total, and this one is bigger.",
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


def error_page(code: str, status: int = 200, page: int | None = None):
    """Render the error page for one of the ERRORS codes.

    The error code shown on the page includes the time (UTC), so when
    someone emails it in, you can find the exact matching entry in
    Render's logs.
    """
    whose, title, what, todo = ERRORS[code]
    if page:
        what = f"The problem is with page {page}. {what}"
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
    expires = premium_until()
    return render_template_string(
        UPLOAD_FORM,
        contact_email=CONTACT_EMAIL,
        kofi_url=KOFI_URL,
        premium_until=nice_date(expires) if expires else None,
        just_unlocked=request.args.get("unlocked") == "1",
        premium_public=PREMIUM_PUBLIC and bool(PREMIUM_KEY_SECRET),
        max_pages=MAX_PAGES,
    )


@app.route("/unlock", methods=["GET", "POST"])
@limiter.limit("20 per hour")
def unlock():
    """Unlock Premium in this browser, from the link in the email (GET) or
    the key box on the upload page (POST). The key is kept in a cookie."""
    key = (request.values.get("key") or "").strip()
    expires = premium_key_expiry(key)
    if not expires:
        return error_page("KEY_INVALID")
    if expires <= time.time():
        return error_page("KEY_EXPIRED")
    response = make_response(redirect("/?unlocked=1#premium"))
    response.set_cookie(
        PREMIUM_COOKIE, key, max_age=int(expires - time.time()),
        httponly=True, secure=request.is_secure, samesite="Lax",
    )
    return response


# ------------------------------------------------- Ko-fi -> automatic keys
# When someone pays on Ko-fi, Ko-fi sends a message to /kofi-webhook. We
# check it really came from Ko-fi, pick the pass from the amount paid, make
# a key and email it straight to the supporter. Nothing is stored: their
# email address is used once to send the key and then forgotten.
#
# Settings (in Render -> Environment):
#   KOFI_VERIFICATION_TOKEN  from Ko-fi: More -> API -> Webhooks -> Advanced
#   GMAIL_ADDRESS            the address keys are sent from (for test emails
#                            and replies)
#   MAIL_SCRIPT_URL          the web-app address of the Google Apps Script
#                            that sends the emails (see send_key_email)
#   MAIL_SCRIPT_SECRET       the same secret as written in that script
KOFI_VERIFICATION_TOKEN = os.environ.get("KOFI_VERIFICATION_TOKEN", "")
GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "")
MAIL_SCRIPT_URL = os.environ.get("MAIL_SCRIPT_URL", "").strip()
MAIL_SCRIPT_SECRET = os.environ.get("MAIL_SCRIPT_SECRET", "").strip()

# Amount paid (in your Ko-fi currency) -> pass. Checked from the top, so
# 10 or more gets a month, 5 or more a week, 1 or more a day.
PASS_PRICES = [(10, "1 month"), (5, "7 days"), (1, "1 day")]

# Ko-fi may send the same message twice; remember recent ones so nobody
# gets two keys for one payment. (Only Ko-fi's message ids, kept in memory.)
_seen_kofi_messages: list[str] = []


def pass_for_amount(amount: float) -> str | None:
    for price, pass_name in PASS_PRICES:
        if amount >= price:
            return pass_name
    return None


KEY_EMAIL_HTML = """<!doctype html>
<html><body style="margin:0; padding:24px; background:#efe7da; font-family:Arial, Helvetica, sans-serif; color:#2b2724;">
  <div style="max-width:520px; margin:0 auto; background:#fffdf9; border-radius:14px; padding:28px; border:1px solid #e6dccb;">
    <h1 style="font-size:22px; margin:0 0 12px;">Thank you for your support{% if name %}, {{ name }}{% endif %}! 💛</h1>
    <p style="font-size:15px; line-height:1.5;">Your <strong>{{ pass_name }}</strong> Premium pass for HQ Digitizer is ready. It's valid until <strong>{{ until }}</strong>.</p>
    <p style="text-align:center; margin:26px 0;">
      <a href="{{ link }}" style="background:#2a6f2a; color:#ffffff; text-decoration:none; font-weight:bold; padding:13px 26px; border-radius:8px; display:inline-block;">Unlock Premium</a>
    </p>
    <p style="font-size:14px; line-height:1.5;">Press the button on each device you want to use, for example your phone and your computer.</p>
    <p style="font-size:14px; line-height:1.5;">Your key, in case you need it:</p>
    <p style="font-family:Consolas, Menlo, monospace; font-size:15px; background:#f7f2ea; padding:12px; border-radius:8px; word-break:break-all;">{{ key }}</p>
    <p style="font-size:13px; color:#6b5d4d; line-height:1.5;">Questions or problems? Just reply to this email.</p>
  </div>
</body></html>"""

KEY_EMAIL_TEXT = """Thank you for your support{% if name %}, {{ name }}{% endif %}!

Your {{ pass_name }} Premium pass for HQ Digitizer is ready. It's valid until {{ until }}.

Unlock Premium (open this link on each device you want to use):
{{ link }}

Your key, in case you need it:
{{ key }}

Questions or problems? Just reply to this email.
"""


def send_key_email(to_address: str, name: str, pass_name: str, key: str, link: str, test: bool = False) -> None:
    """Send the key email through a small Google Apps Script running in the
    hqdigitizersupport Gmail account.

    Render's free plan blocks the normal way of sending email (SMTP ports),
    so instead the app asks that script, over ordinary web traffic, to send
    the email from Gmail. The shared secret stops anyone else using it.
    """
    until = nice_date(premium_key_expiry(key))
    fields = dict(name=name, pass_name=pass_name, until=until, key=key, link=link)
    payload = json.dumps({
        "secret": MAIL_SCRIPT_SECRET,
        "to": to_address,
        "subject": ("[TEST] " if test else "") + f"Your HQ Digitizer Premium pass ({pass_name})",
        "text": render_template_string(KEY_EMAIL_TEXT, **fields),
        "html": render_template_string(KEY_EMAIL_HTML, **fields),
        "replyTo": GMAIL_ADDRESS,
    }).encode("utf-8")
    req = urllib.request.Request(
        MAIL_SCRIPT_URL, data=payload, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        answer = response.read().decode("utf-8", "replace").strip()
    if answer != "ok":
        raise RuntimeError(f"mail script answered {answer[:80]!r}")


@app.route("/kofi-webhook", methods=["POST"])
def kofi_webhook():
    """Ko-fi calls this after every payment. We always answer 200 once the
    message is genuine, so Ko-fi doesn't keep resending it."""
    if not (KOFI_VERIFICATION_TOKEN and PREMIUM_KEY_SECRET and GMAIL_ADDRESS and MAIL_SCRIPT_URL and MAIL_SCRIPT_SECRET):
        app.logger.error("Ko-fi: payment message arrived, but Ko-fi/email settings are missing in Render")
        return "not set up", 503
    try:
        data = json.loads(request.form.get("data", ""))
    except ValueError:
        return "bad request", 400
    if not hmac.compare_digest(str(data.get("verification_token", "")).encode(), KOFI_VERIFICATION_TOKEN.encode()):
        app.logger.warning("Ko-fi: message with a wrong verification token was rejected")
        return "forbidden", 403

    message_id = str(data.get("message_id", ""))
    if message_id and message_id in _seen_kofi_messages:
        return "already handled", 200
    _seen_kofi_messages.append(message_id)
    del _seen_kofi_messages[:-200]

    try:
        amount = float(data.get("amount") or 0)
    except ValueError:
        amount = 0
    pass_name = pass_for_amount(amount)
    if not pass_name:
        app.logger.info("Ko-fi: payment of %s %s is below the cheapest pass, no key sent", amount, data.get("currency"))
        return "ok", 200

    email = str(data.get("email") or "").strip()
    name = str(data.get("from_name") or "").strip()[:60]
    # Ko-fi's "Send test" uses a made-up example address; send those to our
    # own inbox instead, so the whole flow can be tried without a real payment.
    test = not email or email.lower().endswith("@example.com")
    if test:
        email = GMAIL_ADDRESS

    key = make_premium_key(PASSES[pass_name])
    link = request.host_url + "unlock?key=" + key
    try:
        send_key_email(email, name, pass_name, key, link, test=test)
        app.logger.info("Ko-fi: sent a %s key (%s %s)%s", pass_name, amount, data.get("currency"), " [test]" if test else "")
    except Exception as exc:  # noqa: BLE001
        # Only on failure do we log the address, so you can send the key by hand.
        app.logger.error(
            "Ko-fi: KEY EMAIL FAILED (%s). Make a %s key on /admin and send it to %s by hand.",
            exc, pass_name, email,
        )
    return "ok", 200


@app.route("/admin", methods=["GET", "POST"])
@limiter.limit("10 per minute", methods=["POST"])
def admin():
    """Make Premium keys by hand (for testing, gifts or fixing problems).
    Hidden unless ADMIN_PASSWORD and PREMIUM_KEY_SECRET are both set."""
    if not (ADMIN_PASSWORD and PREMIUM_KEY_SECRET):
        return redirect("/")
    page = dict(passes=list(PASSES))
    if request.method == "POST":
        if not hmac.compare_digest(request.form.get("password", "").encode(), ADMIN_PASSWORD.encode()):
            app.logger.warning("Admin: wrong password attempt")
            page["error"] = "Wrong password."
        else:
            pass_name = request.form.get("pass", "1 day")
            days = PASSES.get(pass_name, 1)
            key = make_premium_key(days)
            page.update(key=key, pass_name=pass_name,
                        until=nice_date(premium_key_expiry(key)),
                        link=request.host_url + "unlock?key=" + key)
    return render_template_string(ADMIN_PAGE, **page)


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


def reading_failed(exc: Exception):
    """The page to show when reading failed. The real error goes to
    Render's logs only, never to the visitor's screen."""
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


@app.route("/transcribe-pages", methods=["GET"])
def transcribe_pages_reload():
    return redirect("/#premium")


@app.route("/transcribe-pages", methods=["POST"])
@limiter.limit("3 per minute; 20 per day")
def transcribe_pages():
    """Premium: read several pages, in the order the visitor arranged them."""
    if not premium_until():
        return error_page("PREMIUM_NEEDED")
    uploads = [f for f in request.files.getlist("photos") if f and f.filename]
    if not uploads:
        return error_page("NO_PHOTO")
    if len(uploads) > MAX_PAGES:
        return error_page("TOO_MANY_PAGES")

    pages = []
    for number, upload in enumerate(uploads, start=1):
        if not (upload.mimetype or "").startswith("image/"):
            return error_page("NOT_IMAGE", page=number)
        try:
            image_bytes, media_type = shrink_image_if_needed(upload.read())
        except UnreadableImageError:
            return error_page("UNREADABLE", page=number)
        image_b64 = base64.b64encode(auto_orient(image_bytes)).decode("utf-8")
        pages.append({"media_type": media_type, "image_b64": image_b64})

    context_hint = request.form.get("context", "")[:MAX_CONTEXT_LENGTH]
    follow_story = request.form.get("follow_story") == "1"
    notes, truncated, used_fallback = "", False, False
    try:
        if len(pages) == 1:
            message, used_fallback = read_with_fallback(
                [image_block(pages[0]["media_type"], pages[0]["image_b64"]), {"type": "text", "text": build_prompt(context_hint)}])
            texts = [message_text(message)]
            truncated = message.stop_reason == "max_tokens"
        elif follow_story:
            # All pages in one request, so the story can be followed across them.
            content = []
            for number, page in enumerate(pages, start=1):
                content += [{"type": "text", "text": f"Page {number}:"}, image_block(page["media_type"], page["image_b64"])]
            content.append({"type": "text", "text": build_pages_prompt(len(pages), context_hint, follow_story=True)})
            message, used_fallback = read_with_fallback(content, max_tokens=20000)
            texts, notes = split_pages(message_text(message), len(pages))
            truncated = message.stop_reason == "max_tokens"
            if texts is None:
                # The answer didn't come back split into pages; show it all
                # on page 1 rather than losing it.
                texts = [message_text(message)] + [""] * (len(pages) - 1)
                notes = "The pages came back as one block of text, so it's all shown under page 1."
        else:
            # Pages read one by one, each on its own.
            texts = []
            for page in pages:
                message, fallback = read_with_fallback(
                    [image_block(page["media_type"], page["image_b64"]), {"type": "text", "text": build_prompt(context_hint)}])
                texts.append(message_text(message))
                used_fallback = used_fallback or fallback
                truncated = truncated or message.stop_reason == "max_tokens"
    except Exception as exc:  # noqa: BLE001 — never show a raw technical error to a stranger
        return reading_failed(exc)

    for page, text in zip(pages, texts):
        page["html"] = format_transcription(text)
    all_text = "\n\n".join(
        f"— Page {number} —\n{clean_text_for_copy(text)}" for number, text in enumerate(texts, start=1)
    )
    # Rendered once, straight back into this response — never written to a
    # file, database, or log.
    return render_template_string(
        PAGES_RESULT,
        pages=pages,
        notes=notes,
        all_text=all_text,
        truncated=truncated,
        used_fallback_model=used_fallback,
    )


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

    try:
        message, used_fallback_model = read_with_fallback(
            [image_block(media_type, image_b64), {"type": "text", "text": prompt}])
        text = message_text(message)
    except Exception as exc:  # noqa: BLE001 — never show a raw technical error to a stranger
        return reading_failed(exc)

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
