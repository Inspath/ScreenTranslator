import sys
import os
import io
import json
import time
import ctypes
import mss
import numpy as np
from PIL import Image
from PyQt6.QtCore import Qt, QRect, QPoint, QSize, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QColor, QPixmap, QImage, QAction, QFont, QFontMetrics
from PyQt6.QtWidgets import (
    QApplication, QWidget, QSystemTrayIcon, QMenu, QStyle, QLabel, QVBoxLayout, QFrame
)
from google import genai
from google.genai import types
import keyboard

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
ACTIVATION_HOTKEY = "alt+s"
QUIT_HOTKEY = "alt+q"


WATCH_INTERVAL_MS = 250

STABLE_TICKS_REQUIRED = 2

QUIET_TICK_PIXEL_THRESHOLD = 150
# Above this many changed pixels (per tick, vs the previous tick - not vs the
# last translation), we consider the user actively scrolling/turning pages.
# The moment we cross this, the old translated bubbles are cleared instantly
# instead of lingering (misaligned) until a brand new translation lands.
MOVEMENT_CLEAR_PIXEL_THRESHOLD = 600
# Cap on the longer edge (px) sent to the API. Screenshots are often much
# bigger than needed for OCR; downscaling cuts upload time and inference
# time with negligible accuracy loss for typical manga text sizes.
MAX_DIMENSION = 1400
# Floor on the shorter edge (px) sent to the API, upscaled if the crop is
# smaller than this so small text stays legible.
MIN_DIMENSION = 500

IMAGE_UPLOAD_FORMAT = "PNG" 
JPEG_QUALITY = 90
# Caps the token budget Gemini spends processing the image itself - this is
# the API's actual "detail vs. speed/cost" dial Per Google's current docs:
#   MEDIA_RESOLUTION_LOW    ~280 tokens - fastest, real risk of missing small/dense text
#   MEDIA_RESOLUTION_MEDIUM ~560 tokens - balanced starting point
#   MEDIA_RESOLUTION_HIGH   ~1120 tokens - closer to the model's default, best OCR fidelity
# https://ai.google.dev/gemini-api/docs/media-resolution 
MEDIA_RESOLUTION = "MEDIA_RESOLUTION_MEDIUM"

# Seconds to wait before retrying a transient (503/429) API error.
RETRY_BACKOFF_SECONDS = 1.5

#watcher state colours
WATCH_COLOR_IDLE = "#888888"
WATCH_COLOR_SCROLLING = "#FFA726"   # actively moving - not sending
WATCH_COLOR_SETTLING = "#FFD54F"    # quiet, but not yet confirmed stable
WATCH_COLOR_UNCHANGED = "#00FF7A"   # stable AND matches last translation - nothing to send
WATCH_COLOR_SENDING = "#00E5FF"     # dispatching / in flight
WATCH_COLOR_QUEUED = "#BB86FC"      # newer frame queued behind an in-flight request
# ---------------------------------------------------------------------------

# Windows API constant used to hide a window from screen-capture APIs
# (mss/BitBlt/most screen recorders) while keeping it visible to the user.
# Requires Windows 10 version 2004 (build 19041) or later.
WDA_EXCLUDEFROMCAPTURE = 0x11


def exclude_from_capture(widget):
    """Make a top-level widget invisible to screen-capture APIs while still
    visible on the user's physical display.

    This is critical for BubbleOverlay and WatchedRegionOverlay: both paint
    directly on top of the region that check_region_change() re-screenshots
    every ~600ms. Without this, mss.grab() sees our own rendered translation
    text / border box sitting over the manga page, the pixel-diff against
    the pre-overlay baseline is huge, and the watcher concludes "new panel
    detected" and re-translates its own output -> infinite retranslation
    loop. Excluding these windows from capture means mss only ever sees the
    real page underneath, so the diff correctly goes to ~0 once the page
    itself stops changing.
    """
    if sys.platform != "win32":
        return
    try:
        hwnd = int(widget.winId())
        ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
    except Exception as e:
        print(f"[WARN] Could not set display affinity (capture exclusion): {e}")


# A single shared client is reused across every request instead of
# constructing a new genai.Client() per API call/attempt. Client construction
# does some setup work (transport/session init); reusing it shaves a bit of
# fixed overhead off every request. Created lazily on first use since
# GEMINI_API_KEY may not be set yet at import time in some setups.
_shared_client = None


def get_client():
    global _shared_client
    if _shared_client is None:
        _shared_client = genai.Client(api_key=GEMINI_API_KEY)
    return _shared_client


# Unicode ranges covering common CJK script blocks (Han/Kanji, Hiragana,
# Katakana, Hangul). Used as a cheap heuristic to flag translations that are
# still mostly in the source script - i.e. the model echoed the original
# text instead of translating it.
_CJK_RANGES = (
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs (Han/Kanji/Hanzi)
    (0x3400, 0x4DBF),   # CJK Extension A
    (0x3040, 0x309F),   # Hiragana
    (0x30A0, 0x30FF),   # Katakana
    (0xAC00, 0xD7A3),   # Hangul syllables
)


def _looks_untranslated(text, threshold=0.3):
    """Rough heuristic: if more than `threshold` of a translation's
    non-space characters are CJK script, it's very likely the model
    returned the original text (or a partial mix) instead of translating
    it. Not a hard filter - just used to log a warning so this is visible
    instead of a silent quality failure."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    cjk_count = sum(
        1 for c in chars if any(lo <= ord(c) <= hi for lo, hi in _CJK_RANGES)
    )
    return (cjk_count / len(chars)) > threshold


class TranslationWorker(QThread):
    # Emits a dict: {"bubbles": [{"bbox_2d": [y0,x0,y1,x1], "text": str}, ...],
    #                "plain_text": str, "generation": int, "timings": {...}}
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, pil_image, generation=0):
        super().__init__()
        # Raw (pre-resize) crop. Resize + encode happen inside run(), i.e. on
        # the worker thread rather than the GUI thread - this was previously
        # done in dispatch_image_to_worker() on the main thread, which briefly
        # blocked the UI (and the watcher timer) on every single request.
        # Moving it here also lets us time it as its own stage.
        self.pil_image = pil_image
        # Tags this request with the page/panel "generation" it belongs to,
        # so a slow response that lands after the user has already moved on
        # can be recognized as stale and ignored instead of flashing an old
        # translation over new content.
        self.generation = generation

    def _prepare_image(self):
        img = self.pil_image
        if img.width < MIN_DIMENSION or img.height < MIN_DIMENSION:
            scale = max(MIN_DIMENSION / img.width, MIN_DIMENSION / img.height)
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        elif img.width > MAX_DIMENSION or img.height > MAX_DIMENSION:
            # Downscale oversized crops - shrinks the upload payload and the
            # amount of image data the model has to process, which is one of
            # the biggest levers on end-to-end latency, with negligible OCR
            # accuracy cost since manga text is rarely fine enough to need
            # full screenshot resolution.
            scale = min(MAX_DIMENSION / img.width, MAX_DIMENSION / img.height)
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        if IMAGE_UPLOAD_FORMAT == "JPEG":
            img.convert("RGB").save(buffer, format="JPEG", quality=JPEG_QUALITY)
            mime_type = "image/jpeg"
        else:
            # compress_level=1: fastest PNG encode setting. Default (6) spends
            # noticeably more CPU squeezing the file a bit smaller, not worth
            # it on an already-downscaled image.
            img.save(buffer, format="PNG", compress_level=1)
            mime_type = "image/png"
        return buffer.getvalue(), mime_type

    def run(self):
        t_start = time.time()
        image_bytes, mime_type = self._prepare_image()
        t_prepared = time.time()

        max_attempts = 3
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                client = get_client()
                prompt = (
                    "You are a professional Manga Translator. Identify every distinct speech "
                    "bubble or text block in this manga panel image (Japanese, Chinese, or "
                    "Korean, including vertical text). For each one, provide its bounding box, "
                    "the original text exactly as written (source_text), and a natural ENGLISH "
                    "translation of it (translation). "
                    "The 'translation' field must always be in English, in the Latin alphabet - "
                    "never copy any Chinese, Japanese, or Korean characters into it, even "
                    "partially, even for names or single words. If a word has no natural English "
                    "equivalent (e.g. a proper noun), transliterate/romanize it instead of leaving "
                    "it in the original script. Do not skip a bubble just because it seems simple "
                    "or already looks readable. "
                    "bbox_2d must be [y_min, x_min, y_max, x_max], normalized 0-1000 relative "
                    "to the full image dimensions. Return an entry for every bubble, even ones "
                    "with short text."
                )
                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "bbox_2d": types.Schema(
                                    type=types.Type.ARRAY,
                                    items=types.Schema(type=types.Type.INTEGER),
                                ),
                                # Splitting "the text" into an explicit
                                # source/translation pair (rather than one
                                # overloaded "text" field) measurably reduces
                                # cases where the model just echoes the
                                # original characters back instead of
                                # translating - it now has to actually
                                # produce two distinct outputs instead of one
                                # field that's ambiguous about which
                                # language it should be in.
                                "source_text": types.Schema(type=types.Type.STRING),
                                "translation": types.Schema(type=types.Type.STRING),
                            },
                            required=["bbox_2d", "source_text", "translation"],
                        ),
                    ),
                    # Caps how many tokens the model spends "looking at" the
                    # image (see MEDIA_RESOLUTION constant above) - this is
                    # the real, API-level speed/quality dial, unlike prompt
                    # wording. Passed as a plain string since accepted enum
                    # values are just these string names; if your installed
                    # google-genai version rejects this kwarg, check its
                    # current docs for the equivalent field name.
                    media_resolution=MEDIA_RESOLUTION,
                )
                print("\nCalling Gemini API...")
                api_attempt_start = time.time()
                response = client.models.generate_content(
                    model="gemini-3.1-flash-lite",
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        prompt,
                    ],
                    config=config,
                )
                print(f"[TIMING] Gemini API call took {time.time() - api_attempt_start:.2f}s")
                t_api_done = time.time()

                raw = response.text or "[]"
                try:
                    parsed = json.loads(raw)
                    if not isinstance(parsed, list):
                        parsed = []
                except json.JSONDecodeError:
                    parsed = []
                    print(f"[API WARNING] Could not parse structured JSON. Raw response:\n{raw}")

                bubbles = []
                for entry in parsed:
                    bbox = entry.get("bbox_2d")
                    text = (entry.get("translation") or "").strip()
                    if not text or not bbox or len(bbox) != 4:
                        continue
                    if _looks_untranslated(text):
                        # Best-effort sanity check: the model occasionally
                        # still echoes the original CJK text into the
                        # "translation" field despite the prompt/schema
                        # split above. We still show it (better than
                        # silently dropping a bubble) but flag it loudly in
                        # the console so this is visible and countable
                        # rather than a silent, invisible failure.
                        print(f"[TRANSLATION WARNING] Output still looks untranslated: {text!r} "
                              f"(source: {entry.get('source_text', '')!r})")
                    bubbles.append({"bbox_2d": bbox, "text": text})

                t_parsed = time.time()

                result = {
                    "bubbles": bubbles,
                    "plain_text": "\n\n".join(b["text"] for b in bubbles) if bubbles else "No text found in image.",
                    "generation": self.generation,
                    "timings": {
                        "prep_ms": (t_prepared - t_start) * 1000,
                        # api_ms covers everything from "image ready to send"
                        # to "response received", including any earlier
                        # failed attempts + backoff sleeps in this loop.
                        "api_ms": (t_api_done - t_prepared) * 1000,
                        "parse_ms": (t_parsed - t_api_done) * 1000,
                        "total_ms": (t_parsed - t_start) * 1000,
                    },
                }
                self.finished.emit(result)
                return
            except Exception as e:
                last_error = e
                err_str = str(e)
                print(f"\n[API ERROR - Attempt {attempt}/{max_attempts}] {err_str}")

                if ("503" in err_str or "429" in err_str) and attempt < max_attempts:
                    time.sleep(RETRY_BACKOFF_SECONDS)
                    continue
                break

        self.failed.emit(str(last_error))


class DraggableCard(QFrame):
    """Standalone draggable HUD card that positions cleanly on any monitor.

    Supports an optional 'exclusion rect' (in GLOBAL screen coordinates) that
    the card will not be allowed to overlap while being manually dragged.
    This prevents the user from accidentally dragging the HUD on top of the
    watched region, which would otherwise be picked up as a screen change by
    check_region_change() and cause a false "new panel" trigger.
    """
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFrameShape(QFrame.Shape.StyledPanel)

        self.setStyleSheet("""
            QFrame {
                background-color: rgba(15, 18, 28, 0.92);
                border: 1px solid rgba(0, 255, 122, 0.4);
                border-radius: 8px;
            }
            QLabel {
                color: #FFFFFF;
                border: none;
                background: transparent;
            }
        """)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 8, 10, 8)

        self.title_label = QLabel(title)
        self.title_label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.title_label.setStyleSheet("color: #00FF7A; border-bottom: 1px solid rgba(0,255,122,0.3); padding-bottom: 4px;")
        self.main_layout.addWidget(self.title_label)

        self.drag_position = None

        # --- drag-clamping state (used while the user is manually dragging) ---
        self.exclusion_rect = None   # QRect in GLOBAL screen coords, or None
        self.base_width = 380        # width to return to when not overlapping
        self.min_width = 280         # never shrink narrower than this while dragging

        # --- content-growth clamping state (used when the card resizes
        # itself due to new text/stats, not from being dragged) ---
        self.screen_geo = None
        self.avoid_padding = 20

    def set_reposition_context(self, screen_geo, padding=20):
        """Must be called (e.g. from setup_positions) before enforce_no_overlap()
        can do anything - it needs to know the target monitor's bounds to
        know which direction has room to move into."""
        self.screen_geo = screen_geo
        self.avoid_padding = padding

    def enforce_no_overlap(self):
        """Call this any time the card's size may have changed on its own
        (e.g. after adjustSize() when new text/stats came in) - NOT just
        while dragging. adjustSize()-driven growth doesn't go through
        mouseMoveEvent/_clamp() at all, so without this a card that grows
        taller after the fact could silently creep into the watched
        selection with nothing to stop it. Only repositions (never changes
        width) so it doesn't fight with the drag-clamp's width logic."""
        if not self.exclusion_rect or not self.screen_geo:
            return

        guard = self.exclusion_rect.adjusted(-10, -10, 10, 10)
        current = QRect(self.pos(), self.size())
        if not current.intersects(guard):
            return  # already clear at the current position/size - nothing to do

        padding = self.avoid_padding
        # Prefer moving straight down, below the exclusion rect, if there's room.
        below_y = guard.bottom() + padding
        if below_y + self.height() <= self.screen_geo.bottom():
            self.move(self.x(), below_y)
            return
        # Otherwise try straight up, above the exclusion rect.
        above_y = guard.top() - padding - self.height()
        if above_y >= self.screen_geo.top():
            self.move(self.x(), above_y)
            return
        # Last resort: hop past the exclusion rect's right edge (same escape
        # hatch already used for the drag-clamp case).
        self.move(guard.right() + padding, self.y())

    def set_exclusion_rect(self, rect):
        """rect must be in GLOBAL screen coordinates (e.g. via mapToGlobal)."""
        self.exclusion_rect = rect

    def _clamp(self, pos):
        """Given a candidate top-left position (global coords), return an
        adjusted (pos, width) that avoids the exclusion rect."""
        if not self.exclusion_rect:
            return pos, self.base_width

        width = self.base_width
        candidate = QRect(pos, QSize(width, max(self.height(), 1)))
        guard = self.exclusion_rect.adjusted(-10, -10, 10, 10)

        if candidate.intersects(guard):
            if pos.x() + width / 2 < self.exclusion_rect.center().x():
                # Card is approaching/overlapping from the left: shrink it so
                # its right edge stops just before the watched region.
                width = self.exclusion_rect.left() - pos.x() - 10
                if width < self.min_width:
                    # Not enough room to just shrink - hop to the right side instead.
                    pos = QPoint(self.exclusion_rect.right() + 10, pos.y())
                    width = self.base_width
            else:
                # Card is approaching/overlapping from the right or from inside:
                # hop it past the region's right edge.
                pos = QPoint(self.exclusion_rect.right() + 10, pos.y())

        return pos, width

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton and self.drag_position is not None:
            new_pos = event.globalPosition().toPoint() - self.drag_position
            new_pos, new_width = self._clamp(new_pos)
            self.setFixedWidth(new_width)
            self.move(new_pos)
            event.accept()


class WatchedRegionOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.box_rect = QRect()

    def update_region(self, rect):
        self.box_rect = rect
        screens = QApplication.screens()
        combined_geometry = screens[0].geometry()
        for screen in screens[1:]:
            combined_geometry = combined_geometry.united(screen.geometry())
        self.setGeometry(combined_geometry)
        self.show()
        # Must be called after show() creates the native window (winId is
        # only valid once the widget has an actual HWND).
        exclude_from_capture(self)
        self.update()

    def paintEvent(self, event):
        if self.box_rect.isEmpty():
            return
        painter = QPainter(self)
        pen = QPen(QColor(0, 255, 122), 3, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        painter.drawRect(self.box_rect)


class BubbleOverlay(QWidget):
    """Transparent, click-through overlay that paints translated text directly
    on top of each detected speech bubble, in place over the live selection.

    Operates in the same LOCAL coordinate space as WatchedRegionOverlay /
    MangaSnipper (both span the full virtual desktop starting at the combined
    screen geometry's top-left), so bubble rects computed against
    MangaSnipper.active_rect can be used here unmodified.
    """
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.bubbles = []  # list of {"rect": QRect(local), "text": str, "bg_color": QColor}

    def _sync_geometry(self):
        screens = QApplication.screens()
        combined_geometry = screens[0].geometry()
        for screen in screens[1:]:
            combined_geometry = combined_geometry.united(screen.geometry())
        self.setGeometry(combined_geometry)

    def set_bubbles(self, bubbles):
        self.bubbles = bubbles
        self._sync_geometry()
        self.show()
        # Critical: without this, check_region_change() screenshots this
        # overlay's own rendered text and mistakes it for a new panel.
        exclude_from_capture(self)
        self.update()

    def clear(self):
        self.bubbles = []
        self.update()
        self.hide()

    @staticmethod
    def _fit_font(text, rect, max_pt=30, min_pt=7):
        """Largest point size (within range) whose word-wrapped bounding box
        fits inside rect (minus a small inset)."""
        inner = QRect(0, 0, max(1, rect.width() - 8), max(1, rect.height() - 8))
        font = QFont("Segoe UI", min_pt, QFont.Weight.Bold)
        for pt in range(max_pt, min_pt - 1, -1):
            font.setPointSize(pt)
            metrics = QFontMetrics(font)
            bounds = metrics.boundingRect(
                inner, int(Qt.TextFlag.TextWordWrap) | int(Qt.AlignmentFlag.AlignCenter), text
            )
            if bounds.height() <= inner.height() and bounds.width() <= inner.width():
                font.setPointSize(pt)
                return font
        return font  # fell through: use smallest size anyway

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        for bubble in self.bubbles:
            rect = bubble["rect"]
            bg_color = bubble.get("bg_color") or QColor(255, 255, 255)

            # Cover the original bubble with its (sampled) background color.
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(bg_color)
            painter.drawRoundedRect(rect, 10, 10)

            # Draw the translated text, auto-sized to fit.
            font = self._fit_font(bubble["text"], rect)
            painter.setFont(font)
            # Pick readable text color based on background luminance.
            luminance = 0.299 * bg_color.red() + 0.587 * bg_color.green() + 0.114 * bg_color.blue()
            painter.setPen(QColor(20, 20, 20) if luminance > 128 else QColor(240, 240, 240))
            painter.drawText(
                rect.adjusted(4, 4, -4, -4),
                int(Qt.TextFlag.TextWordWrap) | int(Qt.AlignmentFlag.AlignCenter),
                bubble["text"],
            )


class TranslationHUD:
    """Manages the standalone translation card and stats card."""
    def __init__(self):
        # Create each card as an independent top-level window
        self.translation_card = DraggableCard("TRANSLATION RESULT")
        self.trans_label = QLabel("Waiting for page selection...", self.translation_card)
        self.trans_label.setFont(QFont("Segoe UI", 11))
        self.trans_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.trans_label.setWordWrap(True)
        self.translation_card.main_layout.addWidget(self.trans_label)

        self.stats_card = DraggableCard("API STATS & STATUS")
        self.stats_label = QLabel("Requests: 0 | Success: 0 | Errors: 0\nStatus: Idle", self.stats_card)
        self.stats_label.setWordWrap(True)
        self.stats_label.setFont(QFont("Segoe UI", 9))
        self.stats_card.setStyleSheet(self.stats_card.styleSheet().replace("#00FF7A", "#00E5FF"))
        self.stats_card.title_label.setStyleSheet("color: #00E5FF; border-bottom: 1px solid rgba(0, 229, 255, 0.3); padding-bottom: 4px;")
        self.stats_card.main_layout.addWidget(self.stats_label)

        self.api_requests = 0
        self.api_success = 0
        self.api_errors = 0
        self.target_width = 380
        # Stage breakdown from the most recently completed request, shown in
        # the stats card so you can see exactly where the time is going
        # (image prep vs. the network+model call vs. parsing vs. drawing).
        self.last_timings = None
        # Persistent request-lifecycle status text ("Sending API Request...",
        # "Monitoring screen...", "API Error", etc.) - kept separate from
        # the live watcher-state line below, since they update at different
        # granularities (per-request vs. every watcher tick).
        self.status_msg = "Idle"
        # Live, tick-by-tick readout of what the watcher is currently doing -
        # in particular, whether it's about to send an API request or is
        # deliberately holding off (settling after a scroll, or the frame
        # hasn't actually changed since the last translation) so it's
        # obvious at a glance that API calls aren't being fired needlessly
        # while you're just scrolling or reading.
        self.watch_state_text = "Idle"
        self.watch_state_color = "#888888"
        self._refresh_stats_label()

    def _refresh_stats_label(self):
        lines = [f"Requests: {self.api_requests} | Success: {self.api_success} | Errors: {self.api_errors}"]
        lines.append(
            f'<span style="color:{self.watch_state_color};">&#9679; {self.watch_state_text}</span>'
        )
        lines.append(f"Status: {self.status_msg}")
        if self.last_timings:
            t = self.last_timings
            lines.append(
                f"Prep {t.get('prep_ms', 0):.0f}ms &middot; API {t.get('api_ms', 0):.0f}ms &middot; "
                f"Parse {t.get('parse_ms', 0):.0f}ms"
            )
            lines.append(
                f"Render {t.get('render_ms', 0):.0f}ms &middot; Total {t.get('total_ms', 0):.0f}ms"
            )
        self.stats_label.setText("<br>".join(lines))
        self.stats_card.adjustSize()
        self.stats_card.enforce_no_overlap()

    def set_watch_state(self, text, color="#888888"):
        """Update the live watcher-state readout. Called on essentially every
        watcher tick so the indicator tracks in near real time whether the
        program is about to fire an API request or is intentionally
        skipping one."""
        self.watch_state_text = text
        self.watch_state_color = color
        self._refresh_stats_label()

    def setup_positions(self, rect):
        """rect must be in GLOBAL screen coordinates."""
        # 1. Identify which physical monitor contains the user's selection center
        target_screen = QApplication.screenAt(rect.center())
        if not target_screen:
            target_screen = QApplication.primaryScreen()

        screen_geo = target_screen.geometry()
        padding = 20

        # 2. Dynamically compute translation card width based on available left margin
        left_margin = rect.left() - screen_geo.left() - (padding * 2)
        ideal_width = int(screen_geo.width() * 0.25)

        if left_margin > 200:
            self.target_width = min(ideal_width, int(left_margin * 0.8))
        else:
            self.target_width = max(320, ideal_width)

        self.target_width = max(280, min(500, self.target_width))

        # 3. Position Translation Card -> Top-Left of the TARGET monitor
        t_x = screen_geo.left() + padding
        t_y = screen_geo.top() + padding
        self.translation_card.base_width = self.target_width
        self.translation_card.set_exclusion_rect(rect)
        self.translation_card.set_reposition_context(screen_geo, padding)
        self.translation_card.setFixedWidth(self.target_width)
        self.translation_card.move(t_x, t_y)

        # 4. Position Stats Card -> Top-Right of the TARGET monitor
        s_width = min(350, int(screen_geo.width() * 0.22))
        s_x = screen_geo.right() - s_width - padding
        s_y = screen_geo.top() + padding
        self.stats_card.base_width = s_width
        self.stats_card.set_exclusion_rect(rect)
        self.stats_card.set_reposition_context(screen_geo, padding)
        self.stats_card.setFixedWidth(s_width)
        self.stats_card.move(s_x, s_y)

        self.translation_card.show()
        self.stats_card.show()
        # HUD cards sit outside active_rect by construction (exclusion_rect
        # clamping keeps them from overlapping it), so excluding them from
        # capture isn't strictly required for the false-trigger bug - but it
        # costs nothing and protects against edge cases (e.g. a very wide
        # card on a small monitor) where they might clip into the region.
        exclude_from_capture(self.translation_card)
        exclude_from_capture(self.stats_card)

        # Content hasn't been set yet at this point, but this covers the
        # (rare) case where the placeholder text alone is already tall
        # enough to reach into a selection positioned very close to a
        # screen edge.
        self.translation_card.enforce_no_overlap()
        self.stats_card.enforce_no_overlap()

    def set_translation(self, text):
        self.trans_label.setText(text)

        # IMPORTANT: do not call trans_label.adjustSize() here. adjustSize()
        # resizes the label to its *unconstrained* sizeHint (i.e. as if it
        # were laid out on one line), which ignores word-wrapping and made
        # the card grow wider every time a longer translation came in - wide
        # enough to intrude into the watched selection and cause false
        # "new panel" triggers. Instead we fix the label's width explicitly
        # and let heightForWidth tell us how tall it needs to be once
        # wrapped at that width.
        margins = self.translation_card.main_layout.contentsMargins()
        content_width = max(50, self.target_width - margins.left() - margins.right())

        self.trans_label.setFixedWidth(content_width)
        needed_height = self.trans_label.heightForWidth(content_width)
        if needed_height <= 0:
            needed_height = self.trans_label.sizeHint().height()
        self.trans_label.setFixedHeight(needed_height)

        self.translation_card.setFixedWidth(self.target_width)
        self.translation_card.adjustSize()  # only height changes now
        self.translation_card.enforce_no_overlap()

    def update_stats(self, status_msg="Idle", req_inc=0, succ_inc=0, err_inc=0, timings=None):
        self.api_requests += req_inc
        self.api_success += succ_inc
        self.api_errors += err_inc
        self.status_msg = status_msg
        if timings is not None:
            self.last_timings = timings
        self._refresh_stats_label()

    def hide(self):
        self.translation_card.hide()
        self.stats_card.hide()

    def isVisible(self):
        return self.translation_card.isVisible() or self.stats_card.isVisible()


class MangaSnipper(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )

        self.begin = QPoint()
        self.end = QPoint()
        self.is_selecting = False
        self.desktop_pixmap = None
        self.full_pil_image = None

        self.active_rect = None
        self.last_translated_np = None
        self.stable_frame_count = 0
        self.last_frame_np = None
        self.is_translating = False

        self.translation_generation = 0

        self.pending_dispatch_np = None

        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.worker = None
        self.border_overlay = WatchedRegionOverlay()
        self.bubble_overlay = BubbleOverlay()
        self.hud = TranslationHUD()
        self.current_crop_for_sampling = None  # pre-resize PIL crop used to sample bubble bg colors
        self.hide()

        # Global Hotkey Listeners
        self.hotkey_listener = HotkeyListener(ACTIVATION_HOTKEY)
        self.hotkey_listener.triggered.connect(self.activate)
        self.hotkey_listener.start()

        # ESC key toggles between hiding overlays and re-opening selection mode
        self.esc_listener = HotkeyListener("esc")
        self.esc_listener.triggered.connect(self.toggle_esc)
        self.esc_listener.start()

        self.quit_listener = HotkeyListener(QUIT_HOTKEY)
        self.quit_listener.triggered.connect(self.quit_app)
        self.quit_listener.start()

        self.watch_timer = QTimer(self)
        self.watch_timer.setInterval(WATCH_INTERVAL_MS)
        self.watch_timer.timeout.connect(self.check_region_change)
        self.watch_timer.start()

        self.setup_tray_icon()

    def setup_tray_icon(self):
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))

        tray_menu = QMenu()
        snip_action = QAction("Resnip Region (Alt+S)", self)
        snip_action.triggered.connect(self.activate)
        quit_action = QAction("Quit (Alt+Q)", self)
        quit_action.triggered.connect(self.quit_app)

        tray_menu.addAction(snip_action)
        tray_menu.addSeparator()
        tray_menu.addAction(quit_action)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

    def toggle_esc(self):
        """If overlays are visible, hide them. If hidden, re-trigger selection mode."""
        if self.isVisible() or self.border_overlay.isVisible() or self.hud.isVisible():
            print("[INFO] ESC pressed: Hiding selection overlays.")
            self.active_rect = None
            self.border_overlay.hide()
            self.bubble_overlay.clear()
            self.hud.hide()
            self.hud.set_watch_state("Idle", WATCH_COLOR_IDLE)
            self.hide()
        else:
            print("[INFO] ESC pressed: Re-opening region selector.")
            self.activate()

    def quit_app(self):
        """Completely closes and shuts down the application."""
        print("[INFO] Alt+Q pressed: Quitting application completely...")
        self.watch_timer.stop()
        self.border_overlay.hide()
        self.bubble_overlay.clear()
        self.hud.hide()
        self.hide()
        QApplication.instance().quit()

    def activate(self):
        self.border_overlay.hide()
        self.bubble_overlay.clear()
        self.hud.hide()
        self.begin = QPoint()
        self.end = QPoint()
        self.is_selecting = False

        screens = QApplication.screens()
        combined_geometry = screens[0].geometry()
        for screen in screens[1:]:
            combined_geometry = combined_geometry.united(screen.geometry())
        self.setGeometry(combined_geometry)

        with mss.mss() as sct:
            monitor = sct.monitors[0]
            sct_img = sct.grab(monitor)
            self.full_pil_image = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

        bytes_img = self.full_pil_image.tobytes("raw", "RGB")
        qimage = QImage(bytes_img, self.full_pil_image.width, self.full_pil_image.height, QImage.Format.Format_RGB888)
        self.desktop_pixmap = QPixmap.fromImage(qimage)

        self.show()

    def showEvent(self, event):
        super().showEvent(event)
        self.raise_()
        self.activateWindow()
        self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.desktop_pixmap:
            painter.drawPixmap(self.rect(), self.desktop_pixmap)

        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))

        if self.is_selecting and not self.begin.isNull() and not self.end.isNull():
            rect = QRect(self.begin, self.end).normalized()
            if self.desktop_pixmap:
                painter.drawPixmap(rect, self.desktop_pixmap, rect)

            pen = QPen(QColor(255, 0, 0), 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(rect)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.begin = event.pos()
            self.end = event.pos()
            self.is_selecting = True
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_selecting:
            self.end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.is_selecting:
            self.is_selecting = False
            self.end = event.pos()
            rect = QRect(self.begin, self.end).normalized()
            self.hide()

            if rect.width() > 5 and rect.height() > 5:
                #converting coordinates
                global_rect = QRect(self.mapToGlobal(rect.topLeft()), rect.size())

                self.active_rect = rect  # local coords: matches mss's monitor[0] space used in check_region_change
                self.last_translated_np = None
                self.last_frame_np = None
                self.stable_frame_count = 0

                self.border_overlay.update_region(rect)
                self.bubble_overlay.clear()
                self.hud.setup_positions(global_rect)
                self.hud.set_translation("Translating selection...")

                self.trigger_translation_from_rect(rect)

    def check_region_change(self):
        if not self.active_rect or self.is_selecting or self.isVisible():
            return

        dpr = self.devicePixelRatioF()
        rect = self.active_rect

        # 1. Grab full virtual desktop once (foolproof against multi-monitor offset bugs)
        with mss.mss() as sct:
            monitor = sct.monitors[0]
            sct_img = sct.grab(monitor)
            live_pil = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

        # 2. Crop the active selection box using exact coordinates (same as initial snip)
        crop_box = (
            int(rect.left() * dpr),
            int(rect.top() * dpr),
            int(rect.right() * dpr),
            int(rect.bottom() * dpr)
        )

        # Ensure crop box stays within image bounds
        crop_box = (
            max(0, crop_box[0]),
            max(0, crop_box[1]),
            min(live_pil.width, crop_box[2]),
            min(live_pil.height, crop_box[3])
        )

        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            return

        cropped_image = live_pil.crop(crop_box)
        current_np = np.array(cropped_image.convert("RGB"), dtype=np.uint8)

        # 3. Mask out boundary margins
        h, w, _ = current_np.shape
        b_pad = int(4 * dpr)
        if h > b_pad * 2 and w > b_pad * 2:
            current_np[:b_pad, :, :] = 0
            current_np[-b_pad:, :, :] = 0
            current_np[:, :b_pad, :] = 0
            current_np[:, -b_pad:, :] = 0

        # Initial baseline assignment
        if self.last_translated_np is None:
            self.last_translated_np = current_np.copy()
            self.last_frame_np = current_np.copy()
            self.hud.set_watch_state("Watching...", WATCH_COLOR_IDLE)
            return

        if self.last_frame_np is None or current_np.shape != self.last_frame_np.shape:
            self.last_frame_np = current_np.copy()
            return

        # 4. Compute pixel differences using Grayscale Intensity Thresholding
        curr_gray = np.dot(current_np[..., :3], [0.2989, 0.5870, 0.1140])
        prev_gray = np.dot(self.last_frame_np[..., :3], [0.2989, 0.5870, 0.1140])
        last_trans_gray = np.dot(self.last_translated_np[..., :3], [0.2989, 0.5870, 0.1140])

        diff_tick = np.abs(curr_gray - prev_gray)
        changed_tick_pixels = np.count_nonzero(diff_tick > 25)

        self.last_frame_np = current_np.copy()

        # If a request is already in flight, say so plainly rather than
        # reporting on motion/stability underneath it - the busy state is
        # what actually determines whether a new request could go out.
        if self.is_translating:
            if self.pending_dispatch_np is not None:
                self.hud.set_watch_state("Translating... (next page queued)", WATCH_COLOR_QUEUED)
            else:
                self.hud.set_watch_state("Translating...", WATCH_COLOR_SENDING)

        # 4b. Movement detection: the instant the frame is changing a lot
        # (scrolling / page turn in progress), immediately clear the old
        # translated bubbles rather than leaving them to be painted
        if changed_tick_pixels > MOVEMENT_CLEAR_PIXEL_THRESHOLD:
            if self.bubble_overlay.bubbles:
                self.bubble_overlay.clear()
                self.hud.set_translation("Scrolling...")
            self.stable_frame_count = 0
            if not self.is_translating:
                self.hud.set_watch_state("Scrolling - not sending", WATCH_COLOR_SCROLLING)
        elif changed_tick_pixels < QUIET_TICK_PIXEL_THRESHOLD:
            self.stable_frame_count += 1
            if not self.is_translating and self.stable_frame_count < STABLE_TICKS_REQUIRED:
                self.hud.set_watch_state("Settling...", WATCH_COLOR_SETTLING)
        else:
            self.stable_frame_count = 0
            if not self.is_translating:
                self.hud.set_watch_state("Watching...", WATCH_COLOR_IDLE)

        # 5. Trigger check once stable for STABLE_TICKS_REQUIRED consecutive ticks
        if self.stable_frame_count >= STABLE_TICKS_REQUIRED:
            if curr_gray.shape != last_trans_gray.shape:
                self.last_translated_np = current_np.copy()
                last_trans_gray = curr_gray.copy()

            diff_translated = np.abs(curr_gray - last_trans_gray)
            changed_trans_pixels = np.count_nonzero(diff_translated > 25)

            # Percentage-based floor in addition to the flat 200px floor:
            # a fixed pixel count is trivially exceeded by ordinary capture
            # jitter/antialiasing on large selections, so scale the bar with
            # the size of the watched region.
            total_px = curr_gray.size
            trigger_threshold = max(200, int(total_px * 0.0015))

            print(f"[WATCHER] Screen Stable. Changed Pixels vs Last Translation: {changed_trans_pixels} "
                  f"(threshold: {trigger_threshold})")

            if changed_trans_pixels > trigger_threshold:
                self.stable_frame_count = 0
                self.last_translated_np = current_np.copy()
                print(f"[WATCHER TRIGGER] New text/panel detected ({changed_trans_pixels} pixels changed). Requesting API...")

                if self.is_translating:
                    # A request is already in flight for a previous frame.
                    # Rather than waiting for the next watcher tick after it
                    # completes, stash this newer frame so it's dispatched
                    # the instant the current one finishes.
                    print("[WATCHER] Request already in flight - queuing this frame as pending.")
                    self.pending_dispatch_np = current_np.copy()
                    self.bubble_overlay.clear()
                    self.hud.set_translation("New panel detected. Translating...")
                    self.hud.set_watch_state("Translating... (next page queued)", WATCH_COLOR_QUEUED)
                else:
                    self.bubble_overlay.clear()
                    self.hud.set_translation("New panel detected. Translating...")
                    self.hud.set_watch_state("Sending request...", WATCH_COLOR_SENDING)
                    self.trigger_translation_from_np(current_np)
            elif not self.is_translating:
                # Stable, but the page hasn't actually changed since the
                # last translation - explicitly confirms no API call is
                # being wasted here.
                self.hud.set_watch_state("Up to date - no request needed", WATCH_COLOR_UNCHANGED)

    def trigger_translation_from_rect(self, rect):
        dpr = self.devicePixelRatioF()
        crop_box = (
            int(rect.left() * dpr),
            int(rect.top() * dpr),
            int(rect.right() * dpr),
            int(rect.bottom() * dpr)
        )
        cropped_image = self.full_pil_image.crop(crop_box)
        self.last_translated_np = np.array(cropped_image.convert("RGB"))
        self.dispatch_image_to_worker(cropped_image)

    def trigger_translation_from_np(self, np_array):
        pil_img = Image.fromarray(np_array[:, :, :3])
        self.dispatch_image_to_worker(pil_img)

    def dispatch_image_to_worker(self, pil_image):
        if not GEMINI_API_KEY or self.is_translating:
            return

        self.is_translating = True
        self.translation_generation += 1
        my_generation = self.translation_generation
        print(f"\n[API REQUEST #{self.hud.api_requests + 1}] Dispatching image crop to Gemini API "
              f"(generation {my_generation})...")
        self.hud.update_stats(status_msg="Sending API Request...", req_inc=1)
        self.hud.set_watch_state("Sending request...", WATCH_COLOR_SENDING)

        # Keep the un-resized crop around for bubble background-color sampling
        # once results come back (normalized bbox fractions apply the same
        # regardless of what resolution was actually sent to the API).
        self.current_crop_for_sampling = pil_image

        # Resize + encode now happen inside TranslationWorker.run(), i.e. on
        # the worker thread instead of here on the GUI thread. That keeps
        # the UI (and the 250ms watcher tick) from stalling on image prep
        # for every single request, and lets that stage be timed separately.
        self.worker = TranslationWorker(pil_image, generation=my_generation)
        self.worker.finished.connect(self.on_translation_done)
        self.worker.failed.connect(self.on_translation_failed)
        self.worker.start()

    def _dispatch_pending_if_any(self):
        """Called once a request finishes. If a newer frame arrived while we
        were busy, send it immediately instead of waiting for the next
        watcher tick - keeps fast page-turning/scrolling from feeling like
        it "missed" a page. Returns True if a pending frame was dispatched."""
        if self.pending_dispatch_np is not None:
            pending = self.pending_dispatch_np
            self.pending_dispatch_np = None
            self.trigger_translation_from_np(pending)
            return True
        return False

    def on_translation_done(self, result):
        self.is_translating = False
        if result.get("generation") != self.translation_generation:
            # Stale result from a request that's since been superseded
            # (e.g. the user scrolled further before this one came back).
            # Drop it rather than flashing an outdated translation.
            print("[API SUCCESS] Discarding stale result (superseded by a newer page).")
            if not self._dispatch_pending_if_any():
                self.hud.set_watch_state("Watching...", WATCH_COLOR_IDLE)
            return

        bubbles = result.get("bubbles", [])
        plain_text = result.get("plain_text", "No text found in image.")
        print(f"[API SUCCESS] {len(bubbles)} bubble(s) translated.")
        self.hud.set_translation(plain_text)

        timings = dict(result.get("timings", {}))
        render_start = time.time()
        self.render_bubbles(bubbles)
        timings["render_ms"] = (time.time() - render_start) * 1000
        timings["total_ms"] = timings.get("total_ms", 0) + timings["render_ms"]

        self.hud.update_stats(status_msg="Monitoring screen...", succ_inc=1, timings=timings)
        if not self._dispatch_pending_if_any():
            self.hud.set_watch_state("Watching...", WATCH_COLOR_IDLE)

    def on_translation_failed(self, error_text):
        self.is_translating = False
        print(f"[API FAILED] Request finished with error:\n{error_text}\n")
        self.hud.set_translation(f"Error: {error_text}")
        self.hud.update_stats(status_msg="API Error", err_inc=1)
        if not self._dispatch_pending_if_any():
            self.hud.set_watch_state("Watching...", WATCH_COLOR_IDLE)
        # Leave any previously rendered in-place bubbles showing rather than
        # clearing them on a transient error.

    def render_bubbles(self, bubbles):
        """Map each bubble's normalized bbox onto screen coordinates within
        self.active_rect and hand the result to the BubbleOverlay to paint."""
        if not self.active_rect or not bubbles:
            self.bubble_overlay.clear()
            return

        rect = self.active_rect
        sample_img = self.current_crop_for_sampling
        rendered = []

        for b in bubbles:
            try:
                y0, x0, y1, x1 = b["bbox_2d"]
            except (KeyError, ValueError):
                continue

            x0f, y0f = min(x0, x1) / 1000.0, min(y0, y1) / 1000.0
            x1f, y1f = max(x0, x1) / 1000.0, max(y0, y1) / 1000.0

            bubble_rect = QRect(
                rect.left() + int(x0f * rect.width()),
                rect.top() + int(y0f * rect.height()),
                max(10, int((x1f - x0f) * rect.width())),
                max(10, int((y1f - y0f) * rect.height())),
            )
            bg_color = self._sample_bubble_color(sample_img, x0f, y0f, x1f, y1f)
            rendered.append({"rect": bubble_rect, "text": b["text"], "bg_color": bg_color})

        self.bubble_overlay.set_bubbles(rendered)

    @staticmethod
    def _sample_bubble_color(pil_image, x0f, y0f, x1f, y1f):
        """Median RGB color within the bubble's box in the original crop -
        a simple, fast approximation of the bubble's background fill that
        works reasonably well since background pixels usually outnumber
        text-stroke pixels within the box."""
        if pil_image is None:
            return QColor(255, 255, 255)

        w, h = pil_image.size
        box = (
            max(0, int(x0f * w)), max(0, int(y0f * h)),
            min(w, int(x1f * w)), min(h, int(y1f * h)),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            return QColor(255, 255, 255)

        crop = np.array(pil_image.crop(box).convert("RGB"))
        median = np.median(crop.reshape(-1, 3), axis=0)
        return QColor(int(median[0]), int(median[1]), int(median[2]))


class HotkeyListener(QThread):
    triggered = pyqtSignal()

    def __init__(self, hotkey):
        super().__init__()
        self.hotkey = hotkey

    def run(self):
        keyboard.add_hotkey(self.hotkey, self.triggered.emit)
        keyboard.wait()


def main():
    app = QApplication(sys.argv)
    snipper = MangaSnipper()
    print("Manga Snipper controls:")
    print(f"  {ACTIVATION_HOTKEY.upper()} / ESC: Toggle region selection mode")
    print(f"  {QUIT_HOTKEY.upper()}: Quit program completely")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()