"""Reading the EXP and meso numbers at any size.

Some setups draw the game bigger than normal, and not always evenly (the
text can grow more than the bars around it), so the pixel-exact reader
can't match. This reader doesn't care about size: it splits the number into
characters by the blank columns between them, shrinks each character to a
standard box, and compares its shape with our known characters shrunk the
same way. Punctuation is told apart by its size and position.

Used only when the exact reader finds nothing.
"""
import re

import cv2
import numpy as np

BOX = (12, 16)                    # (w, h) each character is normalised to
TEXT_RE = re.compile(r'^(\d{1,3}(?:,\d{3})*)\[(\d{1,3})\.(\d{2})%\]$')
MESO_RE = re.compile(r'^\d{1,3}(,\d{3})*$')


def _norm(img):
    """Tight-crop a glyph (float 0..1, ink=1) and resize it to BOX."""
    ys, xs = np.where(img > 0.35)
    if len(ys) == 0:
        return None
    g = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    g = cv2.resize(g, BOX, interpolation=cv2.INTER_AREA)
    g -= g.mean()
    n = np.linalg.norm(g)
    return g / n if n > 1e-6 else None


def _digit_templates(bitmaps):
    """{char: [normalised templates]} for digits from 0/1 bitmaps."""
    out = {}
    for ch, bms in bitmaps.items():
        if not (ch.isdigit() or ch in "[]"):
            continue
        for bm in bms:
            b = bm.astype(np.float32)
            rows = np.where(b.any(1))[0]
            b = b[rows.min():rows.max() + 1]
            cols = np.where(b.any(0))[0]
            b = b[:, cols.min():cols.max() + 1]
            # render bigger first so shrinking to BOX behaves like a real glyph
            b = cv2.resize(b, (b.shape[1] * 6, b.shape[0] * 6), interpolation=cv2.INTER_NEAREST)
            t = _norm(cv2.GaussianBlur(b, (0, 0), 2.0))
            if t is not None:
                out.setdefault(ch, []).append(t)
    return out


def _segments(ink, min_gap=1):
    """Column runs of ink (start, end), split at blank columns."""
    cols = ink.max(0) > 0.35
    segs, x = [], 0
    while x < len(cols):
        if cols[x]:
            s = x
            while x < len(cols) and cols[x]:
                x += 1
            segs.append([s, x])
        else:
            x += 1
    return segs


def read_line(ink, templates, kind):
    """Read one line of text. ink: float 0..1 (1 = text), rows = the line.

    kind 'exp' expects 123,456[12.34%], kind 'meso' expects 1,234,567.
    Returns the text or None.
    """
    segs = _segments(ink)
    if len(segs) < 2:
        return None
    rows_any = np.where(ink.max(1) > 0.35)[0]
    if len(rows_any) == 0:
        return None
    # digit height = the most common glyph height among the wider glyphs
    info = []
    for s, e in segs:
        g = ink[:, s:e]
        ys = np.where(g.max(1) > 0.35)[0]
        info.append((s, e, ys.min(), ys.max()))
    heights = sorted(b - a + 1 for s, e, a, b in info if e - s >= 3)
    if not heights:
        return None
    dh = heights[len(heights) // 2]
    tops = sorted(a for s, e, a, b in info if e - s >= 3 and abs((b - a + 1) - dh) <= max(1, dh * 0.2))
    if not tops:
        return None
    top = tops[len(tops) // 2]
    base = top + dh - 1                          # digit baseline
    widths = sorted(e - s for s, e, a, b in info if abs((b - a + 1) - dh) <= max(1, dh * 0.2))
    dw = widths[len(widths) // 2] if widths else dh * 0.7

    text = ''
    for s, e, a, b in info:
        if kind == 'exp' and text.endswith(']'):
            break
        w, h = e - s, b - a + 1
        if h <= dh * 0.65 and w <= dw * 0.6:
            # small mark: in the EXP number and the meso total it can only be
            # a comma; inside the [..%] it can only be the decimal point
            text += '.' if (kind == 'exp' and '[' in text) else ','
            continue
        if w >= dw * 1.25 and h >= dh * 0.8:
            text += '%'
            continue
        n = _norm(ink[a:b + 1, s:e])
        if n is None:
            return None
        narrow = w <= dw * 0.6
        pool = "1[]" if narrow else "0123456789"
        if kind == 'meso':
            pool = "1" if narrow else "0123456789"
        best = sorted(((max(float((n * t).sum()) for t in ts), ch)
                       for ch, ts in templates.items() if ch in pool), reverse=True)
        if not best or best[0][0] < 0.5:
            return None
        if len(best) > 1 and best[0][0] - best[1][0] < 0.03:
            return None
        text += best[0][1]
    if kind == 'exp':
        if ']' in text:
            text = text[:text.index(']') + 1]
        if not TEXT_RE.match(text):
            # a faint decimal point can vanish; the % always has 2 decimals
            m = re.match(r'^(\d{1,3}(?:,\d{3})*)\[(\d{3,4})%\]$', text)
            if m:
                d = m.group(2)
                text = f"{m.group(1)}[{d[:-2]}.{d[-2:]}%]"
        return text if TEXT_RE.match(text) else None
    return text if MESO_RE.match(text) else None


def _ink(region):
    """Brightness of light text on a dark panel as 0..1 'ink'."""
    g = region.max(2).astype(np.float32)
    lo, hi = np.percentile(g, 30), np.percentile(g, 99.7)
    if hi - lo < 50:
        return None
    return np.clip((g - lo) / (hi - lo), 0, 1)


def _text_rows(ink):
    """The rows of the text line sitting just above the bottom of `ink`."""
    frac = (ink > 0.5).mean(1)
    y = len(frac) - 1
    while y >= 0 and frac[y] >= 0.4:          # the bar's own bright border
        y -= 1
    while y >= 0 and frac[y] < 0.02:          # gap between text and bar
        y -= 1
    bottom = y
    while y >= 0 and 0.02 <= frac[y] < 0.6:   # the text itself
        y -= 1
    top = y + 1
    if bottom - top + 1 < 5:
        return None
    return top, bottom + 1


def find_exp_text(frame, bar_tops, templates):
    """Read the EXP text at any size. bar_tops: (y, x) of yellow-green fills."""
    for y, x0 in bar_tops[:40]:
        if y < 20:
            continue
        # how wide is the bar (fill + empty part)? walk right along it
        row = frame[min(y + 2, frame.shape[0] - 1), x0:].max(1)
        dark = np.where(row < 70)[0]
        bar_w = int(dark[0]) if len(dark) else len(row)
        if bar_w < 40:
            continue
        x1 = min(frame.shape[1], x0 + int(bar_w * 1.05))
        region = frame[max(0, y - bar_w // 3):max(0, y - 1), max(0, x0 - 2):x1]
        if region.shape[0] < 8:
            continue
        ink = _ink(region)
        if ink is None:
            continue
        rows = _text_rows(ink)
        if not rows:
            continue
        tail = max(2, (rows[1] - rows[0]) // 3)     # room for a comma's tail
        line = ink[rows[0]:min(ink.shape[0], rows[1] + tail)]
        segs = _segments(line)
        if len(segs) < 6:
            continue
        # skip the "EXP." label: it ends at the first wide gap
        gaps = [segs[i + 1][0] - segs[i][1] for i in range(len(segs) - 1)]
        small = sorted(gaps)[len(gaps) // 3] if gaps else 1
        cut = None
        for i, gp in enumerate(gaps[:6]):
            if gp >= max(3, small * 2.5):
                cut = i + 1
                break
        if cut is None:
            continue
        start = segs[cut][0]
        text = read_line(line[:, start:], templates, 'exp')
        if text:
            m = TEXT_RE.match(text)
            exp = int(m.group(1).replace(',', ''))
            pct = int(m.group(2)) + int(m.group(3)) / 100.0
            if pct < 100:
                return exp, pct, text
    return None
