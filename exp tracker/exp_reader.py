"""Reads your character's EXP straight off the status bar.

The text above the EXP bar ("168,840[36.04%]") is drawn in a tiny pixel
font, so every character always has exactly the same pixels. We find the
yellow-green EXP bar, cut out the strip of text above it, split it into
characters and compare each one against the known bitmaps below.
"""
import re

import cv2
import numpy as np

# 7-row bitmaps of the status-bar font ('#' = lit pixel).
FONT = {
    '0': '.###./#...#/#...#/#...#/#...#/#...#/.###.',
    '1': '.#/##/.#/.#/.#/.#/.#',
    '2': '.###./#...#/....#/...#./..#../.#.../#####',
    '3': '.###./#...#/....#/..##./....#/#...#/.###.',
    '4': '...#./..##./.#.#./#..#./#####/...#./...#.',
    '5': '#####/#..../#..../.###./....#/#...#/.###.',
    '6': '.###./#...#/#..../####./#...#/#...#/.###.',
    '7': '#####/....#/....#/....#/...#./..#../..#..',
    '8': '.###./#...#/#...#/.###./#...#/#...#/.###.',
    '9': '.###./#...#/#...#/.####/....#/#...#/.###.',
    ',': '../../../../../.#/#.',
    '.': '././././././#',
    '[': '##/#./#./#./#./#./##',
    ']': '##/.#/.#/.#/.#/.#/##',
    '%': '.#....#/#.#..#./.#..#../...#.../..#..#./.#..#.#/#....#.',
}
# The "EXP." label just left of the numbers (rows y-17..y-6, cols x0..x0+21).
# Checking it makes sure we're reading from the true start of the number,
# not from a spot shifted into the middle of it.
LABEL = np.array([[c == '#' for c in row] for row in (
    '.....................',
    '######.##..###.######',
    '##.....###.##..##..##',
    '##......#####..##..##',
    '#####...####...##..##',
    '#####....###...######',
    '##......####...###...',
    '##.....######..##....',
    '######.###.###.##..##',
    '#########..###.##..##',
    '.....................',
    '.....................')], np.uint8)
LABEL_MAX_DIFF = 20

TEMPLATES = [(ch, np.array([[c == '#' for c in row] for row in s.split('/')], np.uint8))
             for ch, s in FONT.items()]

HP_TO_EXP = 219          # EXP bar starts this far right of the HP bar
MP_TO_EXP = 113          # ... and this far right of the MP bar
REF_WIDTH = 1920          # the font is pixel-exact at this game width
INK = 150                 # brightness that counts as a lit text pixel
TEXT_RE = re.compile(r'^(\d{1,3}(?:,\d{3})*)\[(\d{1,3})\.(\d{2})%\]$')


def _bar_tops(mask, y0, min_len):
    """Top-left corners of horizontal colour bars in a mask, lowest first."""
    tops = {}
    for yy in np.where(mask.sum(1) >= min_len)[0]:
        xs = np.where(mask[yy])[0]
        sp = np.where(np.diff(xs) > 2)[0]
        for a, b in zip(np.r_[xs[0], xs[sp + 1]], np.r_[xs[sp], xs[-1]]):
            if b - a + 1 < min_len:
                continue
            a = int(a)
            if a not in tops or tops[a][1] < yy - 1:   # new bar at this x
                tops[a] = [int(yy), int(yy)]
            else:
                tops[a][1] = int(yy)
    return sorted(((y0 + t[0], x) for x, t in tops.items()), key=lambda c: -c[0])


def find_exp_bars(frame, limit=10):
    """Candidate (y, left_x) spots for the EXP bar, most likely first.

    The bar is found from its yellow-green fill, and also from the red HP
    and blue MP bars that sit on the same row at fixed distances, because
    right after a level-up the EXP fill is only a few pixels long.
    """
    h = frame.shape[0]
    y0 = int(h * 0.5)
    sub = frame[y0:].astype(np.int16)
    B, G, R = sub[..., 0], sub[..., 1], sub[..., 2]
    cands = []
    for (y, x) in _bar_tops((G > 150) & (R > 110) & (B < 90) & (G >= R), y0, 8)[:limit]:
        cands.append((y, x))
    for (y, x) in _bar_tops((B > 180) & (R < 60) & (G > 80) & (G < 200), y0, 20)[:limit]:
        cands.append((y, x + MP_TO_EXP))
    for (y, x) in _bar_tops((R > 180) & (G < 60) & (B < 60), y0, 20)[:limit]:
        cands.append((y, x + HP_TO_EXP))
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _glyphs(frame, y, x0):
    """Split the text strip above the bar into character bitmaps."""
    if y - 17 < 0:
        return None
    strip = frame[y - 17:y - 5, x0 + 22:x0 + 130]
    if strip.size == 0:
        return None
    b = (strip.max(2) > INK).astype(np.uint8)
    # find the text rows from the left part only (the bar frame / other
    # buttons can poke into the right end of the strip)
    rows = np.where(b[:, :60].sum(1) > 0)[0]
    if len(rows) == 0:
        return None
    b = b[rows.min():rows.max() + 1]
    if b.shape[0] != 7:
        return None
    cols = b.sum(0) > 0
    out, x, last_end = [], 0, None
    while x < len(cols):
        if cols[x]:
            s = x
            while x < len(cols) and cols[x]:
                x += 1
            if last_end is not None and s - last_end > 5:
                break                      # end of the text
            out.append(b[:, s:x])
            last_end = x
        else:
            x += 1
    return out


def _match(g):
    """Best character for a glyph, or None if it's unclear.

    A character is only accepted when it's clearly closer than every other
    one (e.g. a slightly blurred 8 that looks as much like a 6 or 9 is
    rejected rather than guessed).
    """
    ds = sorted((int((t != g).sum()), ch) for ch, t in TEMPLATES if t.shape == g.shape)
    if not ds or ds[0][0] > max(2, g.size // 6):
        return None
    if ds[0][0] > 0 and len(ds) > 1 and ds[1][0] - ds[0][0] < 2:
        return None
    return ds[0][1]


def _label_ok(frame, y, x0):
    if y - 17 < 0 or x0 < 0:
        return False
    c = frame[y - 17:y - 5, x0:x0 + 21]
    if c.shape[:2] != LABEL.shape:
        return False
    return int(((c.max(2) > INK).astype(np.uint8) != LABEL).sum()) <= LABEL_MAX_DIFF


def _read_at(frame, y, x0):
    if not _label_ok(frame, y, x0):
        return None
    gs = _glyphs(frame, y, x0)
    if not gs:
        return None
    text = ''
    for g in gs:
        ch = _match(g)
        if ch is None:
            return None
        text += ch
    m = TEXT_RE.match(text)
    if not m:
        return None
    exp = int(m.group(1).replace(',', ''))
    pct = int(m.group(2)) + int(m.group(3)) / 100.0
    if pct >= 100:
        return None
    return exp, pct, text


VIEWS = ("as-is", "stretched", "16:9-sharp", "1080p", "1920wide")


def view(frame, key):
    """The frame as-is, or undoing a stretch GeForce NOW may have applied.

    On a 16:10 screen (1920x1200, 1680x1050, ...) the 16:9 game picture can
    be stretched taller to fill it; squashing it back to 16:9 restores the
    status bar text. Returns None when a view would be the same as as-is.
    """
    h, w = frame.shape[:2]
    if key == "as-is":
        return frame
    if key in ("16:9", "16:9-sharp"):
        h2 = round(w * 9 / 16)
        if abs(h2 - h) < 4:
            return None
        interp = cv2.INTER_AREA if key == "16:9" else cv2.INTER_NEAREST
        return cv2.resize(frame, (w, h2), interpolation=interp)
    if key == "1080p":
        if (w, h) == (1920, 1080):
            return None
        return cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    if key == "1920wide":
        if w == REF_WIDTH:
            return None
        return cv2.resize(frame, (REF_WIDTH, round(h * REF_WIDTH / w)),
                          interpolation=cv2.INTER_AREA)
    return None


class ExpReader:
    """Remembers where the bar is, so it still reads at 0% after a level-up."""

    def __init__(self, anchor=None):
        self.anchor = tuple(anchor) if anchor else None   # (w, h, y, x0)
        self.view = "as-is"
        self.soft = None          # (w, h, y, x0) for stretched screens

    def read(self, frame):
        """Returns (exp, pct, text) or None if the EXP text isn't visible."""
        if frame is None or frame.size == 0:
            return None
        # Try the way that worked last time first, then the others.
        for key in [self.view] + [k for k in VIEWS if k != self.view]:
            if key == "stretched":
                r = self._read_stretched(frame)
            else:
                img = view(frame, key)
                r = self._read(img) if img is not None else None
            if r:
                self.view = key
                return r
        return None

    def _read_stretched(self, frame):
        sy = stretch_of(frame)
        if sy is None:
            return None
        h, w = frame.shape[:2]
        if self.soft and self.soft[:2] == (w, h):
            r = soft_read_at(frame, self.soft[2], self.soft[3], sy)
            if r:
                return r
        for y, x0 in find_exp_bars(frame):
            for dy in (0, -1, 1, -2, 2):
                r = soft_read_at(frame, y + dy, x0, sy)
                if r:
                    self.soft = (w, h, y + dy, x0)
                    return r
        return None

    def _read(self, frame):
        h, w = frame.shape[:2]
        if self.anchor and self.anchor[:2] == (w, h):
            r = _read_at(frame, self.anchor[2], self.anchor[3])
            if r:
                return r
        for y, x0 in find_exp_bars(frame):
            for dy in (0, -1, 1, -2, 2):
                r = _read_at(frame, y + dy, x0)
                if r:
                    self.anchor = (w, h, y + dy, x0)
                    return r
        return None


# --------------------------------------------------------------------------
# Stretched screens (e.g. 1920x1200)
#
# On a 16:10 screen GeForce NOW can stretch the 16:9 game picture taller.
# Widths stay exact, so the columns between characters are still blank and
# the number splits into characters perfectly; only each character gets
# vertically blurred. We then identify each one by comparing it with the
# same character stretched the same way (normalised correlation), which
# copes with the blur that exact pixel matching can't.
# --------------------------------------------------------------------------
def stretch_of(frame):
    """Vertical stretch factor if the picture looks like a stretched 16:9 one."""
    h, w = frame.shape[:2]
    sy = h / (w * 9 / 16)
    return sy if 1.02 < sy < 1.4 else None


def _soft_tpl(bitmap, sy, pad=3):
    t = np.pad(bitmap.astype(np.float32), ((pad, pad), (0, 0)))
    return cv2.resize(t, (t.shape[1], max(1, round(t.shape[0] * sy))),
                      interpolation=cv2.INTER_LINEAR)


def _ncc_best(patch, tpl):
    if patch.shape[0] < tpl.shape[0] or patch.shape[1] < tpl.shape[1]:
        return -1.0
    if tpl.std() < 1e-6:
        return -1.0
    r = cv2.matchTemplate(patch, tpl, cv2.TM_CCOEFF_NORMED)
    return float(np.nanmax(r))


def _ssd_best(patch, tpl):
    """Smallest squared difference of tpl slid up/down over patch."""
    if patch.shape[0] < tpl.shape[0] or patch.shape[1] != tpl.shape[1]:
        return None
    r = cv2.matchTemplate(patch, tpl, cv2.TM_SQDIFF)
    return float(r.min())


def soft_read_at(frame, y, x0, sy):
    """Read the EXP text of a vertically stretched frame. (exp, pct, text)."""
    top = y - int(round(22 * sy))
    if top < 0 or x0 < 0:
        return None
    band = frame[top:y, x0:x0 + 130].max(2).astype(np.float32)
    if band.shape[1] < 60:
        return None
    # find the "EXP." label rows
    lab = _soft_tpl(LABEL, sy, pad=0)
    r = cv2.matchTemplate(band[:, :21] / 255.0, lab, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(r)
    if score < 0.7:
        return None
    r0 = loc[1]
    # the text sits on label rows 2-8; keep a little margin but stay clear
    # of the bright border lines above and below
    rows = band[r0 + int(round(1.5 * sy)):r0 + int(round(9.5 * sy)), 22:]
    if rows.shape[0] < 6:
        return None
    lo, hi = np.percentile(rows, 20), np.percentile(rows, 99.5)
    if hi - lo < 60:
        return None
    norm = np.clip((rows - lo) / (hi - lo), 0, 1).astype(np.float32)
    # characters are separated by blank columns (widths are not stretched)
    cols = norm.max(0) > 0.25
    segs, x = [], 0
    while x < len(cols):
        if cols[x]:
            s = x
            while x < len(cols) and cols[x]:
                x += 1
            if segs and s - segs[-1][1] > 5:
                break
            segs.append((s, x))
        else:
            x += 1
    text = ''
    for s, e in segs:
        w = e - s
        cands = []
        for ch, t in TEMPLATES:
            tw = t.shape[1]
            tpl = _soft_tpl(t, sy)
            if tw == w:
                windows = [(s, e)]
            elif tw == w + 1:              # a faint edge column got lost
                windows = [(s - 1, e), (s, e + 1)]
            else:
                continue
            for a, b in windows:
                if a < 0 or b > norm.shape[1]:
                    continue
                patch = np.pad(norm[:, a:b], ((3, 3), (0, 0)))
                d = _ssd_best(patch, tpl)
                if d is not None:
                    cands.append((d / tpl.size, ch))
        cands.sort()
        if not cands or cands[0][0] > 0.06:
            return None
        if len(cands) > 1 and cands[1][1] != cands[0][1] and \
                (cands[1][0] - cands[0][0]) / (cands[0][0] + 0.002) < 0.12:
            return None                    # too close to call
        text += cands[0][1]
    m = TEXT_RE.match(text)
    if not m:
        return None
    exp = int(m.group(1).replace(',', ''))
    pct = int(m.group(2)) + int(m.group(3)) / 100.0
    return (exp, pct, text) if pct < 100 else None
