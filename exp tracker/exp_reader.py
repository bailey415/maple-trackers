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
    best = None
    for ch, t in TEMPLATES:
        if t.shape != g.shape:
            continue
        d = int((t != g).sum())
        if best is None or d < best[0]:
            best = (d, ch)
    if best is None or best[0] > max(2, g.size // 6):
        return None
    return best[1]


def _read_at(frame, y, x0):
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


class ExpReader:
    """Remembers where the bar is, so it still reads at 0% after a level-up."""

    def __init__(self, anchor=None):
        self.anchor = tuple(anchor) if anchor else None   # (w, h, y, x0)

    def read(self, frame):
        """Returns (exp, pct, text) or None if the EXP text isn't visible."""
        if frame is None or frame.size == 0:
            return None
        r = self._read(frame)
        if r is None and frame.shape[1] != REF_WIDTH:
            # game drawn at another size: scale to the size the font is exact at
            h, w = frame.shape[:2]
            r = self._read(cv2.resize(frame, (REF_WIDTH, round(h * REF_WIDTH / w)),
                                      interpolation=cv2.INTER_AREA))
        return r

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
