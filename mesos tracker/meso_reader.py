"""Reads the meso total from the bottom of your inventory window.

The number sits in a white box ("37,111") and is drawn in a small pixel
font, black on white. We find that box anywhere on screen, cut the
characters apart and match them against known bitmaps.

Only 3, 7, 1 and the comma have been seen for real so far. The other
digits start out as close guesses; whenever you confirm a reading on the
dashboard, the real pixels of every character in it are saved to
meso_font.json and used from then on.
"""
import json
import re
from pathlib import Path

import cv2
import numpy as np

ROWS = 10          # 8 rows of digit + 2 for the comma's tail
SEEN = {           # taken from a real screenshot
    '3': '.###./#...#/....#/..##./....#/....#/#...#/.###./...../.....',
    '7': '#####/....#/...#./...#./..#../..#../.#.../.#.../...../.....',
    '1': '.#./##./.#./.#./.#./.#./.#./###/.../...',
    ',': '../../../../../../.#/.#/.#/#.',
}
GUESS = {          # same style, not seen yet
    '0': '.###./#...#/#...#/#...#/#...#/#...#/#...#/.###./...../.....',
    '2': '.###./#...#/....#/...#./..#../.#.../#..../#####/...../.....',
    '4': '...#./..##./.#.#./#..#./#####/...#./...#./...#./...../.....',
    '5': '#####/#..../#..../####./....#/....#/#...#/.###./...../.....',
    '6': '.###./#...#/#..../####./#...#/#...#/#...#/.###./...../.....',
    '8': '.###./#...#/#...#/.###./#...#/#...#/#...#/.###./...../.....',
    '9': '.###./#...#/#...#/#...#/.####/....#/#...#/.###./...../.....',
}
NUM_RE = re.compile(r'^\d{1,3}(,\d{3})*$')
WHITE = 245        # box background
INK = 120          # text pixels are darker than this


def _arr(s):
    return np.array([[c == '#' for c in row] for row in s.split('/')], np.uint8)


def _key(g):
    return '/'.join(''.join('#' if v else '.' for v in row) for row in g)


class MesoReader:
    def __init__(self, font_path):
        self.font_path = Path(font_path)
        self.learned = {}                     # bitmap string -> char (confirmed)
        try:
            self.learned = json.loads(self.font_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        self.box = None                       # (frame w, h, x0, y0, x1, y1)
        self.last_glyphs = None
        self.last_crop = None

    # ---- font ----------------------------------------------------------
    def known_chars(self):
        return sorted(set(SEEN) | set(self.learned.values()))

    def unconfirmed_digits(self):
        return [d for d in "0123456789" if d not in self.known_chars()]

    def _match(self, g):
        """(char, confirmed) for one glyph, or (None, False)."""
        k = _key(g)
        if k in self.learned:
            return self.learned[k], True
        for ch, s in SEEN.items():
            if s == k:
                return ch, True
        best = []
        for ch, s in list(SEEN.items()) + list(GUESS.items()):
            t = _arr(s)
            if t.shape != g.shape:
                if t.shape[0] != g.shape[0] or abs(t.shape[1] - g.shape[1]) > 1:
                    continue
                w = max(t.shape[1], g.shape[1])
                a = np.zeros((ROWS, w), np.uint8)
                b = np.zeros((ROWS, w), np.uint8)
                a[:, :t.shape[1]] = t
                b[:, :g.shape[1]] = g
                d = int((a != b).sum()) + 2
            else:
                d = int((t != g).sum())
            best.append((d, ch))
        best.sort()
        if not best or best[0][0] > 8:
            return None, False
        return best[0][1], False

    def learn(self, glyphs, text):
        """Save the real pixels of each character of a confirmed reading."""
        if glyphs is None or len(glyphs) != len(text):
            return False
        for g, ch in zip(glyphs, text):
            self.learned[_key(g)] = ch
        try:
            tmp = self.font_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.learned, indent=1), encoding="utf-8")
            tmp.replace(self.font_path)
        except Exception as e:
            print("could not save meso font:", e)
        return True

    # ---- finding the box -----------------------------------------------
    @staticmethod
    def find_boxes(frame):
        """White boxes about the size of the meso box: (x0, y0, x1, y1)."""
        white = frame.min(2) > WHITE
        out = []
        runs = {}
        for y in np.where(white.sum(1) >= 100)[0]:
            xs = np.where(white[y])[0]
            sp = np.where(np.diff(xs) > 1)[0]
            for a, b in zip(np.r_[xs[0], xs[sp + 1]], np.r_[xs[sp], xs[-1]]):
                L = b - a + 1
                if 100 <= L <= 260:
                    runs.setdefault((int(a), int(L)), []).append(int(y))
        for (a, L), ys in runs.items():
            # the top and bottom rows of the box are completely white
            ys.sort()
            for i in range(len(ys)):
                for j in range(i + 1, len(ys)):
                    h = ys[j] - ys[i] + 1
                    if 12 <= h <= 18:
                        out.append((a, ys[i], a + L, ys[j] + 1))
        return out

    # ---- reading -------------------------------------------------------
    def _glyphs(self, frame, x0, y0, x1, y1):
        box = frame[y0:y1, x0:x1]
        ink = box.max(2) < INK
        if not ink.any():
            return None
        rows = np.where(ink.any(1))[0]
        b = ink[rows.min():rows.min() + ROWS]
        if b.shape[0] < 8:
            return None
        if b.shape[0] < ROWS:
            b = np.vstack([b, np.zeros((ROWS - b.shape[0], b.shape[1]), bool)])
        b = b.astype(np.uint8)
        cols = b.any(0)
        segs, x = [], 0
        while x < len(cols):
            if cols[x]:
                s = x
                while x < len(cols) and cols[x]:
                    x += 1
                segs.append((s, x))
            else:
                x += 1
        if not segs or len(segs) > 16:
            return None
        for i in range(1, len(segs)):
            if segs[i][0] - segs[i - 1][1] > 4:      # text is one tight group
                return None
        return [b[:, s:e] for s, e in segs]

    def _read_box(self, frame, x0, y0, x1, y1):
        gs = self._glyphs(frame, x0, y0, x1, y1)
        if not gs:
            return None
        text, sure = '', True
        for g in gs:
            ch, ok = self._match(g)
            if ch is None:
                return None
            text += ch
            sure = sure and ok
        if not NUM_RE.match(text):
            return None
        return int(text.replace(',', '')), text, sure, gs

    def read(self, frame):
        """Returns (mesos, text, confirmed) or None if the box isn't visible."""
        if frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        cands = []
        if self.box and self.box[:2] == (w, h):
            cands.append(self.box[2:])
        r = None
        for c in cands:
            r = self._read_box(frame, *c)
            if r:
                break
        if r is None:
            for c in self.find_boxes(frame):
                r = self._read_box(frame, *c)
                if r:
                    self.box = (w, h) + tuple(c)
                    break
        if r is None:
            return None
        x0, y0, x1, y1 = self.box[2:]
        self.last_glyphs = r[3]
        self.last_crop = frame[max(0, y0 - 3):y1 + 3, max(0, x0 - 3):x1 + 3].copy()
        return r[0], r[1], r[2]

    def crop_png(self, scale=3):
        if self.last_crop is None:
            return None
        c = cv2.resize(self.last_crop, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_NEAREST)
        ok, buf = cv2.imencode(".png", c)
        return buf.tobytes() if ok else None
