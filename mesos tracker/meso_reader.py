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


VIEWS = ("as-is", "16:9", "16:9-sharp", "1080p")
GAME_SIZES = ((1366, 768), (1280, 720), (1920, 1080), (1600, 900),
              (1024, 768), (800, 600))


def view(frame, key):
    """The frame as-is, or undoing a stretch GeForce NOW may have applied
    (a 16:9 game stretched to fill a 16:10 screen such as 1920x1200)."""
    h, w = frame.shape[:2]
    if key == "as-is":
        return frame
    if key.startswith("fit:"):
        _, size, how = key.split(":")
        gw, gh = (int(v) for v in size.split("x"))
        interp = {"cubic": cv2.INTER_CUBIC, "linear": cv2.INTER_LINEAR}.get(how, cv2.INTER_CUBIC)
        return cv2.resize(frame, (gw, gh), interpolation=interp)
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
    return None


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
        self.view = "as-is"
        self.misses = 0
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
        if len(best) > 1 and best[1][0] - best[0][0] < 2 and best[1][1] != best[0][1]:
            return None, False            # too close to call
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
        runs = []
        for y in np.where(white.sum(1) >= 60)[0]:
            xs = np.where(white[y])[0]
            sp = np.where(np.diff(xs) > 1)[0]
            for a, b in zip(np.r_[xs[0], xs[sp + 1]], np.r_[xs[sp], xs[-1]]):
                L = b - a + 1
                if 60 <= L <= 260:
                    runs.append((int(y), int(a), int(L)))
        # A box starts with a fully white row; follow it down while each row
        # still starts white at the same spot (text, or a comma's tail on a
        # stretched screen, can split a row, so we don't need it whole).
        by_row = {}
        for y, a, L in runs:
            by_row.setdefault(y, []).append((a, L))
        out, seen = [], set()
        for y1, a1, L1 in runs:
            if L1 < 100:
                continue
            if any(abs(a - a1) <= 2 and L >= L1 * 0.3 for a, L in by_row.get(y1 - 1, [])):
                continue                     # not the top row
            y = y1
            while any(abs(a - a1) <= 2 and L >= 0.3 * L1 for a, L in by_row.get(y + 1, [])):
                y += 1
            h = y - y1 + 1
            if 12 <= h <= 22:
                box = (a1, y1, a1 + L1, y + 1)
                if box not in seen:
                    seen.add(box)
                    out.append(box)
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

    def read(self, frame, prefer=None, sizes=None):
        """Returns (mesos, text, confirmed) or None if the box isn't visible.

        Like the EXP reader, also tries undoing a 16:10 stretch. `prefer`
        is the view that worked for the EXP bar, tried first.
        """
        if frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        fits = []
        for gw, gh in (sizes or GAME_SIZES):
            if (gw, gh) != (w, h) and gw <= w + 4 and gh <= h + 4:
                fits += [f"fit:{gw}x{gh}:cubic", f"fit:{gw}x{gh}:linear"]
        order = [self.view] + ([prefer] if prefer and not prefer.startswith(("stretched", "fitsoft")) else []) \
            + list(VIEWS) + fits
        seen = set()
        slow = not (self.misses > 3 and self.misses % 5)
        if not slow:
            order = order[:1] + ["as-is"]
        for key in order:
            if key in seen:
                continue
            seen.add(key)
            img = view(frame, key)
            if img is None:
                continue
            r = self._read_frame(img)
            if r:
                self.view = key
                self.misses = 0
                return r
        if slow:
            r = self._read_stretched(frame)
            for key in fits:
                if r:
                    break
                if key.endswith(":cubic"):
                    r = self._read_soft(view(frame, key), 1.0)
            if r:
                self.misses = 0
                return r
        self.misses += 1
        return None

    # ---- stretched screens (e.g. 1920x1200) ------------------------------
    def _templates(self):
        out = {}
        for ch, s in list(GUESS.items()) + list(SEEN.items()):
            out.setdefault(ch, []).append(_arr(s))
        for k, ch in self.learned.items():
            out.setdefault(ch, []).append(_arr(k))
        return out

    def _read_stretched(self, frame):
        h, w = frame.shape[:2]
        sy = h / (w * 9 / 16)
        if not 1.02 < sy < 1.4:
            return None
        return self._read_soft(frame, sy)

    def _read_soft(self, frame, sy):
        """Like the EXP reader: split characters by blank columns and compare
        each one with the same character stretched the same way."""
        tpls = self._templates()
        for x0, y0, x1, y1 in self.find_boxes(frame):
            ink = 255.0 - frame[y0:y1, x0:x1].min(2).astype(np.float32)
            rows = np.where(ink.max(1) > 120)[0]
            if len(rows) == 0:
                continue
            band = ink[max(0, rows.min() - 2):rows.min() + int(round(ROWS * sy)) + 2]
            norm = np.clip(band / 255.0, 0, 1).astype(np.float32)
            cols = norm.max(0) > 0.3
            segs, x = [], 0
            while x < len(cols):
                if cols[x]:
                    s = x
                    while x < len(cols) and cols[x]:
                        x += 1
                    segs.append((s, x))
                else:
                    x += 1
            if not segs or len(segs) > 16 or any(
                    segs[i][0] - segs[i - 1][1] > 4 for i in range(1, len(segs))):
                continue
            text = ''
            for s, e in segs:
                cands = []
                for ch, bitmaps in tpls.items():
                    for bm in bitmaps:
                        if bm.shape[1] != e - s:
                            continue
                        t = np.pad(bm.astype(np.float32), ((3, 3), (0, 0)))
                        t = cv2.resize(t, (t.shape[1], round(t.shape[0] * sy)),
                                       interpolation=cv2.INTER_LINEAR)
                        patch = np.pad(norm[:, s:e], ((3, 3), (0, 0)))
                        if patch.shape[0] < t.shape[0]:
                            continue
                        d = float(cv2.matchTemplate(patch, t, cv2.TM_SQDIFF).min()) / t.size
                        cands.append((d, ch))
                cands.sort()
                if not cands or cands[0][0] > 0.06:
                    text = None
                    break
                others = [c for c in cands if c[1] != cands[0][1]]
                if others and (others[0][0] - cands[0][0]) / (cands[0][0] + 0.002) < 0.12:
                    text = None
                    break
                text += cands[0][1]
            if text and NUM_RE.match(text):
                self.last_glyphs = None          # can't learn from blurred pixels
                self.last_crop = frame[max(0, y0 - 3):y1 + 3, max(0, x0 - 3):x1 + 3].copy()
                self.view = "as-is"
                return int(text.replace(',', '')), text, True
        return None

    def _read_frame(self, frame):
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
