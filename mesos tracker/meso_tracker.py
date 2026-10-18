"""Mesos tracker for your own MapleStory character (GeForce NOW).

Keep your inventory window open. Every couple of seconds this reads the
meso total at the bottom of it, logs it, and serves a dashboard at
http://127.0.0.1:8766 with mesos/hour over the last 5/10/15/30 minutes,
today's mesos, what you spent, and your current total.
"""
import base64
import csv
import ctypes
import datetime as dt
import json
import os
import sys
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from meso_reader import MesoReader

HERE = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent
BUNDLE = Path(getattr(sys, "_MEIPASS", HERE))
DATA = HERE / "data"
SETTINGS = HERE / "tracker_settings.json"
FONT = HERE / "meso_font.json"

DEFAULTS = {
    "name": "My character",
    "port": 8766,
    "window_titles": ["GeForce NOW", "MapleStory"],
    "sample_seconds": 2.0,
}
WINDOWS = (5, 10, 15, 30)          # minutes, the buttons on the dashboard
GAP_SECONDS = 45                   # longer than this without a read = not counted as playing
HEARTBEAT_SECONDS = 30             # log a row at least this often while visible
LOOT_MAX = 300_000                 # a rise bigger than this must be read twice to count


def load_settings():
    s = dict(DEFAULTS)
    try:
        s.update(json.loads(SETTINGS.read_text(encoding="utf-8")))
    except Exception:
        pass
    return s


def save_settings(s):
    try:
        tmp = SETTINGS.with_suffix(".tmp")
        tmp.write_text(json.dumps(s, indent=2), encoding="utf-8")
        os.replace(tmp, SETTINGS)
    except Exception as e:
        print("could not save settings:", e)


def midnight(t=None):
    d = dt.datetime.fromtimestamp(t or time.time())
    return dt.datetime(d.year, d.month, d.day).timestamp()


# --------------------------------------------------------------------------
# Screen capture (same as the exp tracker)
# --------------------------------------------------------------------------
class Capture:
    """Finds the game window and screenshots it.

    Picks the most likely game window: a GeForce NOW window showing
    MapleStory first, then the MapleStory client itself, then any GeForce
    NOW window. Browser windows that merely mention MapleStory (a wiki
    tab, say) are skipped. If the chosen window never gives a reading,
    the next candidate is tried.
    """
    BROWSERS = ("google chrome", "mozilla firefox", "microsoft edge", "opera",
                "brave", "vivaldi", "discord", "notepad", "explorer")

    def __init__(self, titles):
        self.titles = [t.lower() for t in titles]
        self.sct = None
        self.pick = 0                 # which candidate we're using
        self.title = None             # title of the window being captured
        self.size = None
        self.candidates = []
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

    @staticmethod
    def _rank(title):
        t = title.lower()
        gfn, maple = "geforce now" in t, "maplestory" in t
        if gfn and maple:
            return 4
        if t.strip() == "maplestory":
            return 3                  # the game client itself
        if gfn:
            return 2
        if maple and not any(b in t for b in Capture.BROWSERS):
            return 1
        return 0

    def _windows(self):
        import win32gui
        found = []

        def cb(h, _):
            if win32gui.IsWindowVisible(h) and not win32gui.IsIconic(h):
                title = win32gui.GetWindowText(h)
                if any(t in title.lower() for t in self.titles):
                    rank = self._rank(title)
                    l, t_, r, b = win32gui.GetWindowRect(h)
                    if rank and r - l > 200 and b - t_ > 200:
                        found.append((rank, (r - l) * (b - t_), title, (l, t_, r, b)))
            return True
        win32gui.EnumWindows(cb, 0)
        found.sort(reverse=True)
        return found

    def failed(self):
        """The current window gave no reading for a while: try the next one."""
        if len(self.candidates) > 1:
            self.pick = (self.pick + 1) % len(self.candidates)

    def grab(self):
        import mss
        self.candidates = self._windows()
        if not self.candidates:
            if self.title is not None:
                print("Game window not found any more.")
            self.title = None
            return None
        _, _, title, rect = self.candidates[min(self.pick, len(self.candidates) - 1)]
        if self.sct is None:
            self.sct = mss.mss()
        l, t, r, b = rect
        # stay inside the virtual screen (windows can hang off the edge)
        mon = self.sct.monitors[0]
        l, t = max(l, mon["left"]), max(t, mon["top"])
        r = min(r, mon["left"] + mon["width"])
        b = min(b, mon["top"] + mon["height"])
        if r - l < 200 or b - t < 200:
            return None
        if title != self.title or (r - l, b - t) != self.size:
            self.title, self.size = title, (r - l, b - t)
            print(f'Watching window: "{title}" ({r - l}x{b - t})')
        img = self.sct.grab({"left": l, "top": t, "width": r - l, "height": b - t})
        return cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)


class Diagnostics:
    """Saves a picture of what we captured when nothing can be read."""

    def __init__(self, folder, what):
        self.folder, self.what = folder, what
        self.since = time.time()       # last successful read (or start)
        self.saved = 0.0

    def ok(self):
        self.since = time.time()

    def miss(self, frame):
        now = time.time()
        if frame is None or now - self.since < 20 or now - self.saved < 300:
            return
        self.saved = now
        path = self.folder / "debug_capture.png"
        try:
            cv2.imwrite(str(path), frame)
            print(f"Can't see the {self.what}. Saved what the tracker sees to:")
            print(f"  {path}")
            print("  (send that picture to whoever set this up)")
        except Exception as e:
            print("could not save debug picture:", e)


# --------------------------------------------------------------------------
# Meso bookkeeping
# --------------------------------------------------------------------------
class Tracker:
    """Turns (time, mesos) readings into gains, spending and rates."""

    def __init__(self, settings):
        self.s = settings
        self.lock = threading.Lock()
        self.last = None             # (t, mesos)
        self.events = deque()        # (t, change, active_seconds)
        self.last_seen = 0.0
        self.last_logged = 0.0
        self.pending = None
        self.window = None
        DATA.mkdir(exist_ok=True)
        self._replay()

    def _apply(self, t, mesos):
        if self.last is not None:
            gap = t - self.last[0]
            active = gap if gap <= GAP_SECONDS else 0.0
            change = mesos - self.last[1]
            if change or active:
                self.events.append((t, change, active))
        self.last = (t, mesos)

    def _replay(self):
        start = min(midnight(), time.time() - 3600)
        days = sorted({dt.date.fromtimestamp(start), dt.date.today()})
        rows = []
        for d in days:
            p = DATA / f"mesos_{d.isoformat()}.csv"
            if not p.exists():
                continue
            with open(p, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    try:
                        rows.append((float(r["time"]), int(r["mesos"])))
                    except Exception:
                        continue
        rows.sort()
        for t, m in rows:
            if t >= start:
                self._apply(t, m)
        if self.last:
            self.last_logged = self.last[0]
        print(f"loaded {len(rows)} saved readings")

    def _log(self, t, mesos):
        p = DATA / f"mesos_{dt.date.fromtimestamp(t).isoformat()}.csv"
        new = not p.exists()
        with open(p, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["time", "local_time", "mesos"])
            w.writerow([f"{t:.1f}", dt.datetime.fromtimestamp(t).strftime("%H:%M:%S"), mesos])
        self.last_logged = t

    def on_read(self, t, mesos):
        with self.lock:
            self.last_seen = t
            prev, self.pending = self.pending, mesos
            if self.last is not None and mesos == self.last[1]:
                if t - self.last_logged >= HEARTBEAT_SECONDS:
                    self._apply(t, mesos)
                    self._log(t, mesos)
                return
            # Normal looting is taken at once. A drop (spending) or a big
            # jump has to be read the same twice in a row first.
            if self.last is not None:
                change = mesos - self.last[1]
                if not (0 < change <= LOOT_MAX) and prev != mesos:
                    return
            elif prev != mesos:
                return
            self._apply(t, mesos)
            self._log(t, mesos)

    def correct(self, mesos):
        """You told us the real number: fix the latest reading without counting a gain."""
        with self.lock:
            t = time.time()
            self.last = (t, mesos)
            self.pending = mesos
            self._log(t, mesos)

    def state(self):
        with self.lock:
            now = time.time()
            while self.events and self.events[0][0] < min(midnight(), now - 3600):
                self.events.popleft()
            ev = list(self.events)
            last = self.last

            rates = {}
            for m in WINDOWS:
                lo = now - m * 60
                gained = sum(e[1] for e in ev if e[0] > lo and e[1] > 0)
                spent = -sum(e[1] for e in ev if e[0] > lo and e[1] < 0)
                active = sum(e[2] for e in ev if e[0] > lo)
                per_h = gained * 3600.0 / active if active >= 20 else None
                rates[m] = {"per_h": per_h, "gained": gained, "spent": spent,
                            "minutes": active / 60}

            m0 = midnight(now)
            today = [e for e in ev if e[0] >= m0]
            gained = sum(e[1] for e in today if e[1] > 0)
            spent = -sum(e[1] for e in today if e[1] < 0)
            active = sum(e[2] for e in today)

            def buckets(lo, step=300):
                n = max(1, int((now - lo) // step) + 1)
                g, a = [0.0] * n, [0.0] * n
                for e in ev:
                    if e[0] >= lo:
                        i = min(n - 1, int((e[0] - lo) // step))
                        if e[1] > 0:
                            g[i] += e[1]
                        a[i] += e[2]
                return g, a

            def trim(series, act=None):
                i = 0
                while i < len(series) - 2 and not series[i] and not (act and act[i]):
                    i += 1
                return series[max(0, i - 1):]

            tg, _ = buckets(m0)
            cum, run = [], 0.0
            for g in tg:
                run += g
                cum.append(run)
            rg, ra = buckets(now - 2 * 3600)
            rate_series = [g * 3600.0 / a if a >= 20 else 0 for g, a in zip(rg, ra)]
            return {
                "name": self.s.get("name"),
                "on_screen": now - self.last_seen < 10,
                "window": self.window,
                "last_seen": self.last_seen or None,
                "mesos": last[1] if last else None,
                "rates": rates,
                "today": {"gained": gained, "spent": spent, "net": gained - spent,
                          "minutes": active / 60, "spark": trim(cum)},
                "rate_spark": trim(rate_series, ra),
                "now": now,
            }


# --------------------------------------------------------------------------
# Reader loop + web server
# --------------------------------------------------------------------------
class Live:
    """What the reader saw last, for the dashboard's 'is this right?' check."""

    def __init__(self):
        self.lock = threading.Lock()
        self.id = 0
        self.text = None
        self.sure = True
        self.glyphs = None
        self.png = None
        self.recent = {}          # id -> (text, glyphs) of the last few numbers shown


def reader_loop(tracker, reader, live, settings, stop):
    cap = Capture(settings["window_titles"])
    diag = Diagnostics(HERE, "meso box (is your inventory open?)")
    misses = 0
    while not stop.is_set():
        t0 = time.time()
        frame = None
        try:
            frame = cap.grab()
            r = reader.read(frame) if frame is not None else None
            tracker.window = cap.title
            if r:
                misses = 0
                diag.ok()
                mesos, text, sure = r
                tracker.on_read(time.time(), mesos)
                with live.lock:
                    if text != live.text or (not sure and live.png is None):
                        live.id += 1
                        live.text, live.sure = text, sure
                        live.glyphs = reader.last_glyphs
                        live.png = reader.crop_png()
                        live.recent[live.id] = (text, live.glyphs)
                        for old in [k for k in live.recent if k < live.id - 10]:
                            del live.recent[old]
            else:
                misses += 1
                diag.miss(frame)
                if misses % 5 == 0:
                    cap.failed()
                if misses == 15:
                    if cap.title is None:
                        print("Can't find the game window - is MapleStory open in GeForce NOW?")
                    else:
                        print("Meso box not visible - is your inventory open and on screen?")
        except Exception as e:
            print("read error:", e)
        stop.wait(max(0.2, settings["sample_seconds"] - (time.time() - t0)))


def make_handler(tracker, reader, live):
    page = (BUNDLE / "dashboard.html").read_bytes()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/api/state"):
                st = tracker.state()
                with live.lock:
                    missing = reader.unconfirmed_digits()
                    if not live.sure and live.glyphs and all(reader._match(g)[1] for g in live.glyphs):
                        live.sure = True          # learned from an earlier "Yes"
                    st["check"] = None
                    if missing and live.text and not live.sure and live.png:
                        st["check"] = {"id": live.id, "text": live.text,
                                       "img": base64.b64encode(live.png).decode()}
                    st["missing_digits"] = missing
                self._send(200, json.dumps(st).encode(), "application/json")
            elif self.path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
            except Exception as e:
                return self._send(400, str(e).encode(), "text/plain")
            if self.path.startswith("/api/settings"):
                if "name" in body:
                    tracker.s["name"] = str(body["name"])[:24]
                    save_settings(tracker.s)
                return self._send(200, b"{}", "application/json")
            if self.path.startswith("/api/confirm"):
                # body: {"id": n, "text": "37,111"} - the number the box really shows
                text = str(body.get("text", "")).replace(" ", "")
                digits = text.replace(",", "")
                if not digits.isdigit():
                    return self._send(400, b"type the number shown, e.g. 37,111", "text/plain")
                pretty = f"{int(digits):,}"
                with live.lock:
                    seen = live.recent.get(body.get("id"))
                    if seen is None:
                        return self._send(409, b"that number is too old - check the new one", "text/plain")
                    shown, glyphs = seen
                    if not reader.learn(glyphs, pretty):
                        return self._send(400, b"that doesn't have the same number of characters as the box", "text/plain")
                    wrong = pretty != shown
                    current = body.get("id") == live.id
                    if current:
                        live.sure = True
                if wrong and current:
                    tracker.correct(int(digits))
                return self._send(200, b"{}", "application/json")
            self._send(404, b"not found", "text/plain")
    return H


def main():
    settings = load_settings()
    save_settings(settings)
    tracker = Tracker(settings)
    reader = MesoReader(FONT)
    live = Live()
    stop = threading.Event()
    threading.Thread(target=reader_loop, args=(tracker, reader, live, settings, stop),
                     daemon=True).start()
    port = int(settings["port"])
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(tracker, reader, live))
    url = f"http://127.0.0.1:{port}/"
    print("Mesos tracker running. Dashboard:", url)
    print("Keep your inventory open and this window running. Close it to stop.")
    if "--no-browser" not in sys.argv:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()


if __name__ == "__main__":
    main()
