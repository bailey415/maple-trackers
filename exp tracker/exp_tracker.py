"""EXP tracker for your own MapleStory character (GeForce NOW).

Reads the EXP numbers off your status bar every couple of seconds, logs
them, and serves a small dashboard at http://127.0.0.1:8765 with:
  EXP/hour and %/hour over the last 5/10/15/30 minutes, today's EXP,
  level, the EXP bar and the time until your next level.

It only ever reads your own status bar, so it only tracks your character.
"""
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

from exp_reader import ExpReader

HERE = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent
BUNDLE = Path(getattr(sys, "_MEIPASS", HERE))
DATA = HERE / "data"
SETTINGS = HERE / "tracker_settings.json"

DEFAULTS = {
    "name": "My character",
    "level": None,
    "port": 8765,
    "window_titles": ["GeForce NOW", "MapleStory"],
    "sample_seconds": 2.0,
    "anchor": None,
}
WINDOWS = (5, 10, 15, 30)          # minutes, the buttons on the dashboard
GAP_SECONDS = 45                   # longer than this without a read = not counted as playing
HEARTBEAT_SECONDS = 30             # log a row at least this often while on screen
WRAP_DROP_PCT = 30                 # % falling by more than this = level up


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


def _same_need(a, b):
    """Do two (exp, %) readings point at the same EXP requirement?"""
    def rng(e, p):
        return e * 100.0 / (p + 0.01), e * 100.0 / max(p - 0.01, 0.001)
    (alo, ahi), (blo, bhi) = rng(*a), rng(*b)
    return max(alo, blo) <= min(ahi, bhi) * 1.001


def midnight(t=None):
    d = dt.datetime.fromtimestamp(t or time.time())
    return dt.datetime(d.year, d.month, d.day).timestamp()


# --------------------------------------------------------------------------
# Screen capture
# --------------------------------------------------------------------------
class Capture:
    def __init__(self, titles):
        self.titles = [t.lower() for t in titles]
        self.sct = None
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

    def _window(self):
        import win32gui
        found = []

        def cb(h, _):
            if win32gui.IsWindowVisible(h) and not win32gui.IsIconic(h):
                title = win32gui.GetWindowText(h).lower()
                if any(t in title for t in self.titles):
                    l, t_, r, b = win32gui.GetWindowRect(h)
                    if r - l > 200 and b - t_ > 200:
                        found.append(((r - l) * (b - t_), (l, t_, r, b)))
            return True
        win32gui.EnumWindows(cb, 0)
        return max(found)[1] if found else None

    def grab(self):
        import mss
        rect = self._window()
        if rect is None:
            return None
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
        img = self.sct.grab({"left": l, "top": t, "width": r - l, "height": b - t})
        return cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)


# --------------------------------------------------------------------------
# EXP bookkeeping
# --------------------------------------------------------------------------
class Tracker:
    """Turns raw (time, exp, %) reads into gains, rates and level-ups."""

    def __init__(self, settings):
        self.s = settings
        self.lock = threading.Lock()
        self.last = None             # (t, exp, pct, level)
        self.events = deque()        # (t, gain, active_seconds, level)
        self.levelups = []           # (t, new_level)
        self.need = {}               # level -> EXP needed for that level
        self.last_seen = 0.0
        self.last_logged = 0.0
        self.pending = None          # read waiting for a second matching read
        self.level_since = None
        DATA.mkdir(exist_ok=True)
        self._replay()

    # ---- helpers -------------------------------------------------------
    @property
    def level(self):
        return self.s.get("level")

    def _need_for(self, level, exp, pct):
        """EXP needed for this level.

        The game shows % with two decimals, so each reading says the real
        requirement lies between exp/(pct+0.01%) and exp/(pct-0.01%).
        Intersecting those ranges over many readings pins it down exactly.
        """
        if exp > 0 and pct > 0:
            lo = exp * 100.0 / (pct + 0.01)
            hi = exp * 100.0 / max(pct - 0.01, 0.001)
            old = self.need.get(level)
            if old is not None:
                nlo, nhi = max(lo, old[0]), min(hi, old[1])
                if nlo <= nhi:
                    lo, hi = nlo, nhi
                elif (hi - lo) > (old[1] - old[0]):
                    lo, hi = old           # this reading is the odd one out
            self.need[level] = (lo, hi)
        if level in self.need:
            lo, hi = self.need[level]
            return (lo + hi) / 2.0
        return None

    def _need_now(self):
        if self.level in self.need:
            lo, hi = self.need[self.level]
            return (lo + hi) / 2.0
        return None

    def _apply(self, t, exp, pct, level, replay=False):
        """Account for one accepted reading. Returns the level after it."""
        if self.last is not None:
            lt, lexp, lpct, llevel = self.last
            gap = t - lt
            active = gap if gap <= GAP_SECONDS else 0.0
            wrapped = pct < lpct - WRAP_DROP_PCT
            if replay and level is not None and llevel is not None:
                levelled = level > llevel
            else:
                levelled = wrapped
            if levelled:
                need_prev = self._need_for(llevel, lexp, lpct) or lexp
                gain = max(0.0, need_prev - lexp) + exp
                if not replay:
                    level = (llevel + 1) if llevel is not None else None
                self.levelups.append((t, level))
                self.level_since = t
            else:
                gain = exp - lexp          # negative when you die
            if gain or active:
                self.events.append((t, gain, active, level))
        self._need_for(level, exp, pct)
        self.last = (t, exp, pct, level)
        return level

    def _replay(self):
        """Rebuild today (and the last half hour) from the log files."""
        start = min(midnight(), time.time() - 3600)
        days = {dt.date.fromtimestamp(start), dt.date.today()}
        rows = []
        for d in sorted(days):
            p = DATA / f"exp_{d.isoformat()}.csv"
            if not p.exists():
                continue
            with open(p, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    try:
                        lv = int(r["level"]) if r.get("level") not in (None, "", "None") else None
                        rows.append((float(r["time"]), int(r["exp"]), float(r["pct"]), lv))
                    except Exception:
                        continue
        rows.sort()
        for t, exp, pct, lv in rows:
            if t >= start:
                self._apply(t, exp, pct, lv, replay=True)
        if self.last:
            self.last_logged = self.last[0]
            if self.level is None and self.last[3] is not None:
                self.s["level"] = self.last[3]
        print(f"loaded {len(rows)} saved readings")

    def _log(self, t, exp, pct, level):
        p = DATA / f"exp_{dt.date.fromtimestamp(t).isoformat()}.csv"
        new = not p.exists()
        with open(p, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["time", "local_time", "exp", "pct", "level"])
            w.writerow([f"{t:.1f}", dt.datetime.fromtimestamp(t).strftime("%H:%M:%S"),
                        exp, f"{pct:.2f}", level if level is not None else ""])
        self.last_logged = t

    # ---- live input ----------------------------------------------------
    def on_read(self, t, exp, pct):
        with self.lock:
            self.last_seen = t
            same_as_last = self.last is not None and (exp, pct) == self.last[1:3]
            if same_as_last:
                self.pending = (exp, pct)
                if t - self.last_logged >= HEARTBEAT_SECONDS:
                    self._apply(t, exp, pct, self.level)
                    self._log(t, exp, pct, self.level)
                return
            # A reading whose EXP and % agree with what we've learned is taken
            # at once. Anything odd (a level-up, a death, a glitchy frame, or
            # not knowing the level's EXP yet) is only taken once the next
            # reading carries on from it.
            prev, self.pending = self.pending, (exp, pct)
            need = self._need_now()
            wrapped = self.last is not None and pct < self.last[2] - WRAP_DROP_PCT
            dropped = self.last is not None and exp < self.last[1]
            fits = need is not None and abs(exp * 100.0 / need - pct) <= 0.05
            if not fits or wrapped or dropped:
                confirmed = prev is not None and (
                    prev == (exp, pct) or
                    (0 <= prev[0] <= exp and prev[1] <= pct and _same_need(prev, (exp, pct))))
                if not confirmed:
                    return
                if not fits and not wrapped and not dropped and need is not None:
                    self.need.pop(self.level, None)   # confirmed: re-learn it
            before = self.level
            after = self._apply(t, exp, pct, self.level)
            if after != before:
                self.s["level"] = after
                save_settings(self.s)
                print(f"LEVEL UP -> {after}")
            self._log(t, exp, pct, self.level)

    def set_level(self, level):
        with self.lock:
            self.s["level"] = level
            if self.last:
                self.last = (*self.last[:3], level)
            save_settings(self.s)

    # ---- dashboard numbers ---------------------------------------------
    def state(self):
        with self.lock:
            now = time.time()
            while self.events and self.events[0][0] < min(midnight(), now - 3600):
                self.events.popleft()
            ev = list(self.events)
            last = self.last
            level = self.level
            on_screen = now - self.last_seen < 10
            need = None
            if last:
                need = self._need_now() or (
                    last[1] * 100.0 / last[2] if last[2] > 0 else None)

            def pct_of(g):
                return g * 100.0 / need if need else None

            rates = {}
            for m in WINDOWS:
                lo = now - m * 60
                gain = sum(e[1] for e in ev if e[0] > lo)
                active = sum(e[2] for e in ev if e[0] > lo)
                if active < 20:
                    rates[m] = {"exp_h": None, "pct_h": None, "eta_s": None,
                                "gain": gain, "minutes": active / 60}
                    continue
                per_h = gain * 3600.0 / active
                eta = None
                if last and need and per_h > 0:
                    eta = (need - last[1]) / per_h * 3600.0
                rates[m] = {"exp_h": per_h, "pct_h": pct_of(per_h), "eta_s": eta,
                            "gain": gain, "minutes": active / 60}

            m0 = midnight(now)
            today = [e for e in ev if e[0] >= m0]
            today_gain = sum(e[1] for e in today)
            today_active = sum(e[2] for e in today)

            # sparklines: 5-minute buckets
            def buckets(lo, step=300):
                n = max(1, int((now - lo) // step) + 1)
                b = [0.0] * n
                act = [0.0] * n
                for e in ev:
                    if e[0] >= lo:
                        i = min(n - 1, int((e[0] - lo) // step))
                        b[i] += e[1]
                        act[i] += e[2]
                return b, act
            tb, _ = buckets(m0)
            cum, run = [], 0.0
            for g in tb:
                run += g
                cum.append(run)
            rb, ra = buckets(now - 2 * 3600)
            rate_series = [max(0.0, g * 3600.0 / a) if a >= 20 else 0 for g, a in zip(rb, ra)]

            def trim(series, act=None):
                """Drop the empty stretch before you started playing."""
                i = 0
                while i < len(series) - 2 and not series[i] and not (act and act[i]):
                    i += 1
                return series[max(0, i - 1):]
            cum = trim(cum)
            rate_series = trim(rate_series, ra)

            ups = [{"t": t, "level": lv} for t, lv in self.levelups if t >= m0]
            return {
                "name": self.s.get("name"),
                "level": level,
                "on_screen": on_screen,
                "last_seen": self.last_seen or None,
                "exp": last[1] if last else None,
                "pct": last[2] if last else None,
                "need": need,
                "rates": rates,
                "today": {"exp": today_gain, "pct": pct_of(today_gain),
                          "minutes": today_active / 60, "levelups": ups,
                          "spark": cum},
                "rate_spark": rate_series,
                "level_since": self.level_since,
                "now": now,
            }


# --------------------------------------------------------------------------
# Reader loop + web server
# --------------------------------------------------------------------------
def reader_loop(tracker, settings, stop):
    cap = Capture(settings["window_titles"])
    reader = ExpReader(settings.get("anchor"))
    misses = 0
    while not stop.is_set():
        t0 = time.time()
        try:
            frame = cap.grab()
            r = reader.read(frame) if frame is not None else None
            if r:
                misses = 0
                tracker.on_read(time.time(), r[0], r[1])
                if reader.anchor and list(reader.anchor) != settings.get("anchor"):
                    settings["anchor"] = list(reader.anchor)
                    save_settings(settings)
            else:
                misses += 1
                if misses == 15:
                    print("EXP bar not visible (game hidden, covered or minimised?)")
        except Exception as e:
            print("read error:", e)
        stop.wait(max(0.2, settings["sample_seconds"] - (time.time() - t0)))


def make_handler(tracker):
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
                self._send(200, json.dumps(tracker.state()).encode(), "application/json")
            elif self.path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if not self.path.startswith("/api/settings"):
                return self._send(404, b"not found", "text/plain")
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                if "name" in body:
                    tracker.s["name"] = str(body["name"])[:24]
                    save_settings(tracker.s)
                if "level" in body:
                    lv = int(body["level"])
                    if 1 <= lv <= 250:
                        tracker.set_level(lv)
                self._send(200, b"{}", "application/json")
            except Exception as e:
                self._send(400, str(e).encode(), "text/plain")
    return H


def main():
    settings = load_settings()
    save_settings(settings)
    tracker = Tracker(settings)
    stop = threading.Event()
    threading.Thread(target=reader_loop, args=(tracker, settings, stop), daemon=True).start()
    port = int(settings["port"])
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(tracker))
    url = f"http://127.0.0.1:{port}/"
    print("EXP tracker running. Dashboard:", url)
    print("Keep this window open. Close it (or Ctrl+C) to stop.")
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
