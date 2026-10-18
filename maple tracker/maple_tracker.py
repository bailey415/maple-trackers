"""Maple Tracker: the EXP tracker and the mesos tracker in one app.

One screenshot of the game every couple of seconds is read for both your
EXP (status bar) and your mesos (inventory window), and both show on one
dashboard at http://127.0.0.1:8765.

When run from the source folders it keeps using the data and settings of
the separate "exp tracker" and "mesos tracker" folders, so nothing is lost.
"""
import base64
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)
HERE = Path(sys.executable).parent if FROZEN else Path(__file__).resolve().parent
BUNDLE = Path(getattr(sys, "_MEIPASS", HERE))
if not FROZEN:
    ROOT = HERE.parent
    sys.path.insert(0, str(ROOT / "exp tracker"))
    sys.path.insert(0, str(ROOT / "mesos tracker"))

import exp_tracker as et          # noqa: E402
import meso_tracker as mt         # noqa: E402
from exp_reader import ExpReader  # noqa: E402
from meso_reader import MesoReader  # noqa: E402

# Where each half keeps its data
if FROZEN:
    EXP_DIR = HERE
    et.DATA, et.SETTINGS = HERE / "data" / "exp", HERE / "exp_settings.json"
    mt.DATA, mt.SETTINGS = HERE / "data" / "mesos", HERE / "mesos_settings.json"
    FONT = HERE / "meso_font.json"
else:
    et.DATA = ROOT / "exp tracker" / "data"
    et.SETTINGS = ROOT / "exp tracker" / "tracker_settings.json"
    mt.DATA = ROOT / "mesos tracker" / "data"
    mt.SETTINGS = ROOT / "mesos tracker" / "tracker_settings.json"
    FONT = ROOT / "mesos tracker" / "meso_font.json"
SETTINGS = HERE / "maple_settings.json"
DEFAULTS = {"port": 8765, "sample_seconds": 2.0,
            "window_titles": ["GeForce NOW", "MapleStory"]}


def load_settings():
    s = dict(DEFAULTS)
    try:
        s.update(json.loads(SETTINGS.read_text(encoding="utf-8")))
    except Exception:
        pass
    return s


class App:
    def __init__(self):
        self.s = load_settings()
        et.DATA.mkdir(parents=True, exist_ok=True)
        mt.DATA.mkdir(parents=True, exist_ok=True)
        self.exp_s = et.load_settings()
        self.meso_s = mt.load_settings()
        self.exp = et.Tracker(self.exp_s)
        self.meso = mt.Tracker(self.meso_s)
        self.exp_reader = ExpReader(self.exp_s.get("anchor"))
        self.meso_reader = MesoReader(FONT)
        self.live = mt.Live()
        self.stop = threading.Event()

    # ---- reading -------------------------------------------------------
    def loop(self):
        cap = et.Capture(self.s["window_titles"])
        diag = et.Diagnostics(HERE, "EXP bar")
        exp_miss = meso_miss = 0
        while not self.stop.is_set():
            t0 = time.time()
            try:
                frame = cap.grab()
                self.exp.window = self.meso.window = cap.title
                if frame is None:
                    exp_miss += 1
                    if exp_miss == 15:
                        print("Can't find the game window - is MapleStory open in GeForce NOW?")
                else:
                    exp_miss = self._read_exp(frame, exp_miss)
                    meso_miss = self._read_meso(frame, meso_miss)
                    if exp_miss == 0:
                        diag.ok()
                    else:
                        diag.miss(frame)
                        if exp_miss % 5 == 0 and meso_miss:
                            cap.failed()
            except Exception as e:
                print("read error:", e)
            self.stop.wait(max(0.2, self.s["sample_seconds"] - (time.time() - t0)))

    def _read_exp(self, frame, misses):
        r = self.exp_reader.read(frame)
        if not r:
            if misses + 1 == 15:
                print("EXP bar not visible (game hidden, covered or minimised?)")
            return misses + 1
        self.exp.on_read(time.time(), r[0], r[1])
        a = self.exp_reader.anchor
        if a and list(a) != self.exp_s.get("anchor"):
            self.exp_s["anchor"] = list(a)
            et.save_settings(self.exp_s)
        return 0

    def _read_meso(self, frame, misses):
        r = self.meso_reader.read(frame)
        if not r:
            if misses + 1 == 15:
                print("Meso box not visible - open your inventory to track mesos")
            return misses + 1
        mesos, text, sure = r
        self.meso.on_read(time.time(), mesos)
        live = self.live
        with live.lock:
            if text != live.text or (not sure and live.png is None):
                live.id += 1
                live.text, live.sure = text, sure
                live.glyphs = self.meso_reader.last_glyphs
                live.png = self.meso_reader.crop_png()
                live.recent[live.id] = (text, live.glyphs)
                for old in [k for k in live.recent if k < live.id - 10]:
                    del live.recent[old]
        return 0

    # ---- dashboard -----------------------------------------------------
    def state(self):
        st = {"exp": self.exp.state(), "mesos": self.meso.state()}
        live, reader = self.live, self.meso_reader
        with live.lock:
            missing = reader.unconfirmed_digits()
            if not live.sure and live.glyphs and all(reader._match(g)[1] for g in live.glyphs):
                live.sure = True
            check = None
            if missing and live.text and not live.sure and live.png:
                check = {"id": live.id, "text": live.text,
                         "img": base64.b64encode(live.png).decode()}
            st["mesos"]["check"] = check
            st["mesos"]["missing_digits"] = missing
        st["name"] = self.exp_s.get("name")
        return st

    def set_name(self, name):
        name = str(name)[:24]
        self.exp_s["name"] = name
        et.save_settings(self.exp_s)
        self.meso_s["name"] = name
        mt.save_settings(self.meso_s)

    def confirm(self, body):
        """Returns (http code, message)."""
        text = str(body.get("text", "")).replace(" ", "")
        digits = text.replace(",", "")
        if not digits.isdigit():
            return 400, "type the number shown, e.g. 37,111"
        pretty = f"{int(digits):,}"
        live = self.live
        with live.lock:
            seen = live.recent.get(body.get("id"))
            if seen is None:
                return 409, "that number is too old - check the new one"
            shown, glyphs = seen
            if not self.meso_reader.learn(glyphs, pretty):
                return 400, "that doesn't have the same number of characters as the box"
            current = body.get("id") == live.id
            if current:
                live.sure = True
        if pretty != shown and current:
            self.meso.correct(int(digits))
        return 200, "{}"


def make_handler(app):
    page = (BUNDLE / "dashboard.html").read_bytes()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/api/state"):
                self._send(200, json.dumps(app.state()))
            elif self.path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            else:
                self._send(404, "not found", "text/plain")

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
            except Exception as e:
                return self._send(400, str(e), "text/plain")
            if self.path.startswith("/api/settings"):
                if "name" in body:
                    app.set_name(body["name"])
                if "level" in body:
                    try:
                        lv = int(body["level"])
                    except (TypeError, ValueError):
                        return self._send(400, "level must be a number", "text/plain")
                    if 1 <= lv <= 250:
                        app.exp.set_level(lv)
                return self._send(200, "{}")
            if self.path.startswith("/api/confirm"):
                code, msg = app.confirm(body)
                return self._send(code, msg, "application/json" if code == 200 else "text/plain")
            self._send(404, "not found", "text/plain")
    return H


def main():
    app = App()
    threading.Thread(target=app.loop, daemon=True).start()
    port = int(app.s["port"])
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    except OSError:
        print(f"Port {port} is busy - is the EXP tracker or another Maple Tracker "
              "already running? Close it and try again.")
        input("Press Enter to close...")
        return
    url = f"http://127.0.0.1:{port}/"
    print("Maple Tracker running. Dashboard:", url)
    print("Keep this window open. Open your inventory to track mesos too.")
    if "--no-browser" not in sys.argv:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop.set()
        server.server_close()


if __name__ == "__main__":
    main()
