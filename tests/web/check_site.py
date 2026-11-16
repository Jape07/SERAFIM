"""Does the site still work in a real browser? Run by ci.yml after the tests.

    python -m serafim mock --all-targets          # or real gold in data/gold
    python tests/web/check_site.py --gold data/gold [--browser PATH] [--wait 90]

It serves a copy of web/ with that gold on a local port, opens it in a headless Chromium-based
browser through the DevTools protocol (no extra package but websocket-client), waits for the
viewer image, and fails when:

- the page throws an exception or logs a console error;
- the viewer never shows an image, or shows a broken one;
- a control that this page uses is missing from index.html;
- the viewer does not open an image for the field.

Exit code 1 and the reasons on stdout. Set SERAFIM_BROWSER to pick the browser.
"""

from __future__ import annotations

import argparse
import base64
import functools
import http.server
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"

# Every id app.js reads without a guard, or that a visitor needs to use the challenge features.
REQUIRED_IDS = [
    "frame", "marks", "slider", "region-cards", "sector", "sky-ra", "sky-dec",
    "cnn-out", "csv", "compare-images", "analyze",
    "search-form", "search-ra", "search-dec", "fov",
    "gif-make", "gif-preview", "gif-seasons", "help",
]

BROWSERS = [
    os.environ.get("SERAFIM_BROWSER", ""),
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome", "msedge",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]

STATE_JS = r"""(() => {
  const q = (s) => document.querySelector(s);
  const qa = (s) => [...document.querySelectorAll(s)];
  const frame = q("#frame");
  const broken = [...document.images]
    .filter((i) => i.complete && i.naturalWidth === 0 && i.getAttribute("src"))
    .map((i) => i.src.slice(-60));
  return JSON.stringify({
    frame_src: (frame?.getAttribute("src") || "").split("/gold/")[1] || "",
    frame_width: frame ? frame.naturalWidth : 0,
    missing_ids: __IDS__.filter((id) => !q("#" + id)),
    regions: qa("#region-cards article").length,
    broken_images: broken,
    sector: (q("#sector")?.textContent || "").trim(),
  });
})()""".replace("__IDS__", json.dumps(REQUIRED_IDS))


def find_browser(asked: str | None) -> str:
    for candidate in ([asked] if asked else []) + BROWSERS:
        if not candidate:
            continue
        if Path(candidate).is_file():
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    raise SystemExit("no Chromium-based browser found; pass --browser or set SERAFIM_BROWSER")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args, **kwargs):  # one line per PNG would drown the report
        pass


def serve(folder: Path) -> tuple[http.server.ThreadingHTTPServer, int]:
    handler = functools.partial(QuietHandler, directory=str(folder))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


class Browser:
    """One headless page over the DevTools protocol."""

    def __init__(self, binary: str, profile: Path):
        import websocket

        self.port = free_port()
        self.process = subprocess.Popen(
            [binary, "--headless=new", "--disable-gpu", "--no-sandbox", "--no-first-run",
             f"--user-data-dir={profile}", f"--remote-debugging-port={self.port}",
             "--disk-cache-size=1", "--window-size=1400,2400", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        tabs = None
        for _ in range(120):
            try:
                tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json").read())
                break
            except Exception:
                time.sleep(0.5)
        if not tabs:
            raise SystemExit("the browser did not open its debugging port")
        page = next(tab for tab in tabs if tab["type"] == "page")
        self.ws = websocket.create_connection(page["webSocketDebuggerUrl"], suppress_origin=True)
        self.ws.settimeout(60)
        self.counter = 0
        self.events: list[dict] = []
        for method in ("Runtime.enable", "Log.enable", "Network.enable", "Page.enable"):
            self.send(method)

    def send(self, method: str, **params):
        self.counter += 1
        self.ws.send(json.dumps({"id": self.counter, "method": method, "params": params}))
        while True:
            message = json.loads(self.ws.recv())
            if message.get("id") == self.counter:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message.get("result", {})
            self.events.append(message)

    def evaluate(self, expression: str):
        result = self.send("Runtime.evaluate", expression=expression, returnByValue=True,
                           awaitPromise=True)["result"]
        value = result.get("value")
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return value

    def open(self, url: str, wait_s: float) -> dict:
        """Navigate and wait until the viewer image has loaded, or wait_s seconds."""
        self.events.clear()
        self.send("Page.navigate", url=url)
        started = time.time()
        state = {}
        while time.time() - started < wait_s:
            self.ws.settimeout(1.0)
            try:
                self.events.append(json.loads(self.ws.recv()))
            except Exception:
                pass
            self.ws.settimeout(60)
            state = self.evaluate(STATE_JS) or {}
            if state.get("frame_width") and state.get("regions", 0) >= 5:
                time.sleep(2)  # let the catalog and the spectrum finish
                return self.evaluate(STATE_JS) or {}
        return state

    def problems(self) -> list[str]:
        out = []
        for event in self.events:
            method, params = event.get("method"), event.get("params", {})
            if method == "Runtime.exceptionThrown":
                details = params["exceptionDetails"]
                text = details.get("exception", {}).get("description") or details.get("text")
                out.append(f"exception: {str(text)[:300]}")
            elif method == "Runtime.consoleAPICalled" and params.get("type") == "error":
                parts = [str(a.get("value", a.get("description", "")))[:200]
                         for a in params.get("args", [])]
                out.append(f"console.error: {' '.join(parts)}")
            elif method == "Log.entryAdded" and params["entry"]["level"] == "error":
                url = params["entry"].get("url", "")
                if "favicon" not in url:
                    out.append(f"log: {params['entry']['text'][:200]} {url[-80:]}")
            elif method == "Network.responseReceived":
                response = params["response"]
                if response["status"] >= 400 and "favicon" not in response["url"]:
                    out.append(f"http {response['status']}: {response['url'][-100:]}")
        return out

    def screenshot(self, path: Path) -> None:
        shot = self.send("Page.captureScreenshot", format="png", captureBeyondViewport=True)
        path.write_bytes(base64.b64decode(shot["data"]))

    def close(self) -> None:
        try:
            self.ws.close()
        finally:
            self.process.kill()


def check(state: dict, problems: list[str], *, difference: bool) -> list[str]:
    failures = list(problems)
    if not state.get("frame_width"):
        failures.append(f"the viewer shows no image (#sector: {state.get('sector')!r})")
    if state.get("missing_ids"):
        failures.append(f"missing in index.html: {', '.join(state['missing_ids'])}")
    if state.get("broken_images"):
        failures.append(f"broken images: {state['broken_images']}")
    if state.get("regions", 0) < 5:
        failures.append(f"{state.get('regions')} region cards, expected 5")
    if difference and not str(state.get("frame_src", "")).split("/")[-1].startswith("sum_"):
        failures.append(f"difference mode shows {state.get('frame_src')!r}, not the season sum")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gold", type=Path, default=ROOT / "data" / "gold")
    parser.add_argument("--browser", default=None)
    parser.add_argument("--wait", type=float, default=90.0, help="seconds to wait per page")
    parser.add_argument("--screenshots", type=Path, default=None, help="folder for PNG captures")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    if not (args.gold / "targets.parquet").is_file():
        print(f"no gold in {args.gold}: run `python -m serafim mock --all-targets` first")
        return 1
    binary = find_browser(args.browser)
    with tempfile.TemporaryDirectory(prefix="serafim-site-") as temp:
        site = Path(temp) / "site"
        shutil.copytree(WEB, site, ignore=shutil.ignore_patterns("gold"))
        shutil.copytree(args.gold, site / "gold")
        server, port = serve(site)
        browser = Browser(binary, Path(temp) / "profile")
        failures: list[str] = []
        try:
            base = f"http://127.0.0.1:{port}/?target=barnard"
            pages = [
                ("default view", base, False),
                ("difference view", f"{base}&detector=1&season=1-2&mode=difference", True),
            ]
            for name, url, difference in pages:
                state = browser.open(url, args.wait)
                found = check(state, browser.problems(), difference=difference)
                verdict = "ok" if not found else "FAILED"
                print(f"{name}: {verdict} · image {state.get('frame_src')!r} · "
                      f"{state.get('regions')} regions")
                for item in found:
                    print(f"  - {item}")
                failures += [f"{name}: {item}" for item in found]
                if args.screenshots:
                    args.screenshots.mkdir(parents=True, exist_ok=True)
                    browser.screenshot(args.screenshots / f"{name.replace(' ', '_')}.png")
        finally:
            browser.close()
            server.shutdown()
    print("site check:", "passed" if not failures else f"{len(failures)} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
