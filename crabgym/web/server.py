"""`python -m crabgym.web`: the console -- every run's conversation with ER,
followed while it is written.

    python -m crabgym.web                          # http://127.0.0.1:8765/
    python -m crabgym.web --open                   # ... and open it, if there is a browser here
    python -m crabgym.web --root <dir>             # runs somewhere other than logs/crabgym/

A page and four routes, read-only over the runs' directory:

    GET /                           the page (static/)
    GET /api/runs                   every run, newest first (`record.list_runs`)
    GET /api/runs/<id>?from=<byte>  one run, and its record from that byte on (`record.run_view`)
    GET /runs/<id>/<path>           a file of the run: a picture, a clip (with ranges), a record

The page asks for what is new once a second while a run is live. That is
the whole delay: ER takes 1.5 to 1.8 s to answer (the medians of the three
Gemini runs of 2026-10-08/09; 7.6 s the longest) and two to three seconds
pass between its answers, and a question a second needs no connection held
open per page, no stream to resume, no thread parked on each.

Bound to 127.0.0.1 only, the rule `rl/mjrl/viewer/tensorboard.py` keeps for
TensorBoard: the records are not for the network. Over ssh, forward the
port: `ssh -L 8765:localhost:8765 <host>`.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ..log import LOG_ROOT
from . import PORT, record

STATIC = Path(__file__).resolve().parent / "static"
#: What `/static/` serves, and as what. A name not here is not served.
STATIC_FILES = {"index.html": "text/html; charset=utf-8", "app.js": "text/javascript; charset=utf-8",
                "app.css": "text/css; charset=utf-8"}
#: A run's files that change while it runs, shown as text in the browser.
TEXT_SUFFIXES = (".json", ".jsonl", ".txt", ".log")
#: One byte range, the only kind a `<video>` asks for.
RANGE = re.compile(r"bytes=(\d*)-(\d*)")


class Handler(BaseHTTPRequestHandler):
    """The routes above. `root` is set on a subclass by `make_server`."""

    server_version = "crabgym-web"
    root: Path = LOG_ROOT

    def do_GET(self) -> None:
        try:
            self._route()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the page went away mid-answer: nobody is left to tell

    def _route(self) -> None:
        url = urllib.parse.urlsplit(self.path)
        # Split before unquoting: "%2F" is a character of a segment, not a
        # separator, and ".." smuggled in as "..%2F" stays one segment for
        # `record.run_file` to refuse.
        segments = [urllib.parse.unquote(s) for s in url.path.split("/")[1:]]
        if url.path == "/":
            self._static("index.html")
        elif len(segments) == 2 and segments[0] == "static":
            self._static(segments[1])
        elif segments == ["api", "runs"]:
            self._json(record.list_runs(self.root))
        elif len(segments) == 3 and segments[:2] == ["api", "runs"]:
            start = urllib.parse.parse_qs(url.query).get("from", ["0"])[0]
            view = record.run_view(self.root, segments[2], int(start)) if start.isdigit() else None
            if view is None:
                self._not_found()
            else:
                self._json(view)
        elif len(segments) >= 3 and segments[0] == "runs":
            path = record.run_file(self.root, segments[1], "/".join(segments[2:]))
            if path is None:
                self._not_found()
            else:
                self._file(path)
        else:
            self._not_found()

    # ── Answers ────────────────────────────────────────────────────────

    def _send(self, status: int, body: bytes, content_type: str, **headers: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in headers.items():
            self.send_header(name.replace("_", "-"), value)
        self.end_headers()
        self.wfile.write(body)

    def _not_found(self) -> None:
        self._send(404, b"not found\n", "text/plain; charset=utf-8")

    def _json(self, value: Any) -> None:
        self._send(200, json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8",
                   Cache_Control="no-store")

    def _static(self, name: str) -> None:
        if name not in STATIC_FILES:
            return self._not_found()
        self._send(200, (STATIC / name).read_bytes(), STATIC_FILES[name], Cache_Control="no-cache")

    def _file(self, path: Path) -> None:
        """A file of a run, whole or one range of it. A picture or a clip is
        never written again once it is named, and is kept; the records grow."""
        size = path.stat().st_size
        if path.suffix in TEXT_SUFFIXES:
            content_type, cache = "text/plain; charset=utf-8", "no-store"
        else:
            content_type, cache = mimetypes.guess_type(path.name)[0] or "application/octet-stream", "max-age=3600"
        first, last, status = 0, size - 1, 200
        # A range this does not parse -- several at once, say -- is ignored and
        # the whole file sent, as HTTP allows.
        m = RANGE.fullmatch((self.headers.get("Range") or "").strip())
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                first = int(m.group(1))
                last = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
            else:
                first = max(0, size - int(m.group(2)))
            if first > last or first >= size:
                self._send(416, b"", "text/plain", Content_Range=f"bytes */{size}")
                return
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(last - first + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", cache)
        if status == 206:
            self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
        self.end_headers()
        with path.open("rb") as f:
            f.seek(first)
            remaining = last - first + 1
            while remaining > 0:
                chunk = f.read(min(1 << 16, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        pass  # a line per picture would bury the errors, which still print


def make_server(root: Path = LOG_ROOT, port: int = PORT) -> ThreadingHTTPServer:
    """The console on 127.0.0.1:`port` (0 for any free one), serving `root`."""
    handler = type("Handler", (Handler,), {"root": Path(root)})
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m crabgym.web", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=PORT, help=f"the port on 127.0.0.1 (default {PORT})")
    parser.add_argument("--root", type=Path, default=LOG_ROOT, help="the runs' directory (default logs/crabgym)")
    parser.add_argument("--open", action="store_true", help="open the page in a browser, if there is one here")
    args = parser.parse_args(argv)
    try:
        server = make_server(args.root, args.port)
    except OSError as e:
        # Not the next port up: a bookmark that moves with every start is no
        # bookmark, and the usual reason is a console already serving there.
        print(f"[web] cannot listen on 127.0.0.1:{args.port} ({e.strerror}): a console may be running there "
              f"already -- http://127.0.0.1:{args.port}/ -- or pass --port", file=sys.stderr)
        return 1
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"[web] {url}  -- the runs in {args.root}; Ctrl-C stops it", flush=True)
    if args.open:
        from mjrl.viewer.tensorboard import _can_open_browser

        if _can_open_browser():
            import webbrowser

            webbrowser.open(url)
        else:
            print("[web] no browser here to open (ssh, or no display): forward the port instead", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


__all__ = ["STATIC_FILES", "Handler", "main", "make_server"]
