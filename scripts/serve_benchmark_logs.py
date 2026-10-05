"""Serve the latest benchmark status log using only the Python standard library."""

import argparse
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

LOGS_DIR = Path(__file__).resolve().parents[1] / "benchmark-results" / "batch-logs"


def latest_log(directory: Path) -> Path | None:
    candidates = []
    for path in [
        *directory.glob("*/*.log"),
        *directory.parent.glob("*/gepa-*/progress.log"),
    ]:
        try:
            candidates.append((path.stat().st_mtime_ns, str(path), path))
        except FileNotFoundError:
            continue
    return max(candidates)[2] if candidates else None


def read_tail(path: Path) -> str:
    # Bound memory and response size even when viewing an older, noisy log.
    with path.open("rb") as stream:
        stream.seek(0, 2)
        start = max(0, stream.tell() - 65_536)
        stream.seek(start)
        content = stream.read(65_536)
    if start:
        content = content.partition(b"\n")[2]
    return "\n".join(content.decode("utf-8", errors="replace").splitlines()[-200:])


def make_handler(log_file: Path | None, logs_directory: Path = LOGS_DIR):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            route = urlsplit(self.path).path
            if route not in {"/", "/log"}:
                self.send_error(404)
                return
            try:
                path = log_file if log_file is not None else latest_log(logs_directory)
                title = path.name if path is not None else "Benchmark logs"
                content = (
                    read_tail(path) if path is not None else "Waiting for a run..."
                )
            except FileNotFoundError:
                title, content = "Benchmark logs", "Waiting for the log file..."
            except OSError as error:
                title, content = "Benchmark logs", f"Cannot read log: {error}"

            if route == "/log":
                body = f"{title}\n\n{content}\n"
                content_type = "text/plain; charset=utf-8"
            else:
                body = (
                    '<!doctype html><html><head><meta charset="utf-8">'
                    '<meta name="viewport" content="width=device-width, initial-scale=1">'
                    '<meta http-equiv="refresh" content="5">'
                    f"<title>{html.escape(title)}</title></head><body>"
                    f"<h1>{html.escape(title)}</h1>"
                    "<p>Refreshes every 5 seconds. Run summaries update as work progresses. "
                    'Showing the last 200 lines. <a href="/log">Plain text</a></p>'
                    f"<pre>{html.escape(content)}</pre></body></html>"
                )
                content_type = "text/html; charset=utf-8"
            data = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, _format: str, *args) -> None:
            # Browser polling must not create another growing access log.
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_file",
        nargs="?",
        type=Path,
        help="One log file; default follows the latest run.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    handler = make_handler(args.log_file.resolve() if args.log_file else None)
    with ThreadingHTTPServer((args.host, args.port), handler) as server:
        print(f"Log viewer: http://{args.host}:{server.server_port}", flush=True)
        print("Press Ctrl-C to stop.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
