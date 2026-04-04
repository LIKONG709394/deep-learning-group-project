"""
Start local web + Cloudflare quick tunnel (public HTTPS URL without router port-forward).

Prerequisite: cloudflared (e.g. winget install Cloudflare.cloudflared).

Usage (from classroom_blackboard_analytics):
  python scripts/run_web_public_tunnel.py
  python scripts/run_web_public_tunnel.py --port 8766

Web listens on 127.0.0.1; cloudflared forwards. Ctrl+C stops both.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _wait_port(host: str, port: int, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.3)
    return False


def main() -> None:
    os.chdir(ROOT)

    p = argparse.ArgumentParser(description="Web + Cloudflare quick tunnel")
    p.add_argument("--port", type=int, default=int(os.environ.get("BLACKBOARD_WEB_PORT", "8766")))
    args = p.parse_args()
    port = args.port

    if not shutil.which("cloudflared"):
        print(
            "cloudflared not found. Install e.g.:\n"
            "  winget install --id Cloudflare.cloudflared\n"
            "Then open a new terminal and run this script again.",
            file=sys.stderr,
        )
        sys.exit(1)

    web_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_web.py"),
        "--local",
        "--port",
        str(port),
    ]
    print("Starting local web:", " ".join(web_cmd))
    web = subprocess.Popen(web_cmd, cwd=str(ROOT))
    cf: subprocess.Popen | None = None
    try:
        if not _wait_port("127.0.0.1", port):
            print("Timeout waiting for web port; check run_web errors.", file=sys.stderr)
            web.terminate()
            sys.exit(1)

        tunnel_cmd = ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}"]
        print("Starting Cloudflare Tunnel:", " ".join(tunnel_cmd))
        print("Find your public https://....trycloudflare.com URL in the output below.\n")
        cf = subprocess.Popen(tunnel_cmd)

        def stop(*_: object) -> None:
            for proc in (cf, web):
                if proc is not None and proc.poll() is None:
                    proc.terminate()
            for proc in (cf, web):
                if proc is None:
                    continue
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

        signal.signal(signal.SIGINT, stop)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, stop)

        code = cf.wait() if cf else 1
        stop()
        sys.exit(code if code is not None else 0)
    finally:
        if web.poll() is None:
            web.terminate()
            try:
                web.wait(timeout=5)
            except subprocess.TimeoutExpired:
                web.kill()
        if cf is not None and cf.poll() is None:
            cf.terminate()
            try:
                cf.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cf.kill()


if __name__ == "__main__":
    main()
