# run_web on 127.0.0.1 + cloudflared tunnel; needs cloudflared on PATH

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]


def _project_venv_python() -> Optional[Path]:
    """Prefer classroom_blackboard_analytics/.venv so we do not use repo-root venv missing uvicorn."""
    if sys.platform == "win32":
        cand = ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        cand = ROOT / ".venv" / "bin" / "python"
    return cand if cand.is_file() else None


def _wait_port(host: str, port: int, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.3)
    return False


def _graceful_terminate(proc: Optional[subprocess.Popen], *, timeout: float = 5.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> None:
    os.chdir(ROOT)

    p = argparse.ArgumentParser()
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

    py = _project_venv_python()
    if py is None:
        print(
            "WARNING: No .venv in this project. Using current Python:\n"
            f"  {sys.executable}\n"
            "If you see ModuleNotFoundError: uvicorn, run:\n"
            f"  cd {ROOT}\n"
            "  python -m venv .venv\n"
            "  .venv\\Scripts\\activate    (Windows)\n"
            "  pip install -r requirements.txt\n",
            file=sys.stderr,
        )
        exe = Path(sys.executable)
    else:
        exe = py
    web_cmd = [
        str(exe),
        str(ROOT / "scripts" / "run_web.py"),
        "--local",
        "--port",
        str(port),
    ]
    print("Starting local web:", " ".join(web_cmd))
    web = subprocess.Popen(web_cmd, cwd=str(ROOT))
    cf: Optional[subprocess.Popen] = None
    try:
        if not _wait_port("127.0.0.1", port):
            code = web.poll()
            print(
                "Timeout waiting for web port. If the window above shows ModuleNotFoundError, use:\n"
                f"  cd {ROOT}\n"
                "  .venv\\Scripts\\activate\n"
                "  pip install -r requirements.txt\n"
                f"(web subprocess exit code: {code})",
                file=sys.stderr,
            )
            web.terminate()
            sys.exit(1)

        tunnel_cmd = ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}"]
        print("Starting Cloudflare Tunnel:", " ".join(tunnel_cmd))
        print("Find your public https://....trycloudflare.com URL in the output below.\n")
        cf = subprocess.Popen(tunnel_cmd)

        def stop(*_: object) -> None:
            _graceful_terminate(cf)
            _graceful_terminate(web)

        signal.signal(signal.SIGINT, stop)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, stop)

        code = cf.wait()
        stop()
        sys.exit(code if code is not None else 0)
    finally:
        _graceful_terminate(web)
        _graceful_terminate(cf)


if __name__ == "__main__":
    main()
