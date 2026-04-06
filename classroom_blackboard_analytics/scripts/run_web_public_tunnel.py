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
_REQ = ROOT / "requirements.txt"


def _venv_python_suffix() -> Path:
    return Path("Scripts") / "python.exe" if sys.platform == "win32" else Path("bin") / "python"


def _discovered_venv_pythons() -> list[Path]:
    """Prefer subproject .venv, then subproject venv, then parent .venv / venv (monorepo)."""
    suffix = _venv_python_suffix()
    ordered_dirs = [ROOT, ROOT.parent]
    ordered_names = [".venv", "venv"]
    seen_resolved: set[Path] = set()
    out: list[Path] = []
    for base in ordered_dirs:
        for name in ordered_names:
            cand = (base / name / suffix).resolve()
            if cand.is_file() and cand not in seen_resolved:
                seen_resolved.add(cand)
                out.append(cand)
    return out


def _pick_web_python() -> tuple[Path, bool]:
    """Return (python.exe, from_explicit_venv_dir). If false, caller may warn (fallback to sys.executable)."""
    found = _discovered_venv_pythons()
    if found:
        return found[0], True
    return Path(sys.executable), False


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

    exe, from_venv = _pick_web_python()
    if from_venv:
        print(f"Using Python for web subprocess: {exe}")
    else:
        print(
            "WARNING: No .venv or venv found under this app folder or its parent.\n"
            f"  Using: {exe}\n"
            "Install web dependencies (uvicorn, fastapi, …), for example:\n"
            f"  {exe} -m pip install -r {_REQ}\n"
            "Or create a venv under the app folder:\n"
            f"  cd {ROOT}\n"
            "  python -m venv .venv\n"
            "  .venv\\Scripts\\activate\n"
            f"  pip install -r requirements.txt\n",
            file=sys.stderr,
        )
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
                "Timeout waiting for web port. Typical fix — install app requirements into the same Python:\n"
                f"  {exe} -m pip install -r {_REQ}\n"
                f"(app root: {ROOT}, web subprocess exit code: {code})",
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
