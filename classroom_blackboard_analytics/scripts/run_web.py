# Starts the same pipeline as the CLI, but behind a browser UI (FastAPI + static files).

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    try:
        import uvicorn
    except ModuleNotFoundError:
        print(
            "Missing dependency: uvicorn. Use this project's virtualenv and install requirements:\n"
            f"  cd {ROOT}\n"
            "  python -m venv .venv\n"
            "  .venv\\Scripts\\activate          (Windows PowerShell)\n"
            "  pip install -r requirements.txt\n"
            f"Current Python: {sys.executable}",
            file=sys.stderr,
        )
        sys.exit(1)

    default_host = os.environ.get("BLACKBOARD_WEB_HOST", "0.0.0.0")
    default_port = int(os.environ.get("BLACKBOARD_WEB_PORT", "8766"))

    p = argparse.ArgumentParser()
    p.add_argument(
        "--host",
        default=default_host,
        help="Bind address. 0.0.0.0 = LAN/WAN; 127.0.0.1 = localhost only",
    )
    p.add_argument("--port", type=int, default=default_port, help="TCP port")
    p.add_argument(
        "--local",
        action="store_true",
        help="Same as --host 127.0.0.1",
    )
    args = p.parse_args()
    host = "127.0.0.1" if args.local else args.host

    print(f"Listening on http://{host}:{args.port}")
    if host == "0.0.0.0":
        try:
            import socket

            z = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            z.connect(("8.8.8.8", 80))
            lan = z.getsockname()[0]
            z.close()
            print(f"Try from another device on LAN: http://{lan}:{args.port}")
        except Exception:
            print("On the host PC run ipconfig and use your IPv4 address in the URL.")

    uvicorn.run(
        "web.server:app",
        host=host,
        port=args.port,
        reload=False,
        app_dir=str(ROOT),
    )


if __name__ == "__main__":
    main()
