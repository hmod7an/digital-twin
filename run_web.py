"""
Face Health Digital Twin — Web Launcher

Usage:
    python run_web.py                   # local → http://localhost:7860
    python run_web.py --share           # public HTTPS tunnel via localhost.run
    python run_web.py --share --auth    # with login prompt
    python run_web.py --port 8080       # custom port
"""
import argparse
import sys
import os
import subprocess
import threading
import re

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _print_qr(url: str):
    """Print a scannable QR code for the URL in the terminal."""
    try:
        import qrcode
        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        print()
        qr.print_ascii(invert=True)
        print()
    except ImportError:
        pass  # qrcode package not installed — skip silently


def _start_localtunnel(port: int):
    """
    Open an SSH reverse tunnel to localhost.run and print the public HTTPS URL
    plus a scannable QR code. Runs in a background thread.
    """
    cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-R", f"80:localhost:{port}",
        "nokey@localhost.run",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            m = re.search(r"(https://[^\s]+\.lhr\.life)", line)
            if m:
                url = m.group(1)
                print("\n" + "=" * 60)
                print("  PUBLIC URL (mobile / external access):")
                print(f"  {url}")
                print("=" * 60)
                print("  Scan with your phone camera:")
                _print_qr(url)
                print(flush=True)
    except FileNotFoundError:
        print("  [share] ssh not found — cannot open tunnel", flush=True)
    except Exception as e:
        print(f"  [share] tunnel error: {e}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Face Health Digital Twin — Web App")
    parser.add_argument("--share",  action="store_true",
                        help="Create a public HTTPS tunnel via localhost.run")
    parser.add_argument("--auth",   action="store_true",
                        help="Require username/password (demo / demo1234)")
    parser.add_argument("--port",   type=int, default=7860,
                        help="Local port (default 7860)")
    args = parser.parse_args()

    auth_pair = ("demo", "demo1234") if args.auth else None

    print("=" * 60)
    print("  Face Health Digital Twin  —  Web Interface")
    print("=" * 60)
    if args.share:
        print("  Share mode ON  →  public URL will appear below")
    if args.auth:
        print(f"  Auth ON  →  login: demo / demo1234")
    print(f"  Local URL: http://localhost:{args.port}")
    print("=" * 60)
    print()

    if args.share:
        t = threading.Thread(target=_start_localtunnel, args=(args.port,), daemon=True)
        t.start()

    from app.web_app import launch
    # share=False — we handle the tunnel ourselves via localhost.run
    launch(share=False, auth=auth_pair, port=args.port)


if __name__ == "__main__":
    main()
