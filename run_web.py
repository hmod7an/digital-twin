"""
Face Health Digital Twin — Web Launcher

Usage:
    python run_web.py                   # local → http://localhost:7860
    python run_web.py --share           # public HTTPS via Cloudflare Tunnel (fastest)
    python run_web.py --share --auth    # with login prompt (demo / demo1234)
    python run_web.py --port 8080       # custom port

Tunnel priority (--share):
    1. Cloudflare Tunnel (cloudflared.exe)  — global CDN, lowest latency
    2. localhost.run SSH fallback           — if cloudflared not found
"""
import argparse
import os
import re
import subprocess
import sys
import threading

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Cloudflare binary lives next to this script (or anywhere on PATH)
_CF_BIN = os.path.join(_ROOT, "cloudflared.exe")


# ── QR code ───────────────────────────────────────────────────────────────────

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
        pass


def _announce(url: str, provider: str):
    """Print the public URL banner + QR code."""
    print("\n" + "=" * 60)
    print(f"  PUBLIC URL  [{provider}]")
    print(f"  {url}")
    print("=" * 60)
    print("  Scan with your phone camera:")
    _print_qr(url)
    print(flush=True)


# ── Cloudflare Tunnel ─────────────────────────────────────────────────────────

def _start_cloudflare(port: int) -> bool:
    """
    Start a Cloudflare Quick Tunnel to localhost:port.
    Prints the *.trycloudflare.com URL and QR code when ready.
    Returns True if cloudflared binary was found and launched.
    """
    cf = _CF_BIN if os.path.isfile(_CF_BIN) else "cloudflared"
    try:
        proc = subprocess.Popen(
            [cf, "tunnel", "--url", f"http://localhost:{port}",
             "--no-autoupdate"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError:
        return False

    def _watch():
        printed = False
        for line in proc.stdout:
            if not printed:
                m = re.search(r"(https://[a-z0-9\-]+\.trycloudflare\.com)", line)
                if m:
                    _announce(m.group(1), "Cloudflare — global CDN")
                    printed = True

    threading.Thread(target=_watch, daemon=True).start()
    return True


# ── localhost.run SSH fallback ────────────────────────────────────────────────

def _start_localtunnel(port: int):
    """
    Open an SSH reverse tunnel to localhost.run.
    Used only when Cloudflare binary is not present.
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
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        for line in proc.stdout:
            m = re.search(r"(https://[^\s]+\.lhr\.life)", line)
            if m:
                _announce(m.group(1), "localhost.run SSH")
    except FileNotFoundError:
        print("  [share] ssh not found — cannot open tunnel", flush=True)
    except Exception as e:
        print(f"  [share] tunnel error: {e}", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Face Health Digital Twin — Web App")
    parser.add_argument("--share", action="store_true",
                        help="Create a public HTTPS tunnel (Cloudflare CDN)")
    parser.add_argument("--auth",  action="store_true",
                        help="Require username/password (demo / demo1234)")
    parser.add_argument("--port",  type=int, default=7860,
                        help="Local port (default 7860)")
    args = parser.parse_args()

    auth_pair = ("demo", "demo1234") if args.auth else None

    print("=" * 60)
    print("  Face Health Digital Twin  —  Web Interface")
    print("=" * 60)
    if args.share:
        print("  Share mode ON  →  public URL will appear below")
        print("  Tunnel: Cloudflare CDN (fastest) → localhost.run fallback")
    if args.auth:
        print("  Auth ON  →  login: demo / demo1234")
    print(f"  Local URL: http://localhost:{args.port}")
    print("=" * 60)
    print()

    if args.share:
        # Try Cloudflare first; fall back to SSH tunnel
        if not _start_cloudflare(args.port):
            print("  [share] cloudflared not found — falling back to localhost.run",
                  flush=True)
            t = threading.Thread(target=_start_localtunnel,
                                 args=(args.port,), daemon=True)
            t.start()

    from app.web_app import launch
    launch(share=False, auth=auth_pair, port=args.port)


if __name__ == "__main__":
    main()
