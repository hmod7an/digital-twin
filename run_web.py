"""
Face Health Digital Twin — Web Launcher

Usage:
    python run_web.py                   # local → http://localhost:7860
    python run_web.py --share           # public Gradio link (72-h free tunnel)
    python run_web.py --share --auth    # with login prompt
    python run_web.py --port 8080       # custom port
"""
import argparse
import sys
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    parser = argparse.ArgumentParser(description="Face Health Digital Twin — Web App")
    parser.add_argument("--share",  action="store_true",
                        help="Create a public Gradio tunnel link (72 hours)")
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

    from app.web_app import launch
    launch(share=args.share, auth=auth_pair, port=args.port)


if __name__ == "__main__":
    main()
