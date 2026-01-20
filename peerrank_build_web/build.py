#!/usr/bin/env python3
"""
Unified build script for PeerRank website.

Usage:
    python peerrank_build_web/build.py              # Generate webpage only
    python peerrank_build_web/build.py --serve      # Generate + start local server
    python peerrank_build_web/build.py --watch      # Generate + serve + latency monitor
    python peerrank_build_web/build.py --port 8080  # Custom port (default: 8000)
"""

import argparse
import http.server
import socketserver
import threading
import sys
import asyncio
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from peerrank_build_web.generate_webpage import generate_webpage


def start_server(port: int, directory: Path):
    """Start a simple HTTP server for local testing."""
    import os
    os.chdir(directory)

    handler = http.server.SimpleHTTPRequestHandler
    handler.extensions_map.update({
        '.js': 'application/javascript',
        '.json': 'application/json',
    })

    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"\nServing at http://localhost:{port}")
        print(f"Directory: {directory}")
        print("Press Ctrl+C to stop\n")
        httpd.serve_forever()


def start_latency_monitor(interval: int):
    """Start the latency monitor in a separate thread."""
    from peerrank_build_web.latency_monitor import run_monitor

    def run():
        asyncio.run(run_monitor(interval, once=False))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    print(f"Latency monitor started (interval: {interval}s)")
    return thread


def main():
    parser = argparse.ArgumentParser(description="Build and serve PeerRank website")
    parser.add_argument("--serve", "-s", action="store_true",
                        help="Start local HTTP server after building")
    parser.add_argument("--watch", "-w", action="store_true",
                        help="Start server + latency monitor")
    parser.add_argument("--port", "-p", type=int, default=8000,
                        help="Server port (default: 8000)")
    parser.add_argument("--interval", "-i", type=int, default=60,
                        help="Latency check interval in seconds (default: 60)")
    parser.add_argument("--no-build", action="store_true",
                        help="Skip webpage generation (serve only)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    peerrank_hosted_website_dir = project_root / "peerrank_hosted_website"

    # Generate webpage
    if not args.no_build:
        print("=" * 50)
        print("  Building PeerRank Website")
        print("=" * 50)
        success = generate_webpage()
        if not success:
            print("\nBuild failed!")
            return 1
        print("\nBuild complete!")

    # Start latency monitor if --watch
    if args.watch:
        start_latency_monitor(args.interval)

    # Start server if --serve or --watch
    if args.serve or args.watch:
        try:
            start_server(args.port, peerrank_hosted_website_dir)
        except KeyboardInterrupt:
            print("\nServer stopped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
