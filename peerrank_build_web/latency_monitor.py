#!/usr/bin/env python3
"""
Latency Monitor - Continuously checks model response times.
Pings each model with a minimal query every minute and saves results to JSON.

Usage:
    python peerrank_build_web/latency_monitor.py              # Run continuously
    python peerrank_build_web/latency_monitor.py --once       # Single check, then exit
    python peerrank_build_web/latency_monitor.py --interval 30  # Check every 30 seconds
"""

import asyncio
import json
import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODELS, get_api_key
from providers import call_llm, clear_clients

# Output file for website
OUTPUT_FILE = Path(__file__).parent.parent / "peerrank_hosted_website" / "latency.json"

# Minimal prompt for speed test
TEST_PROMPT = "Reply with only the word 'OK'."

# How many recent checks to keep per model
HISTORY_SIZE = 60  # 1 hour of data at 1-minute intervals


async def check_model_latency(provider: str, model_id: str, name: str) -> dict:
    """Check latency for a single model."""
    try:
        api_key = get_api_key(provider)
        if not api_key:
            return {"model": name, "status": "no_key", "latency_ms": None}

        content, duration, _, _ = await call_llm(
            provider=provider,
            model=model_id,
            prompt=TEST_PROMPT,
            max_tokens=10,
            timeout=30,
            use_web_search=False,
            temperature=0
        )

        return {
            "model": name,
            "status": "ok",
            "latency_ms": round(duration * 1000),
            "response": content[:50]  # Truncate for debugging
        }

    except Exception as e:
        error_msg = str(e)[:100]
        return {
            "model": name,
            "status": "error",
            "latency_ms": None,
            "error": error_msg
        }


async def check_all_models() -> dict:
    """Check latency for all models in parallel."""
    tasks = [
        check_model_latency(provider, model_id, name)
        for provider, model_id, name in MODELS
    ]

    results = await asyncio.gather(*tasks)

    return {
        "timestamp": datetime.now().isoformat(),
        "results": {r["model"]: r for r in results}
    }


def load_existing_data() -> dict:
    """Load existing latency data if it exists."""
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"history": [], "current": None, "models": {}}


def update_data(existing: dict, new_check: dict) -> dict:
    """Update data with new check, maintaining history."""
    # Update current
    existing["current"] = new_check

    # Add to history
    existing["history"].append({
        "timestamp": new_check["timestamp"],
        "latencies": {
            model: data.get("latency_ms")
            for model, data in new_check["results"].items()
        }
    })

    # Trim history
    if len(existing["history"]) > HISTORY_SIZE:
        existing["history"] = existing["history"][-HISTORY_SIZE:]

    # Update per-model stats
    for model, data in new_check["results"].items():
        if model not in existing["models"]:
            existing["models"][model] = {
                "recent_latencies": [],
                "status": "unknown",
                "avg_latency_ms": None,
                "min_latency_ms": None,
                "max_latency_ms": None,
                "last_check": None,
                "uptime_pct": None
            }

        model_data = existing["models"][model]
        model_data["last_check"] = new_check["timestamp"]
        model_data["status"] = data["status"]

        if data["latency_ms"] is not None:
            model_data["recent_latencies"].append(data["latency_ms"])
            # Keep only recent values
            if len(model_data["recent_latencies"]) > HISTORY_SIZE:
                model_data["recent_latencies"] = model_data["recent_latencies"][-HISTORY_SIZE:]

            # Calculate stats
            latencies = model_data["recent_latencies"]
            model_data["avg_latency_ms"] = round(sum(latencies) / len(latencies))
            model_data["min_latency_ms"] = min(latencies)
            model_data["max_latency_ms"] = max(latencies)

        # Calculate uptime from history
        model_history = [
            h["latencies"].get(model) for h in existing["history"]
            if model in h["latencies"]
        ]
        if model_history:
            successful = sum(1 for lat in model_history if lat is not None)
            model_data["uptime_pct"] = round(100 * successful / len(model_history), 1)

    # Update summary stats
    existing["last_updated"] = new_check["timestamp"]
    existing["check_count"] = len(existing["history"])

    return existing


def save_data(data: dict):
    """Save data to JSON file."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def print_status(check: dict):
    """Print current status to console."""
    print(f"\n[{check['timestamp']}] Latency Check")
    print("-" * 50)

    # Sort by latency (None values at end)
    sorted_results = sorted(
        check["results"].items(),
        key=lambda x: (x[1]["latency_ms"] is None, x[1]["latency_ms"] or 99999)
    )

    for model, data in sorted_results:
        if data["status"] == "ok":
            latency = data["latency_ms"]
            # Color coding (ANSI)
            if latency < 1000:
                color = "\033[92m"  # Green
            elif latency < 3000:
                color = "\033[93m"  # Yellow
            else:
                color = "\033[91m"  # Red
            print(f"  {model:<25} {color}{latency:>6}ms\033[0m")
        elif data["status"] == "no_key":
            print(f"  {model:<25} \033[90m   no key\033[0m")
        else:
            print(f"  {model:<25} \033[91m   error\033[0m")


async def run_monitor(interval: int, once: bool):
    """Main monitoring loop."""
    print(f"Starting latency monitor (interval: {interval}s)")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Models: {len(MODELS)}")

    while True:
        # Clear clients before each check to avoid stale connections
        clear_clients()

        # Run check
        check = await check_all_models()

        # Update and save
        data = load_existing_data()
        data = update_data(data, check)
        save_data(data)

        # Print status
        print_status(check)

        if once:
            print("\nSingle check complete.")
            break

        # Wait for next interval
        print(f"\nNext check in {interval}s... (Ctrl+C to stop)")
        await asyncio.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Monitor LLM latency")
    parser.add_argument("--interval", "-i", type=int, default=60,
                        help="Check interval in seconds (default: 60)")
    parser.add_argument("--once", "-o", action="store_true",
                        help="Run single check and exit")
    args = parser.parse_args()

    try:
        asyncio.run(run_monitor(args.interval, args.once))
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
