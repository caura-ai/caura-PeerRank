"""
PeerRank - LLM Peer Evaluation System

4-phase evaluation where LLMs:
  1. Generate questions
  2. Answer all questions (with web search)
  3. Cross-evaluate in 3 bias modes (shuffle, blind, both)
  4. Generate ranked report with bias analysis

Usage:
  python peerrank.py              # Interactive menu
  python peerrank.py --phase 1    # Run specific phase
  python peerrank.py --all        # All phases
  python peerrank.py --resume     # Resume from last completed
  python peerrank.py --models gpt-5.2,claude-opus-4-5  # Include only these models
  python peerrank.py --exclude gemini-3-pro-preview    # Exclude these models
  python peerrank.py --seed 42    # Reproducible shuffle ordering
"""

import asyncio
import argparse

from peerrank import config
from peerrank.config import (
    MODELS, ALL_MODELS, CATEGORIES, ALL_CATEGORIES,
    get_last_completed_phase,
    set_active_models, list_available_models,
    set_active_categories, list_available_categories,
    get_revision, set_revision,
    set_bias_test_config, get_bias_test_config,
    get_phase2_web_search, set_phase2_web_search,
    get_phase3_web_search, set_phase3_web_search,
    get_phase5_judge, set_phase5_judge,
    get_web_grounding_provider, set_web_grounding_provider,
)
from peerrank.providers import health_check
from peerrank_phase1 import phase1_generate_questions
from peerrank_phase2 import phase2_answer_questions
from peerrank_phase3 import phase3_evaluate_answers
from peerrank_phase4 import phase4_generate_report
from peerrank_phase5 import phase5_final_analysis

# Phase functions indexed by phase number (1-5)
PHASES = {
    1: phase1_generate_questions,
    2: phase2_answer_questions,
    3: phase3_evaluate_answers,
    4: phase4_generate_report,
    5: phase5_final_analysis,
}


async def resume_from_last():
    """Resume evaluation from last completed phase."""
    last = get_last_completed_phase()
    print(f"Resuming from phase {last + 1}...")
    for i in range(last + 1, 6):
        phase = PHASES[i]
        if asyncio.iscoroutinefunction(phase):
            await phase()
        else:
            phase()


def show_menu():
    """Display interactive menu and get user choice."""
    print("\n" + "=" * 50)
    print("         PEERRANK.AI - LLM Evaluation")
    print("=" * 50)

    # Setup
    print(f"\n  Revision: {get_revision()}  |  Progress: {get_last_completed_phase()}/5")
    print(f"  Models: {len(MODELS)}/{len(ALL_MODELS)}  |  Categories: {len(CATEGORIES)}/{len(ALL_CATEGORIES)}  |  Questions: {config.NUM_QUESTIONS}/model")

    # Phase settings
    ws2 = "ON" if get_phase2_web_search() else "OFF"
    ws3 = "ON" if get_phase3_web_search() else "OFF"
    seed = get_bias_test_config().get("seed")
    seed_str = str(seed) if seed is not None else "rand"
    judge = get_phase5_judge()[2]
    print(f"  P2: web={ws2}  |  P3: seed={seed_str}, web-grounding={ws3}  |  P5: {judge}")

    provider = get_web_grounding_provider().upper()
    print(f"  Grounding: {provider}")

    print("""
  --- Run ---
  [1-5] Run phase    [A] All    [R] Resume    [H] Health check

  --- Setup ---
  [V] Revision    [M] Models    [N] Questions    [C] Categories    [S] Search provider

  --- Phase Settings ---
  [W] P2 web grounding  [D] P3 seed    [G] P3 web grounding
  [J] P5 judge

  [Q] Quit
""")
    return input("  Select option: ").strip().upper()


def change_settings():
    """Change evaluation settings."""
    print(f"\n  Current questions per model: {config.NUM_QUESTIONS}")
    try:
        num = int(input("  Enter new value (1-100): ").strip())
        if 1 <= num <= 100:
            config.NUM_QUESTIONS = num
            print(f"  Updated to {num} questions per model")
        else:
            print("  Invalid: must be between 1 and 100")
    except ValueError:
        print("  Invalid: enter a number")


def change_grounding_provider():
    """Change web grounding provider."""
    current = get_web_grounding_provider()
    print(f"\n  Current grounding provider: {current.upper()}")
    print("  Available providers:")
    print("    [1] TAVILY  - $0.008/search, includes AI summary")
    print("    [2] SERPAPI - $0.010/search, Google results + knowledge graph")
    choice = input("  Select provider (1-2): ").strip()
    if choice == "1":
        set_web_grounding_provider("tavily")
        print("  Updated to TAVILY")
    elif choice == "2":
        set_web_grounding_provider("serpapi")
        print("  Updated to SERPAPI")
    else:
        print("  Cancelled")


def change_revision():
    """Change revision tag."""
    print(f"\n  Current revision: {get_revision()}")
    rev = input("  Enter new revision tag: ").strip()
    if rev:
        set_revision(rev)
        print(f"  Updated to: {get_revision()}")
    else:
        print("  Revision unchanged")


def change_seed():
    """Set random seed for Phase 3 shuffle ordering."""
    cfg = get_bias_test_config()
    current = cfg['seed'] if cfg['seed'] is not None else "random"
    print(f"\n  Current seed: {current}")
    print("  Phase 3 runs 3 passes: shuffle_only, blind_only, shuffle_blind")
    print("  A fixed seed ensures reproducible answer ordering.\n")

    seed_input = input("  Enter seed (number) or 'random': ").strip()
    if seed_input.lower() == "random" or not seed_input:
        set_bias_test_config(seed=None)
        print("  Seed: random")
    else:
        try:
            seed = int(seed_input)
            set_bias_test_config(seed=seed)
            print(f"  Seed: {seed}")
        except ValueError:
            print("  Invalid: enter a number or 'random'")


def toggle_setting(name, getter, setter, description):
    """Generic toggle for boolean settings."""
    current = getter()
    print(f"\n  Current {name}: {'ON' if current else 'OFF'}")
    print(f"  {description}\n")

    choice = input(f"  Enable {name}? (y/n): ").strip().lower()
    if choice in ("y", "yes"):
        setter(True)
        print(f"  {name}: ON")
    elif choice in ("n", "no"):
        setter(False)
        print(f"  {name}: OFF")
    else:
        print("  Unchanged")


def select_judge():
    """Select judge model for Phase 5 analysis."""
    current = get_phase5_judge()
    all_names = list_available_models()

    print(f"\n  Current Phase 5 judge: {current[2]}")
    print("\n  Available models:")
    for i, name in enumerate(all_names, 1):
        marker = " *" if name == current[2] else ""
        print(f"    {i}. {name}{marker}")

    print("\n  Enter number to select judge, or Enter to keep current:")
    choice = input("  Selection: ").strip()

    if not choice:
        return

    try:
        n = int(choice)
        if 1 <= n <= len(all_names):
            selected_name = all_names[n - 1]
            # Find the full model tuple
            for provider, model_id, name in ALL_MODELS:
                if name == selected_name:
                    set_phase5_judge(provider, model_id, name)
                    print(f"  Phase 5 judge: {name}")
                    return
        print("  Invalid selection")
    except ValueError:
        print("  Invalid input")


def select_items(name, all_items, active_items, set_func, all_count, format_active):
    """Generic interactive selection for models/categories."""
    print(f"\n  Available {name}:")
    for i, item in enumerate(all_items, 1):
        print(f"    {i}. {'[x]' if item in active_items else '[ ]'} {item}")

    print("\n  Enter numbers to toggle, 'all', 'none', or Enter to keep:")
    choice = input("  Selection: ").strip().lower()

    if not choice:
        return
    if choice == "all":
        set_func()
        print(f"  Selected all {all_count} {name}")
        return
    if choice == "none":
        set_func(include=["__none__"])
        print("  Cleared selection")
        return

    try:
        for n in [int(x) for x in choice.replace(",", " ").split()]:
            if 1 <= n <= len(all_items):
                item = all_items[n - 1]
                active_items.remove(item) if item in active_items else active_items.append(item)
        if active_items:
            set_func(include=active_items)
            print(f"  Active: {format_active()}")
        else:
            print(f"  Warning: No {name} selected!")
    except ValueError:
        print("  Invalid input")


async def run_phase(n: int):
    """Run a specific phase by number."""
    phase = PHASES[n]
    if asyncio.iscoroutinefunction(phase):
        await phase()
    else:
        phase()


async def main():
    parser = argparse.ArgumentParser(
        description="PeerRank - LLM Evaluation System",
        epilog=f"Available models: {', '.join(list_available_models())}"
    )
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5], help="Run specific phase")
    parser.add_argument("--resume", action="store_true", help="Resume from last completed")
    parser.add_argument("--all", action="store_true", help="Run all phases")
    parser.add_argument("--health", action="store_true", help="Run API health check")
    parser.add_argument("--models", type=str, help="Models to include (comma-separated)")
    parser.add_argument("--exclude", type=str, help="Models to exclude (comma-separated)")
    parser.add_argument("--categories", type=str, help="Categories to include (comma-separated keywords)")
    parser.add_argument("--exclude-categories", type=str, help="Categories to exclude (comma-separated keywords)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible Phase 3 shuffle ordering")
    parser.add_argument("--web-search", type=str, choices=["on", "off"], help="Enable/disable web grounding in Phase 2")
    parser.add_argument("--web-search-3", type=str, choices=["on", "off"], help="Enable/disable web grounding in Phase 3")
    parser.add_argument("--grounding-provider", type=str, choices=["tavily", "serpapi"], help="Web grounding provider (default: tavily)")
    parser.add_argument("--judge", type=str, help="Judge model for Phase 5 analysis (model name)")
    parser.add_argument("--rev", type=str, help="Set revision tag")
    args = parser.parse_args()

    # Apply revision if provided
    if args.rev:
        set_revision(args.rev)
        print(f"Revision: {args.rev}")

    # Apply model filters
    if args.models or args.exclude:
        set_active_models(
            include=[m.strip() for m in args.models.split(",")] if args.models else None,
            exclude=[m.strip() for m in args.exclude.split(",")] if args.exclude else None
        )
        print(f"Active models: {', '.join(m[2] for m in MODELS)}")

    # Apply category filters
    if args.categories or args.exclude_categories:
        set_active_categories(
            include=[c.strip() for c in args.categories.split(",")] if args.categories else None,
            exclude=[c.strip() for c in args.exclude_categories.split(",")] if args.exclude_categories else None
        )
        print(f"Active categories: {len(CATEGORIES)}")

    # Apply seed for Phase 3
    if args.seed is not None:
        set_bias_test_config(seed=args.seed)
        print(f"Phase 3 seed: {args.seed}")

    # Apply web search setting for Phase 2
    if args.web_search:
        set_phase2_web_search(args.web_search.lower() == "on")
        print(f"Phase 2 web grounding: {args.web_search.upper()}")

    # Apply web search setting for Phase 3
    if args.web_search_3:
        set_phase3_web_search(args.web_search_3.lower() == "on")
        print(f"Phase 3 web grounding: {args.web_search_3.upper()}")

    # Apply grounding provider
    if args.grounding_provider:
        set_web_grounding_provider(args.grounding_provider.lower())
        print(f"Web grounding provider: {args.grounding_provider.upper()}")

    # Apply judge for Phase 5
    if args.judge:
        for provider, model_id, name in ALL_MODELS:
            if name.lower() == args.judge.lower():
                set_phase5_judge(provider, model_id, name)
                print(f"Phase 5 judge: {name}")
                break
        else:
            print(f"Warning: Judge model '{args.judge}' not found")

    # CLI mode
    if args.phase:
        await run_phase(args.phase)
        return
    if args.resume:
        await resume_from_last()
        return
    if args.all:
        for phase in range(1, 6):
            await run_phase(phase)
        return
    if args.health:
        await health_check()
        return

    # Interactive menu
    while True:
        choice = show_menu()

        if choice in ("1", "2", "3", "4", "5"):
            await run_phase(int(choice))
        elif choice == "A":
            for phase in range(1, 6):
                await run_phase(phase)
        elif choice == "R":
            await resume_from_last()
        elif choice == "H":
            await health_check()
        elif choice == "M":
            select_items("models", list_available_models(), [m[2] for m in MODELS],
                         set_active_models, len(ALL_MODELS), lambda: ', '.join(m[2] for m in MODELS))
            continue
        elif choice == "C":
            select_items("categories", list_available_categories(), CATEGORIES.copy(),
                         set_active_categories, len(ALL_CATEGORIES), lambda: f"{len(CATEGORIES)} categories")
            continue
        elif choice == "N":
            change_settings()
            continue
        elif choice == "S":
            change_grounding_provider()
            continue
        elif choice == "W":
            toggle_setting("Phase 2 web grounding", get_phase2_web_search, set_phase2_web_search,
                           "Web search enables models to access current information.")
            continue
        elif choice == "D":
            change_seed()
            continue
        elif choice == "G":
            toggle_setting("Phase 3 web grounding", get_phase3_web_search, set_phase3_web_search,
                           "Native search models can fact-check (excludes Tavily models).")
            continue
        elif choice == "J":
            select_judge()
            continue
        elif choice == "V":
            change_revision()
            continue
        elif choice == "Q":
            print("\nGoodbye!")
            break
        else:
            print("\nInvalid option.")
            continue

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    asyncio.run(main())
