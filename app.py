"""
app.py
------
Command-line interface for the Online Store AI Agent.

Why a separate app.py?
  - Keeps the agent.py module import-safe (no side-effects on import).
  - Allows the same agent to be called from app.py, streamlit_app.py, or
    unit tests without executing a REPL loop.
  - Provides a quick way to smoke-test the agent during development.

Usage:
    python app.py
    python app.py --question "Where is order ORD-1002?"
"""

import argparse
import sys

from agent import run_agent
from utils import get_logger

logger = get_logger(__name__)


def interactive_loop() -> None:
    """Run an interactive REPL loop for the customer agent."""
    print("\n" + "=" * 60)
    print("  Online Store AI Agent  |  Type 'quit' to exit")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        print("\nAgent: ", end="", flush=True)
        try:
            answer = run_agent(question)
            print(answer)
        except Exception as exc:
            logger.error("Unhandled error: %s", exc, exc_info=True)
            print(f"[Error] {exc}")

        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Online Store AI Agent — answers customer questions."
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        default=None,
        help="A single question to answer (non-interactive mode).",
    )
    args = parser.parse_args()

    if args.question:
        # Non-interactive: answer a single question and exit
        print(run_agent(args.question))
        sys.exit(0)
    else:
        # Interactive REPL
        interactive_loop()


if __name__ == "__main__":
    main()
