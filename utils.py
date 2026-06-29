"""
utils.py
--------
Shared utility functions used across the project:
  - Centralised logger factory  (all modules get consistent formatting)
  - Tool-call pretty printer     (human-readable audit trail in logs/console)
  - Response sanitisation helper (strips stray whitespace before returning)

Why a separate utils module?
  Prevents each file from re-implementing logging setup, avoids circular
  imports, and gives one place to upgrade formatting or add metrics later.
"""

import json
import logging
import textwrap
from pathlib import Path
from typing import Any

from config import LOG_FILE, LOG_LEVEL


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Create (or retrieve) a named logger with file + console handlers.

    The first call configures handlers; subsequent calls with the same
    *name* return the cached logger without adding duplicate handlers.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    # Guard: only configure once
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — persists logs between runs
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(formatter)

    # Console handler — visible during development
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Tool-call audit logger
# ---------------------------------------------------------------------------

def log_tool_call(logger: logging.Logger, tool_name: str, args: dict, result: Any) -> None:
    """Pretty-print a tool invocation to the log.

    Format mirrors the assignment requirement:

        Tool Called:
        get_order("ORD-1002")

        Returned:
        { ... }

    Args:
        logger:    The logger to write to.
        tool_name: The function name, e.g. ``"get_order"``.
        args:      The kwargs passed to the tool.
        result:    Whatever the tool returned (serialised to JSON).
    """
    # Build a "function signature" style string, e.g. get_order("ORD-1002")
    arg_str = ", ".join(f'"{v}"' if isinstance(v, str) else str(v) for v in args.values())
    signature = f'{tool_name}({arg_str})'

    try:
        result_str = json.dumps(result, indent=2, default=str)
    except (TypeError, ValueError):
        result_str = str(result)

    separator = "─" * 60
    message = (
        f"\n{separator}\n"
        f"Tool Called:\n  {signature}\n\n"
        f"Returned:\n{textwrap.indent(result_str, '  ')}\n"
        f"{separator}"
    )
    logger.info(message)


# ---------------------------------------------------------------------------
# Response cleanup
# ---------------------------------------------------------------------------

def clean_response(text: str) -> str:
    """Strip leading/trailing whitespace and normalise blank lines.

    Args:
        text: Raw LLM or tool output string.

    Returns:
        Cleaned string safe to return to the user.
    """
    lines = text.splitlines()
    # Collapse runs of blank lines into a single blank line
    cleaned: list[str] = []
    prev_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and prev_blank:
            continue
        cleaned.append(line)
        prev_blank = is_blank

    return "\n".join(cleaned).strip()
