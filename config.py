"""
config.py
---------
Central configuration module for the Online Store AI Agent.

Centralising configuration here means:
  - No hard-coded values scattered across files
  - Easy to swap models or paths without touching business logic
  - Environment variables are the single source of truth for secrets
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
ORDERS_FILE = DATA_DIR / "orders.json"
PRODUCTS_FILE = DATA_DIR / "products.json"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# LLM provider selection
# Supported values: "anthropic" | "openai" | "gemini"
# ---------------------------------------------------------------------------
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic")

# ---------------------------------------------------------------------------
# Anthropic / Claude settings
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")

# ---------------------------------------------------------------------------
# OpenAI settings
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")

# ---------------------------------------------------------------------------
# Google Gemini settings
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

# ---------------------------------------------------------------------------
# Agent behaviour
# ---------------------------------------------------------------------------
MAX_TOOL_ITERATIONS: int = 10          # Safety cap to prevent infinite loops
AGENT_TEMPERATURE: float = 0.0         # 0 = deterministic, reduces hallucination
AGENT_MAX_TOKENS: int = 1024

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: Path = LOG_DIR / "agent.log"

# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def get_active_api_key() -> str:
    """Return the API key for the currently selected provider.

    Raises:
        ValueError: If the required API key is not set.
    """
    mapping = {
        "anthropic": ANTHROPIC_API_KEY,
        "openai": OPENAI_API_KEY,
        "gemini": GEMINI_API_KEY,
    }
    key = mapping.get(LLM_PROVIDER, "")
    if not key:
        raise ValueError(
            f"API key for provider '{LLM_PROVIDER}' is not set. "
            f"Please export the corresponding environment variable."
        )
    return key
