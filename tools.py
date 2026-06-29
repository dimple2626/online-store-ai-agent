"""
tools.py
--------
Pure, LLM-agnostic tool implementations for the Online Store Agent.

Why separate from the agent?
  Tools are the *source of truth* for data retrieval.  By keeping them
  decoupled from LLM code, they can be:
    - Unit-tested without mocking API calls
    - Reused in non-agent contexts (e.g. a REST API)
    - Swapped for live database queries without touching agent logic

Each tool function:
  1. Accepts simple scalar arguments (strings)
  2. Returns a structured dict (JSON-serialisable, suitable for LLM context)
  3. Returns an explicit error dict rather than raising exceptions — the
     agent layer can relay the error message without crashing

Data source: flat JSON files in data/ (easy to replace with a DB later).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import ORDERS_FILE, PRODUCTS_FILE
from utils import get_logger, log_tool_call

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal data-loading helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> list[dict]:
    """Load a JSON file and return its contents as a list of dicts.

    Args:
        path: Absolute path to the JSON file.

    Returns:
        Parsed list of dicts, or an empty list on failure.
    """
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.error("Failed to load %s: %s", path, exc)
        return []


# ---------------------------------------------------------------------------
# Public tool functions — these are registered with the LLM as callable tools
# ---------------------------------------------------------------------------

def get_order(order_id: str) -> dict[str, Any]:
    """Retrieve a single order by its ID.

    Args:
        order_id: The order identifier, e.g. ``"ORD-1002"``.

    Returns:
        The order dict if found, otherwise an error dict with key ``"error"``.

    Example return (success)::

        {
          "order_id": "ORD-1002",
          "customer_name": "Bob Smith",
          "status": "In Transit",
          ...
        }

    Example return (failure)::

        {"error": "Order ORD-9999 not found."}
    """
    orders = _load_json(ORDERS_FILE)
    normalised_id = order_id.strip().upper()

    result: dict[str, Any]
    for order in orders:
        if order.get("order_id", "").upper() == normalised_id:
            result = order
            break
    else:
        result = {"error": f"Order {normalised_id} not found."}

    log_tool_call(logger, "get_order", {"order_id": order_id}, result)
    return result


def get_product(product_id: str) -> dict[str, Any]:
    """Retrieve a single product by its ID.

    Args:
        product_id: The product identifier, e.g. ``"P-101"``.

    Returns:
        The product dict if found, otherwise an error dict with key ``"error"``.

    Example return (success)::

        {
          "product_id": "P-101",
          "name": "Running Shoes Pro",
          "category": "shoes",
          "price": 89.99,
          ...
        }

    Example return (failure)::

        {"error": "Product P-999 not found."}
    """
    products = _load_json(PRODUCTS_FILE)
    normalised_id = product_id.strip().upper()

    result: dict[str, Any]
    for product in products:
        if product.get("product_id", "").upper() == normalised_id:
            result = product
            break
    else:
        result = {"error": f"Product {normalised_id} not found."}

    log_tool_call(logger, "get_product", {"product_id": product_id}, result)
    return result


def search_products(query: str) -> dict[str, Any]:
    """Full-text search over product names, descriptions, categories, and tags.

    The search is intentionally simple (keyword overlap scoring) so the
    agent can run it without a vector database.  In production this would
    call Elasticsearch or a similar service.

    Args:
        query: Natural-language or keyword search string.

    Returns:
        A dict with a ``"results"`` list of matching products (sorted by
        relevance score, then by price ascending), or an error dict.

    Example return::

        {
          "results": [
            {"product_id": "P-102", "name": "Casual Canvas Sneakers", ...},
            ...
          ],
          "count": 3
        }
    """
    products = _load_json(PRODUCTS_FILE)
    query_tokens = set(query.lower().split())

    scored: list[tuple[int, dict]] = []
    for product in products:
        # Build a searchable text blob for this product
        blob = " ".join([
            product.get("name", ""),
            product.get("description", ""),
            product.get("category", ""),
            product.get("brand", ""),
            " ".join(product.get("tags", [])),
        ]).lower()

        blob_tokens = set(blob.split())
        score = len(query_tokens & blob_tokens)

        if score > 0:
            scored.append((score, product))

    # Sort: highest relevance first, then cheapest first for tie-breaking
    scored.sort(key=lambda x: (-x[0], x[1].get("price", 0)))

    matches = [p for _, p in scored]

    if matches:
        result: dict[str, Any] = {"results": matches, "count": len(matches)}
    else:
        result = {"results": [], "count": 0}

    log_tool_call(logger, "search_products", {"query": query}, result)
    return result


# ---------------------------------------------------------------------------
# Tool registry — used by agent.py to map LLM tool-names to Python functions
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, Any] = {
    "get_order": get_order,
    "get_product": get_product,
    "search_products": search_products,
}


# ---------------------------------------------------------------------------
# Tool schema definitions — passed to the LLM so it knows what tools exist
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "get_order",
        "description": (
            "Retrieve details for a specific customer order by its order ID. "
            "Returns order status, items, shipping address, tracking number "
            "(if available), and estimated or actual delivery date."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID, e.g. 'ORD-1002'.",
                }
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "get_product",
        "description": (
            "Retrieve full details for a specific product by its product ID. "
            "Returns name, category, brand, price, stock status, rating, and description."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "The product ID, e.g. 'P-101'.",
                }
            },
            "required": ["product_id"],
        },
    },
    {
        "name": "search_products",
        "description": (
            "Search the product catalogue using a natural-language or keyword query. "
            "Use this to find alternatives, browse categories, or look up products "
            "when only a name or type is known (not an exact product ID). "
            "Results are sorted by relevance and then by price (lowest first)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Keywords or a short phrase describing what to search for, "
                        "e.g. 'cheap running shoes' or 'wireless headphones'."
                    ),
                }
            },
            "required": ["query"],
        },
    },
]
