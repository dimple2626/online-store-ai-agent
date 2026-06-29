# 🛍️ Online Store AI Agent

> An AI-powered customer service agent that answers product and order questions using natural language, autonomous tool calling, and multi-step reasoning.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Flow Diagrams](#flow-diagrams)
- [Installation](#installation)
- [Configuration](#configuration)
- [How to Run](#how-to-run)
- [How the Agent Works](#how-the-agent-works)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Running Tests](#running-tests)
- [Future Improvements](#future-improvements)
- [Screenshots](#screenshots)

---

## Project Overview

The Online Store AI Agent is a production-quality conversational AI built on top of a large language model (Claude / GPT-4o / Gemini). It serves as a customer service assistant that can:

- 📦 **Track orders** — look up status, delivery dates, and tracking numbers
- 🔍 **Search products** — find items by keyword, category, or description
- 📋 **Describe products** — provide detailed, friendly product summaries
- 💰 **Compare prices** — find cheaper alternatives to products in an order
- 🔗 **Chain multiple tools** — autonomously combine tool calls to answer complex questions

The agent **never hallucinates** — it always calls tools before making factual claims, and it returns honest error messages when data is not found.

---

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                   User Interface Layer                  │
│           app.py (CLI) │ streamlit_app.py (Web)         │
└─────────────────────────┬──────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────┐
│                    Agent Layer (agent.py)               │
│                                                        │
│   ReAct Loop:  LLM ←→ Tool Executor                    │
│   • Builds conversation history                        │
│   • Routes tool calls via TOOL_REGISTRY                │
│   • Stops when LLM returns end_turn                    │
└──────────┬────────────────────────┬───────────────────┘
           │                        │
┌──────────▼──────────┐   ┌────────▼────────────────────┐
│   LLM Providers     │   │   Tool Layer (tools.py)     │
│                     │   │                             │
│  • Anthropic Claude │   │  get_order(order_id)        │
│  • OpenAI GPT-4o    │   │  get_product(product_id)    │
│  • Google Gemini    │   │  search_products(query)     │
└─────────────────────┘   └──────────┬──────────────────┘
                                     │
                          ┌──────────▼──────────────────┐
                          │   Data Layer (data/)        │
                          │                             │
                          │  orders.json  (10 orders)   │
                          │  products.json (17 products)│
                          └─────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| **ReAct agentic loop** | LLM decides which tools to call; no hand-coded intent routing |
| **Tool calling over RAG** | Live data retrieval; no stale embeddings |
| **Temperature = 0** | Deterministic, minimises hallucination |
| **Typed tool schemas** | LLM knows exactly what arguments to pass |
| **Provider abstraction** | Swap Claude/GPT-4o/Gemini with one env variable |

---

## Flow Diagrams

### Simple Order Query

```
"Where is order ORD-1002?"
         │
         ▼
  get_order("ORD-1002")
         │
         ▼
  [status: "In Transit", tracking: "TRK-9928374650"]
         │
         ▼
  "Your order is on its way! Expected June 28. Track it with TRK-9928374650."
```

### Cheaper Alternative (Multi-step)

```
"Is there a cheaper alternative to the shoes in ORD-1001?"
         │
         ▼
  get_order("ORD-1001")     → finds P-101 (Running Shoes Pro, $89.99)
         │
         ▼
  get_product("P-101")      → category: "shoes"
         │
         ▼
  search_products("shoes")  → [P-104 Budget Sport Shoes $34.99, ...]
         │
         ▼
  "Yes! Budget Sport Shoes at $34.99 would save you $55."
```

---

## Installation

### Prerequisites

- Python 3.11+
- An API key for at least one LLM provider (Anthropic, OpenAI, or Gemini)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourname/online-store-agent.git
cd online-store-agent

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Configuration

Set environment variables before running the agent:

```bash
# Choose your LLM provider (default: anthropic)
export LLM_PROVIDER=anthropic       # or: openai | gemini

# Set the API key for your chosen provider
export ANTHROPIC_API_KEY=sk-ant-...
# export OPENAI_API_KEY=sk-...
# export GEMINI_API_KEY=AI...

# Optional: override the model
# export ANTHROPIC_MODEL=claude-opus-4-6
```

Or create a `.env` file and load it with `python-dotenv`:

```dotenv
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

---

## How to Run

### Option 1 — Streamlit Web UI (recommended)

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501 in your browser.

### Option 2 — Interactive CLI

```bash
python app.py
```

### Option 3 — Single Question (non-interactive)

```bash
python app.py --question "Where is order ORD-1002?"
```

### Option 4 — Import and use in your own code

```python
from agent import run_agent

answer = run_agent("Is there a cheaper alternative to the shoes in order ORD-1001?")
print(answer)
```

---

## How the Agent Works

### 1. Intent Understanding
The LLM interprets the user's question in natural language. There is no keyword matching or regex — the model understands semantics.

### 2. Tool Selection
The model is given three tool schemas and decides which tool(s) to call based on the question and the system prompt.

### 3. Tool Execution
`agent.py` intercepts the LLM's tool_use blocks, calls the corresponding Python function, and injects the results back into the conversation.

### 4. Multi-step Chaining
If the first tool result reveals information needed to call another tool (e.g., a product ID found in an order), the LLM will request the next tool call automatically.

### 5. Response Generation
Once the LLM has all the information it needs, it composes a customer-friendly final answer in plain English.

### 6. Hallucination Prevention
- **System prompt**: "Never invent or guess information."
- **Tool grounding**: Real data is injected into context before answering.
- **Temperature 0**: No creative deviation.

---

## Project Structure

```
online-store-agent/
│
├── app.py                  # CLI entry point (interactive + single-shot mode)
├── agent.py                # ReAct agentic loop + multi-provider LLM wrappers
├── tools.py                # Tool implementations + schemas + registry
├── prompts.py              # System prompt and few-shot examples
├── utils.py                # Logging factory, tool-call audit printer
├── config.py               # All configuration (API keys, models, paths)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── DESIGN.md               # 2-page design document
│
├── tests/
│   ├── __init__.py
│   └── test_agent.py       # Unit tests (tool-level + mocked agent-level)
│
├── data/
│   ├── orders.json         # 10 sample orders
│   └── products.json       # 17 sample products (5 categories)
│
└── streamlit_app.py        # Web chatbot UI with chat history + log panel
```

---

## Examples

### Order Tracking

```
You: Where is order ORD-1002?

Agent: Hi Bob! Your order (ORD-1002) is currently **in transit** and
       is estimated to arrive by June 28, 2024.
       Tracking number: TRK-9928374650
```

### Product Lookup

```
You: Tell me about product P-203.

Agent: The **Samsung 4K Smart TV 55"** is a 55-inch 4K UHD Smart TV priced
       at $699.99. It features a Crystal Display, built-in Alexa, multiple
       HDR formats, and a slim bezel design. Currently in stock. ⭐ 4.6/5
```

### Cheaper Alternative

```
You: Is there a cheaper alternative to the shoes I ordered in ORD-1001?

Agent: The Running Shoes Pro in your order is priced at $89.99. A more
       affordable option is the **Budget Sport Shoes** at just $34.99 —
       saving you $55! They're great for gym workouts and casual walking.
```

### No Match

```
You: Where is order ORD-9999?

Agent: I'm sorry, I couldn't find an order with ID ORD-9999. Please
       double-check the order ID and try again.
```

---

## Running Tests

```bash
# Run all tests (no API key needed — uses mocks)
python -m pytest tests/ -v

# Run with coverage
pip install pytest-cov
python -m pytest tests/ --cov=. --cov-report=term-missing
```

Tests cover:
- ✅ Valid order lookup
- ✅ Invalid order (not found)
- ✅ Invalid product (not found)
- ✅ Successful product search
- ✅ Empty search result
- ✅ Tool chaining (3-step: order → product → search)
- ✅ Cheaper alternative flow (full chain, mocked)
- ✅ Direct text response (no tools needed)
- ✅ Product price query

---

## Future Improvements

| Priority | Improvement |
|---|---|
| 🔴 High | Replace JSON files with a real database (PostgreSQL + SQLAlchemy ORM) |
| 🔴 High | Add customer authentication so agents only expose the right orders |
| 🔴 High | Deploy as a FastAPI REST service with `/chat` and `/health` endpoints |
| 🟡 Medium | Add `create_return_request(order_id)` write tool for self-service returns |
| 🟡 Medium | Integrate a vector database for semantic/embedding-based product search |
| 🟡 Medium | Persist conversation memory across sessions (Redis-backed) |
| 🟡 Medium | Add a `get_recommendations(customer_id)` personalisation tool |
| 🟢 Low | Multi-language support based on browser locale |
| 🟢 Low | Product image display in Streamlit UI |
| 🟢 Low | Rate limiting, abuse detection, and input sanitisation |
| 🟢 Low | CI/CD pipeline with automated test runs on pull requests |

---

## Screenshots

> **Streamlit Web UI**
> *(Run `streamlit run streamlit_app.py` to see live)*

```
┌──────────────────────────────────────────────────────────┐
│  🛍️  Online Store AI Agent                               │
│  ─────────────────────────────────────────────────────── │
│                                                          │
│  You:   Where is order ORD-1002?                        │
│                                                          │
│  🤖:    Hi Bob! Your order is currently in transit and  │
│         should arrive by June 28. Track it with         │
│         TRK-9928374650.                                  │
│                                                          │
│  ▶ Tool Call Logs                                        │
│  ─────────────────────────────────────────────────────── │
│  [Ask a question...]                           [Send]    │
└──────────────────────────────────────────────────────────┘
```

---

*Built with ❤️ using Python, Anthropic Claude, and Streamlit.*
