# Online Store AI Agent — Design Document

---

## 1. Architecture Overview

### Chosen Pattern: ReAct Agentic Loop

The agent implements the **ReAct** (Reason + Act) pattern, where a large language model (LLM) alternates between reasoning steps and tool executions. The loop continues until the LLM produces a plain-text answer.

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────┐
│  LLM Call (with system prompt + tool schemas)   │
│                                                 │
│  Response type?                                 │
│    ┌──────────────┬──────────────────────────┐  │
│    │  end_turn    │       tool_use           │  │
│    │  (text only) │ (one or more tool calls) │  │
│    └──────┬───────┴──────────┬───────────────┘  │
│           │                  │                  │
│           ▼                  ▼                  │
│     Return answer     Execute tool(s)           │
│     to user           Append results            │
│                       Loop back to LLM ◄────────┘
└─────────────────────────────────────────────────┘
```

### Why ReAct (not a Fixed Intent-Detection Pipeline)?

A rule-based approach (regex + switch/case) would require manually coding every possible intent. ReAct lets the LLM:

- Understand ambiguous natural language ("cheaper alternative to what I ordered")
- Decompose multi-step problems autonomously
- Adapt to phrasing variations without code changes
- Chain tools in any order based on intermediate results

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | All configuration: API keys, model names, file paths |
| `tools.py` | Pure data-retrieval functions (no LLM dependency) |
| `prompts.py` | System prompt and few-shot examples |
| `agent.py` | Agentic loop, LLM factory, tool execution |
| `utils.py` | Logging, tool-call audit trail, response cleanup |
| `app.py` | CLI entry point |
| `streamlit_app.py` | Web chatbot UI |
| `tests/test_agent.py` | Unit + integration tests |

---

## 2. Tool Design Rationale

### Why Tool Calling (not RAG or Fine-Tuning)?

| Approach | Limitation |
|---|---|
| RAG | Embeds static snapshots; can't execute logic (order status changes) |
| Fine-tuning | Bakes data into weights; stale immediately; expensive |
| **Tool calling** | Always reads live data; composable; auditable |

Tool calling is the correct approach because:
1. Order data changes in real time (status, tracking).
2. The same product catalogue must serve search, lookup, and comparison.
3. Each tool can be tested in isolation.

### Tool Design Principles

- **No side effects**: tools are read-only; they never modify data.
- **Structured returns**: every tool returns a dict (JSON-serialisable).
- **Explicit error dicts**: instead of raising exceptions, tools return `{"error": "..."}` so the LLM can relay the message naturally.
- **Swappable backend**: the JSON files can be replaced with database calls without changing any agent or prompt code.

---

## 3. Tool Chaining: The "Cheaper Alternative" Flow

```
User: "Is there a cheaper alternative to the shoes I ordered in ORD-1001?"

Step 1 ──► get_order("ORD-1001")
           Returns: {items: [{product_id: "P-101", name: "Running Shoes Pro", price: 89.99}]}
              │
Step 2 ──► get_product("P-101")
           Returns: {category: "shoes", price: 89.99, ...}
              │
Step 3 ──► search_products("shoes cheaper alternative")
           Returns: [{name: "Budget Sport Shoes", price: 34.99}, ...]
              │
Step 4 ──► LLM compares prices → recommends Budget Sport Shoes ($34.99)
           "Yes! A cheaper alternative is Budget Sport Shoes at $34.99,
            saving you $55 compared to Running Shoes Pro."
```

The LLM decides to chain these three tools autonomously based on the system prompt instruction: *"Chain tools when needed"*. No hard-coded orchestration is required.

---

## 4. Hallucination Prevention

Three complementary mechanisms prevent the agent from inventing data:

### 4.1 System Prompt Grounding
The system prompt explicitly forbids the model from answering factual questions without first calling a tool:
> *"NEVER invent or guess information. Always call tools before making factual claims."*

### 4.2 Tool Results Injected into Context
Every tool result is appended to the conversation history before the next LLM call. The model's answer is therefore always conditioned on real retrieved data, not training-time memorisation.

### 4.3 Temperature = 0
`AGENT_TEMPERATURE = 0.0` makes the model deterministic and eliminates creative embellishment.

---

## 5. Error Handling Strategy

| Error Type | Handling |
|---|---|
| Order not found | Tool returns `{"error": "..."}` → LLM uses exact phrase: *"I'm sorry, I couldn't find an order with that ID."* |
| Product not found | Tool returns `{"error": "..."}` → LLM: *"I couldn't find that product."* |
| Empty search | Tool returns `{"results": [], "count": 0}` → LLM: *"I couldn't find any matching products."* |
| Tool exception | `agent.py` catches the exception, logs it, and injects `{"error": str(exc)}` |
| LLM API failure | Propagated to the caller with a logged traceback |
| Max iterations | Safety cap at 10 tool calls; falls back to a generic apology message |

---

## 6. Customer-Friendliness

The system prompt enforces:
- Warm, conversational tone (not robotic data dumps)
- Use of the customer's name when available
- Short paragraphs and bullet points for readability
- Human-readable summaries, not raw JSON

Example contrast:

**Without prompt guidance:**
> `{"order_id": "ORD-1002", "status": "In Transit", "tracking_number": "TRK-9928374650"}`

**With prompt guidance:**
> "Hi Bob! Your order is currently on its way and should arrive by June 28. Your tracking number is TRK-9928374650 if you'd like to follow it in real time."

---

## 7. Multi-Provider Support

The `_build_llm()` factory function in `agent.py` selects the provider at startup. All three providers expose the same `.chat(messages, tools) → dict` interface, so the main agent loop is completely provider-agnostic. Switching providers requires only changing the `LLM_PROVIDER` environment variable.

---

## 8. Future Improvements

| Priority | Improvement |
|---|---|
| High | Replace JSON files with a real database (PostgreSQL + SQLAlchemy) |
| High | Add authentication so agents only see the requesting customer's orders |
| High | Deploy as a FastAPI microservice with a `/chat` endpoint |
| Medium | Add a `create_return_request(order_id)` tool for self-service returns |
| Medium | Integrate with a vector database for semantic product search |
| Medium | Add conversation memory across sessions (Redis-backed) |
| Low | Support multi-language responses based on browser locale |
| Low | Add product image display in the Streamlit UI |
| Low | Implement rate limiting and abuse detection |
