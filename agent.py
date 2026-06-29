"""
agent.py
--------
Core AI Agent — implements ``run_agent(question: str) -> str``.

## Architecture: ReAct-style agentic loop

The agent follows the ReAct pattern (Reason → Act → Observe → repeat):

  ┌─────────────────────────────────────────────────────┐
  │  User Question                                      │
  │         ↓                                           │
  │  Send to LLM with tool schemas + system prompt      │
  │         ↓                                           │
  │  LLM responds with:                                 │
  │    • text only  → return to user (done)             │
  │    • tool_use   → execute tool(s), append results   │
  │         ↓              ↓                            │
  │  Append tool results to conversation history        │
  │         ↓                                           │
  │  Send updated history back to LLM (loop)            │
  │         ↓                                           │
  │  LLM responds with final text → return to user      │
  └─────────────────────────────────────────────────────┘

## Why this design?
  - The LLM decides WHICH tools to call and in WHAT ORDER — no hard-coded
    intent-detection switch statements.  The model understands natural language
    far better than regex.
  - The agentic loop allows multi-step reasoning: the model can call get_order,
    see the product ID, then call get_product, see the category, then call
    search_products — all autonomously.
  - Hallucination prevention: the system prompt forbids the model from answering
    factual questions without first calling a tool.  Tool results are injected
    into the conversation context, so the model's answer is always grounded.
  - Multi-provider support: a thin factory pattern selects the API client at
    startup; the main loop is provider-agnostic.

## Supported LLM providers
  - Anthropic (Claude)  — default; uses the native tool_use API
  - OpenAI (GPT-4o)     — uses the function_calling / tools API
  - Google Gemini        — uses the function declarations API
"""

import json
from typing import Any

import config
from prompts import SYSTEM_PROMPT
from tools import TOOL_REGISTRY, TOOL_SCHEMAS
from utils import clean_response, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Provider-specific LLM wrappers
# ---------------------------------------------------------------------------

class AnthropicLLM:
    """Thin wrapper around the Anthropic Messages API."""

    def __init__(self) -> None:
        try:
            import anthropic  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            ) from exc

        self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self._model = config.ANTHROPIC_MODEL
        logger.info("Anthropic LLM initialised (model=%s)", self._model)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> dict[str, Any]:
        """Send a messages request and return the raw response body as a dict."""
        response = self._client.messages.create(
            model=self._model,
            max_tokens=config.AGENT_MAX_TOKENS,
            temperature=config.AGENT_TEMPERATURE,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )
        return {
            "stop_reason": response.stop_reason,  # "end_turn" | "tool_use"
            "content": [block.model_dump() for block in response.content],
        }


class OpenAILLM:
    """Thin wrapper around the OpenAI Chat Completions API."""

    def __init__(self) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            ) from exc

        self._client = OpenAI(api_key=config.OPENAI_API_KEY)
        self._model = config.OPENAI_MODEL
        logger.info("OpenAI LLM initialised (model=%s)", self._model)

    @staticmethod
    def _convert_tools(tool_schemas: list[dict]) -> list[dict]:
        """Convert Anthropic-style tool schemas to OpenAI function format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tool_schemas
        ]

    def chat(self, messages: list[dict], tools: list[dict]) -> dict[str, Any]:
        openai_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        openai_tools = self._convert_tools(tools)

        response = self._client.chat.completions.create(
            model=self._model,
            temperature=config.AGENT_TEMPERATURE,
            max_tokens=config.AGENT_MAX_TOKENS,
            tools=openai_tools,
            messages=openai_messages,
        )
        choice = response.choices[0]
        msg = choice.message

        content_blocks: list[dict] = []
        if msg.content:
            content_blocks.append({"type": "text", "text": msg.content})
        if msg.tool_calls:
            for tc in msg.tool_calls:
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments),
                })

        stop_reason = "tool_use" if msg.tool_calls else "end_turn"
        return {"stop_reason": stop_reason, "content": content_blocks}


class GeminiLLM:
    """Thin wrapper around the Google Gemini GenerativeModel API."""

    def __init__(self) -> None:
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "google-generativeai package not installed. "
                "Run: pip install google-generativeai"
            ) from exc

        genai.configure(api_key=config.GEMINI_API_KEY)
        self._genai = genai
        self._model_name = config.GEMINI_MODEL
        logger.info("Gemini LLM initialised (model=%s)", self._model_name)

    @staticmethod
    def _convert_tools(tool_schemas: list[dict]) -> list[dict]:
        """Convert Anthropic-style schemas to Gemini function declarations."""
        declarations = []
        for t in tool_schemas:
            declarations.append({
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            })
        return declarations

    def chat(self, messages: list[dict], tools: list[dict]) -> dict[str, Any]:
        from google.generativeai.types import content_types  # type: ignore

        gen_tools = self._convert_tools(tools)
        model = self._genai.GenerativeModel(
            model_name=self._model_name,
            system_instruction=SYSTEM_PROMPT,
            tools=gen_tools,
        )

        # Build Gemini-compatible history
        gemini_history = []
        for m in messages[:-1]:
            gemini_history.append({"role": m["role"], "parts": [m["content"]]})

        chat = model.start_chat(history=gemini_history)
        last_msg = messages[-1]["content"]
        response = chat.send_message(last_msg)

        content_blocks: list[dict] = []
        stop_reason = "end_turn"

        for part in response.parts:
            if hasattr(part, "text") and part.text:
                content_blocks.append({"type": "text", "text": part.text})
            elif hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                content_blocks.append({
                    "type": "tool_use",
                    "id": f"gemini-{fc.name}",
                    "name": fc.name,
                    "input": dict(fc.args),
                })
                stop_reason = "tool_use"

        return {"stop_reason": stop_reason, "content": content_blocks}


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _build_llm() -> AnthropicLLM | OpenAILLM | GeminiLLM:
    """Instantiate the correct LLM wrapper based on config.LLM_PROVIDER."""
    provider = config.LLM_PROVIDER.lower()
    if provider == "anthropic":
        return AnthropicLLM()
    elif provider == "openai":
        return OpenAILLM()
    elif provider == "gemini":
        return GeminiLLM()
    else:
        raise ValueError(f"Unknown LLM provider: '{provider}'. Choose anthropic/openai/gemini.")


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

def _execute_tool(tool_name: str, tool_input: dict) -> Any:
    """Look up and call a tool by name.

    Args:
        tool_name:  Name matching a key in TOOL_REGISTRY.
        tool_input: Argument dict as provided by the LLM.

    Returns:
        Whatever the tool function returns (a dict).

    Raises:
        KeyError: If the tool name is not in the registry.
    """
    fn = TOOL_REGISTRY.get(tool_name)
    if fn is None:
        raise KeyError(f"Tool '{tool_name}' not found in registry.")
    logger.debug("Executing tool: %s(%s)", tool_name, tool_input)
    return fn(**tool_input)


# ---------------------------------------------------------------------------
# Main agent function
# ---------------------------------------------------------------------------

def run_agent(question: str) -> str:
    """Run the AI agent on a customer question and return a friendly response.

    This is the primary public API of the agent module.

    The function implements a ReAct-style agentic loop:
      1. Send the user question to the LLM with tool schemas.
      2. If the LLM requests tool calls, execute them and feed results back.
      3. Repeat until the LLM produces a plain-text answer.
      4. Return the cleaned final response.

    Args:
        question: The customer's natural-language question.

    Returns:
        A customer-friendly string answer, grounded in tool results.
    """
    logger.info("=" * 70)
    logger.info("User question: %s", question)

    llm = _build_llm()

    # Conversation history — we maintain this across iterations so the LLM
    # has full context of what tools returned in previous steps.
    messages: list[dict] = [{"role": "user", "content": question}]

    for iteration in range(1, config.MAX_TOOL_ITERATIONS + 1):
        logger.debug("Agent loop iteration %d", iteration)

        # ── Step 1: Ask the LLM ────────────────────────────────────────────
        response = llm.chat(messages=messages, tools=TOOL_SCHEMAS)
        stop_reason = response["stop_reason"]
        content_blocks = response["content"]

        logger.debug("LLM stop_reason=%s, blocks=%d", stop_reason, len(content_blocks))

        # ── Step 2: Check if LLM is done ──────────────────────────────────
        if stop_reason == "end_turn":
            # Collect all text blocks into the final answer
            text_parts = [
                block["text"]
                for block in content_blocks
                if block.get("type") == "text" and block.get("text")
            ]
            final_answer = "\n".join(text_parts)
            logger.info("Agent final answer produced after %d iteration(s).", iteration)
            return clean_response(final_answer)

        # ── Step 3: Process tool calls ─────────────────────────────────────
        # The LLM may request multiple tools in a single response.
        # We execute all of them and bundle results back.
        if stop_reason == "tool_use":
            # Append the assistant's response (including tool_use blocks) to history
            messages.append({"role": "assistant", "content": content_blocks})

            # Build the tool_result message with results from every tool call
            tool_result_contents: list[dict] = []

            for block in content_blocks:
                if block.get("type") != "tool_use":
                    continue

                tool_name = block["name"]
                tool_input = block["input"]
                tool_use_id = block["id"]

                logger.info("LLM requested tool: %s with input: %s", tool_name, tool_input)

                try:
                    tool_result = _execute_tool(tool_name, tool_input)
                    result_text = json.dumps(tool_result, indent=2, default=str)
                    is_error = False
                except Exception as exc:
                    logger.error("Tool '%s' raised an exception: %s", tool_name, exc)
                    result_text = json.dumps({"error": str(exc)})
                    is_error = True

                tool_result_contents.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_text,
                    "is_error": is_error,
                })

            # Append all tool results as a single user-role message
            messages.append({"role": "user", "content": tool_result_contents})
            continue  # Loop back to ask the LLM again

        # ── Unexpected stop reason ─────────────────────────────────────────
        logger.warning("Unexpected stop_reason: %s", stop_reason)
        break

    # Safety fallback — should never reach here in normal operation
    logger.error("Agent reached max iterations (%d) without a final answer.", config.MAX_TOOL_ITERATIONS)
    return "I'm sorry, I wasn't able to process your request. Please try again."
