"""
streamlit_app.py
----------------
Streamlit-based chatbot interface for the Online Store AI Agent.

Features:
  - Chat-style message history (persisted within the session)
  - "Clear Chat" button to reset the conversation
  - Expandable "Tool Call Logs" panel updated in real time
  - Sidebar with model info and quick-start example questions
  - Spinner while the agent is thinking

Run with:
    streamlit run streamlit_app.py
"""

import logging
from io import StringIO
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Online Store AI Agent",
    page_icon="🛍️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Lazy imports so the config check runs after st.set_page_config
# ---------------------------------------------------------------------------
import config
from agent import run_agent
from utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# CSS styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }
    .log-box { background:#1e1e1e; color:#d4d4d4; padding:12px;
               border-radius:8px; font-size:12px; white-space:pre-wrap;
               max-height:300px; overflow-y:auto; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🛍️ Online Store Agent")
    st.caption(f"Provider: **{config.LLM_PROVIDER.title()}**")
    st.divider()

    st.subheader("Example Questions")
    examples = [
        "Where is order ORD-1002?",
        "What is the price of product P-203?",
        "Tell me about product P-203.",
        "Is there a cheaper alternative to the shoes I ordered in ORD-1001?",
        "Do you have any books on programming?",
        "What noise-cancelling headphones do you carry?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
            st.session_state.pending_question = ex

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.tool_logs = []
        st.session_state.pending_question = None
        st.rerun()

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tool_logs" not in st.session_state:
    st.session_state.tool_logs = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.title("🛍️ Online Store AI Agent")
st.caption("Ask me anything about your orders or our products!")
st.divider()

# ---------------------------------------------------------------------------
# Chat history display
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Capture logs from this session to display in the log panel
# ---------------------------------------------------------------------------

class SessionLogHandler(logging.Handler):
    """In-memory log handler that appends records to a session list."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            st.session_state.tool_logs.append(self.format(record))
        except Exception:
            pass


def _ensure_session_log_handler() -> None:
    """Attach the session log handler once per app session."""
    root_logger = logging.getLogger()
    for h in root_logger.handlers:
        if isinstance(h, SessionLogHandler):
            return  # Already attached

    handler = SessionLogHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S"
    ))
    root_logger.addHandler(handler)


_ensure_session_log_handler()

# ---------------------------------------------------------------------------
# Input handling — chat input OR sidebar example button
# ---------------------------------------------------------------------------
user_input: str | None = st.chat_input("Ask a question about an order or product…")

# Sidebar example button sets pending_question; pick it up here
if st.session_state.pending_question and not user_input:
    user_input = st.session_state.pending_question
    st.session_state.pending_question = None

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    # Run the agent
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking…"):
            try:
                answer = run_agent(user_input)
            except Exception as exc:
                logger.error("Agent error: %s", exc, exc_info=True)
                answer = f"⚠️ Something went wrong: {exc}"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ---------------------------------------------------------------------------
# Tool call log panel
# ---------------------------------------------------------------------------
st.divider()
with st.expander("🔧 Tool Call Logs", expanded=False):
    if st.session_state.tool_logs:
        log_text = "\n".join(st.session_state.tool_logs[-200:])  # cap at last 200 entries
        st.markdown(f'<div class="log-box">{log_text}</div>', unsafe_allow_html=True)
    else:
        st.caption("No tool calls yet. Ask a question above to see the agent in action!")
