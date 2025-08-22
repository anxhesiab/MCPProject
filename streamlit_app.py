
import asyncio
import traceback
import streamlit as st

# MCP Orchestrator
from mcp_system.client import answer as orchestrate

# Page config
st.markdown("""
    <style>
    .main {
        max-width: 950px;
        margin: 0 auto;
        padding: 2rem;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    textarea, input, button, .stTextInput, .stTextArea {
        width: 100% !important;
        max-width: 100% !important;
    }
    .stTextArea textarea {
        height: 100px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💬 MCP LLM Orchestrator")

# Persistent state for history
if "history" not in st.session_state:
    st.session_state.history: list[tuple[str, str]] = []

# Async wrapper to keep Streamlit responsive
async def ask_llm(query: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, orchestrate, query)

query = st.text_area("🔍 Ask a question about your data:", height=100)

# Chat trigger
if st.button("🧠 Analyze with LLM") and query.strip():
    with st.spinner("Thinking..."):
        try:
            answer = asyncio.run(ask_llm(query))
            st.session_state.history.append((query, answer))
            st.success("✅ Done")
        except Exception:
            st.error("❌ Error – see details below")
            st.text_area("Traceback", traceback.format_exc(), height=300)

# Show chat history
st.divider()
with st.container():
    for q, a in reversed(st.session_state.history):
        st.markdown(f"**🧍‍♂️ You:** {q}", unsafe_allow_html=True)
        st.markdown(
            f"""<div style=" padding: 1rem; border-radius: 8px;
            line-height:1.4; font-size: 15px;">
            <b>🤖 MCP:</b><br>{a}
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown("---")
