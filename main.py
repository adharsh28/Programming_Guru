import os

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from modellearning import chat_model

st.set_page_config(
    page_title="CodeBot", page_icon="⌨️", layout="wide", initial_sidebar_state="expanded"
)

st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Space+Grotesk:wght@300;400;600&display=swap');
  html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; background-color: #0d0f14; color: #c9d1d9; }
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 0 !important; max-width: 100% !important; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] { background-color: #0d1117 !important; border-right: 1px solid #21262d !important; }
  [data-testid="stSidebar"] * { color: #c9d1d9 !important; }
  .sidebar-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem; font-weight: 600;
    color: #6e7681 !important;
    text-transform: uppercase; letter-spacing: 1px;
    padding: 4px 0 12px; border-bottom: 1px solid #21262d; margin-bottom: 12px;
  }
  .history-item {
    display: flex; align-items: flex-start; gap: 8px;
    padding: 8px 10px; border-radius: 8px; margin-bottom: 6px;
    background: #161b22; border: 1px solid #21262d; transition: border-color 0.2s;
  }
  .history-item:hover { border-color: #58a6ff55; }
  .history-icon { font-size: 13px; margin-top: 1px; flex-shrink: 0; }
  .history-text {
    font-size: 0.78rem; line-height: 1.5; color: #8b949e !important;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 190px;
  }
  .history-text.user-text { color: #c9d1d9 !important; font-weight: 500; }
  .empty-history {
    font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;
    color: #4a5568 !important; text-align: center; padding: 24px 0;
  }

  /* ── Main chat ── */
  .chat-header {
    display: flex; align-items: center; gap: 12px;
    padding: 20px 0 14px; border-bottom: 1px solid #21262d;
    max-width: 860px; margin: 0 auto; width: 100%;
  }
  .chat-header .logo {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #58a6ff, #3fb950);
    border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 18px;
  }
  .chat-header h1 { font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 600; color: #e6edf3; margin: 0; }
  .chat-header .sub { font-size: 0.72rem; color: #6e7681; margin-top: 1px; }
  .status-dot { width: 8px; height: 8px; background: #3fb950; border-radius: 50%; margin-left: auto; box-shadow: 0 0 6px #3fb95088; animation: pulse 2s infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
  .chat-area { max-width: 860px; margin: 0 auto; width: 100%; padding: 24px 16px 8px; }
  .msg-row { display: flex; gap: 12px; margin-bottom: 20px; align-items: flex-start; animation: fadeUp 0.25s ease both; }
  @keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
  .msg-row.user { flex-direction: row-reverse; }
  .avatar { width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 14px; flex-shrink: 0; }
  .avatar.bot { background: #161b22; border: 1px solid #30363d; }
  .avatar.user { background: #1f3040; border: 1px solid #1f6feb44; }
  .bubble { max-width: 78%; padding: 12px 16px; border-radius: 12px; font-size: 0.88rem; line-height: 1.65; word-break: break-word; white-space: pre-wrap; }
  .bubble.bot { background: #161b22; border: 1px solid #21262d; border-top-left-radius: 2px; color: #c9d1d9; }
  .bubble.user { background: linear-gradient(135deg, #1f3040, #1a2840); border: 1px solid #1f6feb44; border-top-right-radius: 2px; color: #e6edf3; }
  .empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; height: 55vh; opacity: 0.55; }
  .empty-state .big-icon { font-size: 3rem; }
  .empty-state p { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #6e7681; text-align: center; }
  .thinking { display: flex; gap: 5px; align-items: center; padding: 4px 2px; }
  .thinking span { width: 7px; height: 7px; border-radius: 50%; background: #58a6ff; animation: bounce 1.1s infinite ease-in-out; }
  .thinking span:nth-child(2) { animation-delay: 0.18s; }
  .thinking span:nth-child(3) { animation-delay: 0.36s; }
  @keyframes bounce { 0%, 80%, 100% { transform: translateY(0); opacity: 0.4; } 40% { transform: translateY(-6px); opacity: 1; } }
</style>
""",
    unsafe_allow_html=True,
)


# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    os.getenv("HUGGINGFACEHUB_API_TOKEN")
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct", temperature=0.3, max_new_tokens=1024
    )
    return ChatHuggingFace(llm=llm)


model = load_model()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []  # display dicts
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # LangChain HumanMessage/AIMessage objects

# ── Sidebar: chat history ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">🕘 &nbsp;Chat History</div>', unsafe_allow_html=True
    )

    if not st.session_state.messages:
        # Guard: only render when there are messages — avoids the None crash
        st.markdown(
            '<div class="empty-history">No messages yet.<br>Start a conversation!</div>',
            unsafe_allow_html=True,
        )
    else:
        # Show all messages, newest first
        for msg in reversed(st.session_state.messages):
            is_user = msg["role"] == "user"
            icon = "👤" if is_user else " "
            # Truncate long messages to 60 chars for sidebar preview
            label = (
                msg["content"][:60] + "…"
                if len(msg["content"]) > 60
                else msg["content"]
            )
            if is_user:
                st.markdown(
                    f"""
                <div class="history-item">
                <span class="history-icon">{icon}</span>
                <span class="history-text">{label}</span>
                </div>
            """,
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    # Clear button lives in sidebar — no need for a separate one in main area
    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="chat-header">
  <div class="logo">⌨️</div>
  <div>
    <h1>CodeBot</h1>
    <div class="sub">Llama 3.1 · 8B Instruct · Memory ON 🧠</div>
  </div>
  <div class="status-dot"></div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Chat messages ─────────────────────────────────────────────────────────────
st.markdown('<div class="chat-area">', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown(
        """
    <div class="empty-state">
      <div class="big-icon">💻</div>
      <p>Ask me anything about code.<br>Debugging · Algorithms · Concepts · Reviews</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    role = msg["role"]
    avatar = "🤖" if role == "assistant" else "👤"
    side = "user" if role == "user" else ""
    bclass = "user" if role == "user" else "bot"
    aclass = "user" if role == "user" else "bot"
    st.markdown(
        f"""
    <div class="msg-row {side}">
      <div class="avatar {aclass}">{avatar}</div>
      <div class="bubble {bclass}">{msg["content"]}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask your coding question…")

if prompt:  # ← critical guard: sidebar code only runs when prompt is not None
    st.session_state.messages.append({"role": "user", "content": prompt})

    thinking = st.empty()
    thinking.markdown(
        """
    <div class="chat-area">
      <div class="msg-row">
        <div class="avatar bot">🤖</div>
        <div class="bubble bot">
          <div class="thinking"><span></span><span></span><span></span></div>
        </div>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    response = chat_model(model, prompt, st.session_state.chat_history)
    thinking.empty()

    st.session_state.chat_history.append(HumanMessage(prompt))
    st.session_state.chat_history.append(AIMessage(response))
    st.session_state.messages.append({"role": "assistant", "content": response})

    st.rerun()  # ← triggers a full re-render, sidebar auto-updates with the new message
