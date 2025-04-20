import streamlit as st
import os
import sys
import json
import time
from datetime import datetime

# --- SETUP SECTION ---
#sys.path.append('/content/drive/MyDrive/Colab Notebooks/LangChain')
from service_key import groq_key

os.environ["GROQ_API_KEY"] = groq_key

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- LANGCHAIN SETUP ---
model = ChatGroq(model="llama3-8b-8192")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer all questions to the best of your ability."),
    MessagesPlaceholder(variable_name='messages')
])

chain = prompt_template | model

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if "history_store" not in st.session_state:
        st.session_state.history_store = {}
    if session_id not in st.session_state.history_store:
        st.session_state.history_store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.history_store[session_id]

with_history = RunnableWithMessageHistory(chain, get_session_history)

# --- FILE PATHS ---
SESSION_DIR = "sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

# --- STREAMLIT SETUP ---
st.set_page_config(page_title="AI Buddy 🤖", page_icon="🤖", layout="wide")

# --- INITIALIZE SESSION ---
if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- PAGE HEADER ---
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>
        🤖 AI Buddy
    </h1>
    <h4 style='text-align: center; color: gray;'>
        Powered by LangChain + Streamlit
    </h4>
    <hr style="border:1px solid #4CAF50">
    """,
    unsafe_allow_html=True
)

# --- SIDEBAR (SESSION MANAGEMENT) ---
with st.sidebar:
    st.header("📚 Sessions")
    
    # New Chat
    if st.button("➕ New Chat"):
        st.session_state.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        st.session_state.messages = []
        st.rerun()

    # List Saved Sessions
    saved_sessions = sorted(os.listdir(SESSION_DIR))
    for session_file in saved_sessions:
        if st.button(session_file.replace(".json", "")):
            with open(os.path.join(SESSION_DIR, session_file), "r") as f:
                st.session_state.messages = json.load(f)
                st.session_state.session_id = session_file.replace(".json", "")
            st.rerun()

# --- CHAT CONTAINER ---
chat_container = st.container()

with chat_container:
    for sender, message in st.session_state.messages:
        bubble_style = """
            display: inline-block;
            padding: 10px 15px;
            border-radius: 20px;
            margin: 5px;
            word-wrap: break-word;
            max-width: 80%;
        """

        if sender == "user":
            st.markdown(
                f"<div style='text-align: right;'><div style='{bubble_style} background-color: #4CAF50; color: white; margin-left: auto;'>{message}</div></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='text-align: left;'><div style='{bubble_style} background-color: #f0f0f0; color: black;'>{message}</div></div>",
                unsafe_allow_html=True
            )

    # --- AUTO-SCROLL ---
    st.markdown("""
        <script>
            var chat = window.parent.document.querySelector('section.main');
            if (chat) { chat.scrollTo(0, chat.scrollHeight); }
        </script>
    """, unsafe_allow_html=True)

# --- INPUT BOX ---
user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.messages.append(("user", user_input))

    # Typing animation with moving dots
    with chat_container:
        typing_placeholder = st.empty()
        for dots in ["", ".", "..", "..."]:
            typing_placeholder.markdown(
                f"<div style='text-align: left;'><div style='display: inline-block; padding: 10px 15px; border-radius: 20px; margin: 5px; background-color: #f0f0f0; color: gray;'>🤖 Bot is typing{dots}</div></div>",
                unsafe_allow_html=True
            )
            time.sleep(0.3)
        typing_placeholder.empty()

    # Get real response
    response = with_history.invoke(
        [HumanMessage(content=user_input)],
        config={"configurable": {"session_id": st.session_state.session_id}}
    )
    bot_reply = response.content.replace("\\n", "\n")

    st.session_state.messages.append(("bot", bot_reply))

    # --- SAVE CHAT SESSION ---
    with open(os.path.join(SESSION_DIR, f"{st.session_state.session_id}.json"), "w") as f:
        json.dump(st.session_state.messages, f)

    st.rerun()
