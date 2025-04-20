import streamlit as st
import os
import sys
import time

# --- SETUP SECTION ---
# Add your service_key path
sys.path.append('/content/drive/MyDrive/Colab Notebooks/LangChain')

from service_key import groq_key

# Setting environment variable
os.environ["GROQ_API_KEY"] = groq_key

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- LANGCHAIN SETUP ---
# Initialize model
model = ChatGroq(model="llama3-8b-8192")

# Setup prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer all questions to the best of your ability."),
    MessagesPlaceholder(variable_name='messages')
])

# Create chain
chain = prompt_template | model

# --- MEMORY MANAGEMENT with session_state ---
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if "history_store" not in st.session_state:
        st.session_state.history_store = {}
    if session_id not in st.session_state.history_store:
        st.session_state.history_store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.history_store[session_id]

# Wrap chain with memory
with_history = RunnableWithMessageHistory(chain, get_session_history)

# --- STREAMLIT APP ---
st.set_page_config(page_title="Chatbot using LangChain", page_icon="💬", layout="wide")
st.title("💬 LangChain ChatBot")

# Initialize session variables
if "session_id" not in st.session_state:
    st.session_state.session_id = "abc"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat Container
chat_container = st.container()

# --- CHAT HISTORY ---
with chat_container:
    for sender, message in st.session_state.messages:
        bubble_style = """
            display: inline-block;
            padding: 10px 15px;
            border-radius: 20px;
            margin: 5px;
            word-wrap: break-word;
            max-width: 80%;
            min-width: 10px;
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
    # Save user message
    st.session_state.messages.append(("user", user_input))

    # Typing animation
    with chat_container:
        typing_message = st.empty()
        typing_message.markdown(
            "<div style='text-align: left;'><div style='display: inline-block; padding: 10px 15px; border-radius: 20px; margin: 5px; background-color: #f0f0f0; color: grey;'>🤖 Bot is typing...</div></div>",
            unsafe_allow_html=True
        )

    time.sleep(0.5)  # simulate typing delay

    # Get real bot response
    response = with_history.invoke(
        [HumanMessage(content=user_input)],
        config={"configurable": {"session_id": st.session_state.session_id}}
    )
    bot_reply = response.content.replace("\\n", "\n")

    # Update chat
    typing_message.empty()
    st.session_state.messages.append(("bot", bot_reply))

    st.rerun()
