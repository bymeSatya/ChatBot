import streamlit as st
import os
import sys
import time

# Path for your service key
sys.path.append('/content/drive/MyDrive/Colab Notebooks/LangChain')

from service_key import groq_key

# Setting environment variable
os.environ["GROQ_API_KEY"] = groq_key

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Initialize model
model = ChatGroq(model="llama3-8b-8192")

# Setup prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer all questions to the best of your ability."),
    MessagesPlaceholder(variable_name='messages')
])

# Chain with model
chain = prompt_template | model

# Memory store
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Runnable with history
with_history = RunnableWithMessageHistory(chain, get_session_history)

# --- Streamlit App Starts Here ---
st.set_page_config(page_title="Chatbot using LangChain", page_icon="💬", layout="wide")
st.title("💬 LangChain ChatBot")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = "abc"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Create chat message container
chat_container = st.container()

# --- CHAT HISTORY ---
with chat_container:
    for sender, message in st.session_state.messages:
        if sender == "user":
            st.markdown(
                f"<div style='text-align: right; background-color: #4CAF50; color: white; padding: 10px; border-radius: 10px; margin: 5px; max-width: 70%; margin-left: auto;'>{message}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='text-align: left; background-color: #f0f0f0; color: black; padding: 10px; border-radius: 10px; margin: 5px; max-width: 70%;'>{message}</div>",
                unsafe_allow_html=True
            )

# --- INPUT BOX ---
user_input = st.chat_input("Type your message here...")

if user_input:
    # Save user message
    st.session_state.messages.append(("user", user_input))

    # Show 'Bot is typing...' message
    with chat_container:
        typing_message = st.empty()
        typing_message.markdown(
            "<div style='text-align: left; background-color: #f0f0f0; color: grey; padding: 10px; border-radius: 10px; margin: 5px; max-width: 70%;'>🤖 Bot is typing...</div>",
            unsafe_allow_html=True
        )

    # Simulate a small typing delay (optional)
    time.sleep(0.5)

    # Get real bot response
    response = with_history.invoke(
        [HumanMessage(content=user_input)],
        config={"configurable": {"session_id": st.session_state.session_id}}
    )
    bot_reply = response.content.replace("\\n", "\n")

    # Replace 'Bot is typing...' with real reply
    typing_message.empty()
    st.session_state.messages.append(("bot", bot_reply))

    st.rerun()
