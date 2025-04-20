import streamlit as st
import os
import sys

# Path for your service key
#sys.path.append('/content/drive/MyDrive/Colab Notebooks/LangChain')

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

# Streamlit app
st.set_page_config(page_title="LangChain Chatbot", page_icon=":speech_balloon:")
st.title("Chatbot using LangChain and Streamlit")

# Initialize session
if "session_id" not in st.session_state:
    st.session_state.session_id = "abc"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Text input
user_input = st.text_input("You:", key="user_input")

# Button actions
col1, col2 = st.columns([1, 1])
with col1:
    send_clicked = st.button("Send")
with col2:
    clear_clicked = st.button("Clear Chat")

if send_clicked and user_input:
    # Send user message
    st.session_state.messages.append(("user", user_input))
    
    # Get response
    response = with_history.invoke(
        [HumanMessage(content=user_input)],
        config={"configurable": {"session_id": st.session_state.session_id}}
    )
    bot_reply = response.content.replace("\\n", "\n")
    st.session_state.messages.append(("bot", bot_reply))
    
    # Clear input field
    st.rerun()

if clear_clicked:
    st.session_state.messages = []
    st.rerun()

# Display chat history
for sender, message in st.session_state.messages:
    if sender == "user":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Bot:** {message}")
