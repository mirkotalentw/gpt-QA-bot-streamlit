import os
import time
import logging
from typing import List, Tuple
import streamlit as st
import openai
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from pinecone import Pinecone
from pinecone_plugins.assistant.models.chat import Message

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API keys and environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
USER_PASSWORD = os.getenv('USER_PASSWORD')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

client = openai.OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
assistant = pc.assistant.Assistant(assistant_name="test-1")

ASSISTANT_ICON_URL = "https://cdn-icons-png.flaticon.com/512/7966/7966941.png"
USER_ICON_URL = "https://cdn-icons-png.flaticon.com/512/2503/2503707.png"


def inline_icon_text(icon_url: str, text: str, background_color: str) -> str:
    return f"""
    <div style="display: flex; align-items: center; background-color: {background_color}; padding: 10px; border-radius: 15px; margin: 10px 0;">
        <img src="{icon_url}" style="width: 30px; height: 30px; margin-right: 10px;">
        <h2>{text}</h2>
    </div>
    """

def check_credentials(username: str, password: str) -> bool:
    return username == "talentwunder" and password == USER_PASSWORD

def display_login_form():
    st.title("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            if check_credentials(username, password):
                st.session_state['logged_in'] = True
                st.success("Logged in successfully.")
                st.rerun()
            else:
                st.error("Incorrect username or password.")



def get_pinecone_response(chat_history: List) -> Tuple[str, float, float]:
    start_time = time.time()
    messages = []
    for element in chat_history:
        if isinstance(element, tuple) and len(element) >= 5:
            # Extract Message object from the tuple
            messages.append(element[4])
        elif isinstance(element, Message):
            messages.append(element)
    resp = assistant.chat(messages=messages)
    end_time = time.time()
    response_time = end_time - start_time
    input_tokens = resp['usage']['prompt_tokens']
    return resp['message']['content'], input_tokens, response_time

def display_chat_history(history: List):
    """Display chat history with messages"""
    for message in history:
        if isinstance(message, tuple) and len(message) >= 5:
            # This is a user message with Message object
            query = message[0]
            st.markdown(inline_icon_text(USER_ICON_URL, "You: ", "transparent"), unsafe_allow_html=True)
            st.write(query)
        
        elif isinstance(message, tuple) and len(message) == 4:
            # This is an assistant response with metrics
            query, response, input_tokens, response_time = message
            st.markdown(inline_icon_text(ASSISTANT_ICON_URL, "Assistant: ", "transparent"), unsafe_allow_html=True)
            st.write(response)
            st.write(f"Input tokens: {input_tokens}")
            st.write(f"Response Time: {response_time:.2f} seconds")

        st.write("---")

def display_main_app():
    st.title("AI Assistant")
    st.write("How can we help you today?")

    if "history" not in st.session_state:
        st.session_state.history = []

    chat_container = st.container()

    with st.form(key='user_input_form', clear_on_submit=True):
        user_query = st.text_input("You:", "")
        submit_button = st.form_submit_button("Send")

    if submit_button and user_query:
        # Create user message
        user_message = Message(role="user", content=user_query)
        # Add user message to history with the Message object
        st.session_state.history.append((user_query, "", 0, 0, user_message))
        
        with st.spinner("Thinking..."):
            # Get response from Pinecone
            response, input_tokens, response_time = get_pinecone_response(st.session_state.history)
            # Add assistant response to history
            st.session_state.history.append((user_query, response, input_tokens, response_time))

    with chat_container:
        display_chat_history(st.session_state.history)

def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        display_login_form()
    else:
        display_main_app()

if __name__ == "__main__":
    main()