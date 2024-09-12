import os
import logging
from typing import List, Tuple
import streamlit as st
from dotenv import load_dotenv
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_cohere import CohereEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API keys and environment variables
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
USER_PASSWORD = os.getenv('USER_PASSWORD')

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

def initialize_qa_system() -> RetrievalQA:
    # Initialize Pinecone
    index_name = "demo-index"
    
    # Initialize embeddings using Cohere
    embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model="small")

    # Connect to the Pinecone vectorstore using Langchain
    vectorstore = LangchainPinecone.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
        text_key="text"
    )

    # Initialize ChatOpenAI for the LLM
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")

    # Create and return the RetrievalQA chain
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

def display_chat_history(history: List[Tuple[str, str]]):
    for query, response in history:
        st.markdown(inline_icon_text(USER_ICON_URL, "You: ", "transparent"), unsafe_allow_html=True)
        st.write(query)
        st.markdown(inline_icon_text(ASSISTANT_ICON_URL, "Assistant: ", "transparent"), unsafe_allow_html=True)
        st.write(response)
        st.write("---")

def display_main_app():
    st.title("AI Assistant")
    st.write("How can we help you today?")

    if "history" not in st.session_state:
        st.session_state.history = []
    
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = initialize_qa_system()

    chat_container = st.container()

    with st.form(key='user_input_form', clear_on_submit=True):
        user_query = st.text_input("You:", "")
        submit_button = st.form_submit_button("Send")

    if submit_button and user_query:
        with st.spinner("Thinking..."):
            bot_response = st.session_state.qa_chain.run(user_query)
            st.session_state.history.append((user_query, bot_response))

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