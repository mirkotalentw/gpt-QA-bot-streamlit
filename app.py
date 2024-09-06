import os
import logging
import streamlit as st
from dotenv import load_dotenv
import openai
from pinecone import Pinecone
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_cohere import CohereEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI

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

assistant_icon_url = "https://cdn-icons-png.flaticon.com/512/7966/7966941.png"
user_icon_url = "https://cdn-icons-png.flaticon.com/512/2503/2503707.png"

def inline_icon_text(
        icon_url: str,
        text: str,
        background_color: str
    ) -> str:
    """
    Generates HTML code for displaying an inline icon with text.

    Parameters:
    icon_url (str): The URL of the icon image.
    text (str): The text to be displayed.
    background_color (str): The background color of the container.

    Returns:
    str: The generated HTML code.

    Example:
    >>> inline_icon_text('https://example.com/icon.png', 'Hello World', '#ffffff')
    '<div style="display: flex; align-items: center; background-color: #ffffff; padding: 10px; border-radius: 15px; margin: 10px 0;">\n    <img src="https://example.com/icon.png" style="width: 30px; height: 30px; margin-right: 10px;">\n    <div>Hello World</div>\n</div>'
    """
    return f"""
    <div style="display: flex; align-items: center; background-color: {background_color}; padding: 10px; border-radius: 15px; margin: 10px 0;">
        <img src="{icon_url}" style="width: 30px; height: 30px; margin-right: 10px;">
        <h2>{text}</h2>
    </div>
    """

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

def check_credentials(username, password):
    # Fetch the password from environment variables
    correct_password = os.getenv('USER_PASSWORD')
    return username == "talentwunder" and password == correct_password

def display_login_form():
    st.title("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")
        if login_button:
            if check_credentials(username, password):
                st.session_state['logged_in'] = True
                st.success("Logged in successfully.")
                st.rerun()
            else:
                st.error("Incorrect username or password.")

# Initialize Pinecone
pc = Pinecone(
    api_key=PINECONE_API_KEY
)

# Ensure the Pinecone index exists
index_name = "demo-index"
if index_name not in pc.list_indexes().names():
    raise ValueError(f"Pinecone index '{index_name}' does not exist. Please run the indexing script first.")

# Connect to the existing Pinecone index
index = pc.Index(index_name)

# Initialize embeddings using Cohere
embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model="small")

# Connect to the Pinecone vectorstore using Langchain
vectorstore = LangchainPinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
    text_key="text"
)

# Initialize OpenAI for the LLM
openai.api_key = OPENAI_API_KEY
llm = OpenAI(openai_api_key=OPENAI_API_KEY)

# Load the QA chain
qa_chain = load_qa_chain(llm, chain_type="map_reduce")

def display_main_app():
    # Streamlit app title
    st.title("AI Assistant")

    # Chatbot interface
    st.write("How can we help you today?")

    # Initialize chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Create a container for chat history
    chat_container = st.container()

    # Create a form for user input at the bottom
    with st.form(key='user_input_form', clear_on_submit=True):
        user_query = st.text_input("You:", "")
        submit_button = st.form_submit_button("Send")

    if submit_button and user_query:
        # Perform a similarity search in the vectorstore
        related_docs = vectorstore.similarity_search(query=user_query, k=3)

        # If there are no relevant documents, provide a default response
        print(related_docs)
        if not related_docs:
            bot_response = "Sorry, I couldn't find any relevant documents related to your question."
        else:
            # Run the QA chain on the retrieved documents
            result = qa_chain.run(input_documents=related_docs, question=user_query)
            bot_response = result

        # Append user query and bot response to the chat history
        st.session_state.history.append((user_query, bot_response))

    # Display chat history in the container
    with chat_container:
        for i, (query, response) in enumerate(st.session_state.history):
            html = inline_icon_text(user_icon_url, "You: ", "transparent")
            st.markdown(html, unsafe_allow_html=True)
            st.write(query)
            html = inline_icon_text(assistant_icon_url, f"Assistant: ", "transparent")
            st.markdown(html, unsafe_allow_html=True) 
            st.write(response)  
            st.write("---")         

# Decide which part of the app to display based on login status
if not st.session_state['logged_in']:
    display_login_form()
else:
    display_main_app()