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

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API keys and environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
USER_PASSWORD = os.getenv('USER_PASSWORD')

client = openai.OpenAI(api_key=OPENAI_API_KEY)

ASSISTANT_ICON_URL = "https://cdn-icons-png.flaticon.com/512/7966/7966941.png"
USER_ICON_URL = "https://cdn-icons-png.flaticon.com/512/2503/2503707.png"

SYSTEM_PROMPT = """
Based on user question, you should return the number of the pages where the answer on the question could be found.
Analyze the question and the content of the documentation and return the page numbers in a structured way.
Ensure that all page numbers related to the question are returned, it is better to return more page numbers than less if you are not sure.
At each mention of something in the documentation, return the page number of the mention, ALL PAGES!
It is better to return more page numbers if you have dilemma than to return less. Double check that you cover all the pages that are related to the question.

The generated text should be in the following format:
{ "page_numbers": [x, y, z] }

No other text should be returned.

Here is the documentation:
{DOCUMENTATION}
"""

SYSTEM_PROMPT_2 = """
Based on user question and provided context, return the answer to the question.

Here is the context:
{CONTEXT}

If you cannot answer the question based on the context, return "I cannot answer that question, sorry." Don't add any other text as it is for end user.
If the question is only partially related to the context, answer only the part that is related to the context, and for the rest, inform the user that you can only answer questions related to the Talentwunder documentation.
DO NOT MENTION WORDS like "based on the context" or "based on the provided context"
the context is forbidden to use as you are communicating with the user, and it is not professional for app to put it in the answer.
Do not halucinate are assume thing. ONLY MAKE YOUR ANSWER BASED ON THE CONTEXT.
"""

def read_text_file(file_path):
    """
    Read text file with proper encoding handling.
    """
    encodings = ['utf-8', 'cp1252', 'latin1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()

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

def get_ai_response(user_query: str) -> Tuple[str, float, float]:
    start_time = time.time()
    content_text = read_text_file("./llm_output.txt")
    
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model_name="gpt-4o-2024-08-06",
        temperature=0,
        max_tokens=5000
    )
    
    # First call to get page numbers
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.replace("{DOCUMENTATION}", content_text)
        },
        {
            "role": "user",
            "content": user_query
        }
    ]
    
    response = llm.invoke(messages)
    end_time = time.time()
    price = response.usage_metadata["input_tokens"]*2.50/1000000 + response.usage_metadata["output_tokens"]*10/1000000
    response_time = end_time - start_time
    try:
        page_numbers = json.loads(response.content)["page_numbers"]
    except:
        return "I cannot process that question, sorry.", price, response_time
    
    if len(page_numbers) > 0:
        # Gather context from relevant pages
        context = ""
        for page_number in page_numbers:
            try:
                context += read_text_file(f"./pages/page_{page_number}.txt")
            except:
                continue
        
        # Second call with context to get the answer
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_2.replace("{CONTEXT}", context)
            },
            {
                "role": "user",
                "content": user_query
            }
        ]
        
        response = llm.invoke(messages)
        price += response.usage_metadata["input_tokens"]*2.50/1000000 + response.usage_metadata["output_tokens"]*10/1000000
        end_time = time.time()
        response_time = end_time - start_time
        return response.content, price, response_time
    
    return "I cannot answer that question, sorry.", price, response_time

def get_direct_ai_response(user_query: str) -> Tuple[str, float, float]:
    """Get AI response using the full document approach"""
    start_time = time.time()
    content_text = read_text_file("./output.txt")
    
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model_name="gpt-4o-2024-08-06",
        temperature=0,
        max_tokens=5000
    )
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Your task is to answer users questions only on provided context. If the answer is not in the context, say that you don't know. If topic is not related to the context, inform the user that you can only answer questions related to the Talentwunder documentation. DO NOT MENTION WORDS like based on the context' or 'based on the provided context'. Answer the user's question based on the following documentation:\n\n" + content_text
        },
        {
            "role": "user",
            "content": user_query
        }
    ]
    
    response = llm.invoke(messages)
    end_time = time.time()
    price = response.usage_metadata["input_tokens"]*2.50/1000000 + response.usage_metadata["output_tokens"]*10/1000000
    response_time = end_time - start_time
    return response.content, price, response_time

def display_chat_history(history: List[Tuple[str, str, str, float, float, float, float]]):
    """Display chat history with two responses side by side"""
    for query, response1, response2, price1, price2, response_time1, response_time2 in history:
        st.markdown(inline_icon_text(USER_ICON_URL, "You: ", "transparent"), unsafe_allow_html=True)
        st.write(query)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(inline_icon_text(ASSISTANT_ICON_URL, "Assistant (Page-based): ", "transparent"), unsafe_allow_html=True)
            st.write(response1)
            st.write(f"Price: {price1} USD")
            st.write(f"Response Time: {response_time1} seconds")
            

        with col2:
            st.markdown(inline_icon_text(ASSISTANT_ICON_URL, "Assistant (Full Doc): ", "transparent"), unsafe_allow_html=True)
            st.write(response2)
            st.write(f"Price: {price2} USD")
            st.write(f"Response Time: {response_time2} seconds")
            

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
        with st.spinner("Thinking..."):
            # Get responses from both approaches
            page_based_response, page_based_price, page_based_response_time = get_ai_response(user_query)
            full_doc_response, full_doc_price, full_doc_response_time = get_direct_ai_response(user_query)
            
            # Store both responses in history
            st.session_state.history.append((user_query, page_based_response, full_doc_response, page_based_price, full_doc_price, page_based_response_time, full_doc_response_time))

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