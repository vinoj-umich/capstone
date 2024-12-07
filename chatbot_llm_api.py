import os
from dotenv import load_dotenv
import streamlit as st
import openai
import re
from common.chroma_db import ChromaDBSearcher
from common.chat_model_api import ModelQAApi
from common.cache import ModelCache
import time

# Initialize the model cache
model_cache = ModelCache()

# Streamlit Interface
st.title("AutoSensAI Chatbot")


# Load environment variables from the .env file
load_dotenv()

# Example API credentials (replace with actual values)
gpt4apikey = os.getenv("gpt4apikey") 
endpoint = os.getenv("endpoint") 
deployment_name = "gpt-4"  # Use your model deployment name

# Check if model is already cached
model_qa = model_cache.get(deployment_name)
if model_qa is None:
    # Initialize the ChromaDB searcher
    searcher = ChromaDBSearcher()
    # Create a new ModelQA instance and cache it
    model_qa = ModelQAApi(api_key=gpt4apikey, endpoint=endpoint, deployment_name=deployment_name, searcher=searcher, use_gpu=False)
    model_cache.set(deployment_name, model_qa)

# Define your available document sources (you can customize this list)
# Define your available document sources with labels (you can customize this list)
document_sources = {
    "Fraggles_X500_2024_FMS": "Fraggles X500 (2024) - FMS",  # Custom label for the document
    "Fraggles_X700_2022_HCM": "Fraggles X700 (2022) - HCM"   # Custom label for the document
    # Add more sources with labels as needed
}

# Create a dropdown menu (combo box) for document selection with labeled options
document_source_label = st.selectbox(
    "Select Car Make, Model & Year",  
    list(document_sources.values()),
    index=0,                          # Optional: Set the default selected item (first label)
    help="Please select the car make, model, and year to view related documents."  # Tooltip for additional context
)

# Get the actual internal document name based on the label selected
selected_document_key = [key for key, value in document_sources.items() if value == document_source_label][0]

# Initialize chat history in session state if not already initialized
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Define CSS for chat bubbles layout
st.markdown("""
 <style>
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 15px;
    }
    .user-message {
        align-self: flex-end;
        background-color: #DCF8C6;
        padding: 10px;
        border-radius: 15px;
        max-width: 70%;
        word-wrap: break-word;
        text-decoration: none;  /* Ensure no strike-through */
    }
    .bot-message {
        align-self: flex-start;
        background-color: #F1F0F0;
        padding: 10px;
        border-radius: 15px;
        max-width: 70%;
        word-wrap: break-word;
        text-decoration: none;  /* Ensure no strike-through */
    }
</style>
""", unsafe_allow_html=True)

# Display chat history with custom formatting
with st.container():
    for chat in st.session_state.chat_history:
        # Sanitize text to remove any unwanted characters or HTML tags
        user_message = re.sub(r'<.*?>', '', chat["user"]).replace("~", "")
        bot_message = re.sub(r'<.*?>', '', chat["bot"]).replace("~", "")

        # Display user message on the right
        st.markdown(f'<div class="chat-container"><div class="user-message">{user_message}</div></div>', unsafe_allow_html=True)
        # Display bot response on the left
        st.markdown(f'<div class="chat-container"><div class="bot-message">{bot_message}</div></div>', unsafe_allow_html=True)

# Initialize query_input key in session_state if not already initialized
if 'query_input' not in st.session_state:
    st.session_state.query_input = ""

# Text area for user query
query = st.text_area("Enter your query here:", key="query_input", value=st.session_state.query_input)

# Handle query submission
if st.button("Ask"):
    if query:
        with st.spinner("Generating response..."):
            answer = model_qa.ask(selected_document_key, query)  # No need to pass tokenizer and llm_model if they are in ModelQA
        
        # Only append the user query and bot response to history (not the context)
        st.session_state.chat_history.append({"user": query, "bot": ""})  # Initially add empty bot message for chat bubble layout

        # Create a placeholder for the bot response
        bot_response_container = st.empty()

        # Update the response line by line
        bot_response_text = ""
        for line in answer.splitlines():
            bot_response_text += line + "\n"
            bot_response_container.markdown(f'<div class="chat-container"><div class="bot-message">{bot_response_text}</div></div>', unsafe_allow_html=True)
            time.sleep(0.5)  # Simulate a slight delay to make it look like the response is being typed

        # Update the chat history with the complete bot response
        st.session_state.chat_history[-1]["bot"] = answer
        
        # Clear query input after submission. This happens after the widget renders.
        st.experimental_rerun()  # This will trigger a re-render and clear the input field
    else:
        st.error("Please enter a query.")
