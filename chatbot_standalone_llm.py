import streamlit as st
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

from common.chroma_db import ChromaDBSearcher
from common.chat_model import ModelQA
from common.cache import ModelCache


# Initialize the model cache
model_cache = ModelCache()

# Streamlit Interface
st.title("AutoSensAI Chatbot")

# Select model ID for QA
model_id = 'meta-llama/Llama-2-7b-chat-hf'  # Example model ID

# Check if model is already cached
model_qa = model_cache.get(model_id)
if model_qa is None:
    # Initialize the ChromaDB searcher
    searcher = ChromaDBSearcher()
    # Create a new ModelQA instance and cache it
    model_qa = ModelQA(model_id=model_id, searcher=searcher)
    model_cache.set(model_id, model_qa)

# Load the model only once and store it in session state
if 'tokenizer' not in st.session_state or 'llm_model' not in st.session_state:
    tokenizer, llm_model = model_qa.load_model()
    st.session_state.tokenizer = tokenizer
    st.session_state.llm_model = llm_model

# Fetch the loaded model from session state
tokenizer = st.session_state.tokenizer
llm_model = st.session_state.llm_model

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

# Text area for user query
query = st.text_area("Enter your query here:")

# Handle query submission
if st.button("Ask"):
    if query:
        with st.spinner("Generating response..."):
            answer = model_qa.ask(selected_document_key, query)  # No need to pass tokenizer and llm_model if they are in ModelQA
            
        # Only append the user query and bot response to history (not the context)
        st.session_state.chat_history.append({"user": query, "bot": answer})
        # No need for rerun, Streamlit will automatically update
    else:
        st.error("Please enter a query.")