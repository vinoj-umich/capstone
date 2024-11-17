import streamlit as st
import torch
import re
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.utils import is_flash_attn_2_available
from sentence_transformers import SentenceTransformer
from transformers import LlamaTokenizer

# Cache the model loading function so it's loaded only once
@st.cache_resource
def load_model(model_id='meta-llama/Llama-2-7b-chat-hf', use_quantization=True):
    from transformers import BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "sdpa"
    
    #tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer = LlamaTokenizer.from_pretrained(model_id)


    #'google/gemma-2b-it'
    config = AutoConfig.from_pretrained(model_id)
    config.hidden_activation = "gelu"
    
    llm_model = AutoModelForCausalLM.from_pretrained(model_id,config=config,
                                                     torch_dtype=torch.float16,
                                                     quantization_config=quantization_config if use_quantization else None,
                                                     low_cpu_mem_usage=True,
                                                     attn_implementation=attn_implementation)
    if not use_quantization:
        llm_model.to("cuda")
    
    return tokenizer, llm_model

# Format Prompt
def prompt_formatter(query: str, 
                     context_items: list[str]):
    """
    Augments query with text-based context from context_items.
    """
    # Join context items into one dotted paragraph
    context = "- " + "\n- ".join([item for item in context_items])

    # Create a base prompt with examples to help the model
    # Note: this is very customizable, I've chosen to use 3 examples of the answer style we'd like.
    # We could also write this in a txt file and import it in if we wanted.
    # base_prompt = """Based on the following context items, please answer the query.
    # Give yourself room to think by extracting relevant passages from the context before answering the query.
    # Don't return the thinking, only return the answer.
    # Make sure your answers are as explanatory as possible.
    # Use the following examples as reference for the ideal answer style.
    # \nExample:
    # Query: What are the signs that your car needs an oil change?
    # Answer:  The signs that your car needs an oil change include a warning light on the dashboard, dark and gritty oil, unusual engine noises, decreased fuel efficiency, and a burnt smell from the engine. Regularly checking the oil level and its condition can help determine the right time for an oil change, typically every 5,000 to 7,500 miles..
    # \nNow use the following context items to answer the user query:
    # {context}
    # \nRelevant passages: <extract relevant passages from the context here>
    # User query: {query}
    # Answer:"""

    base_prompt ="""Using the following context items, please answer the user query directly.
                    Extract and incorporate relevant information from the context, but do not mention the context or how you arrived at your answer.
                    Provide a clear, concise, and explanatory answer.
                    Use the following examples as a reference for the ideal answer style.

                    Example:
                    Query: What are the signs that your car needs an oil change?
                    Answer: The signs that your car needs an oil change include a warning light on the dashboard, dark and gritty oil, unusual engine noises, decreased fuel efficiency, and a burnt smell from the engine. Regularly checking the oil level and its condition can help determine the right time for an oil change, typically every 5,000 to 7,500 miles.

                    Now use the following context items to answer the user query:
                    {context}
                    User query: {query}
                    Answer:
                """

    # Update base prompt with context items and query   
    base_prompt = base_prompt.format(context=context, query=query)

    # Create prompt template for instruction-tuned model
    dialogue_template = [
        {"role": "user",
        "content": base_prompt}
    ]

    # Apply the chat template
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                          tokenize=False,
                                          add_generation_prompt=True)
    return prompt

# ChromaDB Searcher
# ChromaDB Searcher
class ChromaDBSearcher:
    def __init__(self, chroma_db_dir="chroma_db_dir", model_name="all-mpnet-base-v2"):
        self.client = chromadb.PersistentClient(path=chroma_db_dir)
        self.collection = self.client.get_collection("pdf_chunks")
        self.model = SentenceTransformer(model_name)

    def search_by_id(self, document_source, query):
        query_embedding = self.model.encode(query, convert_to_tensor=True).cpu().numpy()
        results = self.collection.query(
            query_embedding.tolist(),
            where={"source": document_source},
            n_results=10
        )
        if results and results['documents']:
            # Ensure that the documents returned are strings, not lists
            return [doc if isinstance(doc, str) else str(doc) for doc in results['documents']]
        return []


def ask(document_source, query,tokenizer, llm_model,
        temperature=0.5,
        max_new_tokens=512,
        format_answer_text=True, 
        return_answer_only=True):
    """
    Takes a query, finds relevant resources/context and generates an answer to the query based on the relevant resources.
    """
    searcher = ChromaDBSearcher()
    # Create a list of context items
    context_items = searcher.search_by_id(document_source, query)

    # Format the prompt with context items
    prompt = prompt_formatter(query=query, context_items=context_items)
    
    print(prompt)
    
    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate an output of tokens
    outputs = llm_model.generate(**input_ids,
                                 temperature=temperature,
                                 do_sample=True,
                                 max_new_tokens=max_new_tokens)
    
    # Turn the output tokens into text
    output_text = tokenizer.decode(outputs[0])

    if format_answer_text:
        # Replace special tokens and unnecessary help message
        output_text = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").replace("Sure, here is the answer to the user query:\n\n", "")

    # Only return the answer without the context items
    if return_answer_only:
        return output_text
    
    return output_text, context_items


import streamlit as st

# Streamlit Interface
st.title("AutoSensAI Chatbot")

# Load the model only once and store it in session state
if 'tokenizer' not in st.session_state or 'llm_model' not in st.session_state:
    tokenizer, llm_model = load_model()
    st.session_state.tokenizer = tokenizer
    st.session_state.llm_model = llm_model

# Fetch the loaded model from session_state
tokenizer = st.session_state.tokenizer
llm_model = st.session_state.llm_model

# Define your available document sources (you can customize this list)
document_sources = [
    "Fraggles__X500_2027_FMS",
    "Fraggles__X700_2026_CRV"
    # Add more sources as needed
]

# Create a dropdown menu (combo box) for document selection
document_source = st.selectbox("Select Car Model , Make & Year", document_sources)

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
        #Sanitize text to remove any unwanted characters or HTML tags
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
            answer = ask(document_source, query, tokenizer, llm_model)
        
        # Only append the user query and bot response to history (not the context)
        st.session_state.chat_history.append({"user": query, "bot": answer})
        st.rerun()  # Rerun the script to update the interface
    else:
        st.error("Please enter a query.")




