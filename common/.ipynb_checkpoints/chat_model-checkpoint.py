import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from functools import lru_cache
from common.chroma_db import ChromaDBSearcher

class ModelQA:
    def __init__(self, model_id='meta-llama/Llama-2-7b-chat-hf', use_quantization=True, searcher=None):
        """
        Initializes the ModelQA class with model loading, setup configurations, and a searcher for retrieving context.
        
        :param model_id: The identifier of the model to be loaded.
        :param use_quantization: Whether to use model quantization (for efficiency).
        :param searcher: The searcher object (e.g., ChromaDB) used for context retrieval.
        """
        self.model_id = model_id
        self.use_quantization = use_quantization
        self.searcher = searcher  # Store the searcher object
        self.tokenizer, self.llm_model = self.load_model()

    @lru_cache(maxsize=5)
    def load_model(self):
        """
        Loads the model and tokenizer based on the provided model_id.
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
        import torch

        # Set up quantization if needed
        if self.use_quantization:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        else:
            quantization_config = None
        
        # Load the model configuration and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        config = AutoConfig.from_pretrained(self.model_id)
        config.hidden_activation = "gelu"
        
        # Load the actual language model
        llm_model = AutoModelForCausalLM.from_pretrained(self.model_id, config=config,
                                                         torch_dtype=torch.float16,
                                                         quantization_config=quantization_config,
                                                         low_cpu_mem_usage=True)
        if not self.use_quantization:
            llm_model.to("cuda")
        
        return tokenizer, llm_model

    def prompt_formatter(self, query: str, context_items: list[str]) -> str:
        """
        Formats the prompt by adding context and preparing the final input format.
        
        :param query: The user query to be answered.
        :param context_items: The context or relevant information to answer the query.
        :return: The formatted prompt as a string.
        """
        context = "- " + "\n- ".join(context_items)
        
        base_prompt = """Using the following context items, please answer the user query directly.
                        Extract and incorporate relevant information from the context, but do not mention the context or how you arrived at your answer.
                        Provide a clear, concise, and explanatory answer.
                        
                        Now use the following context items to answer the user query:
                        {context}
                        User query: {query}
                        Answer:
                    """
        return base_prompt.format(context=context, query=query)

    def ask(self, document_source, query, temperature=0.5, max_new_tokens=512, format_answer_text=True, return_answer_only=True):
        """
        Handles querying the model, retrieving context, and generating the answer.
        
        :param document_source: The identifier of the document source to search.
        :param query: The query to be answered by the model.
        :param temperature: Sampling temperature for answer generation.
        :param max_new_tokens: Maximum number of tokens for answer generation.
        :param format_answer_text: Whether to clean up the output answer.
        :param return_answer_only: Whether to return only the generated answer (without context).
        :return: Generated answer (and optionally context).
        """
        # Retrieve context items for the query using the searcher
        context_items = self.searcher.search_by_id(document_source, query)
        
        # Format the prompt with context items
        prompt = self.prompt_formatter(query, context_items)

        # Tokenize the prompt
        input_ids = self.tokenizer(prompt, truncation=True, return_tensors="pt").to("cuda")

        # Generate an output of tokens
        outputs = self.llm_model.generate(**input_ids,
                                          temperature=temperature,
                                          do_sample=True,
                                          max_new_tokens=max_new_tokens)
        
        # Convert the output tokens back to text
        output_text = self.tokenizer.decode(outputs[0])

        # Clean up the output text if required
        if format_answer_text:
            output_text = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").replace("Sure, here is the answer to the user query:\n\n", "")
        
        if return_answer_only:
            return output_text
        return output_text, context_items