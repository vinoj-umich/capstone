import openai

class ModelQAApi:
    use_gpu = False  # Class-level flag to control GPU usage (True = use GPU, False = use CPU)

    def __init__(self, api_key, endpoint, deployment_name, searcher=None, use_gpu=False):
        """
        Initializes the ModelQA class with API credentials, setup configurations, and a searcher for retrieving context.
        
        :param api_key: Azure OpenAI API key.
        :param endpoint: Azure OpenAI endpoint.
        :param deployment_name: Deployment name for GPT model in Azure OpenAI.
        :param searcher: The searcher object (e.g., ChromaDB) used for context retrieval.
        :param use_gpu: Flag to determine whether to use GPU or CPU (affects local model handling, not Azure OpenAI API).
        """
        self.api_key = api_key
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.searcher = searcher  # Store the searcher object
        self.use_gpu = use_gpu  # Instance-level flag for GPU usage
        self.setup_openai()

    def setup_openai(self):
        """
        Configures the OpenAI API with Azure credentials.
        """
        openai.api_type = "azure"
        openai.api_key = self.api_key
        openai.api_base = self.endpoint
        openai.api_version = "2023-05-15"
        
        if self.use_gpu:
            print("GPU is enabled for local processing.")  # This would be for local models or handling.
        else:
            print("GPU is disabled for local processing.")  # This would be for CPU-only usage.

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
                        If there is no relevant context provided or context is empty,  simply say I’m sorry, I don’t have that specific information, but I can suggest some resources for you.
                        Provide a clear, concise, and explanatory answer.
                        
                        Now use the following context items to answer the user query:
                        {context}
                        User query: {query}
                        Answer
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
        
        # print("******************")
        # print(prompt)
        # print("******************")
        
        # Prepare the message format expected by OpenAI's ChatCompletion API
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": prompt}
        ]
        
        # Check if we need to simulate GPU usage (this won't affect Azure API call)
        if self.use_gpu:
            print("Simulating GPU processing...")

        # Call Azure OpenAI API for generating the answer
        try:
            response = openai.ChatCompletion.create(
                engine=self.deployment_name,
                messages=messages,  # Correct way: passing the messages list instead of prompt
                temperature=temperature,
                max_tokens=max_new_tokens
            )

            # Extract and clean up the output text
            output_text = response['choices'][0]['message']['content'].strip()  # Corrected to access 'message'

            if format_answer_text:
                output_text = output_text.replace(prompt, "").strip()

            if return_answer_only:
                return output_text
            return output_text, context_items

        except openai.error.OpenAIError as e:
            print(f"OpenAI API Error: {e}")
            return "There was an issue with the OpenAI API."

        except Exception as e:
            print(f"Unexpected error: {e}")
            return "An unexpected error occurred."