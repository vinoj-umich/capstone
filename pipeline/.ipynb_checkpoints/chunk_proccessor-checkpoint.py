from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.base import BaseEstimator, TransformerMixin
import spacy
import logging
import hashlib

# Natural Language Processing (NLP) and Embeddings
import spacy
from sentence_transformers import SentenceTransformer  # For embeddings

# Set up logging
logger = logging.getLogger(__name__)

class SentenceChunkerWithSummarization(BaseEstimator, TransformerMixin):
    def __init__(self, max_sentences_per_chunk=10, max_summary_length=500, num_beams=4):
        """
        Initialize the SentenceChunkerWithSummarization.
        
        :param max_sentences_per_chunk: The maximum number of sentences per chunk.
        :param max_summary_length: Maximum length of the generated summary.
        :param num_beams: Number of beams for beam search during summary generation.
        """
        self.max_sentences_per_chunk = max_sentences_per_chunk
        self.max_summary_length = max_summary_length
        self.num_beams = num_beams
        
        # Load the SpaCy model and add the sentencizer
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")
        
        # Load the T5 model and tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

    def fit(self, X, y=None):
        """
        Fit method does nothing as the model doesn't require fitting.
        """
        return self

    def generate_summary(self, text):
        """
        Generate a summary for a given text using the T5 model.
        
        :param text: The input text to summarize
        :return: The summarized text
        """
        if not text or not isinstance(text, str):
            logger.warning("Received invalid text input.")
            return "Invalid input: Empty or non-string text"

        # Tokenize the input with the T5 summarization prompt
        input_tokens = self.tokenizer.encode("summarize: " + text, return_tensors='pt')

        # Generate the summary using the model
        # max_length=150,  # Increase the max_length for a longer summary
        # min_length=50,   # Set a minimum length to prevent too short summaries
        # num_beams=4,     # Use beam search for better quality summaries
        # early_stopping=True,  # Stop early when the model is confident
        # length_penalty=1.5,   # Apply a penalty to make sure summaries are not too short
        output = self.model.generate(input_tokens, min_length=50, max_length=self.max_summary_length, num_beams=self.num_beams, early_stopping=True,  length_penalty=1.5)

        # Decode the summary
        summary = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return summary

    def generate_unique_id(self, sentence_chunk):
        """
        Generate a unique ID from a sentence chunk using SHA-256 hash.

        :param sentence_chunk: The input sentence to generate the ID from.
        :return: A unique ID (SHA-256 hash) as a hexadecimal string.
        """
        # Step 1: Preprocess the sentence (optional, you could strip, lowercase, etc.)
        processed_chunk = sentence_chunk.strip().lower()

        # Step 2: Create the SHA-256 hash of the sentence
        unique_id = hashlib.sha256(processed_chunk.encode()).hexdigest()

        return unique_id

    def transform(self, X):
        """
        Transform the input data by chunking sentences and summarizing each chunk.
        
        :param X: List of documents or paragraphs to process
        :return: List of dictionaries with sentence chunks and their summaries
        """
        if not X:
            logger.warning("Input data is empty.")
            return []

        pages_and_chunks = []
        sentences = []
        pages = []

        # Extract sentences and page numbers from the input
        for item in X:
            if isinstance(item, dict):
                if 'formatted_text' in item and 'page_number' in item:
                    text = item['formatted_text'].strip()
                    page_number = item['page_number']
                    if text:  # Check if text is not empty
                        doc = self.nlp(text)  # Process text with SpaCy
                        for sent in doc.sents:
                            sentences.append(sent.text.strip())
                            pages.append(page_number)
                        logger.info(f"Extracted sentences from page: {page_number}")
                    else:
                        logger.warning(f"Empty sentence found in item: {item}")
                else:
                    logger.error(f"Missing keys in item: {item}")
            elif isinstance(item, tuple) and len(item) == 2:
                text = item[0].strip()
                page_number = item[1]
                doc = self.nlp(text)  # Process text with SpaCy
                for sent in doc.sents:
                    sentences.append(sent.text.strip())
                    pages.append(page_number)
            else:
                logger.error(f"Unexpected item format: {item}")

        # Organize sentences by page
        sentences_by_page = {}
        for sentence, page in zip(sentences, pages):
            sentences_by_page.setdefault(page, []).append(sentence)

        # Chunk sentences into fixed-size chunks and generate summaries
        for page, sentences in sentences_by_page.items():
            if not sentences:
                continue

            for i in range(0, len(sentences), self.max_sentences_per_chunk):
                chunk_sentences = sentences[i:i + self.max_sentences_per_chunk]
                chunk_text = " ".join(chunk_sentences)
                
                # Generate the summary for the chunk of sentences
                summary = self.generate_summary(chunk_text)

                # Generate additional information
                chunk_char_count = sum(len(s) for s in chunk_sentences)
                chunk_word_count = sum(len(s.split()) for s in chunk_sentences)
                chunk_token_count = sum(len(s) // 4 for s in chunk_sentences)
                summary_char_count = len(summary)
                summary_word_count = len(summary.split())

                # Create a dictionary with both chunk data and summary data
                chunk_dict = {
                    "sentence_chunk": chunk_text,
                    "chunk_char_count": chunk_char_count,
                    "chunk_word_count": chunk_word_count,
                    "chunk_token_count": chunk_token_count,
                    "page_number": page,  # Include the page number
                    "summary_text": summary,
                    "summary_char_count": summary_char_count,
                    "summary_word_count": summary_word_count,                    
                    "para_id" : self.generate_unique_id(chunk_text)
                }

                # Only include chunks with more than 30 tokens
                if chunk_token_count > 30:
                    #logger.info(f"Generated chunk and summary: {chunk_dict}")
                    pages_and_chunks.append(chunk_dict)

        #logger.info(f"Processed {len(pages_and_chunks)} semantic chunks with summaries.")
        return pages_and_chunks
