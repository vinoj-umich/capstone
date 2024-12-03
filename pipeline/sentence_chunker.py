# Natural Language Processing (NLP) and Embeddings
import spacy
from sentence_transformers import SentenceTransformer  # For embeddings

from sklearn.base import BaseEstimator, TransformerMixin

class SentenceChunker(BaseEstimator, TransformerMixin):
    def __init__(self, max_sentences_per_chunk=5):
        self.max_sentences_per_chunk = max_sentences_per_chunk
        # Load the SpaCy English model and add the sentencizer
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")

    def fit(self, X, y=None):
        return self

    def transform(self, X, document_attributes=None):
        pages_and_chunks = []
        logger.info(f"Input data for transformation: {X}")
        logger.info(f"Input data length: {len(X)}")

        if not X:
            logger.warning("Input data is empty.")
            return []

        sentences = []
        pages = []

        # Extract sentences and page numbers
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
                        #logger.info(f"Extracted sentences from page: {page_number}")
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

        # Organize sentences by pages
        sentences_by_page = {}
        for sentence, page in zip(sentences, pages):
            sentences_by_page.setdefault(page, []).append(sentence)

        for page, sentences in sentences_by_page.items():
            if not sentences:
                continue

            # Chunk sentences into fixed-size chunks
            for i in range(0, len(sentences), self.max_sentences_per_chunk):
                chunk_sentences = sentences[i:i + self.max_sentences_per_chunk]
                chunk_token_count = sum(len(s) // 4 for s in chunk_sentences)
                chunk_dict = {
                    "sentence_chunk": " ".join(chunk_sentences),
                    "chunk_char_count": sum(len(s) for s in chunk_sentences),
                    "chunk_word_count": sum(len(s.split()) for s in chunk_sentences),
                    "chunk_token_count": sum(len(s) // 4 for s in chunk_sentences),  # Adjust if needed
                    "page_number": page  # Include the page number
                }
                if chunk_token_count > 30:
                    logger.info(f"Generated chunk: {chunk_dict}")
                    pages_and_chunks.append(chunk_dict)

        logger.info(f"Processed {len(pages_and_chunks)} semantic chunks.")
        return pages_and_chunks
