import fitz 
import logging
from tqdm.auto import tqdm
from sklearn.base import BaseEstimator, TransformerMixin

class PDFReader(BaseEstimator, TransformerMixin):
    def __init__(self, pdf_path, logger):
        self.pdf_path = pdf_path
        self.logger = logger

    def fit(self, X, y=None):
        return self

    def transform(self, X=None):
        """
        Reads a PDF file and extracts text from each page.

        Returns:
            list: A list of dictionaries, each containing the page number and its text.
        """
        try:
            doc = fitz.open(self.pdf_path)
        except Exception as e:
            self.logger.error(f"Failed to open PDF file: {self.pdf_path}. Error: {e}")
            return []

        pages_and_texts = []
        for page_number in tqdm(range(len(doc)), desc="Reading PDF pages"):
            page = doc[page_number]
            text = page.get_text()
            pages_and_texts.append({"page_number": page_number, "text": text})

        self.logger.info(f"Successfully read {len(pages_and_texts)} pages from {self.pdf_path}")
        return pages_and_texts
