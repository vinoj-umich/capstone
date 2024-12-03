from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cuda")

    def fit(self, X, y=None):
        return self

    def transform(self, X, document_attributes):
        sentences = [chunk["sentence_chunk"] for chunk in X]
        embeddings = self.model.encode(sentences)
        
        for i, chunk in enumerate(X):
            chunk["embedding"] = embeddings[i]
        
        #log_output("Embedding Generator: "+ len(X))  # Log output
        return X