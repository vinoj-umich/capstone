import chromadb
from sklearn.base import BaseEstimator, TransformerMixin

class ChromaDBSaver(BaseEstimator, TransformerMixin):
    def __init__(self, chroma_db_dir="chroma_db_dir"):  # Ensure this points to your local ChromaDB
        self.client = chromadb.PersistentClient(path=chroma_db_dir)
        self.collection = self.client.get_or_create_collection("pdf_chunks")

    def fit(self, X, y=None):
        return self

    def transform(self, X, document_attributes):
        i = 0 
        for chunk, doc_attr in zip(X, document_attributes):

            document_id = f"{doc_attr['make']}_{doc_attr['model']}_{doc_attr['year']}_{doc_attr['style']}"
            
            # Log the chunk being added
            text =  chunk["sentence_chunk"]
            chunk_char_count = chunk["chunk_char_count"]
            chunk_word_count = chunk["chunk_word_count"]
            if chunk["sentence_chunk"].strip():  # Ensure it's not empty
                #print(f"Adding document ID: {document_id}, Content: '{chunk['sentence_chunk']}'")
                
                self.collection.add(
                    documents=[text],
                    embeddings=[chunk["embedding"].tolist()],
                    metadatas=[{"source": document_id}],
                    ids = [f"{document_id}_{chunk['page_number']}_{chunk_word_count}_{chunk_char_count}"]

                )
            else:
                print(f"Skipping empty document for ID: {document_id}")
            i=i+1

        return X