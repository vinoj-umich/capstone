from sklearn.base import BaseEstimator, TransformerMixin

class TextFormatter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        formatted_texts = []
        
        for pages_and_text in X:
            # Replace newlines with spaces and strip leading/trailing spaces
            formatted_text = pages_and_text['text'].replace("\n", " ").strip()
            formatted_page_text = {"page_number": pages_and_text['page_number'], "formatted_text": formatted_text}
            formatted_texts.append(formatted_page_text)
            
        return formatted_texts