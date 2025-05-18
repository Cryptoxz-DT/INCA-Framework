import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticSimilarity:
    def __init__(self, embedder):
        self.embedder = embedder

    def calculate(self, text: str, summary: str) -> float:
        """Compute cosine similarity between text and summary embeddings"""
        text_emb = self.embedder.encode([text])[0]
        summary_emb = self.embedder.encode([summary])[0]
        return float(np.dot(text_emb, summary_emb) / 
                   (np.linalg.norm(text_emb) * np.linalg.norm(summary_emb)))

# if __name__ == "__main__":
#     # Test Semantic Similarity
#     embedder = SentenceTransformer('all-MiniLM-L6-v2')
#     similarity = SemanticSimilarity(embedder)
#     text = "How do I check my account balance?"
#     summary = "account_management includes queries about Check balance, View transactions"
#     print("Similarity score:", similarity.calculate(text, summary))