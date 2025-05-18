import numpy as np
from typing import Dict

class GaussianClassRepresentation:
    def __init__(self):
        self.class_means: Dict[str, np.ndarray] = {}
        self.shared_cov = None
        self.num_classes = 0

    def add_class(self, class_id: str, embeddings: np.ndarray):
        """Update class statistics incrementally"""
        self.class_means[class_id] = np.mean(embeddings, axis=0)
        class_cov = np.cov(embeddings, rowvar=False)
        
        if self.shared_cov is None:
            self.shared_cov = class_cov
        else:
            self.shared_cov = (self.shared_cov * self.num_classes + class_cov) / (self.num_classes + 1)
        self.num_classes += 1

# if __name__ == "__main__":
#     # Test Gaussian Class Representation
#     gcr = GaussianClassRepresentation()
#     dummy_embeddings = np.random.rand(5, 10)  # 5 samples, 10D embeddings
#     gcr.add_class("test_class", dummy_embeddings)
#     print("Class means:", gcr.class_means)
#     print("Shared covariance shape:", gcr.shared_cov.shape)