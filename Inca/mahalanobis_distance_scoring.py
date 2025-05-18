import numpy as np
from typing import Dict, List, Tuple

class CovarianceStabilizer:
    @staticmethod
    def stabilize(matrix: np.ndarray, epsilon=1e-5) -> np.ndarray:
        return matrix + np.eye(matrix.shape[0]) * epsilon   

class MahalanobisScorer:
    def __init__(self, stabilizer=CovarianceStabilizer()):
        self.stabilizer = stabilizer

    def score(self, embedding: np.ndarray, 
             class_means: Dict[str, np.ndarray], 
             shared_cov: np.ndarray) -> List[Tuple[str, float]]:
        """Calculate Mahalanobis distances for all classes"""
        inv_cov = np.linalg.inv(self.stabilizer.stabilize(shared_cov))
        return [
            (cls, float(np.sqrt((embedding - mean) @ inv_cov @ (embedding - mean).T)))
            for cls, mean in class_means.items()
        ]

# if __name__ == "__main__":
#     # Test Mahalanobis Scoring
#     stabilizer = CovarianceStabilizer()
#     scorer = MahalanobisScorer(stabilizer)
    
#     test_means = {
#         "class1": np.array([0.5, 0.5]),
#         "class2": np.array([-0.5, -0.5])
#     }
#     test_cov = np.eye(2)
#     test_embedding = np.array([0.6, 0.6])
    
#     scores = scorer.score(test_embedding, test_means, test_cov)
#     print("Mahalanobis scores:", scores)