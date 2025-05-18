# src/utilities.py
import numpy as np

class UtilityModules:
    def generate_embedding(self, texts):
        return np.random.rand(len(texts), 768)
    
    def stabilize_covariance(self, cov):
        # Increase regularization to 0.1
        return cov + np.eye(cov.shape[0]) * 0.1
    
    # Test UtilityModules 

# if __name__ == "__main__":
#     utilities = UtilityModules()
#     texts = ["Example text 1", "Example text 2"]
#     embeddings = utilities.generate_embedding(texts)
#     print("Embedding Shape:", embeddings.shape)
    
#     # Test covariance stabilization
#     cov = np.array([[1, 0], [0, 0]])
#     stabilized_cov = utilities.stabilize_covariance(cov)
#     print("Stabilized Covariance:\n", stabilized_cov) 
