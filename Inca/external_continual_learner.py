# src/external_continual_learner.py
from gaussian_class_representation import GaussianClassRepresentation
from mahalanobis_distance_scoring import MahalanobisDistanceScoring 
from utilities import UtilityModules

class ExternalContinualLearner:
    def __init__(self, utility: UtilityModules):
        self.utility = utility
        self.gcr = GaussianClassRepresentation()
        self.mds = MahalanobisDistanceScoring(utility)
    
    def add_class(self, class_id, tags):
        embeddings = self.utility.generate_embedding(tags)
        self.gcr.add_class(class_id, embeddings)
    
    def score_classes(self, input_text, top_k=2):
        input_embeddings = self.utility.generate_embedding([input_text])
        distances = self.mds.compute_distances(
            input_embeddings[0],
            self.gcr.class_means,
            self.gcr.shared_cov
        )
        return sorted(distances.items(), key=lambda x: x[1])[:top_k]