# src/in_context_learner.py
from class_summary_generator import ClassSummaryGenerator
from semantic_similarity_calculator import SemanticSimilarityCalculator

class InContextLearner:
    def __init__(self):
        self.csg = ClassSummaryGenerator()
        self.ssc = SemanticSimilarityCalculator()

    def add_class_summary(self, class_id, training_examples):
        self.csg.generate_summary(class_id, training_examples)
    
    def predict_class(self, input_text, candidate_classes):
        best_class = None
        best_similarity = -1
        for cls in candidate_classes:
            summary = self.csg.get_summary(cls)
            similarity = self.compute_similarity(input_text, summary)
            if similarity > best_similarity:
                best_similarity = similarity
                best_class = cls
        return best_class

    