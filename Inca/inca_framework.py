import numpy as np
import json
from sentence_transformers import SentenceTransformer
from typing import Dict, List
from tag_generator import TagGenerator
from gaussian_class_representation import GaussianClassRepresentation
from mahalanobis_distance_scoring import MahalanobisScorer
from class_summary_generator import ClassSummaryGenerator
from semantic_similarity_calculator import SemanticSimilarity
from out_of_domain_detector import OutOfDomainDetector

class InCAFramework:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.tag_generator = TagGenerator()
        self.gaussian_model = GaussianClassRepresentation()
        self.scorer = MahalanobisScorer()
        self.summary_gen = ClassSummaryGenerator()
        self.semantic = SemanticSimilarity(self.embedder)
        self.ood_detector = OutOfDomainDetector()

    def add_class(self, class_id: str, examples: List[str]):
        """Add new class to the framework"""
        # Generate and store class summary
        self.summary_gen.generate_summary(class_id, examples)
        
        # Process examples
        tags = [self.tag_generator.generate_tags(ex) for ex in examples]
        embeddings = self.embedder.encode([' '.join(t) for t in tags])
        self.gaussian_model.add_class(class_id, embeddings)

    def predict(self, text: str, top_k: int = 3) -> Dict:
        """Make prediction for input text"""
        # Tag generation
        tags = self.tag_generator.generate_tags(text)
        query_embed = self.embedder.encode([' '.join(tags)])[0]
        
        # Get candidate classes
        scores = self.scorer.score(query_embed, 
                                 self.gaussian_model.class_means,
                                 self.gaussian_model.shared_cov)
        sorted_scores = sorted(scores, key=lambda x: x[1])[:top_k]
        
        # Semantic verification
        best_class, max_score = None, -1
        for cls, _ in sorted_scores:
            similarity = self.semantic.calculate(text, 
                                                self.summary_gen.summaries[cls])
            if similarity > max_score:
                max_score = similarity
                best_class = cls
                
        # OOD detection
        stats_score = 1 - (sorted_scores[0][1] / sum(s[1] for s in sorted_scores))
        is_ood = self.ood_detector.is_ood(stats_score, max_score)
        
        return {
            "prediction": best_class,
            "confidence": (stats_score + max_score) / 2,
            "is_ood": is_ood
        }

if __name__ == "__main__":
    # Test Full Framework
    inca = InCAFramework()
    
    # Add sample classes
    inca.add_class("funds_transfer", 
                 ["Send money to another account", "Wire transfer instructions"])
    inca.add_class("balance_inquiry", 
                 ["Check account balance", "Current balance"])
    
    # Make prediction
    result = inca.predict("How do I send money to my sister?")
    print(json.dumps(result, indent=2))
    
    def add_new_class(self, class_id, training_examples):
        # Generate tags from training examples
        tags = []
        for example in training_examples:
            generated_tags = self.tag_generator.generate_tags(example)
            tags.extend(generated_tags)
        # Add to ECL and InContextLearner
        self.ecl.add_class(class_id, tags)
        self.in_context_learner.add_class_summary(class_id, training_examples)
    
    def predict(self, input_text):
        tags = self.tag_generator.generate_tags(input_text)
        candidate_classes = self.ecl.score_classes(input_text)
        candidate_class_ids = [cls for cls, _ in candidate_classes]
        predicted_class = self.in_context_learner.predict_class(input_text, candidate_class_ids)
        confidence = 1.0 if candidate_classes else 0.0
        is_ooc = self.ooc_detector.detect(confidence)
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "is_ooc": is_ooc
        }