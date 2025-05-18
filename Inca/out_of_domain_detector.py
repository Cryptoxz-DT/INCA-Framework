class OutOfDomainDetector:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def is_ood(self, stats_score: float, semantic_score: float) -> bool:
        """Determine if input is out-of-domain"""
        return (stats_score + semantic_score) / 2 < self.threshold

# if __name__ == "__main__":
#     # Test OOD Detection
#     ood = OutOfDomainDetector()
#     print("Low scores OOD:", ood.is_ood(0.3, 0.4))
#     print("High scores OOD:", ood.is_ood(0.6, 0.7))