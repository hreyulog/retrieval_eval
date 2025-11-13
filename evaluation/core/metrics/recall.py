from typing import List, Dict
from geomas.evaluation.evaluation.core.metrics.ranking_metric import RankingMetric

class RecallAtK(RankingMetric):
    def __init__(self, k: int = 5):
        super().__init__() 
        self.k = k

    def get_name(self) -> str:
        return f"recall"

    def compute(self, pred_sorted_doc_ids: List[str], relevant_doc_ids: Dict[str, float]) -> float:
        """
        Recall@K: fraction of relevant docs found in top-K predictions.
        """
        if not relevant_doc_ids:
            return 0.0

        top_k_preds = pred_sorted_doc_ids[:self.k]
        hits = sum(1 for doc_id in top_k_preds if doc_id in relevant_doc_ids)
        recall = hits / len(relevant_doc_ids)

        self.total += recall
        self.count += 1
        return recall

    def get_average(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0
