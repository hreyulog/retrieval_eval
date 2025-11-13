from typing import List, Dict
from geomas.evaluation.evaluation.core.metrics.ranking_metric import RankingMetric

class F1AtK(RankingMetric):
    def __init__(self, k: int = 5):
        super().__init__() 
        self.k = k

    def get_name(self) -> str:
        return f"f1"

    def compute(self, pred_sorted_doc_ids: List[str], relevant_doc_ids: Dict[str, float]) -> float:
        """
        F1@K: harmonic mean of Precision@K and Recall@K
        """
        if not pred_sorted_doc_ids or not relevant_doc_ids:
            return 0.0

        top_k_preds = pred_sorted_doc_ids[:self.k]
        hits = sum(1 for doc_id in top_k_preds if doc_id in relevant_doc_ids)

        precision = hits / len(top_k_preds)
        recall = hits / len(relevant_docs := relevant_doc_ids)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        self.total += f1
        self.count += 1
        return f1

    def get_average(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0
