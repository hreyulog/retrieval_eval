import math
from typing import List, Dict
from .ranking_metric import RankingMetric

class NDCG(RankingMetric):
    def __init__(self, k: int = 5):
        super().__init__() 
        self.k = k

    def get_name(self) -> str:
        return f"ndcg@{self.k}"

    def compute(self, pred_sorted_doc_ids: List[str], relevant_doc_ids: Dict[str, float]) -> float:
        """
        Normalized Discounted Cumulative Gain
        """

        if not relevant_doc_ids:
            return 0.0

        dcg = 0.0
        for i, doc_id in enumerate(pred_sorted_doc_ids[:self.k], start=1):
            rel = relevant_doc_ids.get(doc_id, 0.0)
            if rel > 0:
                dcg += (2 ** rel - 1) / math.log2(i + 1)

        ideal_rels = sorted(relevant_doc_ids.values(), reverse=True)[:self.k]
        idcg = sum((2 ** rel - 1) / math.log2(i + 1) for i, rel in enumerate(ideal_rels, start=1))

        ndcg = dcg / idcg if idcg > 0 else 0.0

        self.total += ndcg
        self.count += 1
        return ndcg

    def get_average(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0
