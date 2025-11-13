from typing import List, Dict
from geomas.evaluation.evaluation.core.metrics.ranking_metric import RankingMetric

class MRR(RankingMetric):

    def get_name(self) -> str:
        return "mrr"

    def compute(self, pred_sorted_doc_ids: List[str], relevant_doc_ids: Dict[str, float]) -> float:
        if not relevant_doc_ids:
            return 0.0

        rr = 0.0
        for rank, doc_id in enumerate(pred_sorted_doc_ids, start=1):
            if doc_id in relevant_doc_ids:
                rr = 1.0 / rank
                break

        self.total += rr
        self.count += 1
        return rr

    def get_average(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0