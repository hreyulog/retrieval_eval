import re
from .mrr import MRR
from .ndcg import NDCG
from .precision import PrecisionAtK
from .recall import RecallAtK
from .ranking_metric import RankingMetric
from .f1 import F1AtK
# 基本工厂，只存类，不存具体 K 值
_BASE_FACTORIES = {
    'mrr':MRR, 
    'ndcg':NDCG, 
    'precision':PrecisionAtK, 
    'recall':RecallAtK,
    'f1':F1AtK
}

def create_metrics(name: str) -> RankingMetric:
    """
    Create a metric instance from a string like "recall@5", "precision@1", "MRR", "NDCG@10".
    """
    name_lower = name.lower()

    match = re.match(r"(\w+)(?:@(\d+))?", name_lower)
    if not match:
        raise ValueError(f"Invalid metric name: {name}")

    metric_base, k_str = match.groups()
    metric_class = _BASE_FACTORIES.get(metric_base)
    if metric_class is None:
        raise ValueError(f"Unknown metric base name: {metric_base}")

    if k_str is not None:
        k = int(k_str)
        return metric_class(k=k)
    else:
        return metric_class()
