import re
from geomas.evaluation.evaluation.core.metrics.mrr import MRR
from geomas.evaluation.evaluation.core.metrics.ndcg import NDCG
from geomas.evaluation.evaluation.core.metrics.precision import PrecisionAtK
from geomas.evaluation.evaluation.core.metrics.recall import RecallAtK
from geomas.evaluation.evaluation.core.metrics.ranking_metric import RankingMetric

# 基本工厂，只存类，不存具体 K 值
_BASE_FACTORIES = {
    MRR.get_name():MRR, 
    NDCG.get_name():NDCG, 
    PrecisionAtK.get_name():PrecisionAtK, 
    RecallAtK.get_name():RecallAtK
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
