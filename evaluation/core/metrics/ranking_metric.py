import abc
from typing import List, Dict, Any, Optional

class RankingMetric(abc.ABC):
    
    def __init__(self):
        self.total = 0.0
        self.count = 0
        
    def get_name(self) ->str:
        raise NotImplementedError('Not Implemented')
    
    def compute(self, pred_sorted_doc_ids: List[int], relevant_doc_ids: Dict[str, float]) -> Optional[float]:
        raise NotImplementedError('Not Implemented')
    
    def get_average(self) -> float:
        raise NotImplementedError('Not Implemented')
