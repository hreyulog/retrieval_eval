import time
from typing import Dict, List
from ..encoder.encoder_config import create_encode
from .dataset import TestDataset
from .metrics.metrics_config import create_metrics  
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RetrievalEvaluator:
    """
    Evaluator for retrieval performance in RAG systems.
    Calculates Recall@K, Precision@K, MRR, nDCG, etc., 
    and records retrieval time.
    """

    def __init__(self, dataset: str, embedding_model: str, top_k: int = 5, metrics: list=["recall@1","recall@5","mrr","ndcg","f1","precision@3"]):
        """
        Args:
            dataset (str): Dataset name or path for evaluation.
            embedding_model (str): Name of embedding model for retrieval.
            top_k (int): Top-K documents to consider for metrics.
        """
        self.dataset_name = dataset
        self.top_k = top_k

        self.test_dataset = self._load_dataset(dataset)

        self.retrieval_model = create_encode('automodel', embedding_model)

        self.metrics_list = [create_metrics(name) for name in metrics]

    def _load_dataset(self, dataset: str) -> TestDataset:
        """
        Load evaluation dataset.
        Returns a dataset object containing queries and ground truth docs.
        """
        return TestDataset(dataset)

    def get_embedding(self, texts: List[str]):
        """
        Encode a list of texts into embeddings using the retrieval model.
        """
        return self.retrieval_model.encode(texts)

    def search(self, query_embeddings, doc_embeddings) -> Dict[str, List[str]]:
        """
        Perform retrieval and return top-K predicted doc IDs for each query.

        Returns:
            Dict[str, List[str]]: {query_id: [pred_doc_id1, pred_doc_id2, ...]}
        """
        predictions = {}
        sim_matrix = cosine_similarity(query_embeddings, doc_embeddings)

        doc_ids = [d for d in list(self.test_dataset.docs.keys())]
        for i, query_id in enumerate(list(self.test_dataset.queries.keys())):
            top_indices = np.argsort(sim_matrix[i])[::-1][:self.top_k]
            top_doc_ids = [doc_ids[idx] for idx in top_indices]

            predictions[query_id] = top_doc_ids

        return predictions
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate retrieval performance.

        Returns:
            Dict[str, float]: Metrics including Recall@K, Precision@K, MRR, nDCG, retrieval_time_sec
        """
        start_time = time.time()
        query_embeddings = self.get_embedding(list(self.test_dataset.queries.values()))
        doc_embeddings = self.get_embedding(list(self.test_dataset.docs.values()))

        predictions = self.search(query_embeddings, doc_embeddings)

        for qid in list(self.test_dataset.queries.keys()):
            pred_docs = predictions.get(qid, [])
            relevant_docs = self.test_dataset.get_relevant_docs(qid)  
            for metric in self.metrics_list:
                metric.compute(pred_docs, relevant_docs)

        metrics_result = {metric.get_name(): metric.get_average() for metric in self.metrics_list}

        end_time = time.time()
        metrics_result["retrieval_time_sec"] = end_time - start_time
        return metrics_result
