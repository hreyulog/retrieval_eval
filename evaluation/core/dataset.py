import csv
import json
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
class TestDataset:
    """
    Dataset handler for RAG evaluation.
    
    Expects three files:
    - queries.csv: query_id, query_text
    - index.jsonl: doc_id, text
    - annotations.jsonl: query_id, relevant_doc_ids (dict)
    """

    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        queries_path = self.dataset_dir / "queries.csv"
        index_path = self.dataset_dir / "index.jsonl"
        annotations_path = self.dataset_dir / "annotations.jsonl"

        self.queries = self._load_queries(queries_path)
        self.docs = self._load_index(index_path)
        self.annotations = self._load_annotations(annotations_path)

    def _load_queries(self, query_path) -> Dict[str, str]:
        queries = pd.read_csv(query_path)
        return {str(row["query_id"]): row["query"] for _, row in queries.iterrows()}


    def _load_index(self, index_path) -> Dict[str, str]:
        index = {}
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                index[str(item["doc_id"])] = item["code"]
        return index

    def _load_annotations(self, annotations_path) -> Dict[str, Dict[str,float]]:
        annotations = {}
        with open(annotations_path, "r", encoding="utf-8") as f:
            for line in f:
                ann = json.loads(line)
                annotations[str(ann["query_id"])] = ann.get("doc_id2rels",{})
        return annotations

    def get_query(self, query_id: str) -> Optional[str]:
        for q in self.queries:
            if q["query_id"] == query_id:
                return q["query_text"]
        return None

    def get_relevant_docs(self, query_id: str) -> List[str]:
        return self.annotations.get(query_id, {})

    def get_doc_text(self, doc_id: str) -> Optional[str]:
        return self.index.get(doc_id)

    def __len__(self):
        return len(self.queries)
