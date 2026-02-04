from typing import Dict, List
import torch
from .encoder_cls import Encoder
import logging
import numpy

logger = logging.getLogger(__name__)


class SentencetransformerEncoder(Encoder):
    @staticmethod
    def code_name() -> str:
        return "sentence_transformer"

    def __init__(self, model_name, **kwargs):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name,trust_remote_code=True)
        self.max_length = kwargs.get("max_length", 1024)

    def encode(self, texts: List[str], batch_size: int = 2):
        logging.info(
            f"Encoding {len(texts)} texts with batch size {batch_size} and max length {self.max_length}..."
        )
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
            )
            
            logging.info(
                f"Encoded {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}."
            )

            return embeddings
        except Exception as e:
            logging.error(f"Encoding failed: {str(e)}", exc_info=True)
            return numpy.array([])
