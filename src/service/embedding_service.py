import logging
import os
from typing import List

import numpy as np
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

logger = logging.getLogger(__name__)

# Load the Sentence Transformer Model
SENTENCE_MODEL_ID = str(os.getenv('SENTENCE_MODEL_ID'))


class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer(SENTENCE_MODEL_ID, trust_remote_code=True)
        self.batch_size = 16

    async def get_embeddings(self, sentences: List[str]):
        """
        Asynchronously computes embeddings for a list of sentences using a batch-wise approach.

        Args:
            sentences (List[str]): A list of input sentences for which embeddings are to be computed.

        Returns:
            np.ndarray: A NumPy array of shape (num_sentences, embedding_dim) containing the computed embeddings.

        Notes:
            - Uses `torch.no_grad()` to disable gradient calculations for efficiency.
            - Processes sentences in batches to optimize memory usage.
        """
        embeddings = []  # List to store computed embeddings

        with torch.no_grad():  # Disable gradient computation for efficiency
            for i in range(0, len(sentences), self.batch_size):
                # Extract the current batch of sentences
                batch = sentences[i:i + self.batch_size]

                # Compute embeddings for the batch
                batch_embeddings = self.model.encode(batch, show_progress_bar=False)

                # Append batch embeddings to the final list
                embeddings.extend(batch_embeddings)

        # Convert the list of embeddings to a NumPy array with dtype float32
        return np.asarray(embeddings, dtype=np.float32)


embedding_service = EmbeddingService()
