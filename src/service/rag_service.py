import logging
import os

import numpy as np
import torch

from typing import List
from sentence_transformers import SentenceTransformer

from src.model.status_enum import Status
from src.service.index_store import IndexStore
from src.service.index_tools import index_tools

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Load the Sentence Transformer Model ID
SENTENCE_MODEL_ID = str(os.getenv('SENTENCE_MODEL_ID'))

# Load the sentence model
MODEL = SentenceTransformer(SENTENCE_MODEL_ID, trust_remote_code=True)


class RAGService:
    def __init__(self):
        # Initialize an empty vector store.
        # Because it wasn't certain of the optimal parameters
        self.index_store = None
        self.batch_size = 16

        # Initialize a list of HNSW Hyperparameters
        self.m_values = [value for value in range(4, 64, 8)]
        self.ef_construction_values = [value for value in range(10, 200, 40)]
        self.ef_search_values = [value for value in range(10, 200, 40)]

        # Hyperparameters
        self.ef_construction = None
        self.ef_search = None
        self.recall = None
        self.m = None

        self.optimization_results = None

    async def get_embeddings(self, sentences: List[str]):
        embeddings = []
        with torch.no_grad():
            # Process sentences in batches
            for i in range(0, len(sentences), self.batch_size):
                batch = sentences[i:i + self.batch_size]
                batch_embeddings = MODEL.encode(batch, show_progress_bar=False)
                embeddings.extend(batch_embeddings)
        return np.asarray(embeddings, dtype=torch.float32)

    async def get_all_hyperparameters(self):
        assert self.optimization_results is not None, "Hyperparameters have not been initialized yet!"
        return [
            {
                "ID": idx,
                "m": result['m'],
                "Query Time (s)": f"{result['query_time']:.10f}",
                "Recall": f"{result['recall']:.2%}",
                "ef_construction": f"{result['ef_construction']}",
                "ef_search": f"{result['ef_search']}"
            }
            for idx, result in enumerate(self.optimization_results, start=1)
        ]

    async def get_optimal_hyperparameters(self):
        assert self.optimization_results is not None, "Hyperparameters have not been optimized yet!"
        return {
            'M': self.m,
            'Recall': self.recall,
            'ef_construction': self.ef_construction,
            'ef_search': self.ef_search,
            'data_count': len(self.index_store)
        }

    async def configure_vector_store(self, docs: List[str]):
        assert len(docs) >= 2, "There must be at least two documents!"

        # Create embeddings
        embeddings = await self.get_embeddings(docs)

        # Select the optimal hyperparameters
        optimal_result, results = await index_tools.get_optimal_hyperparameters(
            vectors=embeddings,
            ef_construction_values=self.ef_construction_values,
            ef_search_values=self.ef_search_values,
            m_values=self.m_values
        )
        self.optimization_results = results

        # Parse the selected hyperparameters
        self.ef_construction = optimal_result['ef_construction']
        self.ef_search = optimal_result['ef_search']
        self.recall = optimal_result['recall']
        self.m = optimal_result['m']

        # Initialize the Vector Store
        logger.info('Building the Vector Store')
        self.index_store = IndexStore(
            input_size=embeddings.shape[1],
            ef_construction=self.ef_construction,
            ef_search=self.ef_search,
            m=self.m
        )

        # Add the embeddings and the labels to the Vector Store
        await self.index_store.add_index(vectors=embeddings, labels=docs)
        logger.info('Indexing completed successfully.')

        return {'status': Status.SUCCESS.value}

    async def search(
            self,
            query: str | np.ndarray,
            k: int = 10,
            return_embeddings: bool = False
    ):
        assert self.index_store is not None, "RAG initialization required!"

        # Generate the query embeddings
        if isinstance(query, str):
            logger.info('Generating the query embedding')
            query = await self.get_embeddings([query])

        # Search the Vector Store
        logger.info('Querying the vector store')
        results = self.index_store.search(
            query_embeddings=query,
            top_k=k,
            return_embeddings=return_embeddings
        )
        logger.info(f'Generated a total of {len(results)} results.')

        return results
