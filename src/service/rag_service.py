import logging
from typing import List

import numpy as np

from src.model.status_enum import Status
from src.service.embedding_service import embedding_service
from src.service.index_store import IndexStore
from src.service.index_tools import index_tools

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        # Initialize an empty vector store.
        # Because it wasn't certain of the optimal parameters
        self.index_store = None

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

    async def get_all_hyperparameters(self):
        assert self.optimization_results is not None, "Hyperparameters have not been initialized yet!"
        return [
            {
                "ID": idx,
                "m": result['m'],
                "Query Time (s)": f"{result['query_time']:.10f}",
                "Recall": f"{result['recall']:.2%}",
                "ef_construction": f"{result['ef_construction']}",
                "ef_search": f"{result['ef_search']}",
                "Memory Usage": f"{result['memory_usage']:.6f}"
            }
            for idx, result in enumerate(self.optimization_results, start=1)
        ]

    def get_optimal_hyperparameters(self):
        assert self.optimization_results is not None, "Hyperparameters have not been optimized yet!"
        return {
            'm': self.m,
            'ef_construction': self.ef_construction,
            'ef_search': self.ef_search,
        }

    async def configure_vector_store(self, session_id: str, embeddings: np.ndarray, docs: List[str]):
        # Select the optimal hyperparameters
        results, optimal_result = index_tools.get_optimal_hyperparameters(
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
        self.index_store = IndexStore(embeddings.shape[1])

        # Create a new session for storing the index
        await self.index_store.create_session_for_index(
            session_id=session_id,
            ef_construction=self.ef_construction,
            ef_search=self.ef_search,
            m=self.m
        )

        # Add the embeddings and the labels to the Vector Store
        await self.index_store.add_index(
            session_id=session_id,
            vectors=embeddings,
            labels=docs
        )
        logger.info('Indexing completed successfully.')

        return {'status': Status.SUCCESS.value}

    async def search(
            self,
            query: str | np.ndarray,
            k: int = 0,
            return_embeddings: bool = False
    ):
        assert self.index_store is not None, "RAG initialization required!"

        # Generate the query embeddings
        if isinstance(query, str):
            logger.info('Generating the query embedding')
            query = await embedding_service.get_embeddings([query])

        # Search the Vector Store
        logger.info('Querying the vector store')
        results = await self.index_store.search(
            query_embeddings=query,
            top_k=k,
            return_embeddings=return_embeddings
        )
        logger.info(f'Generated a total of {len(results)} results.')

        return results
