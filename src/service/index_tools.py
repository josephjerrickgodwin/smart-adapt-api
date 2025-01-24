import gc
import itertools
import logging
import time

import faiss
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class IndexTools:
    def __init__(self):
        self.weights = {
            'recall': 0.9,
            'query_time': 0.1
        }

    @staticmethod
    async def evaluate_index(
            vectors: np.ndarray,
            query_vector: np.ndarray,
            ef_construction_values: list,
            ef_search_values: list,
            m_values: list
    ):
        """
        Evaluate accuracy, query time, and memory usage for m values

        :param vectors: Numpy vectors of the document chunks
        :param query_vector: Numpy vector of query
        :param ef_construction_values: HNSW construction values
        :param ef_search_values: HNSW ef search values
        :param m_values: HNSW m values

        :return: A list of dictionaries with the evaluation data
        """
        k = 1

        # Compute the exact neighbors using a brute-force (flat) index
        flat_index = faiss.IndexFlatL2(vectors.shape[1])
        flat_index.add(vectors)
        _, exact_indices = flat_index.search(query_vector, k)

        # Iterate over all combinations of m, ef_construction, and ef_search
        hyperparameter_combinations = list(itertools.product(
            m_values, ef_construction_values, ef_search_values
        ))
        total_combinations = len(hyperparameter_combinations)
        logger.debug(f"Total hyperparameter combinations to evaluate: {total_combinations}")

        results = []
        for idx, (m, ef_construction, ef_search) in tqdm(
                enumerate(hyperparameter_combinations, 1),
                total=len(hyperparameter_combinations),
                desc="Processing"
        ):
            # Create an index with the current hyperparameters
            index = faiss.IndexHNSWFlat(vectors.shape[1], m)

            # Set efConstruction and efSearch
            index.hnsw.efConstruction = ef_construction
            index.hnsw.efSearch = ef_search

            # Add the dataset to the index
            index.add(vectors)

            # Start the HNSW search
            start_time = time.time()
            distances, indices = index.search(query_vector, k)
            end_time = time.time()

            # Calculate overall query time
            query_time = end_time - start_time

            # Calculate recall
            recall = float(1 / (1 + distances[0][0]))

            # Collect results for this combination of hyperparameters
            results.append({
                'm': m,
                'recall': recall,
                'ef_search': ef_search,
                'query_time': query_time,
                'ef_construction': ef_construction
            })

            # Cleanup
            del index
            gc.collect()

        return results

    async def get_optimal_hyperparameters(
            self,
            vectors: np.ndarray,
            ef_construction_values: list,
            ef_search_values: list,
            m_values: list,
            epsilon: int = 1e-6
    ):
        """
        Get optimal HNSW hyperparameters for the given vectors

        :param vectors: Numpy array of document vectors
        :param ef_construction_values: HNSW construction values
        :param ef_search_values: HNSW ef search values
        :param m_values: HNSW m values
        :param epsilon: Hyperparameter

        :return:
        """
        # Generate the query vector randomly from the document vectors
        query_vector_pos = np.random.randint(0, vectors.shape[0])
        query_vector = vectors[query_vector_pos:query_vector_pos + 1]

        # Evaluate for a range of m values
        results = await self.evaluate_index(
            vectors=vectors,
            query_vector=query_vector,
            ef_construction_values=ef_construction_values,
            ef_search_values=ef_search_values,
            m_values=m_values
        )

        # Extract metrics from results
        recalls = np.array([res['recall'] for res in results])
        query_times = np.array([res['query_time'] for res in results])

        # Invert query times and memory usages because lower is better
        inv_query_times = query_times.max() - query_times

        # Normalize the metrics to [0, 1] range
        norm_recalls = (recalls - recalls.min()) / (recalls.max() - recalls.min() + epsilon)
        norm_query_times = (inv_query_times - inv_query_times.min()) / (
                    inv_query_times.max() - inv_query_times.min() + epsilon)

        # Compute a combined score using the weights
        scores = (self.weights['recall'] * norm_recalls) + (self.weights['query_time'] * norm_query_times)

        # Find the index of the best score
        best_index = np.argmax(scores)
        best_result = results[best_index]

        # Add the score to the best_result for reference
        best_result['score'] = scores[best_index]

        return results, best_result


index_tools = IndexTools()
