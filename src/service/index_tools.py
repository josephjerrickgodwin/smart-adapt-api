import gc
import itertools
import logging
import time
import tracemalloc

import faiss
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class IndexTools:
    def __init__(self):
        self.weights = {
            'recall': 0.7,
            'query_time': 0.15,
            'memory_usage': 0.15
        }

    @staticmethod
    def evaluate_index(
            vectors: np.ndarray,
            query_vector: np.ndarray,
            ef_construction_values: list,
            ef_search_values: list,
            m_values: list
    ):
        """
        Evaluates the performance of FAISS HNSW (Hierarchical Navigable Small World) index
        for different hyperparameter values by comparing recall and query time.

        Args:
            vectors (np.ndarray): The dataset of vectors to be indexed (shape: [num_samples, dim]).
            query_vector (np.ndarray): The query vector to search (shape: [1, dim]).
            ef_construction_values (list): List of `efConstruction` values for HNSW index.
            ef_search_values (list): List of `efSearch` values for HNSW index.
            m_values (list): List of `M` values, which determine the number of neighbors in HNSW graph.

        Returns:
            list: A list of dictionaries, each containing:
                - 'm' (int): The `M` value used.
                - 'recall' (float): The recall metric calculated as `1 / (1 + distance)`.
                - 'ef_search' (int): The `efSearch` value used.
                - 'query_time' (float): Time taken to execute the search (in seconds).
                - 'ef_construction' (int): The `efConstruction` value used.
        """
        vector_dimension = vectors.ndim
        query_dimension = query_vector.ndim
        assert query_dimension == 2 and vector_dimension == 2, "Please ensure that the vector shape is 2D!"

        # Define the topK based on the query size
        k = query_vector.shape[0]

        # Compute the exact neighbors using a brute-force (flat) index
        flat_index = faiss.IndexFlatL2(vectors.shape[1])
        flat_index.add(vectors)
        _, exact_indices = flat_index.search(query_vector, k)

        # Flatten the exact indices
        exact_indices = exact_indices[query_dimension].flatten()

        # Iterate over all combinations of m, ef_construction, and ef_search
        hyperparameter_combinations = list(itertools.product(
            m_values, ef_construction_values, ef_search_values
        ))
        total_combinations = len(hyperparameter_combinations)

        results = []
        for idx, (m, ef_construction, ef_search) in tqdm(
                enumerate(hyperparameter_combinations, 1),
                total=total_combinations,
                desc="Processing"
        ):
            # Start memory tracking
            tracemalloc.start()

            # Create an index with the current hyperparameters
            index = faiss.IndexHNSWFlat(vectors.shape[1], m)

            # Set efConstruction and efSearch
            index.hnsw.efConstruction = ef_construction
            index.hnsw.efSearch = ef_search

            # Add the dataset to the index
            index.add(vectors)

            # Start the HNSW search
            start_time = time.time()
            _, approx_indices = index.search(query_vector, k)
            end_time = time.time()

            # Measure memory usage after index is built
            current, peak = tracemalloc.get_traced_memory()
            memory_usage = peak / (1024 * 1024)  # Convert bytes to MB
            tracemalloc.stop()

            # Convert the results to a single dimension
            approx_indices = approx_indices[query_dimension].flatten()

            # Calculate overall query time
            query_time = end_time - start_time

            # Calculate recall
            true_positives = np.sum(approx_indices == exact_indices)
            recall = round(true_positives / k, 2)

            # Collect results for this combination of hyperparameters
            results.append({
                'm': m,
                'recall': recall,
                'ef_search': ef_search,
                'query_time': query_time,
                'memory_usage': memory_usage,
                'ef_construction': ef_construction
            })

            # Cleanup
            del index
            gc.collect()

        return results

    def get_optimal_hyperparameters(
            self,
            vectors: np.ndarray,
            ef_construction_values: list,
            ef_search_values: list,
            m_values: list,
            epsilon: int = 1e-6,
            num_queries: int = 5
    ):
        """
        Get optimal HNSW hyperparameters for the given vectors

        :param vectors: Numpy array of document vectors
        :param ef_construction_values: HNSW construction values
        :param ef_search_values: HNSW ef search values
        :param m_values: HNSW m values
        :param epsilon: Hyperparameter
        :param num_queries: Number of queries to extract from vectors

        :return: A list of results of the experiment long with the optimal hyperparameters
        """
        assert vectors.ndim > num_queries, f"Number of queries: {num_queries} must be greater than number of vectors!"

        # Generate the query vector randomly from the document vectors
        query_vector_positions = np.random.choice(vectors.shape[0], num_queries, replace=False)
        query_vector = vectors[query_vector_positions]

        # Evaluate for a range of m values
        results = self.evaluate_index(
            vectors=vectors,
            query_vector=query_vector,
            ef_construction_values=ef_construction_values,
            ef_search_values=ef_search_values,
            m_values=m_values
        )

        # Extract metrics from results
        recalls = np.array([res['recall'] for res in results])
        query_times = np.array([res['query_time'] for res in results])
        memory_usages = np.array([res['memory_usage'] for res in results])

        # Invert query times and memory usages because lower is better
        inv_query_times = query_times.max() - query_times
        inv_memory_usages = memory_usages.max() - memory_usages

        # Normalize the metrics to [0, 1] range
        norm_recalls = (recalls - recalls.min()) / (recalls.max() - recalls.min() + epsilon)
        norm_query_times = (inv_query_times - inv_query_times.min()) / (
                inv_query_times.max() - inv_query_times.min() + epsilon
        )
        norm_memory_usages = (inv_memory_usages - inv_memory_usages.min()) / (
                inv_memory_usages.max() - inv_memory_usages.min() + epsilon
        )

        # Compute a combined score using the weights
        scores = (self.weights['recall'] * norm_recalls) + \
                 (self.weights['query_time'] * norm_query_times) + \
                 (self.weights['memory_usage'] * norm_memory_usages)

        # Find the index of the best score
        best_index = np.argmax(scores)
        best_result = results[best_index]

        # Add the score to the best_result for reference
        best_result['score'] = scores[best_index]

        return results, best_result


index_tools = IndexTools()
