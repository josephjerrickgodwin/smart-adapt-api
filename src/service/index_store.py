from typing import List
import numpy as np
import torch
import torch.nn as nn
import faiss
from torch import Tensor


class IndexStore:
    def __init__(
        self,
        input_size: int,
        ef_construction: int,
        ef_search: int,
        m: int
    ):
        # Create the HNSW index
        self.index = faiss.IndexHNSWFlat(input_size, m)

        # Apply the hyperparameters
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search

        self.labels = []
        self.embeddings = None

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def _reciprocal_rank_fusion(similarity_scores: np.ndarray):
        return 1 / (1 + np.argsort(similarity_scores))

    def _get_levels(self):
        levels = faiss.vector_to_array(self.index.hnsw.levels)
        return np.bincount(levels)

    async def add_index(self, vectors: np.ndarray, labels: List[str]):
        # Add the embeddings to the index
        self.index.add(vectors)

        # Update the labels list
        self.labels.extend(labels)

        # Update the embeddings array
        if self.embeddings is None:
            self.embeddings = vectors
        else:
            self.embeddings = np.vstack((self.embeddings, vectors))

    async def get_top_k_similar_embeddings(
            self,
            labels: List[str],
            embeddings: Tensor,
            query_embedding: np.ndarray,
            top_k: int,
            return_embeddings: bool = False
    ):
        """
        Computes cosine similarity between a query embedding and a list of embeddings,
        then returns the top-k embeddings sorted by similarity score.

        Args:
            labels (list): A list of labels corresponding to the embeddings.
            embeddings (list): A list of embeddings (numpy arrays or tensors).
            query_embedding (numpy.ndarray or list): The query embedding vector.
            top_k (int): The number of top similar embeddings to return.
            return_embeddings (bool): Whether to include embeddings in the result.

        Returns:
            list: A list of dictionaries containing 'label', 'score', and optionally 'embeddings' keys.
        """
        # Convert query_embedding to tensors
        query_tensor = torch.tensor(query_embedding, dtype=torch.float32)
        if query_tensor.dim() == 1:
            query_tensor = query_tensor.unsqueeze(0)

        # Compute cosine similarity
        cosine_similarities = self.cos(embeddings, query_tensor)

        # Ensure top-k does not exceed available samples
        top_k = min(top_k, len(cosine_similarities))

        # Get top-k indices and scores
        top_k_scores, top_k_indices = torch.topk(cosine_similarities, k=top_k)

        # Prepare the result
        result = []
        for score, idx in zip(top_k_scores, top_k_indices):
            label = labels[idx]
            entry = {'label': label, 'score': score.item()}
            if return_embeddings:
                entry['embeddings'] = embeddings[idx].tolist()
            result.append(entry)

        return result

    async def search_by_threshold(
            self,
            query: np.ndarray,
            initial_k: int = 1e+60,
            return_embeddings: bool = False
    ):
        """
        Perform an asynchronous search based on a similarity threshold.

        Args:
            query (np.ndarray):
                A NumPy array representing the query vector used to search for similar items.
            initial_k (int, optional):
                The initial number of top results to retrieve before applying the threshold filter.
                Defaults to 1000.
            return_embeddings (bool, optional):
                If True, includes the embeddings of the retrieved items in the results.
                Defaults to False.

        Returns:
            list:
                A list of results that meet the similarity threshold. Each result may include
                metadata such as the item ID, similarity score, and optionally the item's embedding
                if `return_embeddings` is set to True.

        Raises:
            ValueError:
                If the query array is empty or not a valid NumPy array.
            Exception:
                For any other errors encountered during the search process.
        """
        assert self.index is not None, "Index has not been initialized"

        # Search using HNSW
        distances, indices = self.index.search(query, initial_k)

        results = []
        for i in range(len(query)):
            label_list, embedding_list, distance_scores = [], [], []
            query_embedding = torch.tensor(query[i])
            for idx in indices[i]:
                label = self.labels[idx]
                embedding = torch.tensor(
                    self.embeddings[idx],
                    dtype=torch.float32
                ).reshape(1, -1)

                # Compute the distance score
                distance = distances[i][idx]
                if distance != -1:
                    distance = 1 / (1 + distance)

                    # Update the list
                    label_list.append(label)
                    embedding_list.append(embedding)
                    distance_scores.append(distance)

            # Stack the embeddings
            embedding_list = torch.cat(embedding_list, dim=0)

            # Compute cosine similarity
            cs_similarity_scores = self.cos(query_embedding, embedding_list)

            # Compute the mean scores of both cs and HNSW
            updated_scores = [
                (cosine_score.item() + (1 / (1 + distance_score))) / 2
                for cosine_score, distance_score in zip(cs_similarity_scores, distance_scores)
            ]

            # Compute the threshold
            threshold = np.mean(updated_scores)

            for idx in indices[i]:
                # Get the HNSW distance score
                distance = distances[i][idx]
                if distance != -1:
                    embedding = torch.tensor(
                        self.embeddings[idx],
                        dtype=torch.float32
                    ).reshape(1, -1)

                    # Compute score from the hnsw distance score
                    distance = 1 / (1 + distance)

                    # Get the cosine similarity score
                    cs_score = self.cos(query_embedding, embedding).item()

                    # Compute the overall score
                    score = (distance + cs_score) / 2.0

                    if score >= threshold:
                        label = self.labels[idx]
                        entry = {'label': label, 'score': score}
                        if return_embeddings:
                            entry['embeddings'] = embedding.reshape(1, -1)
                        results.append(entry)

            # Sort the results
            return sorted(results, key=lambda x: x['score'], reverse=True) if results else []

    async def search(
            self,
            query_embeddings: np.ndarray,
            top_k: int = 0,
            return_embeddings: bool = False
    ):
        assert self.index is not None, "Index has not been initialized"

        # Search using HNSW
        distances, indices = self.index.search(query_embeddings, top_k)

        results = []
        for i in range(len(query_embeddings)):
            labels, embeddings = [], []
            for idx in indices[i]:
                label = self.labels[idx]
                embedding = self.embeddings[idx]

                # Reshape to (1, -1)
                embedding = torch.tensor(embedding, dtype=torch.float32).reshape(1, -1)

                labels.append(label)
                embeddings.append(embedding)

            # Stack the embeddings
            embeddings = torch.cat(embeddings, dim=0)

            # Get the top-k results for each query
            query_embedding: np.ndarray = query_embeddings[i]

            # Based on the top-k, determine the type of search
            if top_k:
                current_results = await self.get_top_k_similar_embeddings(
                    labels=labels,
                    embeddings=embeddings,
                    query_embedding=query_embedding,
                    top_k=top_k,
                    return_embeddings=return_embeddings
                )
            else:
                current_results = await self.search_by_threshold(
                    query=query_embedding,
                    return_embeddings=return_embeddings
                )
            results.extend(current_results)

        return results
