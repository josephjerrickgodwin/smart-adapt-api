from typing import List

import faiss
import numpy as np
import torch
import torch.nn as nn


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

    async def search_by_top_k(
            self,
            query_embedding: np.ndarray,
            top_k: int,
            return_embeddings: bool = False
    ):
        """
        Computes cosine similarity between a query embedding and a list of embeddings,
        Then returns the top-k embeddings sorted by similarity score.

        Args:
            query_embedding (numpy.ndarray or list): The query embedding vector.
            top_k (int): The number of top similar embeddings to return.
            return_embeddings (bool): Whether to include embeddings in the result.

        Returns:
            list: A list of dictionaries containing 'label', 'score', and optionally 'embeddings' keys.
        """
        labels, embeddings = [], []

        # Search using HNSW
        _, indices = self.index.search(query_embedding, top_k)

        # Reshape the indices to 1D
        indices = indices.reshape(-1)

        for idx in indices:
            label = self.labels[idx]
            embedding = self.embeddings[idx]

            # Reshape to (1, -1)
            embedding = torch.tensor(embedding, dtype=torch.float32).reshape(1, -1)

            labels.append(label)
            embeddings.append(embedding)

        # Stack the embeddings
        embeddings = torch.cat(embeddings, dim=0)

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
            query_embedding: np.ndarray,
            return_embeddings: bool = False
    ):
        """
        Perform an asynchronous search based on a similarity threshold.

        Args:
            query_embedding (np.ndarray):
                A NumPy array representing the query vector used to search for similar items.
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
        distances, indices = self.index.search(query_embedding, len(self.labels))

        # Reshape the result to 1D
        distances = distances.reshape(-1)
        indices = indices.reshape(-1)

        # Convert the query to a tensor
        query_embedding = torch.tensor(query_embedding, dtype=torch.float32)
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)

        results, scores = [], []
        for idx in indices:
            # Compute the distance score
            distance = distances[idx]
            if distance != -1:
                embedding = torch.tensor(self.embeddings[idx], dtype=torch.float32).reshape(1, -1)
                score = self.cos(query_embedding, embedding).item()
                scores.append(score)
                if return_embeddings:
                    results.append({
                        'label': self.labels[idx],
                        'embeddings': embedding,
                        'score': score
                    })
                else:
                    results.append({
                        'label': self.labels[idx],
                        'score': score
                    })

        # Compute the threshold
        threshold = np.mean(scores)

        # Filter out results where the score is below the threshold
        results = [entry for entry in results if entry['score'] >= threshold]

        # Sort the results
        return sorted(results, key=lambda x: x['score'], reverse=True) if results else []

    async def search(
            self,
            query_embeddings: np.ndarray,
            top_k: int = 0,
            return_embeddings: bool = False
    ):
        assert self.index is not None, "Index has not been initialized"

        results = []
        for i in range(len(query_embeddings)):
            # Get the current query
            query_embedding: np.ndarray = query_embeddings[i].reshape(1, -1)

            # Based on the top-k, determine the type of search
            if top_k:
                # Rank using cosine similarity
                current_results = await self.search_by_top_k(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    return_embeddings=return_embeddings
                )
            else:
                current_results = await self.search_by_threshold(
                    query_embedding=query_embedding,
                    return_embeddings=return_embeddings
                )
            results.extend(current_results)

        return results
