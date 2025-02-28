from typing import List, Dict

import faiss
import numpy as np
import torch
import torch.nn as nn


class IndexStore:
    def __init__(self, input_size: int):
        self.input_size = input_size

        # Store multiple indices based on session IDs
        self.indices: Dict[str, faiss.IndexHNSWFlat] = {}
        self.labels: Dict[str, List[str]] = {}
        self.embeddings: Dict[str, np.ndarray | None] = {}

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def _reciprocal_rank_fusion(similarity_scores: np.ndarray):
        return 1 / (1 + np.argsort(similarity_scores))

    async def create_session_for_index(
            self,
            session_id: str,
            ef_construction: int,
            ef_search: int,
            m: int
    ):
        """Creates a new index for a given session ID."""
        if session_id in self.indices:
            return  # Index already exists for this session

        # Create a new FAISS index for the session
        index = faiss.IndexHNSWFlat(self.input_size, m)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search

        # Store in the dictionary
        self.indices[session_id] = index
        self.labels[session_id] = []
        self.embeddings[session_id] = None

    async def add_index(self, session_id: str, vectors: np.ndarray, labels: List[str]):
        assert session_id in self.indices, "Session index not found. Call create_session_index first."

        # Add the embeddings to the index
        index = self.indices[session_id]

        # Add embeddings to the FAISS index
        index.add(vectors)

        # Update labels and embeddings for the session
        self.labels[session_id].extend(labels)

        if self.embeddings[session_id] is None:
            self.embeddings[session_id] = vectors
        else:
            self.embeddings[session_id] = np.vstack((self.embeddings[session_id], vectors))

    async def remove_index(self, session_id: str):
        """Removes the FAISS index for a given session."""
        if session_id in self.indices:
            del self.indices[session_id]
            del self.labels[session_id]
            del self.embeddings[session_id]

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

        Raises:
            ValueError:
                If the query array is empty or not a valid NumPy array or the index does not exist.
            Exception:
                For any other errors encountered during the search process.
        """
        if not self.indices:
            raise ValueError('Index is not initialized!')

        results = []

        # Search using HNSW
        for session_id, index in self.indices.items():
            _, indices = index.search(query_embedding, top_k)

            # Reshape the indices to 1D
            indices = indices.reshape(-1)

            labels, embeddings = [], []
            for idx in indices:
                label = self.labels[session_id][idx]
                embedding = self.embeddings[session_id][idx]

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
            for score, idx in zip(top_k_scores, top_k_indices):
                label = labels[idx]
                entry = {'label': label, 'score': score.item()}
                if return_embeddings:
                    entry['embeddings'] = embeddings[idx].tolist()
                results.append(entry)

        return results

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
                If the query array is empty or not a valid NumPy array or the index does not exist.
            Exception:
                For any other errors encountered during the search process.
        """
        if not self.indices:
            raise ValueError('Index is not initialized!')

        results, scores = [], []

        for session_id, index in self.indices.items():
            current_labels = self.labels[session_id]
            current_embeddings = self.embeddings[session_id]

            # Search using HNSW
            distances, indices = index.search(query_embedding, len(current_labels))

            # Reshape the result to 1D
            distances = distances.reshape(-1)
            indices = indices.reshape(-1)

            # Convert the query to a tensor
            query_embedding = torch.tensor(query_embedding, dtype=torch.float32)
            if query_embedding.dim() == 1:
                query_embedding = query_embedding.unsqueeze(0)

            for idx in indices:
                # Compute the distance score
                distance = distances[idx]
                if distance == -1:
                    continue    # Skip is the distance is too long against the query

                embedding = torch.tensor(current_embeddings[idx], dtype=torch.float32).reshape(1, -1)
                score = self.cos(query_embedding, embedding).item()
                scores.append(score)
                if return_embeddings:
                    results.append({
                        'label': current_labels[idx],
                        'embeddings': embedding,
                        'score': score
                    })
                else:
                    results.append({
                        'label': current_labels[idx],
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
        assert self.indices is not None, "Index has not been initialized"

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
