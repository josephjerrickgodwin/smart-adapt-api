import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch
from main import app


class TestRAGController(unittest.TestCase):
    """Tests for rag_controller endpoints."""

    def setUp(self):
        self.client = TestClient(app)
        self.user_id = "test_user"

    @patch("src.service.storage_manager.storage_manager.read")
    def test_rag_search_success(self, mock_read):
        """Test RAG search with valid input."""
        mock_read.return_value.search.return_value = ["retrieved result"]

        response = self.client.post(f"/rag/search?user_id={self.user_id}", json={"query": "test", "top_k": 5})
        self.assertEqual(response.status_code, 200)
        self.assertIn("retrieved result", response.json())

    def test_rag_search_missing_index(self):
        """Test RAG search when index does not exist."""
        response = self.client.post(f"/rag/search?user_id=non_existing_user", json={"query": "test"})
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
