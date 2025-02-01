import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch
from main import app


class TestIndexController(unittest.TestCase):
    """Tests for index_controller endpoints."""

    def setUp(self):
        self.client = TestClient(app)
        self.user_id = "test_user"

    @patch("src.service.storage_manager.storage_manager.read")
    def test_get_index_success(self, mock_read):
        """Test successful retrieval of index."""
        mock_read.return_value = {"data": b"mock_index_data"}

        response = self.client.get(f"/index/get/{self.user_id}")
        self.assertEqual(response.status_code, 200)

    def test_get_index_not_found(self):
        """Test fetching non-existent index (404)."""
        response = self.client.get(f"/index/get/non_existing_user")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
