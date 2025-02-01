import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch
from main import app


class TestDatasetController(unittest.TestCase):
    """Tests for dataset_controller endpoints."""

    def setUp(self):
        self.client = TestClient(app)
        self.user_id = "test_user"

    @patch("src.service.fine_tuning.data_preprocessor.bytes_to_text")
    @patch("src.service.fine_tuning.data_preprocessor.preprocess_text")
    def test_pre_process_dataset_success(self, mock_preprocess_text, mock_bytes_to_text):
        """Test dataset preprocessing with valid input."""
        mock_bytes_to_text.return_value = "clean text"
        mock_preprocess_text.return_value = ["cleaned dataset"]

        response = self.client.post(f"/dataset/prune?user_id={self.user_id}",
                                    json={"dataset": [104, 101, 108, 108, 111]})
        self.assertEqual(response.status_code, 200)
        self.assertIn("cleaned dataset", response.json())

    def test_pre_process_dataset_invalid_input(self):
        """Test preprocessing with invalid input (e.g., missing dataset)."""
        response = self.client.post(f"/dataset/prune?user_id={self.user_id}", json={})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
