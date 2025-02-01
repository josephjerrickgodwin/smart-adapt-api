import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch
from main import app


class TestInferenceController(unittest.TestCase):
    """Tests for inference_controller endpoints."""

    def setUp(self):
        self.client = TestClient(app)
        self.user_id = "test_user"

    @patch("src.service.fine_tuning.model_service.start_inference")
    def test_model_inference_success(self, mock_inference):
        """Test successful model inference."""
        mock_inference.return_value = iter(["generated response"])

        response = self.client.post("/v1/completions", json={
            "user_id": self.user_id,
            "history": [{"role": "user", "content": "Hello"}]
        })
        self.assertEqual(response.status_code, 200)

    def test_model_inference_invalid_request(self):
        """Test inference with invalid input (missing fields)."""
        response = self.client.post("/v1/completions", json={})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
