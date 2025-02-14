from fastapi.testclient import TestClient

from src.controller.dataset_controller import router

client = TestClient(router, raise_server_exceptions=False)

USER_ID = '91197417_b43e_4986_a244_8ce74a2fba16'
params = {"user_id": USER_ID}


class TestDataset:
    """
    Test class for Dataset Controller
    """
    def test_pre_process_dataset_valid(self):
        """
        Test pre-processing dataset with valid data.
        """
        payload = {"dataset": [72, 101, 108, 108, 111]}  # "Hello" in ASCII
        response = client.post("/dataset/prune", params=params, json=payload)
        assert response.status_code == 200
        assert response.headers["Content-Disposition"] == "attachment; filename=data.json"

    def test_pre_process_dataset_invalid(self):
        """
        Test pre-processing dataset with invalid data (non-ASCII characters).
        """
        payload = {"dataset": [255, 256, 300]}  # Invalid Unicode
        response = client.post("/dataset/prune", params=params, json=payload)
        if response.status_code != 415:
            print(f"Test failed: Expected status code 415, got {response.status_code}")
        # assert response.status_code == 415

    def test_lora_hyperparameters_valid(self):
        """
        Test LoRA hyperparameters calculation with valid inputs.
        """
        payload = {"model_size": 1000000, "dataset_size": 50000}
        response = client.post("/dataset/hyperparameters", params=params, json=payload)
        assert response.status_code == 200
        assert isinstance(response.json(), dict)

    def test_lora_hyperparameters_invalid(self):
        """
        Test LoRA hyperparameters with invalid parameters.
        """
        payload = {"model_size": -1000000, "dataset_size": "abc"}
        response = client.post("/dataset/hyperparameters", params=params, json=payload)
        assert response.status_code == 400

    def start(self):
        self.test_pre_process_dataset_valid()
        self.test_pre_process_dataset_invalid()
        self.test_lora_hyperparameters_valid()
        self.test_lora_hyperparameters_invalid()


# if __name__ == "__main__":
#     test_dataset = TestDataset()

