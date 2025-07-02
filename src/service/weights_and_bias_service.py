import asyncio
import json
import wandb
from dotenv import load_dotenv
import os

load_dotenv()

# Import Weights and Bias variables
WANDB_PROJECT = str(os.getenv("WANDB_PROJECT"))
WANDB_ENTITY = str(os.getenv("WANDB_ENTITY"))
WANDB_API_KEY = str(os.getenv("WANDB_API_KEY"))


class WeightsAndBiasService:
    """
    An asynchronous service to interact with the Weights & Biases API
    using the official wandb library.
    """
    def __init__(self):
        """
        Initializes the service and the wandb API client.
        The wandb.Api() call automatically authenticates using environment variables.
        """
        self.entity = WANDB_ENTITY
        self.project = WANDB_PROJECT
        self.api_key = WANDB_API_KEY
        
        if not all([self.entity, self.project, self.api_key]):
            raise ValueError(
                "WANDB_ENTITY, WANDB_PROJECT, and WANDB_API_KEY must be set in your environment."
            )
            
        # Initialize the official wandb API client
        self.api = wandb.Api(api_key=self.api_key)

    async def _sync_get_run_id_by_tags(self, knowledge_id: str, user_id: str):
        """Synchronous method to find a run ID using wandb.Api."""
        tags = [knowledge_id, user_id]
        
        # Use a server-side filter to find runs containing ALL specified tags.
        # This is highly efficient as the filtering happens on W&B's servers.
        runs = self.api.runs(
            path=f"{self.entity}/{self.project}",
            filters={"tags": {"$all": tags}}
        )
        
        # Return the ID of the first matched run, or None if no runs are found.
        return runs[0].id if runs else None

    async def fetch_run_history(self, knowledge_id: str, user_id: str):
        """
        Fetches the history (logged metrics) for a specific run identified by its tags.
        """
        run_id = await self._sync_get_run_id_by_tags(knowledge_id, user_id)
        if run_id is None:
            return []
        
        run_path = f"{self.entity}/{self.project}/{run_id}"
        run = self.api.run(run_path)
        
        # run.history() returns a pandas DataFrame
        history_df = run.history()
        
        # Convert the DataFrame to a list of dictionaries to match the
        # original desired output format.
        json_data = history_df.to_json(orient='records')
        return json.loads(json_data)


wb_service = WeightsAndBiasService()
