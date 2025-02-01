import os
import pickle
import re
from typing import Any

from src.exception import DuplicateUserError


class StorageManager:
    """
    A class to manage storing and retrieving pickle files for a user.

    This class provides CRUD (Create, Read, Update, Delete) operations for pickle files,
    with each file identified by an user_id (sanitized to be filename-friendly).

    Files are stored in a directory structure: 
    /data/username/filename.pkl
    """

    def __init__(self, base_dir: str = 'data'):
        """
        Initialize the PickleStorageManager.

        :param base_dir: Base directory for storing pickle files (default: 'data')
        """
        # Create the base directory if it doesn't exist
        self.base_dir = os.path.join(os.getcwd(), base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

    async def get_user_data_path(self, user_id: str):
        pruned_username = await self._sanitize_filename(user_id)
        return os.path.join(self.base_dir, pruned_username)

    @staticmethod
    async def _sanitize_filename(user_id: str) -> str:
        """
        Sanitize the user_id to create a safe filename.

        Replaces special characters with underscores and ensures 
        the filename is safe for filesystem use.

        :param user_id: user_id to be converted to a filename
        :return: Sanitized filename
        """
        # Remove any non-alphanumeric characters except periods and @ symbol
        sanitized = re.sub(r'[^a-zA-Z0-9.@]', '_', user_id)

        # Replace remaining special characters with underscore
        sanitized = re.sub(r'[/\\:]', '_', sanitized)

        return sanitized

    async def _get_user_dir(self, user_id: str) -> str:
        """
        Create and return the directory path for a specific user.

        :param user_id: Unique ID of the user
        :return: Path to the user's directory
        """
        username = await self._sanitize_filename(user_id)
        user_dir = os.path.join(self.base_dir, username)
        os.makedirs(user_dir, exist_ok=True)
        return user_dir

    async def create(self, user_id: str, filename: str, data: Any) -> str:
        """
        Create a new pickle file for a user.

        :param user_id: Unique ID of the user
        :param filename: Name of the file (without .pkl extension)
        :param data: Data to be stored
        :return: Full path of the created file
        :raises FileExistsError: If file already exists
        """
        # Sanitize filename and get user directory
        sanitized_filename = await self._sanitize_filename(filename)
        user_dir = await self._get_user_dir(user_id)

        # Construct full file path
        file_path = os.path.join(user_dir, f"{sanitized_filename}.pkl")

        # Check if file already exists to prevent overwriting
        if os.path.exists(file_path):
            raise DuplicateUserError(f"File {file_path} already exists")

        # Write data to pickle file
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

        return file_path

    async def read(self, user_id: str, filename: str) -> Any:
        """
        Read data from a user's pickle file.

        :param user_id: Unique ID of the user
        :param filename: Name of the file (without .pkl extension)
        :return: Data stored in the pickle file
        :raises FileNotFoundError: If file does not exist
        """
        # Sanitize filename and get user directory
        sanitized_filename = await self._sanitize_filename(filename)
        user_dir = await self._get_user_dir(user_id)

        # Construct full file path
        file_path = os.path.join(user_dir, f"{sanitized_filename}.pkl")

        # Read and return data from pickle file
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    async def update(self, user_id: str, filename: str, data: Any) -> str:
        """
        Update an existing pickle file for a user.

        :param user_id: Unique ID of the user
        :param filename: Name of the file (without .pkl extension)
        :param data: New data to be stored
        :return: Full path of the updated file
        :raises FileNotFoundError: If file does not exist
        """
        # Sanitize filename and get user directory
        sanitized_filename = await self._sanitize_filename(filename)
        user_dir = await self._get_user_dir(user_id)

        # Construct full file path
        file_path = os.path.join(user_dir, f"{sanitized_filename}.pkl")

        # Check if file exists before updating
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist")

        # Write updated data to pickle file
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

        return file_path

    async def delete(self, user_id: str, filename: str):
        """
        Delete a pickle file for a user.

        :param user_id: Unique ID of the user
        :param filename: Name of the file (without .pkl extension)
        :return: True if file was deleted, False if file did not exist
        """
        # Sanitize filename and get user directory
        sanitized_filename = await self._sanitize_filename(filename)
        user_dir = await self._get_user_dir(user_id)

        # Construct full file path
        file_path = os.path.join(user_dir, f"{sanitized_filename}.pkl")

        # Delete file if it exists, else raise FileNotFoundError
        os.remove(file_path)

    async def check_file_exists(self, user_id: str, filename: str):
        """
        Check if a file exists under the user's directory

        :param user_id: Unique ID of the user
        :param filename: Name of the file (without an extension)

        :return: True if file exists, False if it doesn't exist
        """
        # Sanitize both the username and the filename
        sanitized_user_id = await self._sanitize_filename(user_id)
        sanitized_filename = await self._sanitize_filename(filename)

        # Define the full file path
        file_path = os.path.join(self.base_dir, sanitized_user_id, sanitized_filename)

        return os.path.exists(file_path)

    async def list_files(self, user_id: str) -> list:
        """
        List all pickle files for a specific user.

        :param user_id: Unique ID of the user
        :return: List of pickle filenames
        """
        # Get user directory
        user_dir = await self._get_user_dir(user_id)

        # List all .pkl files in the user's directory
        return [f for f in os.listdir(user_dir) if f.endswith('.pkl')]
    
    
storage_manager = StorageManager()
    