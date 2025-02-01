from typing import List

import nltk
import pandas as pd

from src.exception.unicode_decode_error import UnicodeDecodeErrors


class DataPreprocessor:
    """Utility class for preprocessing text data."""

    @staticmethod
    def initialize_nltk():
        """Initialize NLTK resources."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            nltk.download('punkt_tab')

    @classmethod
    def bytes_to_text(cls, bytes_data: List[int]) -> str:
        """
        Convert bytes/integers to text.

        Args:
            bytes_data (List[int]): List of integer byte values

        Returns:
            str: Decoded text from bytes
        """
        try:
            # Convert list of integers to bytes and then decode
            byte_stream = bytes(bytes_data)
            return byte_stream.decode('latin-1', errors='ignore')
        except Exception as e:
            raise UnicodeDecodeErrors(f"Data format not supported! More Details: {str(e)}")

    @classmethod
    def preprocess_text(cls, data_contents: str, chunk_size: int = 128) -> List[str]:
        """
        Preprocess text by cleaning and chunking.

        Args:
            data_contents (str): Input text to preprocess
            chunk_size (int): Size of text chunks

        Returns:
            List[str]: Preprocessed text chunks
        """
        cls.initialize_nltk()

        # Remove problematic characters
        data_contents = (data_contents
                         .replace('\n', ' ')
                         .replace('\xa0', ' ')
                         .replace("'", ''))

        # Tokenize words
        words = nltk.word_tokenize(data_contents)

        # Create chunks
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            word = word.strip()
            if current_length + len(word) > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(word)
            current_length += len(word) + 1

        # Add last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    @classmethod
    def save_to_jsonl(cls, data: List[str], filename: str = 'dataset.jsonl'):
        """
        Save preprocessed data to JSONL format.

        Args:
            data (List[str]): Preprocessed text chunks
            filename (str): Output filename

        Returns:
            str: Path to saved JSONL file
        """
        df = pd.DataFrame({'text': data})
        df.to_json(filename, orient='records', lines=True)
        return filename


data_preprocessor = DataPreprocessor()
