import logging
import sys

import ftfy
from langchain_unstructured import UnstructuredLoader

from src.service.env import GLOBAL_LOG_LEVEL

logging.basicConfig(stream=sys.stdout, level=GLOBAL_LOG_LEVEL)
log = logging.getLogger(__name__)


class Loader:
    """
    Provides methods to load documents from files, apply text normalization, extract metadata,
    And return structured text chunks with contextual headers. Selects the appropriate loader
    Based on file type and preserves document structure such as headings and page numbers.
    """

    def load_and_extract(self, filename: str, file_path: str, chunk_size: int = 256) -> list[str]:
        """
        Loads a document from the specified file using the appropriate loader based on
        file extension and content type, applies text fixing, and returns a list of Document objects
        with context-aware chunks.

        Args:
            filename (str): Name of the file to load.
            file_path (str): Path to the file.
            chunk_size (int): Size of text chunks

        Returns:
            list[str]: List of processed document chunks, each prefixed with a structured header.
        """
        # Get the appropriate loader instance and chunk the file
        loader = self._get_loader(
            file_path=file_path,
            chunk_size=chunk_size
        )
        docs = loader.load()

        chunks = []
        headers = ""

        # Process each document (which might be a page, a paragraph, or a structural element)
        for doc in docs:
            # 1. Fix any inconsistencies in the content
            content = ftfy.fix_text(doc.page_content)
            if not content:
                continue
            content = content.strip()

            # 2. Extract standard metadata
            metadata = doc.metadata
            page_number = metadata.get('page_number', None)

            # 3. Attempt to identify if this document represents a heading
            doc_category = metadata.get('category', '') or metadata.get('type', '')
            doc_category = doc_category.lower()

            # 4. If the document's category is a known heading type
            if doc_category in ['title']:
                headers = content
            elif doc_category.startswith('h'):
                headers += f'\n{headers}'

            # 5. Make sure that the current heading is not the content to avoid duplicates
            if content.strip() == headers:
                continue

            # 6. Add filename and page number
            parts = f"file name: {filename}"
            if page_number is not None:
                parts += f", page number: {page_number}"

            # 7. Add current heading if available
            if headers:
                parts += f", heading: {headers}"

            # 8. Assemble the header string
            chunk = f'{parts}\nContent: {content}'
            chunks.append(chunk)

        return chunks

    @staticmethod
    def _get_loader(file_path: str, chunk_size: int = 256):
        """
        Selects and returns the appropriate document loader that preserves document elements like headers.

        Args:
            file_path (str): Path to the file.
            chunk_size (int): Size of text chunks

        Returns:
            An instance of a document loader suitable for the file type.
        """
        # Break the document into its constituent parts, including headers, titles, and body text.
        return UnstructuredLoader(file_path, max_characters=chunk_size)
