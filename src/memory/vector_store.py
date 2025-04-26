import logging
import uuid
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

from memory.memory import Memory

logger = logging.getLogger(__name__)


class VectorStoreMemory(Memory):
    """
    Memory implementation using ChromaDB for storing and retrieving
    text snippets based on semantic similarity.
    """

    def __init__(
        self,
        db_path: str,
        collection_name: str,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        clear_on_init: bool = False,
    ):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model_name
            )
        )
        logger.info(f"Using embedding model: {embedding_model_name}")

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,  # type: ignore
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"Accessed or created ChromaDB collection: '{self.collection_name}'"
        )

    async def add(self, text: str, metadata: dict) -> Optional[str]:
        """
        Adds a text snippet and its metadata to the ChromaDB collection. Returns the ID of the added document.

        Args:
            text: The text content to store.
            metadata: A dictionary containing metadata (e.g., source_url, title).

        Returns:
            The ID of the added document.
        """
        if not text:
            logger.warning("Attempted to add empty text to memory. Skipping.")
            return None

        document_id = str(uuid.uuid4())
        self.collection.add(documents=[text], metadatas=[metadata], ids=[document_id])

        return document_id

    async def query(self, query_text: str) -> list[dict]:
        """
        Queries the collection for text snippets semantically similar to the query_text.

        Args:
            query_text: The text to query the collection with.

        Returns:
            A list of dictionaries containing the text and metadata of the results.
        """
        if not query_text:
            logger.warning("Attempted to query with empty text. Returning empty list.")
            return []

        results: chromadb.QueryResult = self.collection.query(
            query_texts=[query_text], include=["documents", "metadatas"], n_results=5
        )

        docs = results["documents"]
        metadatas = results["metadatas"]

        if docs is None or metadatas is None:
            logger.warning("No results found in ChromaDB. Returning empty list.")
            return []

        result = []
        for docs, metas in zip(docs, metadatas):
            for doc, meta in zip(docs, metas):
                result.append({
                    "document": doc,
                    "metadata": meta,
                })

        logger.info(f"Found {len(result)} results in ChromaDB.")
        logger.info(result)

        return result

    def clear(self):
        """Deletes the collection. Use with caution!"""
        self.client.delete_collection(name=self.collection_name)
        logger.info(f"Collection '{self.collection_name}' deleted.")

    def __len__(self) -> int:
        """Returns the number of items in the collection."""
        return self.collection.count()
