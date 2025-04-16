import logging
import uuid

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
        self, collection_name: str, embedding_model_name: str = "all-MiniLM-L6-v2"
    ):
        self.collection_name = collection_name
        # TODO: Implement persisted client
        self.client = chromadb.Client()
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

    async def add(self, text: str, metadata: dict):
        """
        Adds a text snippet and its metadata to the ChromaDB collection.

        Args:
            text: The text content to store.
            metadata: A dictionary containing metadata (e.g., source_url, title).
        """
        if not text:
            logger.warning("Attempted to add empty text to memory. Skipping.")
            return

        document_id = str(uuid.uuid4())
        self.collection.add(documents=[text], metadatas=[metadata], ids=[document_id])

    async def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        """
        Queries the collection for text snippets semantically similar to the query_text.

        Args:
            query_text: The text to query the collection with.
            top_k: The number of results to return.

        Returns:
            A list of dictionaries containing the text and metadata of the results.
        """
        if not query_text:
            logger.warning("Attempted to query with empty text. Returning empty list.")
            return []

        results: chromadb.QueryResult = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["documents", "metadatas"],
        )

        docs = results["documents"]
        metadatas = results["metadatas"]

        if docs is None or metadatas is None:
            logger.warning("No results found in ChromaDB. Returning empty list.")
            return []

        return [
            {"document": document, "metadata": metadata}
            for document, metadata in zip(docs, metadatas)
        ]

    async def clear(self):
        """Deletes the collection. Use with caution!"""
        self.client.delete_collection(name=self.collection_name)
        logger.info(f"Collection '{self.collection_name}' deleted.")

    def __len__(self) -> int:
        """Returns the number of items in the collection."""
        return self.collection.count()
