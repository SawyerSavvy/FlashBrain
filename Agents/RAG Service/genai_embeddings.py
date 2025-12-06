"""
Lightweight async wrapper around the Gemini Embeddings API.
"""

import os
import asyncio
from typing import List
from google import genai
from google.genai import types


class GeminiEmbeddingClient:
    """Async-friendly wrapper for Google Gemini embeddings."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-embedding-001",
        output_dimensionality: int = 768
    ):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY (or GEMINI_API_KEY) is required for embeddings")
        self.model = model
        self.output_dimensionality = output_dimensionality
        self.client = genai.Client(api_key=self.api_key)

    async def aembed_documents(
        self,
        texts: List[str],
        task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> List[List[float]]:
        """Embed multiple texts asynchronously."""
        return await asyncio.to_thread(self._embed, texts, task_type)

    async def aembed_query(
        self,
        text: str,
        task_type: str = "RETRIEVAL_QUERY"
    ) -> List[float]:
        """Embed a single query asynchronously."""
        [vec] = await self.aembed_documents([text], task_type=task_type)
        return vec

    def _embed(self, texts: List[str], task_type: str) -> List[List[float]]:
        config = types.EmbedContentConfig(
            output_dimensionality=self.output_dimensionality,
            task_type=task_type
        )
        result = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=config
        )
        return [embedding.values for embedding in result.embeddings]
