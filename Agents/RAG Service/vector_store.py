"""
Vector Store Operations for Supabase pgvector

This module handles storing and retrieving document chunks with embeddings
using Supabase's pgvector extension.
"""

import os
import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
import json
from supabase import create_client, Client
from psycopg import sql
from psycopg_pool import AsyncConnectionPool
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages vector storage and retrieval using Supabase pgvector."""

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        postgres_connection: Optional[str] = None
    ):
        """
        Initialize vector store with Supabase connections.

        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase service role key
            postgres_connection: Direct PostgreSQL connection string for pgvector operations
        """
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.postgres_connection = postgres_connection or os.getenv("SUPABASE_POOLER")

        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")

        # Initialize Supabase client for REST API operations
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)

        # Initialize PostgreSQL connection pool for direct pgvector operations
        # Note: Pool will be opened in async context (startup event)
        self.pool: Optional[AsyncConnectionPool] = None
        if self.postgres_connection:
            try:
                self.pool = AsyncConnectionPool(
                    conninfo=self.postgres_connection,
                    min_size=2,
                    max_size=5,
                    kwargs={"autocommit": True}
                )
                logger.info("PostgreSQL connection pool created (will be opened on first use)")
            except Exception as e:
                logger.warning(f"Failed to create PostgreSQL pool: {e}. Will use Supabase REST API only.")

    async def store_document(
        self,
        title: str,
        source: str,
        source_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        client_id: Optional[UUID] = None,
        project_id: Optional[UUID] = None
    ) -> UUID:
        """
        Store document metadata.

        Args:
            title: Document title
            source: Source URL or file path
            source_type: Type of source ('pdf', 'text', 'url', 'file')
            metadata: Additional metadata dictionary
            client_id: Optional client ID
            project_id: Optional project ID

        Returns:
            Document UUID
        """
        try:
            response = self.supabase.table("documents").insert({
                "title": title,
                "source": source,
                "source_type": source_type,
                "metadata": metadata or {},
                "client_id": str(client_id) if client_id else None,
                "project_id": str(project_id) if project_id else None
            }).execute()

            if response.data and len(response.data) > 0:
                doc_id = UUID(response.data[0]["id"])
                logger.info(f"Stored document: {doc_id} - {title}")
                return doc_id
            else:
                raise ValueError("No document ID returned from insert")
        except Exception as e:
            logger.error(f"Failed to store document: {e}")
            raise

    async def create_document_with_chunks(
        self,
        title: str,
        source: str,
        source_type: str,
        doc_type: str = "auto",
        format_type: str = "auto",
        chunks: List[Dict[str, Any]] = None,
        client_id: Optional[UUID] = None,
        project_id: Optional[UUID] = None
    ) -> UUID:
        """
        Convenience method to create a document and store its chunks in one operation.

        Args:
            title: Document title
            source: Source URL or file path
            source_type: Type of source ('pdf', 'text', 'url', 'file')
            doc_type: Document type ('technical', 'legal', 'security', etc.)
            format_type: Format type ('markdown', 'html', 'pdf', 'plain')
            chunks: List of chunk dictionaries with embeddings
            client_id: Optional client ID
            project_id: Optional project ID

        Returns:
            Document UUID
        """
        # Ensure pool is opened
        if self.pool and not self.pool._opened:
            await self.pool.open()

        # Create document metadata
        metadata = {
            "doc_type": doc_type,
            "format_type": format_type
        }

        document_id = await self.store_document(
            title=title,
            source=source,
            source_type=source_type,
            metadata=metadata,
            client_id=client_id,
            project_id=project_id
        )

        # Store chunks if provided
        if chunks:
            await self.store_chunks(document_id, chunks)

        return document_id

    async def store_chunks(
        self,
        document_id: UUID,
        chunks: List[Dict[str, Any]]
    ) -> int:
        """
        Store document chunks with embeddings using batch insert for performance.

        Args:
            document_id: Document UUID
            chunks: List of chunk dictionaries with:
                - chunk_index: int
                - content: str
                - embedding: list[float] (768 dimensions for Google gemini-embedding-001)
                - metadata: Optional[Dict[str, Any]]

        Returns:
            Number of chunks stored
        """
        if not self.pool:
            raise RuntimeError("PostgreSQL connection pool not initialized. Cannot store embeddings.")

        # Ensure pool is opened
        if not self.pool._opened:
            await self.pool.open()

        if not chunks:
            logger.warning(f"No chunks to store for document {document_id}")
            return 0

        try:
            # Prepare batch data: convert embeddings to PostgreSQL vector format
            # Batch processing is much faster than individual inserts
            batch_data = []
            for chunk in chunks:
                # Strip NUL bytes that PostgreSQL text columns cannot store
                content = (chunk["content"] or "").replace("\x00", "")

                # Convert embedding list to PostgreSQL vector format
                embedding_str = "[" + ",".join(map(str, chunk["embedding"])) + "]"

                batch_data.append((
                    str(document_id),
                    chunk["chunk_index"],
                    content,
                    embedding_str,
                    json.dumps(chunk.get("metadata", {}))
                ))

            # Batch insert using executemany for optimal performance
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.executemany(
                        sql.SQL("""
                            INSERT INTO document_chunks
                            (document_id, chunk_index, content, embedding, metadata)
                            VALUES (%s, %s, %s, %s::vector, %s)
                            ON CONFLICT (document_id, chunk_index)
                            DO UPDATE SET
                                content = EXCLUDED.content,
                                embedding = EXCLUDED.embedding,
                                metadata = EXCLUDED.metadata
                        """),
                        batch_data
                    )

            stored_count = len(batch_data)
            logger.info(f"Batch stored {stored_count} chunks for document {document_id}")
            return stored_count
        except Exception as e:
            logger.error(f"Failed to store chunks: {e}")
            raise

    async def similarity_search(
        self,
        query_embedding: List[float],
        match_threshold: float = 0.7,
        match_count: int = 10,
        document_id: Optional[UUID] = None,
        client_id: Optional[UUID] = None,
        project_id: Optional[UUID] = None,
        query_text: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search using pgvector.

        Args:
            query_embedding: Query embedding vector (768 dimensions)
            match_threshold: Minimum similarity threshold (0-1)
            match_count: Maximum number of results
            document_id: Optional filter by document ID
            client_id: Optional filter by client ID
            project_id: Optional filter by project ID
            query_text: Optional raw query text for sparse (lexical) scoring

        Returns:
            List of matching chunks with similarity scores
        """
        if not self.pool:
            raise RuntimeError("PostgreSQL connection pool not initialized. Cannot perform vector search.")

        # Ensure pool is opened
        if not self.pool._opened:
            await self.pool.open()

        try:
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
            lexical_weight = 0.3

            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        sql.SQL("""
                            SELECT 
                                dc.id,
                                dc.document_id,
                                dc.chunk_index,
                                dc.content,
                                dc.metadata,
                                1 - (dc.embedding <=> %s::vector) as dense_similarity,
                                CASE
                                    WHEN %s::text IS NULL THEN 0
                                    ELSE ts_rank_cd(dc.content_tsv, plainto_tsquery('english', %s::text))
                                END AS lexical_score,
                                (1 - (dc.embedding <=> %s::vector)) + (%s * CASE
                                    WHEN %s::text IS NULL THEN 0
                                    ELSE ts_rank_cd(dc.content_tsv, plainto_tsquery('english', %s::text))
                                END) AS hybrid_score
                            FROM document_chunks dc
                            JOIN documents d ON d.id = dc.document_id
                            WHERE
                                dc.embedding IS NOT NULL
                                AND (1 - (dc.embedding <=> %s::vector)) >= %s
                                AND (%s::uuid IS NULL OR dc.document_id = %s::uuid)
                                AND (%s::uuid IS NULL OR d.client_id = %s::uuid)
                                AND (%s::uuid IS NULL OR d.project_id = %s::uuid)
                            ORDER BY hybrid_score DESC
                            LIMIT %s
                        """),
                        (
                            embedding_str,
                            query_text,
                            query_text,
                            embedding_str,
                            lexical_weight,
                            query_text,
                            query_text,
                            embedding_str,
                            match_threshold,
                            str(document_id) if document_id else None,
                            str(document_id) if document_id else None,
                            str(client_id) if client_id else None,
                            str(client_id) if client_id else None,
                            str(project_id) if project_id else None,
                            str(project_id) if project_id else None,
                            match_count
                        )
                    )

                    results = []
                    async for row in cur:
                        results.append({
                            "id": str(row[0]),
                            "document_id": str(row[1]),
                            "chunk_index": row[2],
                            "content": row[3],
                            "metadata": json.loads(row[4]) if isinstance(row[4], str) else row[4],
                            "similarity": float(row[5]),  # dense similarity
                            "lexical_score": float(row[6]),
                            "hybrid_score": float(row[7])
                        })

            logger.info(f"Found {len(results)} matching chunks")
            return results
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            raise

    async def get_document_by_id(
        self,
        document_id: UUID
    ) -> Optional[Dict[str, Any]]:
        """
        Get document by ID to verify it exists.

        Args:
            document_id: Document UUID

        Returns:
            Document dictionary if found, None otherwise
        """
        try:
            response = self.supabase.table("documents").select("*").eq("id", str(document_id)).execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Failed to get document by ID: {e}")
            raise

    async def get_document_chunks(
        self,
        document_id: UUID
    ) -> List[Dict[str, Any]]:
        """
        Get all chunks for a document.

        Args:
            document_id: Document UUID

        Returns:
            List of chunks ordered by chunk_index
        """
        try:
            response = self.supabase.table("document_chunks").select(
                "id, chunk_index, content, metadata"
            ).eq("document_id", str(document_id)).order("chunk_index").execute()

            chunks = []
            for row in response.data:
                chunks.append({
                    "id": row["id"],
                    "chunk_index": row["chunk_index"],
                    "content": row["content"],
                    "metadata": row.get("metadata", {})
                })

            return chunks
        except Exception as e:
            logger.error(f"Failed to get document chunks: {e}")
            raise

    async def delete_document(self, document_id: UUID) -> bool:
        """
        Delete a document and all its chunks (cascade).

        Args:
            document_id: Document UUID

        Returns:
            True if successful
        """
        try:
            self.supabase.table("documents").delete().eq("id", str(document_id)).execute()
            logger.info(f"Deleted document: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            raise

    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Closed PostgreSQL connection pool")
