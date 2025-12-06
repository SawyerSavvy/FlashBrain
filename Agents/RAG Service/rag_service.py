"""
RAG Service with MCP Tools

Provides two MCP tools:
1. process_document - Intelligently decompose documents into chunks
2. search_knowledge_base - Semantic similarity search
"""

import os
import logging
import json
import asyncio
from typing import Dict, Any, Optional, List
from uuid import UUID
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from vector_store import VectorStore
from document_processor import DocumentProcessor
from genai_embeddings import GeminiEmbeddingClient

load_dotenv()

logger = logging.getLogger(__name__)


class ProcessDocumentRequest(BaseModel):
    """Request model for process_document tool."""
    title: str = Field(..., description="Document title")
    client_id: Optional[str] = Field(None, description="Client ID for access control")
    file_path: Optional[str] = Field(None, description="Path to local file")
    file_url: Optional[str] = Field(None, description="URL to fetch file from")
    content: Optional[str] = Field(None, description="Direct text content")
    supabase_bucket_name: Optional[str] = Field(None, description="Supabase Storage bucket name (e.g., 'documents')")
    supabase_bucket_path: Optional[str] = Field(None, description="Path to file in Supabase Storage bucket (e.g., 'project-123/doc.pdf')")
    source_type: str = Field("pdf", description="Type: 'pdf', 'text', 'url', 'markdown', 'html'")
    doc_type: str = Field("auto", description="Document type: 'technical', 'legal', 'security', 'compliance', etc.")
    force_llm_validation: bool = Field(False, description="Force LLM validation for critical documents")


class ProcessDocumentResponse(BaseModel):
    """Response model for process_document tool."""
    document_id: str
    chunk_count: int
    title: str
    source: str


class SearchKnowledgeBaseRequest(BaseModel):
    """Request model for search_knowledge_base tool."""
    query: str = Field(..., description="Search query")
    match_threshold: float = Field(0.7, description="Minimum similarity threshold (0-1)")
    match_count: int = Field(10, description="Maximum number of results")
    document_id: Optional[str] = Field(None, description="Filter by document ID")
    client_id: Optional[str] = Field(None, description="Filter by client ID")
    project_id: Optional[str] = Field(None, description="Filter by project ID")


class SearchResult(BaseModel):
    """Single search result."""
    id: str
    document_id: str
    chunk_index: int
    content: str
    similarity: float
    metadata: Dict[str, Any]


class SearchKnowledgeBaseResponse(BaseModel):
    """Response model for search_knowledge_base tool."""
    results: List[SearchResult]
    query: str
    total_results: int


class RAGService:
    """RAG service providing document processing and semantic search."""

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        postgres_connection: Optional[str] = None
    ):
        """Initialize RAG service."""
        self.vector_store = VectorStore(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            postgres_connection=postgres_connection
        )
        self.document_processor = DocumentProcessor()
        logger.info("RAG Service initialized")

    async def process_document(self, request: ProcessDocumentRequest) -> ProcessDocumentResponse:
        """
        Process a document with hybrid semantic chunking.

        Automatically creates the document record in Supabase, processes it into
        semantic chunks, and stores chunks with embeddings.

        MCP Tool: process_document
        """
        try:
            # Convert client_id to UUID if provided
            client_uuid = None
            if request.client_id:
                try:
                    client_uuid = UUID(request.client_id)
                except ValueError:
                    raise ValueError(f"Invalid client_id format: {request.client_id}")

            logger.info(f"Processing document: {request.title}")

            # Process document using hybrid chunking
            result = await self.document_processor.process_document(
                file_path=request.file_path,
                file_url=request.file_url,
                content=request.content,
                supabase_bucket_name=request.supabase_bucket_name,
                supabase_bucket_path=request.supabase_bucket_path,
                source_type=request.source_type,
                doc_type=request.doc_type,
                force_llm_validation=request.force_llm_validation,
                supabase_client=self.vector_store.supabase
            )

            chunks = result["chunks"]
            doc_metadata = result["document_metadata"]

            # Create document and store chunks with embeddings
            document_uuid = await self.vector_store.create_document_with_chunks(
                title=request.title,
                source=doc_metadata["source"],
                source_type=doc_metadata["source_type"],
                doc_type=doc_metadata.get("doc_type", "auto"),
                format_type=doc_metadata.get("format_type", "auto"),
                chunks=chunks,
                client_id=client_uuid
            )

            logger.info(f"Created document {document_uuid} with {len(chunks)} chunks")

            return ProcessDocumentResponse(
                document_id=str(document_uuid),
                chunk_count=len(chunks),
                title=request.title,
                source=doc_metadata["source"]
            )
        except Exception as e:
            logger.error(f"Failed to process document: {e}")
            raise

    async def search_knowledge_base(self, request: SearchKnowledgeBaseRequest) -> SearchKnowledgeBaseResponse:
        """
        Perform semantic similarity search.

        MCP Tool: search_knowledge_base
        """
        try:
            # Generate query embedding
            embeddings = GeminiEmbeddingClient(output_dimensionality=768)
            query_embedding = await embeddings.aembed_query(request.query)

            # Convert IDs to UUID if provided
            document_uuid = None
            if request.document_id:
                try:
                    document_uuid = UUID(request.document_id)
                except ValueError:
                    logger.warning(f"Invalid document_id format: {request.document_id}")
                    # Continue without document_id filter
            
            client_uuid = None
            if request.client_id:
                try:
                    client_uuid = UUID(request.client_id)
                except ValueError:
                    logger.warning(f"Invalid client_id format: {request.client_id}")
            
            project_uuid = None
            if request.project_id:
                try:
                    project_uuid = UUID(request.project_id)
                except ValueError:
                    logger.warning(f"Invalid project_id format: {request.project_id}")
            
            results = await self.vector_store.similarity_search(
                query_embedding=query_embedding,
                match_threshold=request.match_threshold,
                match_count=request.match_count,
                document_id=document_uuid,
                client_id=client_uuid,
                project_id=project_uuid,
                query_text=request.query
            )

            # Convert to response format
            search_results = [
                SearchResult(
                    id=r["id"],
                    document_id=r["document_id"],
                    chunk_index=r["chunk_index"],
                    content=r["content"],
                    similarity=r["similarity"],
                    metadata=r.get("metadata", {})
                )
                for r in results
            ]

            logger.info(f"Found {len(search_results)} results for query: {request.query}")

            return SearchKnowledgeBaseResponse(
                results=search_results,
                query=request.query,
                total_results=len(search_results)
            )
        except Exception as e:
            logger.error(f"Failed to search knowledge base: {e}")
            raise

    async def close(self):
        """Cleanup resources."""
        await self.vector_store.close()
