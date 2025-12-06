"""
Intelligent Document Processor

Uses hybrid semantic chunking to understand document structure and create
semantic chunks with preserved boundaries (headers, sections, paragraphs).

Uses three-stage approach:
1. Structure Parsing - Parse document hierarchy
2. Embedding-Based Refinement - Detect semantic boundaries
3. Optional LLM Validation - Validate boundaries for critical documents
"""

import os
import logging
import asyncio
import json
import re
import httpx
import tempfile
from typing import List, Dict, Any, Optional
from uuid import uuid4
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from dotenv import load_dotenv
from chunking import HybridChunker
from genai_embeddings import GeminiEmbeddingClient

load_dotenv()

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Processes documents with hybrid semantic chunking."""

    def __init__(
        self,
        max_chunk_size: int = 1000,
        embedding_threshold: float = 0.5,
        validate_with_llm: bool = False
    ):
        """
        Initialize document processor with hybrid chunker.

        Args:
            max_chunk_size: Maximum chunk size in characters
            embedding_threshold: Similarity threshold for splitting (0-1)
            validate_with_llm: Whether to validate with LLM by default
        """
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.embeddings = GeminiEmbeddingClient(
            output_dimensionality=768
        )

        # Initialize hybrid chunker
        self.chunker = HybridChunker(
            embeddings_model=self.embeddings,
            llm=self.llm,
            max_chunk_size=max_chunk_size,
            embedding_threshold=embedding_threshold,
            validate_with_llm=validate_with_llm
        )

        # Keep text splitter as fallback
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    async def process_document(
        self,
        file_path: Optional[str] = None,
        file_url: Optional[str] = None,
        content: Optional[str] = None,
        supabase_bucket_name: Optional[str] = None,
        supabase_bucket_path: Optional[str] = None,
        source_type: str = "pdf",
        supabase_client: Optional[Any] = None,
        doc_type: str = "auto",
        force_llm_validation: bool = False
    ) -> Dict[str, Any]:
        """
        Process a document with hybrid semantic chunking.

        Args:
            file_path: Path to local file
            file_url: URL to fetch file from
            content: Direct text content (if source_type is 'text')
            supabase_bucket_name: Supabase Storage bucket name (e.g., 'documents')
            supabase_bucket_path: Path to file in Supabase Storage bucket
            source_type: Type of document ('pdf', 'text', 'url', 'markdown', 'html')
            supabase_client: Supabase client instance for storage access
            doc_type: Document type ('technical', 'legal', 'security', 'compliance', etc.)
            force_llm_validation: Force LLM validation regardless of doc_type

        Returns:
            Dictionary with:
                - chunks: List of processed chunks with embeddings
                - document_metadata: Document metadata
        """
        try:
            # Load document
            documents = await self._load_document(
                file_path, file_url, content,
                supabase_bucket_name, supabase_bucket_path,
                source_type, supabase_client
            )
            if not documents:
                raise ValueError("No documents loaded")

            # Combine all pages/sections into single text
            full_text = "\n\n".join([doc.page_content for doc in documents])
            metadata = documents[0].metadata if documents else {}

            # Determine format type for chunking
            format_type = self._determine_format_type(source_type, full_text)

            # Use hybrid chunker to create semantic chunks
            chunks = await self.chunker.chunk_document(
                text=full_text,
                doc_type=doc_type,
                format_type=format_type,
                metadata=metadata,
                force_llm_validation=force_llm_validation
            )

            # Generate embeddings for chunks
            chunk_texts = [chunk["content"] for chunk in chunks]
            embeddings_list = await self.embeddings.aembed_documents(chunk_texts)

            # Add embeddings to chunks
            for i, chunk in enumerate(chunks):
                chunk["embedding"] = embeddings_list[i]
                logger.info(f"Size of embedding: {len(embeddings_list[i])}")

            logger.info(f"Processed document into {len(chunks)} semantic chunks using hybrid approach")

            return {
                "chunks": chunks,
                "document_metadata": {
                    "title": metadata.get("title", "Untitled Document"),
                    "source": metadata.get("source", file_path or file_url or "direct_input"),
                    "source_type": source_type,
                    "doc_type": doc_type,
                    "format_type": format_type,
                    "page_count": len(documents),
                    "chunk_count": len(chunks)
                }
            }
        except Exception as e:
            logger.error(f"Failed to process document: {e}")
            raise

    def _determine_format_type(self, source_type: str, text: str) -> str:
        """
        Determine the format type for chunking.

        Args:
            source_type: Original source type
            text: Document text

        Returns:
            Format type for chunker ('markdown', 'html', 'pdf', 'plain')
        """
        # Map source types to format types
        format_mapping = {
            'markdown': 'markdown',
            'html': 'html',
            'pdf': 'pdf',
            'text': 'plain',
            'url': 'auto'  # Auto-detect from content
        }

        format_type = format_mapping.get(source_type, 'auto')

        # Auto-detect if needed
        if format_type == 'auto':
            format_type = self.chunker._detect_format(text)

        return format_type

    async def _load_document(
        self,
        file_path: Optional[str],
        file_url: Optional[str],
        content: Optional[str],
        supabase_bucket_name: Optional[str],
        supabase_bucket_path: Optional[str],
        source_type: str,
        supabase_client: Optional[Any] = None
    ) -> List[Document]:
        """
        Load document from various sources.
        
        Priority order:
        1. Direct content
        2. Supabase Storage (recommended for production)
        3. Local file path
        4. External URL
        """
        if content:
            # Direct text content
            return [Document(page_content=content, metadata={"source": "direct_input"})]
        elif supabase_bucket_name and supabase_bucket_path and supabase_client:
            # Supabase Storage - RECOMMENDED for production
            try:
                logger.info(f"Downloading from Supabase Storage: {supabase_bucket_name}/{supabase_bucket_path}")
                # Download file from Supabase Storage
                file_data = supabase_client.storage.from_(supabase_bucket_name).download(supabase_bucket_path)
                
                # Save temporarily and load
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf" if source_type == "pdf" else ".txt") as tmp:
                    tmp.write(file_data)
                    tmp_path = tmp.name

                try:
                    loop = asyncio.get_event_loop()
                    if source_type == "pdf":
                        loader = PyPDFLoader(tmp_path)
                    else:
                        loader = TextLoader(tmp_path)
                    docs = await loop.run_in_executor(None, loader.load)
                    # Update metadata with Supabase path
                    for doc in docs:
                        doc.metadata["source"] = f"supabase://{supabase_bucket_name}/{supabase_bucket_path}"
                        doc.metadata["supabase_bucket"] = supabase_bucket_name
                        doc.metadata["supabase_path"] = supabase_bucket_path
                    return docs
                finally:
                    os.unlink(tmp_path)
            except Exception as e:
                logger.error(f"Failed to load from Supabase Storage: {e}")
                raise ValueError(f"Failed to load document from Supabase Storage: {e}")
        elif file_path:
            # Local file - run in executor since loaders are synchronous
            loop = asyncio.get_event_loop()
            if source_type == "pdf":
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path)
            return await loop.run_in_executor(None, loader.load)
        elif file_url:
            # URL - download first
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(file_url)
                response.raise_for_status()
                # Save temporarily and load
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf" if source_type == "pdf" else ".txt") as tmp:
                    tmp.write(response.content)
                    tmp_path = tmp.name

                try:
                    loop = asyncio.get_event_loop()
                    if source_type == "pdf":
                        loader = PyPDFLoader(tmp_path)
                    else:
                        loader = TextLoader(tmp_path)
                    return await loop.run_in_executor(None, loader.load)
                finally:
                    os.unlink(tmp_path)
        else:
            raise ValueError("Must provide content, supabase_bucket_name+supabase_bucket_path, file_path, or file_url")
