"""
MCP RAG Service Entry Point

Starts an HTTP server exposing MCP tools for document processing and semantic search.
"""

import os
import logging
import sys
import json
from typing import Dict, Any
import click
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import ValidationError

from rag_service import RAGService, ProcessDocumentRequest, SearchKnowledgeBaseRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Service MCP Server",
    description="MCP server providing document processing and semantic search tools",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG service instance
rag_service: RAGService = None


@app.on_event("startup")
async def startup():
    """Initialize RAG service on startup."""
    global rag_service
    try:
        rag_service = RAGService()
        # Open PostgreSQL pool if it exists
        if rag_service.vector_store.pool:
            await rag_service.vector_store.pool.open()
        logger.info("RAG Service started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG Service: {e}")
        sys.exit(1)


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    global rag_service
    if rag_service:
        await rag_service.close()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "RAG Service MCP Server"}


@app.post("/mcp/tools/process_document")
async def mcp_process_document(request_data: Dict[str, Any]):
    """
    MCP Tool: process_document
    
    Intelligently decompose documents into semantic chunks using LangChain and store in vector database.
    
    NOTE: The document must already exist in Supabase. This endpoint only processes and stores chunks.
    
    ACCESS CONTROL: The client_id must match the document's client_id. Clients can only process their own documents.
    
    Request body:
    {
        "document_id": "required UUID of existing document in Supabase",
        "client_id": "required UUID - must match document's client_id for access control",
        "file_path": "optional/path/to/file.pdf",
        "file_url": "optional https://example.com/doc.pdf",
        "content": "optional direct text content",
        "supabase_bucket_name": "optional documents",
        "supabase_bucket_path": "optional project-123/doc.pdf",
        "source_type": "pdf|text|url|file"
    }
    
    Priority order for file source:
    1. content (direct text)
    2. supabase_bucket_name + supabase_bucket_path (RECOMMENDED for production)
    3. file_path (local file)
    4. file_url (external URL)
    """
    try:
        request = ProcessDocumentRequest(**request_data)
        result = await rag_service.process_document(request)
        return JSONResponse(content=result.model_dump())
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mcp/tools/search_knowledge_base")
async def mcp_search_knowledge_base(request_data: Dict[str, Any]):
    """
    MCP Tool: search_knowledge_base

    Perform semantic similarity search with BGE reranking. Uses a 2-stage pipeline:
    1. Vector similarity search (fast recall) - retrieves top candidates
    2. BGE reranker (precision ranking) - scores query-passage relevance

    The BGE reranker (bge-reranker-v2-m3) is a specialized model trained specifically
    for ranking tasks, providing more accurate relevance scores than cosine similarity alone.

    Request body:
    {
        "query": "search query text",
        "match_threshold": 0.7,
        "match_count": 10,
        "document_id": "optional UUID",
        "client_id": "optional client ID",
        "project_id": "optional project ID"
    }
    """
    try:
        request = SearchKnowledgeBaseRequest(**request_data)
        result = await rag_service.search_knowledge_base(request)
        return JSONResponse(content=result.model_dump())
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error searching knowledge base: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mcp/tools")
async def list_mcp_tools():
    """List available MCP tools."""
    return {
        "tools": [
            {
                "name": "process_document",
                "description": "Intelligently decompose documents into semantic chunks using LangChain and store in vector database. Document must already exist in Supabase. Client ID must match document's client_id for access control.",
                "parameters": {
                    "document_id": {"type": "string", "required": True, "description": "UUID of existing document in Supabase"},
                    "client_id": {"type": "string", "required": True, "description": "Client UUID - must match document's client_id for access control"},
                    "file_path": {"type": "string", "optional": True, "description": "Path to local file"},
                    "file_url": {"type": "string", "optional": True, "description": "URL to fetch file from"},
                    "content": {"type": "string", "optional": True, "description": "Direct text content"},
                    "supabase_bucket_name": {"type": "string", "optional": True, "description": "Supabase Storage bucket name (e.g., 'documents') - RECOMMENDED"},
                    "supabase_bucket_path": {"type": "string", "optional": True, "description": "Path to file in Supabase Storage bucket (e.g., 'project-123/doc.pdf')"},
                    "source_type": {"type": "string", "default": "pdf", "description": "Type: 'pdf', 'text', 'url', 'file'"}
                }
            },
            {
                "name": "search_knowledge_base",
                "description": "Semantic search with BGE reranking. 2-stage pipeline: (1) Vector similarity retrieves candidates, (2) BGE reranker (bge-reranker-v2-m3) scores query-passage relevance for precision ranking. Returns most contextually relevant chunks.",
                "parameters": {
                    "query": {"type": "string", "required": True, "description": "Search query"},
                    "match_threshold": {"type": "float", "default": 0.7, "description": "Minimum similarity threshold (0-1) for initial retrieval"},
                    "match_count": {"type": "integer", "default": 10, "description": "Maximum number of results (after BGE reranking)"},
                    "document_id": {"type": "string", "optional": True, "description": "Filter by document ID"},
                    "client_id": {"type": "string", "optional": True},
                    "project_id": {"type": "string", "optional": True}
                }
            }
        ]
    }


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=8012)
def main(host, port):
    """Start the RAG Service MCP server."""
    logger.info(f'Starting RAG Service MCP Server on {host}:{port}')
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()

