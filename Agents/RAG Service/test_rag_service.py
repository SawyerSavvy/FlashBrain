import asyncio
import os
import logging
import json
from dotenv import load_dotenv

import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """
    Integration test via HTTP APIs (no direct class calls):
    1. Process a sample document through the RAG service API.
    2. Query the knowledge base through the search API.
    """
    load_dotenv()

    # Config from env or defaults
    base_url = os.getenv("RAG_API_BASE_URL", "http://localhost:8012")
    process_endpoint = f"{base_url.rstrip('/')}/mcp/tools/process_document"
    search_endpoint = f"{base_url.rstrip('/')}/mcp/tools/search_knowledge_base"

    TITLE = "Latent Collaboration In Multi-Agent Systems"
    CLIENT_ID = "9a76be62-0d44-4a34-913d-08dcac008de5"
    SUPABASE_BUCKET = "test"
    SUPABASE_PATH = "Latent Collaboration In Multi-Agent Systems.pdf"

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            # 1) Process document
            logger.info("\n--- TEST: Process Document (HTTP) ---")
            payload = {
                "title": TITLE,
                "client_id": CLIENT_ID,
                "supabase_bucket_name": SUPABASE_BUCKET,
                "supabase_bucket_path": SUPABASE_PATH,
                "source_type": "pdf",
                "doc_type": "technical"
            }
            resp = await client.post(process_endpoint, json=payload)
            resp.raise_for_status()
            process_response = resp.json()
            logger.info(f"Document processed successfully! {json.dumps(process_response, indent=2)}")

            document_id = process_response.get("document_id")
            if not document_id:
                raise ValueError("No document_id returned from process_document API")

            # 2) Search knowledge base
            logger.info("\n--- TEST: Search Knowledge Base (HTTP) ---")
            queries = [
                "What is Latent Collaboration In Multi-Agent Systems?",
                "How can Latent Collaboration affect agents performance?",
                "What are some recent papers on Multi-Agent System communications?"
            ]

            for query in queries:
                logger.info(f"\nQuery: '{query}'")
                search_payload = {
                    "query": query,
                    "match_count": 3,
                    "match_threshold": 0.5,
                    "client_id": CLIENT_ID,
                    "document_id": document_id
                }
                s_resp = await client.post(search_endpoint, json=search_payload)
                s_resp.raise_for_status()
                search_response = s_resp.json()
                logger.info(f"Search response: {json.dumps(search_response, indent=2)}")

        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
