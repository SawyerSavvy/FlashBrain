"""
Helper functions for calling other A2A agents via HTTP.
"""
import requests
from typing import Dict, Any, Optional
import logging

import httpx
from uuid import uuid4
from a2a.client import A2ACardResolver, ClientFactory, ClientConfig
from a2a.types import AgentCard, Message

import asyncio

logger = logging.getLogger(__name__)


def call_a2a_agent(
    agent_url: str,
    message_text: str,
    metadata: Optional[Dict[str, Any]] = None,
    context_id: str = "default",
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Calls an A2A agent via HTTP POST to /messages endpoint.

    Args:
        agent_url: Base URL of the A2A agent (e.g., "http://localhost:8011")
        message_text: The text message to send
        metadata: Optional metadata dict to include in the request
        context_id: Context/session ID for the conversation
        timeout: Request timeout in seconds

    Returns:
        Dict containing the response from the agent

    Raises:
        requests.RequestException: If the HTTP request fails
    """
    # Build A2A message payload
    payload = {
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": message_text}]
        },
        "context_id": context_id
    }

    # Add metadata if provided
    if metadata:
        payload["metadata"] = metadata

    # Send POST request to A2A server
    response = requests.post(
        f"{agent_url.rstrip('/')}/messages",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=timeout
    )

    response.raise_for_status()
    return response.json()


def extract_a2a_response_text(result: Dict[str, Any]) -> str:
    """
    Extracts the text content from an A2A response.

    Args:
        result: The JSON response from an A2A agent

    Returns:
        The extracted text content
    """
    # Check for artifact (task completed)
    if "artifact" in result:
        artifacts = result.get("artifact", [])
        if artifacts and len(artifacts) > 0:
            content = artifacts[0].get("content", {})
            if isinstance(content, dict):
                return content.get("text", "")
            return str(content)
        return "Task completed successfully."

    # Check for message (task in progress or needs input)
    if "message" in result:
        msg_content = result["message"].get("content", [])
        if msg_content and len(msg_content) > 0:
            return msg_content[0].get("text", "Response received")
        return "Task in progress."

    # Check for task status
    if "task" in result:
        task = result["task"]
        state = task.get("state", "unknown")
        if state == "completed":
            return "Task completed."
        elif state == "failed":
            return f"Task failed: {task.get('error', 'Unknown error')}"
        elif state == "input_required":
            return "Waiting for additional input."
        else:
            return f"Task status: {state}"

    # Fallback
    return f"Response: {str(result)}"


async def call_project_decomp_agent(
    agent_url: str,
    message: str,
    project_id: Optional[str] = None,
    client_id: Optional[str] = None,
    job_id: Optional[str] = None,
    exist: bool = False,
    context_id: str = "default"
):
    """
    Calls Project Decomposition Agent via A2A and yields streaming updates.

    Args:
        agent_url: URL of the Project Decomp agent
        message: User message
        project_id: Project ID
        client_id: Client ID
        job_id: Job ID from orchestrator (for async tracking)
        exist: Whether the project already exists
        context_id: Context ID

    Yields:
        String updates as they arrive from the A2A agent
    """
    try:
        async with httpx.AsyncClient(timeout=60.0) as httpx_client:
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url=agent_url,
            )
            
            try:
                agent_card = await resolver.get_agent_card()
            except Exception:
                resp = await httpx_client.get(f"{agent_url}/a2a/agent.json")
                resp.raise_for_status()
                agent_card = AgentCard.model_validate(resp.json())
            
            client_config = ClientConfig(httpx_client=httpx_client)
            client = await ClientFactory.connect(
                agent=agent_card,
                client_config=client_config
            )
            
            message_data = {
                'role': 'user',
                'parts': [{'kind': 'text', 'text': message}],
                'message_id': uuid4().hex,
                'metadata': {
                    'project_id': project_id,
                    'client_id': client_id,
                    'job_id': job_id,
                    'exist': exist
                }
            }
            a2a_message = Message(**message_data)
            
            async for item in client.send_message(a2a_message):
                text_content = None
                if isinstance(item, tuple):
                    task, update = item
                    if hasattr(update, 'status') and update.status:
                        if hasattr(update.status, 'message') and update.status.message:
                            msg = update.status.message
                            if hasattr(msg, 'parts') and msg.parts:
                                for part in msg.parts:
                                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                        text_content = part.root.text
                else:
                    if hasattr(item, 'parts') and item.parts:
                        for part in item.parts:
                            if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                text_content = part.root.text
                
                # Yield each update as it arrives
                if text_content:
                    yield text_content
    except Exception as e:
        logger.error(f"Project Decomp Agent call failed: {e}")
        yield f"Failed to contact Project Decomposition Agent: {e}"


async def call_freelancer_agent(
    agent_url: str,
    message: str,
    project_id: Optional[str] = None,
    client_id: Optional[str] = None,
    context_id: str = "default"
):
    """
    Calls Select Freelancer Agent via A2A and yields streaming updates.

    Args:
        agent_url: URL of the Freelancer agent
        message: User message
        project_id: Project ID
        client_id: Client ID
        context_id: Context ID

    Yields:
        String updates as they arrive from the A2A agent
    """
    try:
        async with httpx.AsyncClient(timeout=60.0) as httpx_client:
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url=agent_url,
            )
            
            try:
                agent_card = await resolver.get_agent_card()
            except Exception:
                resp = await httpx_client.get(f"{agent_url}/a2a/agent.json")
                resp.raise_for_status()
                agent_card = AgentCard.model_validate(resp.json())
            
            client_config = ClientConfig(httpx_client=httpx_client)
            client = await ClientFactory.connect(
                agent=agent_card,
                client_config=client_config
            )
            
            message_data = {
                'role': 'user',
                'parts': [{'kind': 'text', 'text': message}],
                'message_id': uuid4().hex,
                'metadata': {
                    'project_id': project_id,
                    'client_id': client_id
                }
            }
            a2a_message = Message(**message_data)
            
            async for item in client.send_message(a2a_message):
                text_content = None
                if isinstance(item, tuple):
                    task, update = item
                    if hasattr(update, 'status') and update.status:
                        if hasattr(update.status, 'message') and update.status.message:
                            msg = update.status.message
                            if hasattr(msg, 'parts') and msg.parts:
                                for part in msg.parts:
                                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                        text_content = part.root.text
                else:
                    if hasattr(item, 'parts') and item.parts:
                        for part in item.parts:
                            if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                text_content = part.root.text
                
                # Yield each update as it arrives
                if text_content:
                    yield text_content
    except Exception as e:
        logger.error(f"Freelancer Agent call failed: {e}", exc_info=True)
        yield f"Failed to contact Freelancer Selection Agent: {e}"


def call_summarization_agent(
    agent_url: str,
    messages: list,
    keep_last: int = 2,
    context_id: str = "default"
) -> Optional[Dict[str, Any]]:
    """
    Convenience function to call the Summarization Agent.

    Args:
        agent_url: URL of the Summarization agent
        messages: List of messages to summarize
        keep_last: Number of recent messages to keep
        context_id: Context ID

    Returns:
        Dict with summarized messages or None if failed
    """
    try:
        import json

        # Serialize messages
        message_text = json.dumps({
            "messages": messages,
            "keep_last": keep_last
        })

        metadata = {"keep_last": keep_last}

        result = call_a2a_agent(agent_url, message_text, metadata, context_id, timeout=30)
        return result

    except requests.RequestException as e:
        logger.error(f"Summarization Agent call failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Error calling Summarization Agent: {e}")
        return None
