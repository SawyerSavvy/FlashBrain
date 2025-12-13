"""
GitHub Onboarding Agent

A ReAct agent for analyzing GitHub profiles and onboarding freelance developers.
"""

import os
import logging
from typing import AsyncIterator, Dict, Any, Optional
from supabase import create_client
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_google_genai import ChatGoogleGenerativeAI

from github_client import GitHubClient
from extraction_tools import create_extraction_tools
from analysis_tools import create_analysis_tools
from prompts import ONBOARDING_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class GitHubOnboardingAgent:
    """
    A ReAct agent for analyzing GitHub developer profiles.

    Features:
    - Extracts GitHub data (profile, repos, commits)
    - Analyzes commit patterns and calculates scores
    - Generates developer stories
    - Stores results in Supabase
    """

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_service_role_key: Optional[str] = None,
        supabase_pooler: Optional[str] = None,
        github_token: Optional[str] = None,
        google_api_key: Optional[str] = None,
    ):
        """
        Initialize the GitHub Onboarding Agent.

        Args:
            supabase_url: Supabase project URL
            supabase_service_role_key: Supabase service role key
            supabase_pooler: PostgreSQL connection string
            github_token: GitHub personal access token
            google_api_key: Google AI API key
        """
        # Load environment variables
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_service_role_key = supabase_service_role_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.supabase_pooler = supabase_pooler or os.getenv("SUPABASE_POOLER")
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")

        # Validate required environment variables
        if not all([self.supabase_url, self.supabase_service_role_key, self.github_token, self.google_api_key]):
            raise ValueError(
                "Missing required environment variables. Please set: "
                "SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, GITHUB_TOKEN, GOOGLE_API_KEY"
            )

        # Initialize clients
        self.supabase_client = create_client(self.supabase_url, self.supabase_service_role_key)
        self.github_client = GitHubClient(token=self.github_token)

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=self.google_api_key,
        )

        # Create tools
        extraction_tools = create_extraction_tools(
            github_client=self.github_client,
            supabase_client=self.supabase_client
        )

        analysis_tools = create_analysis_tools(
            supabase_client=self.supabase_client,
            llm=self.llm
        )

        self.tools = extraction_tools + analysis_tools

        logger.info("GitHub Onboarding Agent initialized successfully")

    def _create_agent(self, checkpointer=None):
        """Create a ReAct agent with optional checkpointer."""
        return create_react_agent(
            model=self.llm,
            tools=self.tools,
            checkpointer=checkpointer,
            prompt=ONBOARDING_SYSTEM_PROMPT,
        )

    async def stream(
        self,
        message: str,
        thread_id: str = "default",
        client_id: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream agent responses for a given message.

        Args:
            message: User message to process
            thread_id: Thread identifier for conversation state
            client_id: Optional client ID for multi-tenancy

        Yields:
            Event dictionaries with agent updates
        """
        logger.info(f"Processing message: {message[:100]}...")

        # Prepare config
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        # Add client_id to message if provided
        if client_id:
            message = f"[Client ID: {client_id}]\n{message}"

        # Stream agent execution with fresh checkpointer per request
        try:
            if self.supabase_pooler:
                # Create fresh checkpointer for this request
                async with AsyncPostgresSaver.from_conn_string(
                    self.supabase_pooler
                ) as checkpointer:
                    await checkpointer.setup()
                    agent = self._create_agent(checkpointer=checkpointer)

                    async for event in agent.astream(
                        {"messages": [("user", message)]},
                        config=config,
                        stream_mode="values"
                    ):
                        yield event
            else:
                # No checkpointer - create agent without persistence
                agent = self._create_agent()
                async for event in agent.astream(
                    {"messages": [("user", message)]},
                    config=config,
                    stream_mode="values"
                ):
                    yield event

        except Exception as e:
            logger.error(f"Error during agent execution: {e}", exc_info=True)
            yield {
                "error": str(e),
                "message": f"❌ Agent execution failed: {str(e)}"
            }

    async def invoke(
        self,
        message: str,
        thread_id: str = "default",
        client_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Invoke agent synchronously (non-streaming).

        Args:
            message: User message to process
            thread_id: Thread identifier for conversation state
            client_id: Optional client ID for multi-tenancy

        Returns:
            Final agent state
        """
        logger.info(f"Invoking agent with message: {message[:100]}...")

        # Prepare config
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        # Add client_id to message if provided
        if client_id:
            message = f"[Client ID: {client_id}]\n{message}"

        try:
            if self.supabase_pooler:
                # Create fresh checkpointer for this request
                async with AsyncPostgresSaver.from_conn_string(
                    self.supabase_pooler
                ) as checkpointer:
                    await checkpointer.setup()
                    agent = self._create_agent(checkpointer=checkpointer)

                    result = await agent.ainvoke(
                        {"messages": [("user", message)]},
                        config=config
                    )
                    return result
            else:
                # No checkpointer - create agent without persistence
                agent = self._create_agent()
                result = await agent.ainvoke(
                    {"messages": [("user", message)]},
                    config=config
                )
                return result

        except Exception as e:
            logger.error(f"Error during agent invocation: {e}", exc_info=True)
            return {
                "error": str(e),
                "message": f"❌ Agent invocation failed: {str(e)}"
            }


# ============================================================
# Convenience Function
# ============================================================

def create_onboarding_agent(**kwargs) -> GitHubOnboardingAgent:
    """
    Factory function to create a GitHub Onboarding Agent.

    Args:
        **kwargs: Optional overrides for environment variables

    Returns:
        Initialized GitHubOnboardingAgent instance
    """
    return GitHubOnboardingAgent(**kwargs)