"""
GitHub Onboarding Agent Server

A2A server entry point for the GitHub Onboarding Agent.
Provides both A2A protocol and simple REST endpoints.
"""

import logging
import os
import sys
from typing import Optional

import click
import httpx
import uvicorn
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    ContentType,
)

from agent_executor import OnboardingAgentExecutor
from onboarding_agent import GitHubOnboardingAgent

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supported content types
SUPPORTED_CONTENT_TYPES = [ContentType.TEXT]

# Global agent instance for REST endpoints
_agent_instance: Optional[GitHubOnboardingAgent] = None


class AnalyzeRequest(BaseModel):
    """Request model for /analyze endpoint."""
    github_username: str = Field(..., description="GitHub username to analyze")
    client_id: Optional[str] = Field(None, description="Optional client ID for multi-tenancy")


class AnalyzeResponse(BaseModel):
    """Response model for /analyze endpoint."""
    status: str
    profile_id: Optional[str] = None
    github_username: str
    message: str
    analysis: Optional[str] = None


class VerifySkillsRequest(BaseModel):
    """Request model for /verify-skills endpoint."""
    freelancer_id: str = Field(..., description="Freelancer UUID to verify skills for")
    skill_id: Optional[str] = Field(None, description="Optional specific skill UUID to verify")
    max_skills: int = Field(20, description="Maximum number of skills to verify in one call")


async def analyze_endpoint(request: Request) -> JSONResponse:
    """
    Simple REST endpoint to analyze a GitHub user.

    POST /analyze
    {
        "github_username": "octocat",
        "client_id": "optional-client-id"
    }
    """
    global _agent_instance

    try:
        body = await request.json()
        github_username = body.get("github_username")
        client_id = body.get("client_id")

        if not github_username:
            return JSONResponse(
                {"status": "error", "message": "github_username is required"},
                status_code=400
            )

        if _agent_instance is None:
            _agent_instance = GitHubOnboardingAgent()

        logger.info(f"REST API: Analyzing @{github_username}")

        # Invoke the agent
        result = await _agent_instance.invoke(
            message=f"Analyze GitHub developer @{github_username}",
            thread_id=f"rest-{github_username}",
            client_id=client_id
        )

        # Extract final message
        messages = result.get("messages", [])
        final_message = ""
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "content"):
                final_message = last_msg.content

        return JSONResponse({
            "status": "success",
            "github_username": github_username,
            "message": f"Analysis complete for @{github_username}",
            "analysis": final_message
        })

    except Exception as e:
        logger.error(f"REST API error: {e}", exc_info=True)
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )


async def health_endpoint(request: Request) -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "service": "github-onboarding-agent",
        "version": "1.0.0"
    })


async def verify_skills_endpoint(request: Request) -> JSONResponse:
    """
    On-demand skill verification endpoint.

    POST /verify-skills
    {
        "freelancer_id": "uuid-here",
        "skill_id": "optional-skill-uuid",
        "max_skills": 20
    }

    This endpoint directly invokes the verify_skill_evidence tool
    to validate freelancer skills against GitHub evidence.
    """
    global _agent_instance

    try:
        body = await request.json()
        freelancer_id = body.get("freelancer_id")
        skill_id = body.get("skill_id")
        max_skills = body.get("max_skills", 20)

        if not freelancer_id:
            return JSONResponse(
                {"status": "error", "message": "freelancer_id is required"},
                status_code=400
            )

        if _agent_instance is None:
            _agent_instance = GitHubOnboardingAgent()

        logger.info(f"REST API: Verifying skills for freelancer {freelancer_id}")

        # Build the tool call message for the agent
        if skill_id:
            message = f"Verify skill {skill_id} for freelancer {freelancer_id}"
        else:
            message = f"Verify all skills needing verification for freelancer {freelancer_id} (max {max_skills} skills)"

        # Invoke the agent to run the verification
        result = await _agent_instance.invoke(
            message=message,
            thread_id=f"verify-skills-{freelancer_id}",
        )

        # Extract the tool result from messages
        messages = result.get("messages", [])
        verification_result = None

        # Look for tool message with verify_skill_evidence results
        for msg in reversed(messages):
            if hasattr(msg, 'content'):
                content = msg.content
                if isinstance(content, str) and '"skills_verified"' in content:
                    import json
                    try:
                        verification_result = json.loads(content)
                        break
                    except json.JSONDecodeError:
                        pass

        # If no tool result found, extract from AI response
        if not verification_result and messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "content"):
                verification_result = {"raw_response": last_msg.content}

        return JSONResponse({
            "status": "success",
            "freelancer_id": freelancer_id,
            "skill_id": skill_id,
            **verification_result if verification_result else {"message": "Verification completed"}
        })

    except Exception as e:
        logger.error(f"REST API verify-skills error: {e}", exc_info=True)
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )


# REST routes to add to the Starlette app
REST_ROUTES = [
    Route("/analyze", analyze_endpoint, methods=["POST"]),
    Route("/verify-skills", verify_skills_endpoint, methods=["POST"]),
    Route("/health", health_endpoint, methods=["GET"]),
]


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=8013)
def main(host, port):
    """Start the GitHub Onboarding Agent server."""
    try:
        # Check required environment variables
        required_vars = [
            'SUPABASE_URL',
            'SUPABASE_SERVICE_ROLE_KEY',
            'GITHUB_TOKEN',
            'GOOGLE_API_KEY'
        ]
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            logger.error(f'Missing required environment variables: {", ".join(missing_vars)}')
            logger.error('Please set these variables in your .env file or environment.')
            sys.exit(1)

        # Optional variables
        if not os.getenv('SUPABASE_POOLER'):
            logger.warning('SUPABASE_POOLER not set. Checkpointing will be disabled.')

        if not os.getenv('RAG_SERVICE_URL'):
            logger.warning('RAG_SERVICE_URL not set. README vectorization will be disabled.')

        # Define agent capabilities
        capabilities = AgentCapabilities(streaming=True, push_notifications=True)

        # Define agent skills
        onboarding_skill = AgentSkill(
            id='github_onboarding',
            name='GitHub Developer Onboarding',
            description='Analyzes GitHub profiles to extract skills, tech stack, commit patterns, and creates comprehensive developer profiles for freelancer matching.',
            tags=['github', 'onboarding', 'developer', 'analysis', 'skills'],
            examples=[
                'Analyze GitHub developer @octocat',
                'Onboard freelancer with GitHub username torvalds',
                'What are the skills of @kentcdodds?',
                'Create a profile for GitHub user @gaearon'
            ],
        )

        profile_analysis_skill = AgentSkill(
            id='developer_profile_analysis',
            name='Developer Profile Analysis',
            description='Calculates consistency scores, domain expertise, impact scores, and generates developer stories from stored GitHub data.',
            tags=['analysis', 'scoring', 'expertise', 'story'],
            examples=[
                'Calculate impact score for profile abc-123',
                'Generate developer story for profile xyz-456',
                'What is the domain expertise for this developer?'
            ],
        )

        # Use environment variable for public URL (Cloud Run), fallback to host:port for local
        public_url = os.getenv('ONBOARDING_PUBLIC_URL', f'http://{host}:{port}')

        agent_card = AgentCard(
            name='GitHub Onboarding Agent',
            description='A specialized agent for analyzing GitHub developer profiles and creating comprehensive skill assessments for freelancer matching. Extracts commit patterns, tech stack, domain expertise, and generates developer stories.',
            url=public_url,
            version='1.0.0',
            default_input_modes=SUPPORTED_CONTENT_TYPES,
            default_output_modes=SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[onboarding_skill, profile_analysis_skill],
        )

        # Create request handler
        httpx_client = httpx.AsyncClient()
        push_config_store = InMemoryPushNotificationConfigStore()
        push_sender = BasePushNotificationSender(
            httpx_client=httpx_client,
            config_store=push_config_store
        )

        request_handler = DefaultRequestHandler(
            agent_executor=OnboardingAgentExecutor(),
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender=push_sender
        )

        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler
        )

        # Build app and add REST routes
        starlette_app = server.build()
        starlette_app.routes.extend(REST_ROUTES)

        logger.info('=' * 60)
        logger.info('GitHub Onboarding Agent')
        logger.info('=' * 60)
        logger.info(f'Server: {host}:{port}')
        logger.info(f'Public URL: {public_url}')
        logger.info(f'Skills: {len(agent_card.skills)}')
        logger.info('')
        logger.info('REST Endpoints:')
        logger.info(f'  POST {public_url}/analyze       - Analyze GitHub user')
        logger.info(f'  POST {public_url}/verify-skills - Verify freelancer skills against GitHub evidence')
        logger.info(f'  GET  {public_url}/health        - Health check')
        logger.info('')
        logger.info('A2A Endpoints:')
        logger.info(f'  GET  {public_url}/.well-known/agent.json - Agent card')
        logger.info(f'  POST {public_url}/ - Send message to agent')
        logger.info('')
        logger.info('Ready to analyze GitHub developers!')
        logger.info('=' * 60)

        uvicorn.run(starlette_app, host=host, port=port)

    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}', exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
