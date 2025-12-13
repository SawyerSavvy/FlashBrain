"""
GitHub Onboarding Agent Executor

A2A protocol wrapper for the GitHub Onboarding Agent.
"""

import logging
from typing import Optional

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import (
    new_agent_text_message,
    new_task,
)
from a2a.utils.errors import ServerError

from onboarding_agent import GitHubOnboardingAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OnboardingAgentExecutor(AgentExecutor):
    """GitHub Onboarding AgentExecutor for A2A protocol."""

    def __init__(self):
        """Initialize the executor with the onboarding agent."""
        self.agent = GitHubOnboardingAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        Execute the onboarding agent with A2A protocol.

        Context config:
            message: Message
                context_id: str
                message_id: str
                parts: List[Part] (Contains user input, artifact, or system message)
                metadata: {
                    'github_username': str (optional - GitHub username to analyze)
                    'profile_id': str (optional - Existing profile ID to analyze)
                    'client_id': str (optional - Client ID for multi-tenancy)
                }
        """
        error = self._validate_request(context)
        if error:
            raise ServerError(error=InvalidParamsError())

        # Extract user input from message parts
        query = context.get_user_input()

        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            # Extract metadata
            github_username: Optional[str] = None
            profile_id: Optional[str] = None
            client_id: Optional[str] = None

            if hasattr(context, 'message') and context.message:
                metadata = getattr(context.message, 'metadata', None)
                if metadata:
                    github_username = metadata.get('github_username')
                    profile_id = metadata.get('profile_id')
                    client_id = metadata.get('client_id')

            logger.info(
                f"Executing onboarding agent: query='{query[:50]}...', "
                f"github_username={github_username}, profile_id={profile_id}, "
                f"client_id={client_id}"
            )

            # Prepare enhanced query with metadata
            enhanced_query = query
            if github_username:
                enhanced_query = f"[GitHub: @{github_username}] {query}"
            if profile_id:
                enhanced_query = f"[Profile ID: {profile_id}] {query}"

            # Stream responses from the agent
            async for event in self.agent.stream(
                message=enhanced_query,
                thread_id=task.context_id,
                client_id=client_id,
            ):
                # Extract messages from event
                messages = event.get('messages', [])
                if not messages:
                    continue

                # Get the last message (most recent)
                last_message = messages[-1]

                # Check if it's an AI message
                if hasattr(last_message, 'type') and last_message.type == 'ai':
                    content = last_message.content

                    # Check if this is a tool call
                    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                        # Agent is calling a tool - send working status
                        tool_names = [tc['name'] for tc in last_message.tool_calls]
                        status_msg = f"🔧 Calling tools: {', '.join(tool_names)}"

                        await updater.update_status(
                            TaskState.working,
                            new_agent_text_message(
                                status_msg,
                                task.context_id,
                                task.id,
                            ),
                        )
                    else:
                        # Final response from agent
                        await updater.add_artifact(
                            [Part(root=TextPart(text=content))],
                            name='onboarding_analysis',
                        )
                        await updater.complete()
                        break

                # Check for tool messages (tool responses)
                elif hasattr(last_message, 'type') and last_message.type == 'tool':
                    # Tool has returned - optionally show progress
                    tool_name = getattr(last_message, 'name', 'unknown')
                    logger.info(f"Tool '{tool_name}' completed")

        except Exception as e:
            logger.error(f'An error occurred while streaming the response: {e}', exc_info=True)
            raise ServerError(error=InternalError()) from e

    def _validate_request(self, context: RequestContext) -> bool:
        """
        Validate the incoming request.

        Args:
            context: Request context

        Returns:
            True if invalid, False if valid
        """
        # No strict validation - agent handles all cases
        return False

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Cancel operation (not supported)."""
        raise ServerError(error=UnsupportedOperationError())
