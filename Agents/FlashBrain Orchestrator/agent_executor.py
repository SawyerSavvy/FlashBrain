"""
FlashBrain Agent Executor

This module provides the A2A AgentExecutor wrapper for FlashBrain agents.
Works with both the legacy FlashBrainAgent and the new FlashBrainReActAgent
since they share the same stream() interface.
"""

import logging
import os
from typing import Dict, Any, Union

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlashBrainAgentExecutor(AgentExecutor):
    """
    FlashBrain Orchestrator AgentExecutor.
    
    Works with any agent that implements the stream() interface:
        async def stream(query, context_id, client_id, project_id) -> AsyncGenerator
    """

    def __init__(self, agent=None):
        """
        Initialize the executor with an agent instance.
        
        Args:
            agent: FlashBrainAgent or FlashBrainReActAgent instance
        """
        if agent is None:
            # Default to ReAct agent if none provided
            agent_type = os.getenv("FLASHBRAIN_AGENT_TYPE", "react").lower()
            if agent_type == "legacy":
                from FlashBrain import FlashBrainAgent
                agent = FlashBrainAgent()
            else:
                from FlashBrain_ReAct import FlashBrainReActAgent
                agent = FlashBrainReActAgent()
        self.agent = agent

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        error = self._validate_request(context)
        if error:
            raise ServerError(error=InvalidParamsError())

        logger.info(f"Context: {context}")
        query = context.get_user_input()
        logger.info(f"Query: {query}")
        task = context.current_task
        logger.info(f"Task: {task}")
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            # Extract additional parameters from message metadata if available
            client_id = None
            project_id = None

            if hasattr(context, 'message') and context.message:
                metadata = getattr(context.message, 'metadata', None)
                if metadata:
                    client_id = metadata.get('client_id')
                    project_id = metadata.get('project_id')

            # Stream responses from the agent
            async for item in self.agent.stream(
                query,
                context_id=task.context_id,
                client_id=client_id,
                project_id=project_id
            ):
                is_task_complete = item.get('is_task_complete', False)
                require_user_input = item.get('require_user_input', False)
                content = item.get('content', '')

                if not is_task_complete and not require_user_input:
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(
                            content,
                            task.context_id,
                            task.id,
                        ),
                    )
                elif require_user_input:
                    await updater.update_status(
                        TaskState.input_required,
                        new_agent_text_message(
                            content,
                            task.context_id,
                            task.id,
                        ),
                        final=True,
                    )
                    break
                else:
                    await updater.add_artifact(
                        [Part(root=TextPart(text=content))],
                        name='flashbrain_response',
                    )
                    await updater.complete()
                    break

        except Exception as e:
            logger.error(f'An error occurred while streaming the response: {e}')
            raise ServerError(error=InternalError()) from e

    def _validate_request(self, context: RequestContext) -> bool:
        return False

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise ServerError(error=UnsupportedOperationError())
