import logging

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

from project_decomp_agent import ProjectDecompAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProjectDecompAgentExecutor(AgentExecutor):
    """Project Decomposition AgentExecutor."""

    def __init__(self):
        self.agent = ProjectDecompAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        context config: 
            message: Message
                context_id: str
                message_id: str
                parts: List[Part] (Contains the user input, artifact, or system message)
                metadata: { 
                            'project_id': str,
                            'client_id': str,
                            'exist': bool (True if project exists, Flase if project does not exist)
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
            # Extract additional parameters from message metadata if available
            project_id = None
            client_id = None
            job_id = None
            exist = False

            if hasattr(context, 'message') and context.message:
                metadata = getattr(context.message, 'metadata', None)
                if metadata:
                    project_id = metadata.get('project_id')
                    client_id = metadata.get('client_id')
                    job_id = metadata.get('job_id')
                    exist = metadata.get('exist', False)

            # Stream responses from the agent
            async for item in self.agent.stream(
                query,
                context_id=task.context_id,
                project_id=project_id,
                client_id=client_id,
                job_id=job_id,
                exist=exist
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
                        name='decomposition_result',
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
