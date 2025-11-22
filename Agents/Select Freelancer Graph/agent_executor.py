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

from select_freelancer_agent import SelectFreelancerAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SelectFreelancerAgentExecutor(AgentExecutor):
    """Select Freelancer AgentExecutor."""

    def __init__(self):
        self.agent = SelectFreelancerAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        error = self._validate_request(context)
        if error:
            raise ServerError(error=InvalidParamsError())

        query = context.get_user_input()
        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            # Extract project_id from message metadata if available
            project_id = None

            if hasattr(context, 'message') and context.message:
                metadata = getattr(context.message, 'metadata', None)
                if metadata:
                    project_id = metadata.get('project_id')

            # Stream responses from the agent
            async for item in self.agent.stream(
                query,
                context_id=task.context_id,
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
                        name='freelancer_selection_result',
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
