import os
from typing import List, Union, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, message_to_dict, messages_from_dict
from python_a2a import run_server, AgentCard, AgentSkill

import logging
logging.basicConfig(level=logging.DEBUG)

class SummarizationAgent:

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain', 'application/json']

    def __init__(self):
        # Initialize the model
        self.model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

    async def stream(self, query: str, context_id: str = "default", keep_last: int = 2):
        """
        Streams responses from the Summarization process for A2A protocol.

        Yields:
            Dict with keys:
                - is_task_complete: bool
                - require_user_input: bool
                - content: str
        """
        try:
            # Parse the query as messages if it's JSON
            import json
            messages = []

            try:
                parsed = json.loads(query)
                if isinstance(parsed, dict):
                    messages = parsed.get("messages", [])
                    keep_last = parsed.get("keep_last", keep_last)
                elif isinstance(parsed, list):
                    messages = parsed
            except:
                # If not JSON, treat query as a single message to summarize
                messages = [{"role": "user", "content": query}]

            # Yield initial status
            yield {
                'is_task_complete': False,
                'require_user_input': False,
                'content': f'Summarizing {len(messages)} message(s)...'
            }

            # Perform summarization
            result = self.summarize(messages, keep_last)
            summarized_messages = result.get("messages", [])

            # Build response
            summary_text = json.dumps(summarized_messages, indent=2)

            yield {
                'is_task_complete': True,
                'require_user_input': False,
                'content': summary_text
            }

        except Exception as e:
            yield {
                'is_task_complete': False,
                'require_user_input': True,
                'content': f"An error occurred during summarization: {str(e)}"
            }

    def summarize(
        self, 
        messages: List[Any], 
        keep_last: int = 2
    ) -> Dict[str, List[Any]]:
        """
        Summarizes the provided messages.
        
        Args:
            messages: List of message objects (dicts or BaseMessage).
            keep_last: Number of recent messages to preserve.
            
        Returns:
            Dict with 'messages' key containing the updated list.
        """
        # Handle JSON payload parsing if input is a string (A2A compat)
        if isinstance(messages, str):
            import json
            try:
                parsed = json.loads(messages)
                if isinstance(parsed, dict):
                    messages = parsed.get("messages", parsed.get("content", []))
                    keep_last = parsed.get("keep_last", keep_last)
                elif isinstance(parsed, list):
                    messages = parsed
            except:
                pass

        if not messages or not isinstance(messages, list):
            return {"messages": messages}

        # Convert dicts to BaseMessage objects if needed
        converted_messages = []
        for m in messages:
            if isinstance(m, dict):
                # Try langchain conversion first
                try:
                    if "type" in m:
                        converted_messages.extend(messages_from_dict([m]))
                    else:
                        # Manual fallback
                        role = m.get("role", "user")
                        content = m.get("content", "")
                        if role == "user":
                            converted_messages.append(HumanMessage(content=content))
                        elif role == "system":
                            converted_messages.append(SystemMessage(content=content))
                        elif role == "ai" or role == "assistant":
                            converted_messages.append(AIMessage(content=content))
                        else:
                            converted_messages.append(HumanMessage(content=content)) # Default
                except:
                    pass
            elif isinstance(m, BaseMessage):
                converted_messages.append(m)
        
        if not converted_messages:
            return {"messages": messages}

        # Determine split point
        if keep_last >= len(converted_messages):
            return {"messages": [message_to_dict(m) for m in converted_messages]}
        
        if keep_last <= 0:
            msgs_to_summarize = converted_messages
            kept_msgs = []
        else:
            msgs_to_summarize = converted_messages[:-keep_last]
            kept_msgs = converted_messages[-keep_last:]
            
        # Generate summary text
        conversation_text = "\n".join([f"{m.type}: {m.content}" for m in msgs_to_summarize])
        summary_text = self._summarize_text(conversation_text)
        
        # Create summary message
        summary_msg = SystemMessage(content=f"Previous Conversation Summary: {summary_text}")
        
        # Construct new list
        new_messages = [summary_msg] + kept_msgs
        
        # Return as dicts for A2A serialization
        return {"messages": [message_to_dict(m) for m in new_messages]}

    def _summarize_text(self, text: str) -> str:
        prompt = f"""
        Distill the following conversation into a concise summary, capturing key decisions, context, and user preferences.
        
        Conversation:
        {text}
        """
        try:
            response = self.model.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error summarizing text: {e}"

'''
Deploying to Google Run
'''
requirements = [
    "google-cloud-aiplatform[agent_engines,langchain]",
    # any other dependencies
]

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8013))
    print(f"Starting Summarization Agent on port {port}...")
    agent = SummarizationAgent()
    run_server(agent, port=port)
