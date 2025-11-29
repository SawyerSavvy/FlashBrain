import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage

# Import the agent and state
from FlashBrain import FlashBrainAgent, BrainState, END

class TestAsyncOrchestration(unittest.IsolatedAsyncioTestCase):
    """
    Tests for the Async Job Orchestration features in FlashBrain.
    Covers:
    1. Async Job Creation (planning_agent_node)
    2. Guard Logic (orchestrator_node with pending job)
    3. Resume Logic (check_dependency_node)
    """

    def setUp(self):
        # Initialize agent with mocked dependencies
        self.agent = FlashBrainAgent()
        self.agent.supabase_client = MagicMock()
        self.agent.model_clients = {} # Mock model clients if needed
        
        # Mock logger to reduce noise
        self.agent.logger = MagicMock()

    async def test_planning_agent_async_job_creation(self):
        """Test that planning_agent_node creates a job and returns waiting state."""
        print("\n=== Testing Async Job Creation ===")
        
        # Setup state
        state = {
            "messages": [HumanMessage(content="Create a project plan for a mobile app")],
            "pending_tasks": [{"type": "PROJECT_DECOMP", "question": "Create plan", "priority": 1}],
            "current_task_idx": 0,
            "thread_id": "test-thread-123"
        }
        
        # Mock environment variables
        with patch("os.getenv") as mock_getenv:
            def getenv_side_effect(key, default=None):
                env_vars = {
                    "PROJECT_DECOMP_AGENT_URL": "http://localhost:8011",
                    "ORCHESTRATOR_CALLBACK_URL": "http://localhost:8010/callbacks/project-complete"
                }
                return env_vars.get(key, default)
            mock_getenv.side_effect = getenv_side_effect
            
            # Mock Supabase insert for brain_jobs
            mock_insert = MagicMock()
            mock_insert.insert.return_value.execute.return_value.data = [{"id": "job-123"}]
            mock_update = MagicMock()
            mock_update.update.return_value.eq.return_value.execute.return_value = None
            
            def table_side_effect(table_name):
                if table_name == "brain_jobs":
                    return mock_update if hasattr(self, '_job_created') else mock_insert
                return MagicMock()
            
            self.agent.supabase_client.table.side_effect = table_side_effect
            
            # Mock httpx.AsyncClient with proper async response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"project_id": "proj-456", "status": "started"}
            mock_response.raise_for_status = MagicMock()  # No-op
            
            # Create mock async client that returns the response
            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            
            # Mock the async context manager
            mock_client_class = MagicMock()
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)
            
            with patch("httpx.AsyncClient", mock_client_class):
                # Run the node
                self._job_created = False
                result = await self.agent.planning_agent_node(state)
                self._job_created = True
                
                # Verify job creation in brain_jobs table
                self.agent.supabase_client.table.assert_called_with("brain_jobs")
                
                # Verify HTTP call to external agent was made
                mock_client_instance.post.assert_called_once()
                call_args = mock_client_instance.post.call_args
                self.assertIn("callback_url", call_args.kwargs["json"])
                self.assertIn("job_id", call_args.kwargs["json"])
                
                # Verify returned state indicates waiting
                self.assertIn("waiting_for_project_id", result)
                self.assertIn("active_job_id", result)
                self.assertIsNotNone(result["active_job_id"])  # Should have a job_id (UUID)
                self.assertEqual(result["next_step"], END)
                print("✅ Async job creation verified")

    async def test_check_dependency_resume_completed(self):
        """Test that check_dependency_node resumes when job is COMPLETED."""
        print("\n=== Testing Resume Logic (Completed) ===")
        
        # Setup state (waiting for job)
        state = {
            "waiting_for_project_id": "proj-abc",
            "active_job_id": "job-123",
            "suspended_tasks": [{"type": "SELECT_FREELANCER", "question": "Find devs"}]
        }
        
        # Mock Supabase select returning COMPLETED status
        mock_select = MagicMock()
        mock_eq = MagicMock()
        mock_single = MagicMock()
        mock_execute = MagicMock()
        
        # Mock job data
        job_data = {
            "status": "COMPLETED", 
            "result": {"summary": "Plan created successfully"}
        }
        mock_execute.execute.return_value.data = job_data
        
        # Chain the mocks
        self.agent.supabase_client.table.return_value = mock_select
        mock_select.select.return_value = mock_eq
        mock_eq.eq.return_value = mock_single
        mock_single.single.return_value = mock_execute
        
        # Run the node
        result = self.agent.check_dependency_node(state)
        
        # Verify result
        self.assertEqual(result["next_step"], "task_executor")
        self.assertIsNotNone(result.get("pending_tasks")) # Should have restored suspended tasks
        self.assertIsNone(result.get("waiting_for_project_id")) # Should be cleared
        print("✅ Resume logic verified (Workflow resumed)")

    async def test_check_dependency_still_running(self):
        """Test that check_dependency_node waits when job is RUNNING."""
        print("\n=== Testing Resume Logic (Still Running) ===")
        
        state = {
            "waiting_for_project_id": "proj-abc",
            "active_job_id": "job-123"
        }
        
        # Mock Supabase returning RUNNING
        mock_execute = MagicMock()
        mock_execute.execute.return_value.data = {"status": "RUNNING"}
        
        # Setup mock chain
        self.agent.supabase_client.table.return_value.select.return_value.eq.return_value.single.return_value = mock_execute
        
        # Run node
        result = self.agent.check_dependency_node(state)
        
        # Verify result
        self.assertEqual(result["next_step"], END)
        self.assertNotIn("pending_tasks", result) # Should NOT resume yet
        print("✅ Resume logic verified (Still waiting)")

if __name__ == "__main__":
    unittest.main()
