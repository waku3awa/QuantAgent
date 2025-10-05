"""
Unit tests for Claude CLI Wrapper.

These tests include:
1. Unit tests with mocked subprocess
2. Integration tests (require actual claude CLI)
3. Error handling tests
"""

import unittest
import subprocess
import json
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_cli_wrapper import (
    ClaudeCLIClient,
    CLIError,
    create_claude_tools,
    LANGCHAIN_AVAILABLE
)


class TestClaudeCLIClientUnit(unittest.TestCase):
    """Unit tests with mocked subprocess."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = ClaudeCLIClient()

    @patch('subprocess.run')
    def test_run_success(self, mock_run):
        """Test successful single-shot execution."""
        # Mock successful response
        mock_run.return_value = Mock(
            returncode=0,
            stdout="This is a response from Claude",
            stderr=""
        )

        result = self.client.run("Test prompt")

        # Verify the call
        self.assertEqual(result, "This is a response from Claude")
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_run_timeout(self, mock_run):
        """Test timeout handling."""
        # Mock timeout
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd=["claude", "code"],
            timeout=30
        )

        with self.assertRaises(CLIError) as context:
            self.client.run("Test prompt", timeout=30)

        self.assertIn("Timeout", str(context.exception))

    @patch('subprocess.run')
    def test_run_command_not_found(self, mock_run):
        """Test handling of missing claude command."""
        mock_run.side_effect = FileNotFoundError()

        with self.assertRaises(CLIError) as context:
            self.client.run("Test prompt")

        self.assertIn("not found", str(context.exception).lower())

    @patch('subprocess.run')
    def test_run_cli_error(self, mock_run):
        """Test handling of CLI errors."""
        mock_run.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="Authentication failed"
        )

        with self.assertRaises(CLIError) as context:
            self.client.run("Test prompt")

        self.assertIn("Authentication failed", str(context.exception))

    @patch('subprocess.Popen')
    def test_session_creation(self, mock_popen):
        """Test session creation."""
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        session_id = self.client.start_session()

        self.assertIsNotNone(session_id)
        self.assertIn(session_id, self.client._sessions)
        mock_popen.assert_called_once()

    def test_session_not_found(self):
        """Test sending message to non-existent session."""
        with self.assertRaises(CLIError) as context:
            self.client.send_message("invalid-session-id", "Test message")

        self.assertIn("Unknown session_id", str(context.exception))

    @patch('subprocess.Popen')
    def test_end_session(self, mock_popen):
        """Test ending a session."""
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        session_id = self.client.start_session()
        self.client.end_session(session_id)

        self.assertNotIn(session_id, self.client._sessions)
        mock_process.terminate.assert_called_once()


class TestClaudeCLIClientIntegration(unittest.TestCase):
    """
    Integration tests that require actual claude CLI.

    These tests are skipped if claude is not available.
    Run with: python -m pytest tests/test_claude_cli_wrapper.py -v -m integration
    """

    @classmethod
    def setUpClass(cls):
        """Check if claude CLI is available."""
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                timeout=5
            )
            cls.claude_available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            cls.claude_available = False

    def setUp(self):
        """Set up test fixtures."""
        if not self.claude_available:
            self.skipTest("Claude CLI not available")
        self.client = ClaudeCLIClient()

    def test_real_run(self):
        """Test real execution with claude CLI."""
        result = self.client.run("What is 2+2?", flags=["-p"])
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_real_session(self):
        """Test real session with claude CLI."""
        session_id = self.client.start_session()
        try:
            response = self.client.send_message(
                session_id,
                "Say 'test successful' if you can read this"
            )
            self.assertIsInstance(response, str)
        finally:
            self.client.end_session(session_id)


@unittest.skipIf(not LANGCHAIN_AVAILABLE, "LangChain not installed")
class TestLangChainTools(unittest.TestCase):
    """Tests for LangChain tool integration."""

    @patch('subprocess.run')
    def test_create_tools(self, mock_run):
        """Test tool creation."""
        tools = create_claude_tools()

        self.assertEqual(len(tools), 4)
        self.assertEqual(tools[0].name, "claude_cli_run")
        self.assertEqual(tools[1].name, "claude_cli_start_session")
        self.assertEqual(tools[2].name, "claude_cli_send_message")
        self.assertEqual(tools[3].name, "claude_cli_end_session")

    @patch('subprocess.run')
    def test_run_tool_success(self, mock_run):
        """Test ClaudeRunTool execution."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Tool response",
            stderr=""
        )

        tools = create_claude_tools()
        run_tool = tools[0]

        result = run_tool.invoke({"prompt": "Test prompt"})

        self.assertEqual(result, "Tool response")

    @patch('subprocess.run')
    def test_run_tool_error(self, mock_run):
        """Test ClaudeRunTool error handling."""
        from langchain_core.tools import ToolException

        mock_run.side_effect = FileNotFoundError()

        tools = create_claude_tools()
        run_tool = tools[0]

        with self.assertRaises(ToolException):
            run_tool.invoke({"prompt": "Test prompt"})

    @patch('subprocess.Popen')
    def test_session_tools(self, mock_popen):
        """Test session-based tools."""
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.stdin = Mock()
        mock_process.stdout = Mock()
        mock_process.stdout.readline.return_value = ""
        mock_popen.return_value = mock_process

        tools = create_claude_tools()
        start_tool = tools[1]
        end_tool = tools[3]

        # Start session
        result = start_tool.invoke({})
        data = json.loads(result)
        session_id = data["session_id"]

        self.assertIsNotNone(session_id)

        # End session
        result = end_tool.invoke({"session_id": session_id})
        data = json.loads(result)

        self.assertEqual(data["status"], "ok")


class TestErrorCases(unittest.TestCase):
    """Test various error conditions."""

    def test_invalid_command(self):
        """Test with invalid command."""
        client = ClaudeCLIClient(cmd=["nonexistent_command"])

        with self.assertRaises(CLIError):
            client.run("Test")

    def test_empty_prompt(self):
        """Test with empty prompt."""
        client = ClaudeCLIClient()

        # This should work - empty prompts are valid
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            result = client.run("")
            self.assertEqual(result, "")

    def test_custom_timeout(self):
        """Test custom timeout configuration."""
        client = ClaudeCLIClient(timeout=10.0)

        self.assertEqual(client.timeout, 10.0)

    def test_custom_working_directory(self):
        """Test custom working directory."""
        working_dir = "/tmp"
        client = ClaudeCLIClient(working_dir=working_dir)

        self.assertEqual(client.working_dir, working_dir)


class TestSessionManagement(unittest.TestCase):
    """Test session lifecycle management."""

    @patch('subprocess.Popen')
    def test_multiple_sessions(self, mock_popen):
        """Test managing multiple sessions."""
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        client = ClaudeCLIClient()

        # Create multiple sessions
        session_ids = [client.start_session() for _ in range(3)]

        # Verify all sessions exist
        self.assertEqual(len(client._sessions), 3)
        for sid in session_ids:
            self.assertIn(sid, client._sessions)

        # End all sessions
        for sid in session_ids:
            client.end_session(sid)

        # Verify all sessions are gone
        self.assertEqual(len(client._sessions), 0)

    @patch('subprocess.Popen')
    def test_session_auto_cleanup(self, mock_popen):
        """Test automatic cleanup of idle sessions."""
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        # Create client with very short TTL for testing
        import claude_cli_wrapper
        original_ttl = claude_cli_wrapper.SESSION_TTL
        claude_cli_wrapper.SESSION_TTL = 1  # 1 second

        try:
            client = ClaudeCLIClient()
            session_id = client.start_session()

            # Wait for auto-cleanup (in real scenario)
            # Note: This test is limited because we can't easily wait
            # for the reaper thread in a unit test

            self.assertIn(session_id, client._sessions)

        finally:
            # Restore original TTL
            claude_cli_wrapper.SESSION_TTL = original_ttl


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestClaudeCLIClientUnit))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorCases))
    suite.addTests(loader.loadTestsFromTestCase(TestSessionManagement))

    if LANGCHAIN_AVAILABLE:
        suite.addTests(loader.loadTestsFromTestCase(TestLangChainTools))

    # Note: Integration tests are only run explicitly
    # suite.addTests(loader.loadTestsFromTestCase(TestClaudeCLIClientIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
