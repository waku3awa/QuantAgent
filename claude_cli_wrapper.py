"""
Claude CLI Wrapper for LangChain Integration

This module provides a wrapper for the claude.ai/code CLI tool that can be used
as a LangChain Tool. It supports both single-shot execution and interactive sessions.

Architecture:
- ClaudeCLIClient: Core wrapper handling subprocess management and I/O
- ClaudeRunTool: LangChain Tool for single-shot execution
- ClaudeStartSessionTool: LangChain Tool for starting sessions
- ClaudeSendMessageTool: LangChain Tool for sending messages in sessions
- ClaudeEndSessionTool: LangChain Tool for ending sessions
"""

import os
import sys
import time
import uuid
import json
import threading
import subprocess
import shutil
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from pathlib import Path

try:
    import pexpect
    HAS_PEXPECT = True
except ImportError:
    HAS_PEXPECT = False

# LangChain imports
try:
    from langchain_core.tools import BaseTool, ToolException
    from langchain_core.callbacks import CallbackManagerForToolRun
    from pydantic import BaseModel, Field
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseTool = object
    BaseModel = object


# Constants
def _find_claude_command():
    """Find the claude command in PATH."""
    claude_path = shutil.which('claude')
    if claude_path:
        return [claude_path]
    # Fallback to just 'claude' if not found (will fail later with clear error)
    return ["claude"]

DEFAULT_CMD = _find_claude_command()
DEFAULT_TIMEOUT = 60.0  # seconds
SESSION_TTL = 15 * 60  # 15 minutes of inactivity before auto-kill


class CLIError(Exception):
    """Exception raised for errors in the Claude CLI execution."""
    pass


@dataclass
class Session:
    """Represents an active Claude CLI session."""
    session_id: str
    proc: Any  # subprocess.Popen or pexpect.spawn
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)
    working_dir: Optional[str] = None


class ClaudeCLIClient:
    """
    Core client for interacting with claude code CLI.

    Supports both single-shot execution and interactive sessions.
    Thread-safe session management with automatic cleanup of idle sessions.
    """

    def __init__(
        self,
        cmd: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: float = DEFAULT_TIMEOUT,
        working_dir: Optional[str] = None,
        expect_prompt: Optional[str] = None,
        end_marker: Optional[str] = None,
    ):
        """
        Initialize the Claude CLI client.

        Args:
            cmd: Command to execute (default: auto-detected claude path)
            env: Environment variables (merged with os.environ)
            timeout: Default timeout for operations in seconds
            working_dir: Working directory for CLI execution
            expect_prompt: Regex pattern for REPL prompt (for pexpect)
            end_marker: Sentinel marker to detect end of output
        """
        self.cmd = cmd or DEFAULT_CMD.copy()
        self.env = {**os.environ, **(env or {})}
        self.timeout = timeout
        self.working_dir = working_dir
        self.expect_prompt = expect_prompt
        self.end_marker = end_marker
        self._sessions: Dict[str, Session] = {}
        self._reaper_thread = threading.Thread(target=self._reap_idle, daemon=True)
        self._reaper_thread.start()

    def run(self, prompt: str, timeout: Optional[float] = None, flags: Optional[List[str]] = None) -> str:
        """
        Execute a single-shot Claude CLI command.

        Args:
            prompt: The prompt to send to Claude
            timeout: Optional timeout override
            flags: Additional CLI flags (e.g., ["-p"])

        Returns:
            The output from Claude

        Raises:
            CLIError: If execution fails or times out
        """
        to = timeout or self.timeout
        cmd = self.cmd.copy()

        # Add flags if provided
        if flags:
            cmd.extend(flags)

        # Use -p flag for prompt mode if not already in flags
        if "-p" not in cmd and "--prompt" not in cmd:
            cmd.extend(["-p", prompt])
        else:
            cmd.append(prompt)

        try:
            proc = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                timeout=to,
                cwd=self.working_dir,
                env=self.env,
                encoding='utf-8',
                errors='replace',  # Replace invalid characters instead of failing
            )
        except subprocess.TimeoutExpired as e:
            raise CLIError(f"Timeout after {to}s") from e
        except FileNotFoundError as e:
            raise CLIError(f"Claude CLI not found. Please ensure 'claude' is installed and in PATH.") from e
        except Exception as e:
            raise CLIError(f"Failed to execute {self.cmd}: {e}") from e

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            raise CLIError(f"CLI error (code {proc.returncode}): {stderr}")

        return (proc.stdout or "").strip()

    def start_session(self, working_dir: Optional[str] = None, resume_session_id: Optional[str] = None) -> str:
        """
        Start a Claude CLI session.

        Note: This creates a logical session that uses --resume functionality
        to maintain conversation context across multiple run() calls.

        Args:
            working_dir: Optional working directory for this session
            resume_session_id: Optional Claude session ID to resume

        Returns:
            Session ID for future operations

        Raises:
            CLIError: If session creation fails
        """
        sid = str(uuid.uuid4())
        wd = working_dir or self.working_dir

        # Create a session object that stores context for multi-turn conversation
        session = Session(
            session_id=sid,
            proc={"resume_id": resume_session_id, "type": "logical"},  # Logical session, not a subprocess
            working_dir=wd
        )

        self._sessions[sid] = session
        return sid

    def send_message(
        self,
        session_id: str,
        message: str,
        timeout: Optional[float] = None
    ) -> str:
        """
        Send a message to an active session.

        This uses Claude CLI's --resume functionality to maintain context.
        The first message in a session gets a Claude session ID that is
        then reused for subsequent messages.

        Args:
            session_id: The session ID from start_session()
            message: The message to send
            timeout: Optional timeout override

        Returns:
            The response from Claude

        Raises:
            CLIError: If session doesn't exist or communication fails
        """
        to = timeout or self.timeout
        session = self._sessions.get(session_id)
        if not session:
            raise CLIError(f"Unknown session_id: {session_id}")

        with session.lock:
            session.last_used_at = time.time()

            # Build command with session resume if available
            cmd = self.cmd.copy()
            cmd.extend(["-p", "--output-format", "json"])

            proc_info = session.proc
            if isinstance(proc_info, dict) and proc_info.get("type") == "logical":
                # Check if we have a resume ID from previous message
                resume_id = proc_info.get("resume_id")
                if resume_id:
                    cmd.extend(["--resume", resume_id])

            cmd.append(message)

            try:
                result = subprocess.run(
                    cmd,
                    text=True,
                    capture_output=True,
                    timeout=to,
                    cwd=session.working_dir,
                    env=self.env,
                    encoding='utf-8',
                    errors='replace',
                )

                if result.returncode != 0:
                    stderr = (result.stderr or "").strip()
                    raise CLIError(f"CLI error (code {result.returncode}): {stderr}")

                # Parse JSON response to extract session ID and content
                try:
                    response_data = json.loads(result.stdout)
                    # Update session with Claude's session ID for resume
                    if "sessionId" in response_data:
                        proc_info["resume_id"] = response_data["sessionId"]

                    # Extract the actual response text
                    if "response" in response_data:
                        return response_data["response"]
                    elif "content" in response_data:
                        return response_data["content"]
                    else:
                        # Fallback: return the whole JSON as string
                        return result.stdout.strip()

                except json.JSONDecodeError:
                    # If not JSON, return as-is
                    return result.stdout.strip()

            except subprocess.TimeoutExpired as e:
                raise CLIError(f"Session {session_id} timeout after {to}s") from e
            except Exception as e:
                raise CLIError(f"Session {session_id} send failed: {e}") from e

    def end_session(self, session_id: str) -> None:
        """
        End an active session.

        Args:
            session_id: The session ID to terminate
        """
        self._sessions.pop(session_id, None)
        # No process termination needed for logical sessions

    def _terminate(self, session: Session) -> None:
        """Terminate a session (no-op for logical sessions)."""
        pass  # Logical sessions don't have subprocesses to terminate

    def _reap_idle(self) -> None:
        """Background thread to clean up idle sessions."""
        while True:
            try:
                now = time.time()
                to_end = [
                    sid for sid, s in self._sessions.items()
                    if now - s.last_used_at > SESSION_TTL
                ]
                for sid in to_end:
                    sess = self._sessions.pop(sid, None)
                    if sess:
                        self._terminate(sess)
            except Exception:
                pass
            time.sleep(10)

    @staticmethod
    def _supports_tty() -> bool:
        """Check if the platform supports TTY (for pexpect)."""
        return sys.platform != "win32"

    def __del__(self):
        """Cleanup all sessions on deletion."""
        for session in list(self._sessions.values()):
            self._terminate(session)


# LangChain Tool Implementations
if LANGCHAIN_AVAILABLE:

    class RunInput(BaseModel):
        """Input schema for ClaudeRunTool."""
        prompt: str = Field(..., description="Prompt to send to Claude CLI.")
        timeout: Optional[float] = Field(None, description="Optional timeout in seconds.")

    class SessionStartInput(BaseModel):
        """Input schema for ClaudeStartSessionTool."""
        working_dir: Optional[str] = Field(None, description="Optional working directory.")

    class SessionSendInput(BaseModel):
        """Input schema for ClaudeSendMessageTool."""
        session_id: str = Field(..., description="Existing session ID.")
        message: str = Field(..., description="Message to send.")
        timeout: Optional[float] = Field(None, description="Optional timeout in seconds.")

    class SessionEndInput(BaseModel):
        """Input schema for ClaudeEndSessionTool."""
        session_id: str = Field(..., description="Session ID to end.")

    class ClaudeRunTool(BaseTool):
        """LangChain Tool for single-shot Claude CLI execution."""

        name: str = "claude_cli_run"
        description: str = (
            "Execute a single-shot Claude CLI call with a prompt and return the output. "
            "Useful for code generation, explanation, and other tasks supported by Claude."
        )
        args_schema: type = RunInput

        client: ClaudeCLIClient = None

        def __init__(self, client: Optional[ClaudeCLIClient] = None, **kwargs):
            super().__init__(**kwargs)
            self.client = client or ClaudeCLIClient()

        def _run(
            self,
            prompt: str,
            timeout: Optional[float] = None,
            run_manager: Optional[CallbackManagerForToolRun] = None
        ) -> str:
            """Execute the tool."""
            try:
                return self.client.run(prompt, timeout=timeout)
            except Exception as e:
                raise ToolException(str(e))

    class ClaudeStartSessionTool(BaseTool):
        """LangChain Tool for starting Claude CLI sessions."""

        name: str = "claude_cli_start_session"
        description: str = "Start an interactive Claude CLI session and return a session_id."
        args_schema: type = SessionStartInput

        client: ClaudeCLIClient = None

        def __init__(self, client: Optional[ClaudeCLIClient] = None, **kwargs):
            super().__init__(**kwargs)
            self.client = client or ClaudeCLIClient()

        def _run(
            self,
            working_dir: Optional[str] = None,
            run_manager: Optional[CallbackManagerForToolRun] = None
        ) -> str:
            """Execute the tool."""
            try:
                sid = self.client.start_session(working_dir=working_dir)
                return json.dumps({"session_id": sid})
            except Exception as e:
                raise ToolException(str(e))

    class ClaudeSendMessageTool(BaseTool):
        """LangChain Tool for sending messages to Claude CLI sessions."""

        name: str = "claude_cli_send_message"
        description: str = "Send a message to an existing Claude CLI session and return the reply."
        args_schema: type = SessionSendInput

        client: ClaudeCLIClient = None

        def __init__(self, client: Optional[ClaudeCLIClient] = None, **kwargs):
            super().__init__(**kwargs)
            self.client = client or ClaudeCLIClient()

        def _run(
            self,
            session_id: str,
            message: str,
            timeout: Optional[float] = None,
            run_manager: Optional[CallbackManagerForToolRun] = None
        ) -> str:
            """Execute the tool."""
            try:
                return self.client.send_message(session_id, message, timeout=timeout)
            except Exception as e:
                raise ToolException(str(e))

    class ClaudeEndSessionTool(BaseTool):
        """LangChain Tool for ending Claude CLI sessions."""

        name: str = "claude_cli_end_session"
        description: str = "End an existing Claude CLI session."
        args_schema: type = SessionEndInput

        client: ClaudeCLIClient = None

        def __init__(self, client: Optional[ClaudeCLIClient] = None, **kwargs):
            super().__init__(**kwargs)
            self.client = client or ClaudeCLIClient()

        def _run(
            self,
            session_id: str,
            run_manager: Optional[CallbackManagerForToolRun] = None
        ) -> str:
            """Execute the tool."""
            try:
                self.client.end_session(session_id)
                return json.dumps({"status": "ok", "message": "Session ended successfully"})
            except Exception as e:
                raise ToolException(str(e))


def create_claude_tools(client: Optional[ClaudeCLIClient] = None) -> List[BaseTool]:
    """
    Convenience function to create all Claude CLI tools.

    Args:
        client: Optional shared ClaudeCLIClient instance

    Returns:
        List of LangChain tools

    Raises:
        ImportError: If LangChain is not installed
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required to use Claude CLI tools. Install with: pip install langchain")

    client = client or ClaudeCLIClient()
    return [
        ClaudeRunTool(client=client),
        ClaudeStartSessionTool(client=client),
        ClaudeSendMessageTool(client=client),
        ClaudeEndSessionTool(client=client),
    ]


# Example usage
if __name__ == "__main__":
    # Example 1: Single-shot execution
    print("=== Example 1: Single-shot execution ===")
    client = ClaudeCLIClient()
    try:
        response = client.run("What is 2+2?", flags=["-p"])
        print(f"Response: {response}")
    except CLIError as e:
        print(f"Error: {e}")

    # Example 2: Interactive session
    print("\n=== Example 2: Interactive session ===")
    try:
        sid = client.start_session()
        print(f"Session started: {sid}")

        response1 = client.send_message(sid, "Hello, Claude!")
        print(f"Response 1: {response1}")

        response2 = client.send_message(sid, "What's the weather like?")
        print(f"Response 2: {response2}")

        client.end_session(sid)
        print("Session ended")
    except CLIError as e:
        print(f"Error: {e}")

    # Example 3: LangChain integration (if available)
    if LANGCHAIN_AVAILABLE:
        print("\n=== Example 3: LangChain Tool ===")
        try:
            tools = create_claude_tools()
            tool = tools[0]  # ClaudeRunTool
            result = tool.invoke({"prompt": "Explain Python decorators briefly"})
            print(f"Tool result: {result}")
        except Exception as e:
            print(f"Error: {e}")
