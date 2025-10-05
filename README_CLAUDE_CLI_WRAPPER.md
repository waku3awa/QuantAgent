# Claude CLI Wrapper for LangChain

A comprehensive wrapper for the `claude` CLI (`claude.ai/code`) that integrates seamlessly with LangChain.

## Features

- ✅ **Single-shot execution**: Run quick prompts without session overhead
- ✅ **Interactive sessions**: Maintain context across multiple exchanges
- ✅ **LangChain integration**: Use as standard LangChain Tools
- ✅ **Thread-safe**: Safe for concurrent use
- ✅ **Auto-cleanup**: Idle sessions are automatically terminated
- ✅ **Robust error handling**: Comprehensive exception handling with clear messages
- ✅ **Cross-platform**: Works on Windows, Linux, and macOS (with platform-specific optimizations)

## Installation

### Prerequisites

1. Install the Claude CLI:
   ```bash
   # Follow instructions at https://claude.ai/code
   npm install -g @anthropics/claude-code
   ```

   **Note**: The wrapper automatically detects the `claude` command in your PATH.

2. Install Python dependencies:
   ```bash
   pip install langchain langchain-core  # For LangChain integration (optional)
   pip install pexpect  # For better interactive support on Unix (optional)
   ```

### Setup

Simply copy `claude_cli_wrapper.py` to your project:

```bash
cp claude_cli_wrapper.py /path/to/your/project/
```

## Quick Start

### 1. Single-shot Execution

```python
from claude_cli_wrapper import ClaudeCLIClient, CLIError

client = ClaudeCLIClient()

try:
    response = client.run("What is 2+2?", flags=["-p"])
    print(response)
except CLIError as e:
    print(f"Error: {e}")
```

### 2. Interactive Sessions

```python
from claude_cli_wrapper import ClaudeCLIClient

client = ClaudeCLIClient()

# Start a session
session_id = client.start_session()

try:
    # Multiple turns of conversation
    response1 = client.send_message(session_id, "Hello!")
    print(response1)

    response2 = client.send_message(session_id, "What's the weather?")
    print(response2)
finally:
    # Always clean up
    client.end_session(session_id)
```

### 3. LangChain Tool Integration

```python
from claude_cli_wrapper import create_claude_tools
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Create Claude tools
claude_tools = create_claude_tools()

# Use with any LangChain agent
llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with access to Claude."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_openai_functions_agent(llm, claude_tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=claude_tools)

result = agent_executor.invoke({
    "input": "Use Claude to explain Python decorators"
})
print(result)
```

## API Reference

### ClaudeCLIClient

Main client for interacting with Claude CLI.

#### Constructor

```python
ClaudeCLIClient(
    cmd: Optional[List[str]] = None,           # Command to execute (default: auto-detected claude)
    env: Optional[Dict[str, str]] = None,      # Environment variables
    timeout: float = 60.0,                     # Default timeout in seconds
    working_dir: Optional[str] = None,         # Working directory
    expect_prompt: Optional[str] = None,       # REPL prompt regex (for pexpect)
    end_marker: Optional[str] = None,          # Sentinel marker for output detection
)
```

#### Methods

##### `run(prompt: str, timeout: Optional[float] = None, flags: Optional[List[str]] = None) -> str`

Execute a single-shot command.

**Parameters:**
- `prompt`: The prompt to send to Claude
- `timeout`: Optional timeout override
- `flags`: Additional CLI flags (e.g., `["-p"]`)

**Returns:** The output from Claude

**Raises:** `CLIError` if execution fails

##### `start_session(working_dir: Optional[str] = None) -> str`

Start an interactive session.

**Parameters:**
- `working_dir`: Optional working directory for this session

**Returns:** Session ID for future operations

**Raises:** `CLIError` if session creation fails

##### `send_message(session_id: str, message: str, timeout: Optional[float] = None) -> str`

Send a message to an active session.

**Parameters:**
- `session_id`: The session ID from `start_session()`
- `message`: The message to send
- `timeout`: Optional timeout override

**Returns:** The response from Claude

**Raises:** `CLIError` if session doesn't exist or communication fails

##### `end_session(session_id: str) -> None`

End an active session.

**Parameters:**
- `session_id`: The session ID to terminate

### LangChain Tools

Four tools are available for LangChain integration:

1. **claude_cli_run**: Single-shot execution
2. **claude_cli_start_session**: Start a session
3. **claude_cli_send_message**: Send message to session
4. **claude_cli_end_session**: End a session

#### `create_claude_tools(client: Optional[ClaudeCLIClient] = None) -> List[BaseTool]`

Convenience function to create all tools.

**Parameters:**
- `client`: Optional shared `ClaudeCLIClient` instance

**Returns:** List of LangChain tools

## Architecture

### Design Principles

Based on recommendations from both Gemini and GPT-5:

1. **Separation of Concerns**:
   - `ClaudeCLIClient`: Core subprocess management
   - `Claude*Tool` classes: LangChain integration layer

2. **Thread Safety**:
   - Per-session locks prevent race conditions
   - Background reaper thread for auto-cleanup

3. **Platform Optimization**:
   - Uses `pexpect` on Unix-like systems for better TTY control
   - Falls back to `subprocess.Popen` on Windows

4. **Resource Management**:
   - Automatic cleanup of idle sessions (15 min TTL)
   - Graceful shutdown with timeout escalation (terminate → kill)

### Data Flow

```
User Request
    ↓
LangChain Tool (optional)
    ↓
ClaudeCLIClient
    ↓
subprocess/pexpect
    ↓
claude CLI
    ↓
Response parsing
    ↓
Return to user
```

## Error Handling

All errors are wrapped in `CLIError` with descriptive messages:

```python
from claude_cli_wrapper import ClaudeCLIClient, CLIError

client = ClaudeCLIClient()

try:
    response = client.run("Test", timeout=0.1)
except CLIError as e:
    if "Timeout" in str(e):
        print("Request took too long")
    elif "not found" in str(e).lower():
        print("Claude CLI not installed")
    else:
        print(f"Other error: {e}")
```

## Configuration

### Custom Command

```python
# The wrapper auto-detects claude in PATH, but you can override
client = ClaudeCLIClient(
    cmd=["path/to/claude"],
    working_dir="/project/directory"
)
```

### Custom Timeout

```python
# Global timeout
client = ClaudeCLIClient(timeout=120.0)

# Per-request timeout
response = client.run("prompt", timeout=30.0)
```

### Environment Variables

```python
client = ClaudeCLIClient(
    env={"CUSTOM_VAR": "value"}
)
```

## Testing

### Unit Tests

Run unit tests (no Claude CLI required):

```bash
python tests/test_claude_cli_wrapper.py
```

### Integration Tests

Run integration tests (requires Claude CLI):

```bash
python -m pytest tests/test_claude_cli_wrapper.py -v -m integration
```

## Examples

See [`examples/claude_cli_example.py`](examples/claude_cli_example.py) for comprehensive examples:

```bash
python examples/claude_cli_example.py
```

## Troubleshooting

### "Claude CLI not found"

Ensure the `claude` command is in your PATH:

```bash
which claude  # Unix
where claude  # Windows
```

### Session Timeout

Increase the timeout:

```python
client = ClaudeCLIClient(timeout=120.0)  # 2 minutes
```

### Hanging Sessions

Sessions are automatically cleaned up after 15 minutes of inactivity. To force cleanup:

```python
client.end_session(session_id)
```

### Windows TTY Issues

The wrapper uses `subprocess.Popen` on Windows (no TTY support). For better interactive control on Unix, install `pexpect`:

```bash
pip install pexpect
```

## Advanced Usage

### Custom Prompt Detection

If your Claude CLI prints a custom prompt:

```python
client = ClaudeCLIClient(
    expect_prompt=r"^\(claude\)\s*>",  # Regex for prompt
)
```

### Output Markers

Use a sentinel marker to detect output completion:

```python
client = ClaudeCLIClient(
    end_marker="<<<END>>>"
)
```

### Shared Client Across Tools

For better resource management:

```python
from claude_cli_wrapper import ClaudeCLIClient, create_claude_tools

# Single shared client
client = ClaudeCLIClient()

# All tools use the same client
tools = create_claude_tools(client=client)
```

## Limitations

- **Streaming**: Not currently supported (planned for future)
- **Batch Processing**: Not currently supported (planned for future)
- **Windows TTY**: Limited interactive support on Windows (no `pexpect`)

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass
2. Code follows PEP 8
3. New features include tests and documentation

## License

This project is provided as-is for use with the Claude CLI. Please refer to Anthropic's terms of service for the Claude CLI.

## Acknowledgments

Design inspired by:
- Gemini's recommendations on robust subprocess management
- GPT-5's best practices for LangChain tool integration
- LangChain's tool design patterns

## See Also

- [Claude CLI Documentation](https://claude.ai/code)
- [LangChain Documentation](https://python.langchain.com/)
- [Subprocess Best Practices](https://docs.python.org/3/library/subprocess.html)
