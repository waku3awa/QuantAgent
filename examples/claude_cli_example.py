"""
Examples demonstrating the usage of Claude CLI Wrapper.

This file shows various ways to use the wrapper:
1. Direct client usage (single-shot)
2. Interactive sessions
3. LangChain Tool integration
4. Error handling
"""

import sys
import os

# Add parent directory to path to import the wrapper
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_cli_wrapper import (
    ClaudeCLIClient,
    CLIError,
    create_claude_tools,
    LANGCHAIN_AVAILABLE
)


def example_1_single_shot():
    """Example 1: Single-shot execution."""
    print("=" * 60)
    print("Example 1: Single-shot execution")
    print("=" * 60)

    client = ClaudeCLIClient()

    try:
        # Simple question
        print("\n1. Simple calculation:")
        response = client.run("What is 127 * 384?", flags=["-p"])
        print(f"Response: {response}\n")

        # Code generation
        print("2. Code generation:")
        response = client.run(
            "Write a Python function to calculate Fibonacci numbers",
            flags=["-p"]
        )
        print(f"Response:\n{response}\n")

        # With custom timeout
        print("3. With custom timeout:")
        response = client.run(
            "Explain async/await in Python",
            timeout=30.0,
            flags=["-p"]
        )
        print(f"Response:\n{response}\n")

    except CLIError as e:
        print(f"Error: {e}")


def example_2_interactive_session():
    """Example 2: Interactive session."""
    print("=" * 60)
    print("Example 2: Interactive session")
    print("=" * 60)

    client = ClaudeCLIClient()

    try:
        # Start session
        session_id = client.start_session()
        print(f"\nSession started: {session_id}\n")

        # Multiple turns of conversation
        messages = [
            "Hello! Can you help me with Python programming?",
            "I need to read a CSV file. What's the best library?",
            "Can you show me a simple example using pandas?",
        ]

        for i, msg in enumerate(messages, 1):
            print(f"User ({i}): {msg}")
            response = client.send_message(session_id, msg)
            print(f"Claude ({i}): {response}\n")

        # End session
        client.end_session(session_id)
        print(f"Session {session_id} ended.\n")

    except CLIError as e:
        print(f"Error: {e}")
        # Cleanup on error
        try:
            client.end_session(session_id)
        except:
            pass


def example_3_context_manager():
    """Example 3: Using context manager for session management."""
    print("=" * 60)
    print("Example 3: Context manager for safe session handling")
    print("=" * 60)

    # Note: Context manager support requires implementing __enter__ and __exit__
    # in a SessionContext class (not implemented in basic wrapper)
    # This is a placeholder example

    print("\nContext manager example (conceptual):")
    print("""
    with ClaudeCLISession(client) as session:
        response1 = session.send("First message")
        response2 = session.send("Second message")
        # Session automatically closed on exit
    """)


def example_4_error_handling():
    """Example 4: Error handling."""
    print("=" * 60)
    print("Example 4: Error handling")
    print("=" * 60)

    client = ClaudeCLIClient()

    print("\n1. Handling timeout:")
    try:
        # Very short timeout to demonstrate timeout handling
        response = client.run(
            "Write a comprehensive guide to machine learning",
            timeout=0.1  # Intentionally too short
        )
    except CLIError as e:
        print(f"Caught expected error: {e}")

    print("\n2. Handling non-existent session:")
    try:
        client.send_message("non-existent-session-id", "Hello")
    except CLIError as e:
        print(f"Caught expected error: {e}")

    print("\n3. Handling invalid command (if CLI not installed):")
    bad_client = ClaudeCLIClient(cmd=["nonexistent-command"])
    try:
        response = bad_client.run("Test")
    except CLIError as e:
        print(f"Caught expected error: {e}")


def example_5_langchain_integration():
    """Example 5: LangChain Tool integration."""
    print("=" * 60)
    print("Example 5: LangChain Tool integration")
    print("=" * 60)

    if not LANGCHAIN_AVAILABLE:
        print("\nLangChain is not installed. Install with: pip install langchain")
        return

    try:
        # Create tools
        tools = create_claude_tools()

        print(f"\nCreated {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")

        # Use the run tool
        run_tool = tools[0]  # ClaudeRunTool
        print("\nUsing claude_cli_run tool:")
        result = run_tool.invoke({
            "prompt": "What are the key principles of clean code?",
            "timeout": 30.0
        })
        print(f"Result:\n{result}\n")

        # Session-based tools
        start_tool = tools[1]  # ClaudeStartSessionTool
        send_tool = tools[2]   # ClaudeSendMessageTool
        end_tool = tools[3]    # ClaudeEndSessionTool

        print("Using session-based tools:")
        # Start session
        session_result = start_tool.invoke({})
        import json
        session_data = json.loads(session_result)
        session_id = session_data["session_id"]
        print(f"Session started: {session_id}")

        # Send messages
        response1 = send_tool.invoke({
            "session_id": session_id,
            "message": "What is a design pattern?"
        })
        print(f"Response 1: {response1}")

        response2 = send_tool.invoke({
            "session_id": session_id,
            "message": "Give me an example of the Strategy pattern"
        })
        print(f"Response 2: {response2}")

        # End session
        end_result = end_tool.invoke({"session_id": session_id})
        print(f"Session ended: {end_result}\n")

    except Exception as e:
        print(f"Error: {e}")


def example_6_working_directory():
    """Example 6: Using custom working directory."""
    print("=" * 60)
    print("Example 6: Custom working directory")
    print("=" * 60)

    # Get a valid directory (current directory)
    working_dir = os.getcwd()
    print(f"\nWorking directory: {working_dir}")

    client = ClaudeCLIClient(working_dir=working_dir)

    try:
        response = client.run(
            "What files are in the current directory? (Note: Claude CLI may not have direct file access)",
            flags=["-p"]
        )
        print(f"Response: {response}\n")
    except CLIError as e:
        print(f"Error: {e}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Claude CLI Wrapper - Usage Examples")
    print("=" * 60 + "\n")

    print("NOTE: These examples require the 'claude' CLI to be installed")
    print("and available in your PATH. Install from: https://claude.ai/code\n")

    input("Press Enter to continue with examples...")

    examples = [
        ("Single-shot execution", example_1_single_shot),
        ("Interactive session", example_2_interactive_session),
        ("Context manager (conceptual)", example_3_context_manager),
        ("Error handling", example_4_error_handling),
        ("LangChain integration", example_5_langchain_integration),
        ("Custom working directory", example_6_working_directory),
    ]

    for name, func in examples:
        print(f"\n\nRunning: {name}")
        input("Press Enter to continue...")
        try:
            func()
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Unexpected error in {name}: {e}")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
