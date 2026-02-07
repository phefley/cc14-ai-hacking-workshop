"""
MCP Server - File System Tools

Provides file operations scoped to a sandbox directory.
Demonstrates all three MCP primitives:
  - Tools:    read_file, list_directory, write_file
  - Resource: file://sandbox/readme
  - Prompt:   analyze_file
"""

import os
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("file-tools")

SANDBOX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox")


def _validate_path(path: str) -> str:
    """Basic path validation - checks that path doesn't start with '..'"""
    if path.startswith(".."):
        raise ValueError("Access denied: path cannot start with '..'")
    return os.path.join(SANDBOX_DIR, path)


# ============ Tools ============

@mcp.tool()
def read_file(path: str) -> str:
    """Read the contents of a file. The path should be relative to the sandbox directory."""
    try:
        full_path = _validate_path(path)
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except ValueError as e:
        return str(e)
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except Exception as e:
        return f"Error reading file: {e}"


@mcp.tool()
def list_directory(path: str = ".") -> str:
    """List files and directories. The path should be relative to the sandbox directory."""
    try:
        full_path = _validate_path(path)
        entries = []
        for entry in sorted(os.listdir(full_path)):
            entry_path = os.path.join(full_path, entry)
            prefix = "[DIR] " if os.path.isdir(entry_path) else "[FILE]"
            entries.append(f"{prefix} {entry}")
        return "\n".join(entries) if entries else "Directory is empty."
    except ValueError as e:
        return str(e)
    except FileNotFoundError:
        return f"Error: Directory not found: {path}"
    except Exception as e:
        return f"Error listing directory: {e}"


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write content to a file. The path should be relative to the sandbox directory."""
    try:
        full_path = _validate_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Error writing file: {e}"


# ============ Resource ============

@mcp.resource("file://sandbox/readme")
def sandbox_readme() -> str:
    """Information about the file sandbox environment."""
    return (
        "File Tools MCP Server\n"
        "=====================\n"
        "This server provides access to files in the sandbox directory.\n\n"
        "Available tools:\n"
        "  - read_file: Read the contents of a file\n"
        "  - list_directory: List files and directories\n"
        "  - write_file: Write content to a file\n\n"
        "All paths are relative to the sandbox directory."
    )


# ============ Prompt ============

@mcp.prompt()
def analyze_file(filename: str) -> str:
    """Generate a prompt to analyze a file's contents for security issues."""
    return (
        f"Analyze the following file for potential security issues.\n"
        f"Look for: hardcoded credentials, API keys, sensitive data exposure,\n"
        f"insecure configurations, and any other security concerns.\n\n"
        f"File to analyze: {filename}\n\n"
        f"Provide a brief security assessment with findings and recommendations."
    )


if __name__ == "__main__":
    mcp.run()
