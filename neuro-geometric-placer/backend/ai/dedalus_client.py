"""
Dedalus Labs MCP Client

Uses Dedalus SDK for MCP server orchestration and agent execution.
"""

import os
import asyncio
from typing import Dict, Any, Optional, List, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DedalusClient:
    """Client for Dedalus Labs SDK integration."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Dedalus client.

        Args:
            api_key: Dedalus API key (if None, reads from env)
        """
        self.api_key = api_key or os.getenv("DEDALUS_API_KEY")
        self.client = None
        self.runner = None

        if not self.api_key:
            raise ValueError("DEDALUS_API_KEY not found in environment")

        # Try to import dedalus-labs SDK
        try:
            from dedalus_labs import AsyncDedalus, DedalusRunner
            self.AsyncDedalus = AsyncDedalus
            self.DedalusRunner = DedalusRunner
        except ImportError:
            raise ImportError("dedalus-labs SDK not installed. Run: pip install dedalus-labs")

    async def _get_runner(self) -> 'DedalusRunner':
        """Get or create Dedalus runner."""
        if self.runner is not None:
            return self.runner

        self.client = self.AsyncDedalus()
        self.runner = self.DedalusRunner(self.client)
        return self.runner

    async def run_optimization(
        self,
        input_text: str,
        mcp_server_slug: str,
        model: str = "xai/grok-2-1212",
        tools: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Run PCB placement optimization using Dedalus MCP server.

        Args:
            input_text: Natural language description of optimization task
            mcp_server_slug: Slug of deployed MCP server (e.g., "abiralshakya/ngp")
            model: Model to use (default: xAI Grok)
            tools: Optional additional tools

        Returns:
            Optimization results
        """
        runner = await self._get_runner()

        try:
            # Prepare MCP servers list
            mcp_servers = [mcp_server_slug]

            # Run the optimization
            response = await runner.run(
                input=input_text,
                model=model,
                mcp_servers=mcp_servers,
                tools=tools or []
            )

            return {
                "output": response.final_output,
                "success": True,
                "model": model,
                "mcp_servers": mcp_servers
            }

        except Exception as e:
            return {
                "error": f"Dedalus optimization failed: {str(e)}",
                "success": False,
                "model": model,
                "mcp_servers": [mcp_server_slug]
            }

    async def call_mcp_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        mcp_server_slug: str,
        model: str = "xai/grok-2-1212"
    ) -> Dict[str, Any]:
        """
        Call a specific MCP tool.

        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments for the tool
            mcp_server_slug: MCP server slug
            model: Model to use

        Returns:
            Tool execution results
        """
        # Craft input to trigger specific tool
        input_text = f"Execute the {tool_name} tool with these parameters: {tool_args}"

        runner = await self._get_runner()

        try:
            response = await runner.run(
                input=input_text,
                model=model,
                mcp_servers=[mcp_server_slug]
            )

            return {
                "output": response.final_output,
                "success": True,
                "tool": tool_name,
                "args": tool_args
            }

        except Exception as e:
            return {
                "error": f"MCP tool call failed: {str(e)}",
                "success": False,
                "tool": tool_name,
                "args": tool_args
            }

    # Synchronous wrapper for backward compatibility
    def run_agent(self, prompt: str, tools: Optional[List] = None) -> Dict[str, Any]:
        """
        Synchronous wrapper for backward compatibility.
        Use run_optimization_async for new code.
        """
        try:
            # This is a simplified synchronous version
            # In production, you'd want to use asyncio.run()
            return {"error": "Use async methods", "success": False}
        except Exception as e:
            return {"error": str(e), "success": False}

