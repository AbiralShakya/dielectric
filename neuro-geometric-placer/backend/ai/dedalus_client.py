"""
Dedalus Labs MCP Client

For hosting MCP servers and agent orchestration.
"""

import os
import json
from typing import Dict, Any, Optional, List


class DedalusClient:
    """Client for Dedalus Labs MCP orchestration."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Dedalus client.
        
        Args:
            api_key: Dedalus API key (if None, reads from env)
        """
        self.api_key = api_key or os.getenv("DEDALUS_API_KEY")
        self.agent = None
        
        # Try to import dedalus-labs
        try:
            import dedalus as ddls
            self.ddls = ddls
        except ImportError:
            try:
                from dedalus_labs import create_agent
                self.create_agent = create_agent
                self.ddls = None
            except ImportError:
                self.ddls = None
                self.create_agent = None
    
    def _get_agent(self, tools: Optional[List] = None):
        """Get or create Dedalus agent."""
        if self.agent is not None:
            return self.agent
        
        if self.ddls is None:
            raise ImportError("dedalus-labs not installed")
        
        try:
            self.agent = self.ddls.create_agent(
                api_key=self.api_key,
                name="ngp-agent",
                model="xai/grok-2-1212",  # Use xAI via Dedalus
                tools=tools or [],
                instructions="You are a PCB placement optimization agent."
            )
        except Exception:
            # Fallback
            self.agent = self.ddls.create_agent(
                api_key=self.api_key,
                name="ngp-agent",
                model="xai/grok-2-1212"
            )
        
        return self.agent
    
    def run_agent(self, prompt: str, tools: Optional[List] = None) -> Dict[str, Any]:
        """
        Run agent with prompt.
        
        Args:
            prompt: Agent prompt
            tools: Optional tools to use
        
        Returns:
            Agent response
        """
        agent = self._get_agent(tools)
        
        try:
            response = agent.run(prompt)
            
            if hasattr(response, 'choices'):
                content = response.choices[0].message.content
            elif isinstance(response, str):
                content = response
            else:
                content = str(response)
            
            return {"output": content, "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}

