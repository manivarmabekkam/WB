import weave
from typing import Dict, List, Any
import requests
import json
import time
import random
import os

@weave.op()
class MCPToolManager:
    """MCP Tool Manager with fallback to existing tools"""
    
    def __init__(self):
        self.tools = {}
        self._register_tools()
    
    def _register_tools(self):
        """Register MCP-style tools"""
        self.tools = {
            "web_search": {"function": self._web_search, "schema": {"type": "object"}},
            "calculator": {"function": self._calculator, "schema": {"type": "object"}},
            "weather": {"function": self._weather, "schema": {"type": "object"}},
            "time": {"function": self._time, "schema": {"type": "object"}}
        }
    
    @weave.op()
    def _web_search(self, query: str) -> Dict[str, Any]:
        """MCP-style web search"""
        serper_api_key = os.getenv("SERPER_API_KEY")
        
        if not serper_api_key:
            return {
                "results": [{"title": f"MCP Search: {query}", "url": "https://example.com", "snippet": f"MCP-powered search for {query}"}],
                "query": query,
                "source": "mcp-simulated"
            }
        
        try:
            url = "https://google.serper.dev/search"
            headers = {"X-API-KEY": serper_api_key, "Content-Type": "application/json"}
            response = requests.post(url, headers=headers, data=json.dumps({"q": query}), timeout=10)
            data = response.json()
            
            results = []
            for item in data.get("organic", [])[:5]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", "")
                })
            
            return {"results": results, "query": query, "source": "mcp-serper"}
        except Exception as e:
            return {"error": f"MCP Serper error: {str(e)}", "query": query}
    
    @weave.op()
    def _calculator(self, expression: str) -> Dict[str, Any]:
        """MCP-style calculator"""
        try:
            allowed_chars = set('0123456789+-*/.() ')
            if all(c in allowed_chars for c in expression):
                result = eval(expression)
                return {"result": result, "expression": expression, "source": "mcp-calculator"}
            return {"error": "Invalid expression", "source": "mcp-calculator"}
        except Exception as e:
            return {"error": str(e), "source": "mcp-calculator"}
    
    @weave.op()
    def _weather(self, location: str) -> Dict[str, Any]:
        """MCP-style weather"""
        return {
            "location": location,
            "temperature": random.randint(-10, 35),
            "condition": random.choice(["sunny", "cloudy", "rainy"]),
            "source": "mcp-weather"
        }
    
    @weave.op()
    def _time(self, timezone: str = "UTC") -> Dict[str, Any]:
        """MCP-style time"""
        return {
            "timestamp": time.time(),
            "formatted": time.ctime(),
            "timezone": timezone,
            "source": "mcp-time"
        }
    
    @weave.op()
    def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP tool"""
        if tool_name not in self.tools:
            return {"error": f"MCP tool '{tool_name}' not found"}
        
        try:
            if tool_name == "calculator":
                return self.tools[tool_name]["function"](args.get("expression", ""))
            else:
                return self.tools[tool_name]["function"](args.get("query", ""))
        except Exception as e:
            return {"error": str(e)}
    
    def list_tools(self) -> List[str]:
        """List available MCP tools"""
        return list(self.tools.keys())