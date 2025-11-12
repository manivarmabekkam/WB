import weave
from typing import Dict, List, Any, TypedDict
import openai
import time
import json
from .mcp_tools import MCPToolManager

class AgentState(TypedDict):
    messages: List[str]
    tools_used: List[str]
    tool_results: Dict[str, Any]
    query: str
    response: str

class LangGraphAgent:
    """LangGraph-based agent with MCP tools"""
    
    def __init__(self):
        self.mcp_tools = MCPToolManager()
        self.use_mock = self._detect_llm_availability()
        if not self.use_mock:
            self.client = openai.OpenAI()
    
    def _detect_llm_availability(self) -> bool:
        """Detect if LLM is available, return True if should use mock"""
        import os
        from dotenv import load_dotenv
        load_dotenv()  # Ensure .env is loaded
        
        api_key = os.getenv("OPENAI_API_KEY")
        return not bool(api_key and len(api_key.strip()) > 10)  # Use mock if no valid API key
    
    @weave.op()
    def analyze_query(self, state: AgentState) -> AgentState:
        """Analyze query and determine tools needed"""
        query = state["query"].lower()
        tools_needed = []
        
        if "weather" in query:
            tools_needed.append("weather")
        if any(word in query for word in ["calculate", "+", "-", "*", "/"]):
            tools_needed.append("calculator")
        if "search" in query or "find" in query:
            tools_needed.append("web_search")
        if "time" in query:
            tools_needed.append("time")
        
        state["tools_used"] = tools_needed
        return state
    
    @weave.op()
    def execute_tools(self, state: AgentState) -> AgentState:
        """Execute MCP tools"""
        results = {}
        
        for tool_name in state["tools_used"]:
            if tool_name == "calculator":
                import re
                match = re.search(r'[\d+\-*/().\s]+', state["query"])
                expression = match.group(0).strip() if match else state["query"]
                result = self.mcp_tools.call_tool(tool_name, {"expression": expression})
            else:
                result = self.mcp_tools.call_tool(tool_name, {"query": state["query"]})
            
            results[tool_name] = result
        
        state["tool_results"] = results
        return state
    
    @weave.op()
    def generate_response(self, state: AgentState) -> AgentState:
        """Generate final response"""
        if self.use_mock:
            tools_info = f" using MCP tools: {state['tools_used']}" if state["tools_used"] else ""
            response = f"LangGraph Mock Response: Processed '{state['query'][:50]}...'{tools_info}. Results: {json.dumps(state['tool_results'], indent=2)}"
        else:
            messages = [
                {"role": "system", "content": "Generate a helpful response using the MCP tool results."},
                {"role": "user", "content": f"Query: {state['query']}\nMCP Tool Results: {json.dumps(state['tool_results'])}"}
            ]
            
            response_obj = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=300
            )
            response = response_obj.choices[0].message.content
        
        state["response"] = response
        return state
    
    @weave.op()
    def process(self, query: str) -> Dict[str, Any]:
        """Process query through LangGraph pipeline"""
        start_time = time.time()
        
        # Initialize state
        state = AgentState(
            messages=[query],
            tools_used=[],
            tool_results={},
            query=query,
            response=""
        )
        
        # Execute pipeline
        state = self.analyze_query(state)
        state = self.execute_tools(state)
        state = self.generate_response(state)
        
        return {
            "query": query,
            "response": state["response"],
            "tools_used": state["tools_used"],
            "tool_results": state["tool_results"],
            "processing_time": time.time() - start_time,
            "framework": "langgraph-mcp",
            "mode": "mock" if self.use_mock else "llm"
        }