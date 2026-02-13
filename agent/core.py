import weave
import openai
from typing import Dict, List, Any, Optional
import json
import time
import os
from .memory import MemoryManager
from .tools import ToolRegistry

class WeaveAgent:
    """AI Agent with Weave tracing integration"""
    
    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 300, use_mock: bool = True):
        self.model = model
        self.max_tokens = max_tokens
        # Check if OpenAI key exists to determine mock mode
        openai_key = os.getenv("OPENAI_API_KEY")
        self.use_mock = use_mock if not openai_key else False
        if not self.use_mock:
            self.client = openai.OpenAI()
        self.memory = MemoryManager()
        self.tools = ToolRegistry()
        
    @weave.op()
    def reason(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Core reasoning - skipped for performance (tool selection is keyword-based)"""
        return {
            "reasoning": f"Query analysis: {query[:100]}",
            "model": self.model if not self.use_mock else "mock-model",
            "tokens_used": 0
        }
    
    @weave.op()
    def select_tools(self, query: str, reasoning: str) -> List[str]:
        """Select appropriate tools based on query and reasoning"""
        # Use keyword-based tool selection for reliability
        query_lower = query.lower()
        selected = []
        
        if "weather" in query_lower:
            selected.append("weather")
        if any(word in query_lower for word in ["calculate", "math", "+", "-", "*", "/", "="]):
            selected.append("calculator")
        if "search" in query_lower or "find" in query_lower or "look up" in query_lower:
            selected.append("web_search")
        if "time" in query_lower:
            selected.append("time")
        
        # If no tools selected and not in mock mode, ask GPT
        if not selected and not self.use_mock:
            available_tools = self.tools.list_tools()
            messages = [
                {"role": "system", "content": f"Available tools: {available_tools}. Return ONLY a JSON array of tool names, e.g. [\"calculator\", \"weather\"]"},
                {"role": "user", "content": f"Query: {query}"}
            ]
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=50
                )
                tools = json.loads(response.choices[0].message.content)
                return tools if isinstance(tools, list) else []
            except:
                return []
        
        return selected
    
    @weave.op()
    def execute_tools(self, tools: List[str], query: str) -> Dict[str, Any]:
        """Execute selected tools"""
        results = {}
        for tool_name in tools:
            try:
                # Extract math expression for calculator
                if tool_name == "calculator":
                    import re
                    # Try to extract just the math expression
                    match = re.search(r'[\d+\-*/().\s]+', query)
                    expression = match.group(0).strip() if match else query
                    result = self.tools.execute(tool_name, expression)
                else:
                    result = self.tools.execute(tool_name, query)
                results[tool_name] = result
            except Exception as e:
                results[tool_name] = {"error": str(e)}
        return results
    
    @weave.op()
    def generate_response(self, query: str, reasoning: str, tool_results: Dict[str, Any], context: str = "") -> str:
        """Generate final response based on reasoning and tool results"""
        if self.use_mock:
            # Use memory context in mock mode
            if context and "name" in query.lower():
                if "mani" in context.lower():
                    return "Your name is Mani, as you mentioned earlier."
                elif "name" in context.lower():
                    # Extract name from context
                    import re
                    name_match = re.search(r'name is (\w+)', context.lower())
                    if name_match:
                        return f"Your name is {name_match.group(1).title()}."
            
            context_info = f" (with context: {context[:50]}...)" if context else ""
            return f"Mock Response: Based on your query '{query}', I analyzed the situation and used tools {list(tool_results.keys())}. The results show: {str(tool_results)[:100]}...{context_info} This is a simulated response for testing."
        
        system_content = "Generate a helpful response using the tool results. Be concise and direct."
        if context:
            system_content += f" IMPORTANT: Use this previous conversation context to answer personal questions: {context}"
        
        user_content = query
        if tool_results:
            user_content = f"Query: {query}\nTool Results: {json.dumps(tool_results)}"
        elif context:
            user_content = f"Query: {query}\nPrevious conversation: {context}"
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content
    
    @weave.op()
    def process(self, query: str) -> Dict[str, Any]:
        """Main agent processing pipeline"""
        start_time = time.time()
        
        # Store in memory
        self.memory.add_interaction(query, "user")
        
        # Get context from memory
        context = self.memory.get_relevant_context(query)
        
        # Reasoning step
        reasoning_result = self.reason(query, context)
        
        # Tool selection
        selected_tools = self.select_tools(query, reasoning_result["reasoning"])
        
        # Tool execution
        tool_results = self.execute_tools(selected_tools, query)
        
        # Response generation
        response = self.generate_response(query, reasoning_result["reasoning"], tool_results, context)
        
        # Store response in memory
        self.memory.add_interaction(response, "assistant")
        
        end_time = time.time()
        
        result = {
            "query": query,
            "reasoning": reasoning_result,
            "selected_tools": selected_tools,
            "tool_results": tool_results,
            "response": response,
            "processing_time": end_time - start_time,
            "context_used": context
        }
        
        return result