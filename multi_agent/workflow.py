import weave
from typing import Dict, List, Any, Optional
import openai
import asyncio
import json
import time
from agent.tools import ToolRegistry

class SpecialistAgent:
    """Specialized agent for specific domains"""
    
    def __init__(self, specialty: str, use_mock: bool = True):
        self.specialty = specialty
        self.use_mock = use_mock
        self.system_prompt = self._get_system_prompt()
        self.tools = ToolRegistry()
        if not use_mock:
            self.client = openai.OpenAI()
    
    def _get_system_prompt(self) -> str:
        """Get system prompt based on specialty"""
        prompts = {
            "research": "You are a research specialist. Focus on finding and analyzing information.",
            "analysis": "You are an analysis specialist. Focus on data analysis and insights.",
            "writing": "You are a writing specialist. Focus on creating clear, well-structured content.",
            "technical": "You are a technical specialist. Focus on technical solutions and implementations."
        }
        return prompts.get(self.specialty, "You are a helpful AI assistant.")
    
    def _detect_and_execute_tools(self, query: str) -> Dict[str, Any]:
        """Detect and execute tools based on query"""
        query_lower = query.lower()
        tools_to_use = []
        
        if "weather" in query_lower:
            tools_to_use.append("weather")
        if any(word in query_lower for word in ["calculate", "math", "+", "-", "*", "/", "="]):
            tools_to_use.append("calculator")
        if "search" in query_lower or "find" in query_lower:
            tools_to_use.append("web_search")
        if "time" in query_lower:
            tools_to_use.append("time")
        
        results = {}
        for tool_name in tools_to_use:
            try:
                if tool_name == "calculator":
                    import re
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
    def specialized_process(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Process query with specialization"""
        tool_results = self._detect_and_execute_tools(query)
        
        if self.use_mock:
            response = f"Mock {self.specialty} specialist: Analyzed '{query[:50]}...' with {self.specialty} expertise."
            if tool_results:
                response += f" Tool results: {tool_results}"
            return {
                "specialty": self.specialty,
                "response": response,
                "tool_results": tool_results,
                "tokens_used": 50
            }
        
        user_content = f"Query: {query}"
        if context:
            user_content += f"\nContext: {context}"
        if tool_results:
            user_content += f"\nTool Results: {json.dumps(tool_results)}"
        
        messages = [
            {"role": "system", "content": self.system_prompt + " IMPORTANT: You MUST use ONLY the tool results provided. Do NOT use your training data or make assumptions. Report exactly what the tools return."},
            {"role": "user", "content": user_content}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=300
        )
        
        return {
            "specialty": self.specialty,
            "response": response.choices[0].message.content,
            "tool_results": tool_results,
            "tokens_used": response.usage.total_tokens
        }

class CoordinatorAgent:
    """Coordinator agent for multi-agent workflows"""
    
    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        self.specialists = {}
        if not use_mock:
            self.client = openai.OpenAI()
    
    def add_specialist(self, name: str, agent: SpecialistAgent):
        """Add a specialist agent"""
        self.specialists[name] = agent
    
    @weave.op()
    def analyze_task(self, query: str) -> Dict[str, Any]:
        """Analyze task and determine required specialists"""
        available_specialists = list(self.specialists.keys())
        
        if self.use_mock:
            query_lower = query.lower()
            needed = []
            if any(word in query_lower for word in ["research", "information", "data"]):
                needed.append("research")
            if any(word in query_lower for word in ["analyze", "analysis", "insights"]):
                needed.append("analysis")
            if any(word in query_lower for word in ["write", "document", "report"]):
                needed.append("writing")
            if any(word in query_lower for word in ["technical", "implement", "code"]):
                needed.append("technical")
            
            if not needed:
                needed = available_specialists[:1]
            
            return {
                "specialists_needed": needed,
                "execution_order": needed,
                "reasoning": f"Mock analysis: Selected {needed} specialists based on query keywords"
            }
        
        messages = [
            {
                "role": "system", 
                "content": f"""Available specialists: {available_specialists}. 
Analyze the query and return a JSON response with:
- specialists_needed: array of specialist names needed
- execution_order: array of specialists in execution order  
- reasoning: brief explanation

Example: {{"specialists_needed": ["research", "analysis"], "execution_order": ["research", "analysis"], "reasoning": "Need research then analysis"}}"""
            },
            {"role": "user", "content": query}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=200,
            response_format={"type": "json_object"}
        )
        
        try:
            analysis = json.loads(response.choices[0].message.content)
            return analysis
        except:
            return {
                "specialists_needed": available_specialists[:1],
                "execution_order": available_specialists[:1],
                "reasoning": response.choices[0].message.content
            }
    
    @weave.op()
    def coordinate_specialists(self, query: str, specialist_assignments: Dict[str, Any], memory_context: str = "") -> Dict[str, Any]:
        """Coordinate specialist agents"""
        results = {}
        execution_order = specialist_assignments.get("execution_order", [])
        
        query_lower = query.lower()
        is_simple_query = any([
            "calculate" in query_lower or any(op in query for op in ["+", "-", "*", "/"]),
            "weather" in query_lower,
            "time" in query_lower
        ])
        
        if is_simple_query and execution_order:
            specialist_name = execution_order[0]
            if specialist_name in self.specialists:
                specialist = self.specialists[specialist_name]
                result = specialist.specialized_process(query, memory_context)
                results[specialist_name] = result
            return results
        
        context = memory_context
        for specialist_name in execution_order:
            if specialist_name in self.specialists:
                specialist = self.specialists[specialist_name]
                result = specialist.specialized_process(query, context)
                results[specialist_name] = result
                context = result["response"]
        
        return results
    
    @weave.op()
    def synthesize_results(self, query: str, specialist_results: Dict[str, Any], memory_context: str = "") -> str:
        """Synthesize results from multiple specialists"""
        if self.use_mock:
            specialists_used = ", ".join(specialist_results.keys())
            context_info = f" (with context: {memory_context[:50]}...)" if memory_context else ""
            return f"Mock synthesis: Combined insights from {specialists_used} specialists for query '{query[:50]}...'. Final recommendation based on multi-agent analysis.{context_info}"
        
        results_summary = "\n".join([
            f"{name}: {result['response']}" 
            for name, result in specialist_results.items()
        ])
        
        system_content = "Synthesize specialist responses into a coherent answer. Use ONLY the information provided by specialists. Do NOT add information from your training data."
        if memory_context:
            system_content += f" Previous conversation context: {memory_context}"
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Query: {query}\n\nResponses:\n{results_summary}"}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=300
        )
        
        return response.choices[0].message.content

@weave.op()
class MultiAgentWorkflow:
    """Multi-agent workflow orchestrator"""
    
    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        self.coordinator = CoordinatorAgent(use_mock=use_mock)
        self._setup_specialists()
        from agent.memory import MemoryManager
        self.memory = MemoryManager()
    
    def _setup_specialists(self):
        """Setup specialist agents"""
        specialists = [
            ("research", SpecialistAgent("research", use_mock=self.use_mock)),
            ("analysis", SpecialistAgent("analysis", use_mock=self.use_mock)),
            ("writing", SpecialistAgent("writing", use_mock=self.use_mock)),
            ("technical", SpecialistAgent("technical", use_mock=self.use_mock))
        ]
        
        for name, agent in specialists:
            self.coordinator.add_specialist(name, agent)
    
    @weave.op()
    def run(self, query: str) -> Dict[str, Any]:
        """Execute multi-agent workflow"""
        start_time = time.time()
        
        # Store in memory and get context
        self.memory.add_interaction(query, "user")
        context = self.memory.get_relevant_context(query)
        
        query_lower = query.lower()
        
        # Fast path: detect tool-only queries and skip task analysis
        is_tool_query = any([
            "calculate" in query_lower or any(op in query for op in ["+", "-", "*", "/"]),
            "weather" in query_lower,
            "time" in query_lower
        ])
        
        if is_tool_query:
            specialist = list(self.coordinator.specialists.values())[0]
            result = specialist.specialized_process(query, None)
            specialist_results = {"research": result}
            
            tools_used = list(result.get('tool_results', {}).keys())
            tool_results = result.get('tool_results', {})
            final_response = f"Results: {json.dumps(tool_results, indent=2)}" if tool_results else result['response']
            
            # Store response in memory
            self.memory.add_interaction(final_response, "assistant")
            
            return {
                "query": query,
                "task_analysis": {"specialists_needed": ["research"], "execution_order": ["research"]},
                "specialist_results": specialist_results,
                "final_response": final_response,
                "processing_time": time.time() - start_time,
                "agents_used": ["research"],
                "tools_used": tools_used,
                "tool_results": tool_results,
                "context_used": context
            }
        
        # Full workflow for complex queries
        task_analysis = self.coordinator.analyze_task(query)
        specialist_results = self.coordinator.coordinate_specialists(query, task_analysis, context)
        final_response = self.coordinator.synthesize_results(query, specialist_results, context)
        
        tools_used = []
        tool_results = {}
        for agent_name, result in specialist_results.items():
            if 'tool_results' in result and result['tool_results']:
                for tool_name, tool_result in result['tool_results'].items():
                    if tool_name not in tools_used:
                        tools_used.append(tool_name)
                    tool_results[f"{agent_name}_{tool_name}"] = tool_result
        
        # Store response in memory
        self.memory.add_interaction(final_response, "assistant")
        
        return {
            "query": query,
            "task_analysis": task_analysis,
            "specialist_results": specialist_results,
            "final_response": final_response,
            "processing_time": time.time() - start_time,
            "agents_used": list(specialist_results.keys()),
            "tools_used": tools_used,
            "tool_results": tool_results,
            "context_used": context
        }
    
    def execute_workflow(self, query: str) -> Dict[str, Any]:
        """Alias for run() method for backward compatibility"""
        return self.run(query)
