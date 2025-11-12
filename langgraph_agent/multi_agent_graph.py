import weave
from typing import Dict, List, Any, TypedDict
import openai
import time
import json
from .mcp_tools import MCPToolManager
from .graph_agent import LangGraphAgent

class MultiAgentState(TypedDict):
    query: str
    specialist_results: Dict[str, Any]
    final_response: str
    agents_used: List[str]
    tools_used: List[str]
    tool_results: Dict[str, Any]

class SpecialistLangGraphAgent(LangGraphAgent):
    """Specialized LangGraph agent for specific domains"""
    
    def __init__(self, specialty: str):
        super().__init__()
        self.specialty = specialty
        self.system_prompt = self._get_system_prompt()
    
    def _get_system_prompt(self) -> str:
        prompts = {
            "research": "You are a research specialist. Focus on finding and analyzing information using available tools.",
            "analysis": "You are an analysis specialist. Focus on data analysis and insights from provided information.",
            "writing": "You are a writing specialist. Focus on creating clear, well-structured content.",
            "technical": "You are a technical specialist. Focus on technical solutions and implementations."
        }
        return prompts.get(self.specialty, "You are a helpful AI assistant.")
    
    @weave.op()
    def generate_response(self, state):
        """Override to use specialty-specific prompts"""
        if self.use_mock:
            response = f"{self.specialty.title()} Specialist: Analyzed '{state['query'][:50]}...' with {self.specialty} expertise."
            if state["tools_used"]:
                response += f" Used tools: {state['tools_used']}"
        else:
            try:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Query: {state['query']}\nTool Results: {json.dumps(state['tool_results'])}"}
                ]
                
                response_obj = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=300
                )
                response = response_obj.choices[0].message.content
            except Exception as e:
                response = f"{self.specialty.title()} Specialist Error: {str(e)[:100]}..."
        
        state["response"] = response
        return state

class MultiAgentLangGraph:
    """Multi-agent system using LangGraph architecture"""
    
    def __init__(self):
        self.specialists = {
            "research": SpecialistLangGraphAgent("research"),
            "analysis": SpecialistLangGraphAgent("analysis"), 
            "writing": SpecialistLangGraphAgent("writing"),
            "technical": SpecialistLangGraphAgent("technical")
        }
        self.use_mock = self.specialists["research"].use_mock
    
    @weave.op()
    def analyze_task(self, query: str) -> List[str]:
        """Determine which specialists are needed"""
        query_lower = query.lower()
        needed = []
        
        if any(word in query_lower for word in ["research", "information", "data", "search", "find"]):
            needed.append("research")
        if any(word in query_lower for word in ["analyze", "analysis", "insights", "compare"]):
            needed.append("analysis")
        if any(word in query_lower for word in ["write", "document", "report", "summary"]):
            needed.append("writing")
        if any(word in query_lower for word in ["technical", "implement", "code", "solution"]):
            needed.append("technical")
        
        # Default to research + writing for general queries
        if not needed:
            needed = ["research", "writing"]
        
        return needed
    
    @weave.op()
    def coordinate_specialists(self, query: str, specialists_needed: List[str]) -> Dict[str, Any]:
        """Execute specialists in sequence"""
        results = {}
        context = None
        all_tools_used = []
        all_tool_results = {}
        
        for specialist_name in specialists_needed:
            if specialist_name in self.specialists:
                specialist = self.specialists[specialist_name]
                
                # Create state for specialist
                state = {
                    "query": query if not context else f"{query}\n\nContext: {context}",
                    "tools_used": [],
                    "tool_results": {},
                    "response": ""
                }
                
                # Process through specialist pipeline
                state = specialist.analyze_query(state)
                state = specialist.execute_tools(state)
                state = specialist.generate_response(state)
                
                results[specialist_name] = {
                    "response": state["response"],
                    "tools_used": state["tools_used"],
                    "tool_results": state["tool_results"]
                }
                
                # Collect all tools and results
                all_tools_used.extend(state["tools_used"])
                for tool, result in state["tool_results"].items():
                    all_tool_results[f"{specialist_name}_{tool}"] = result
                
                # Pass context to next specialist
                context = state["response"]
        
        return {
            "specialist_results": results,
            "tools_used": list(set(all_tools_used)),
            "tool_results": all_tool_results
        }
    
    @weave.op()
    def synthesize_results(self, query: str, coordination_results: Dict[str, Any]) -> str:
        """Synthesize final response from specialist results"""
        specialist_results = coordination_results["specialist_results"]
        
        if self.use_mock:
            specialists_used = ", ".join(specialist_results.keys())
            return f"Multi-Agent LangGraph Response: Combined insights from {specialists_used} specialists for '{query[:50]}...'. Coordinated analysis complete."
        
        # Use writing specialist or first available for synthesis
        synthesizer = self.specialists.get("writing", list(self.specialists.values())[0])
        
        results_summary = "\n".join([
            f"{name}: {result['response']}" 
            for name, result in specialist_results.items()
        ])
        
        try:
            messages = [
                {"role": "system", "content": "Synthesize specialist responses into a coherent final answer."},
                {"role": "user", "content": f"Query: {query}\n\nSpecialist Responses:\n{results_summary}"}
            ]
            
            response_obj = synthesizer.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=400
            )
            return response_obj.choices[0].message.content
        except Exception as e:
            return f"Synthesis complete. Combined insights from {len(specialist_results)} specialists."
    
    @weave.op()
    def process(self, query: str) -> Dict[str, Any]:
        """Main multi-agent processing pipeline"""
        start_time = time.time()
        
        # Task analysis
        specialists_needed = self.analyze_task(query)
        
        # Coordinate specialists
        coordination_results = self.coordinate_specialists(query, specialists_needed)
        
        # Synthesize final response
        final_response = self.synthesize_results(query, coordination_results)
        
        return {
            "query": query,
            "response": final_response,
            "agents_used": specialists_needed,
            "specialist_results": coordination_results["specialist_results"],
            "tools_used": coordination_results["tools_used"],
            "tool_results": coordination_results["tool_results"],
            "processing_time": time.time() - start_time,
            "framework": "langgraph-multi-agent",
            "mode": "mock" if self.use_mock else "llm"
        }