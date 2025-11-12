#!/usr/bin/env python3
"""
W&B Weave Agent Evaluation Runner
Comprehensive evaluation suite for agent performance
"""

import weave
from dotenv import load_dotenv
from agent import WeaveAgent
from evaluation.evaluators import ResponseQualityEvaluator, ToolUsageEvaluator
from evaluation.scorers import WeaveScorers
from multi_agent.workflow import MultiAgentWorkflow
import json
import time
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

@weave.op()
class AgentEvaluationSuite:
    """Comprehensive agent evaluation suite"""
    
    def __init__(self):
        self.agent = WeaveAgent()
        self.multi_agent = MultiAgentWorkflow()
        self.quality_evaluator = ResponseQualityEvaluator()
        self.tool_evaluator = ToolUsageEvaluator()
        
    @weave.op()
    def create_test_dataset(self) -> List[Dict[str, Any]]:
        """Create comprehensive test dataset"""
        return [
            {
                "query": "What's the weather in New York?",
                "expected_tools": ["weather"],
                "category": "tool_usage",
                "difficulty": "easy"
            },
            {
                "query": "Calculate the compound interest on $1000 at 5% for 3 years",
                "expected_tools": ["calculator"],
                "category": "mathematical",
                "difficulty": "medium"
            },
            {
                "query": "Search for recent developments in quantum computing and explain the key concepts",
                "expected_tools": ["web_search"],
                "category": "research",
                "difficulty": "hard"
            },
            {
                "query": "What time is it and what's the weather like?",
                "expected_tools": ["time", "weather"],
                "category": "multi_tool",
                "difficulty": "medium"
            },
            {
                "query": "Explain machine learning algorithms in simple terms",
                "expected_tools": [],
                "category": "explanation",
                "difficulty": "medium"
            },
            {
                "query": "Find information about Python programming and calculate 15 * 23",
                "expected_tools": ["web_search", "calculator"],
                "category": "mixed",
                "difficulty": "hard"
            }
        ]
    
    @weave.op()
    def evaluate_single_agent(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate single agent performance"""
        results = []
        
        for test_case in test_cases:
            print(f"🧪 Testing: {test_case['query'][:50]}...")
            
            # Run agent
            start_time = time.time()
            agent_result = self.agent.process(test_case["query"])
            end_time = time.time()
            
            # Evaluate quality
            quality_metrics = self.quality_evaluator.evaluate(
                test_case["query"], 
                agent_result["response"]
            )
            
            # Evaluate tool usage
            tool_metrics = self.tool_evaluator.evaluate(
                test_case["query"],
                agent_result["selected_tools"],
                agent_result["tool_results"],
                self.agent.tools.list_tools()
            )
            
            # Comprehensive scoring
            comprehensive_scores = WeaveScorers.comprehensive_scorer(
                test_case["query"], 
                agent_result
            )
            
            # Compile results
            test_result = {
                "test_case": test_case,
                "agent_result": agent_result,
                "quality_metrics": quality_metrics,
                "tool_metrics": tool_metrics,
                "comprehensive_scores": comprehensive_scores,
                "evaluation_time": end_time - start_time
            }
            
            results.append(test_result)
        
        return self._analyze_results(results)
    
    @weave.op()
    def evaluate_multi_agent(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate multi-agent workflow performance"""
        results = []
        
        # Filter for complex queries suitable for multi-agent
        complex_cases = [tc for tc in test_cases if tc["difficulty"] in ["medium", "hard"]]
        
        for test_case in complex_cases:
            print(f"🤖🤖 Multi-agent testing: {test_case['query'][:50]}...")
            
            # Run multi-agent workflow
            start_time = time.time()
            workflow_result = self.multi_agent.execute_workflow(test_case["query"])
            end_time = time.time()
            
            # Evaluate final response quality
            quality_metrics = self.quality_evaluator.evaluate(
                test_case["query"],
                workflow_result["final_response"]
            )
            
            # Compile results
            test_result = {
                "test_case": test_case,
                "workflow_result": workflow_result,
                "quality_metrics": quality_metrics,
                "evaluation_time": end_time - start_time
            }
            
            results.append(test_result)
        
        return self._analyze_multi_agent_results(results)
    
    def _analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze single agent evaluation results"""
        if not results:
            return {"error": "No results to analyze"}
        
        # Aggregate metrics
        total_tests = len(results)
        avg_quality = sum(sum(r["quality_metrics"].values()) / len(r["quality_metrics"]) for r in results) / total_tests
        avg_tool_selection = sum(r["tool_metrics"]["tool_selection_score"] for r in results) / total_tests
        avg_tool_execution = sum(r["tool_metrics"]["tool_execution_score"] for r in results) / total_tests
        avg_processing_time = sum(r["agent_result"]["processing_time"] for r in results) / total_tests
        
        # Category analysis
        category_performance = {}
        for result in results:
            category = result["test_case"]["category"]
            if category not in category_performance:
                category_performance[category] = []
            
            quality_score = sum(result["quality_metrics"].values()) / len(result["quality_metrics"])
            category_performance[category].append(quality_score)
        
        # Calculate category averages
        for category in category_performance:
            scores = category_performance[category]
            category_performance[category] = {
                "avg_score": sum(scores) / len(scores),
                "test_count": len(scores)
            }
        
        return {
            "summary": {
                "total_tests": total_tests,
                "avg_quality_score": avg_quality,
                "avg_tool_selection_score": avg_tool_selection,
                "avg_tool_execution_score": avg_tool_execution,
                "avg_processing_time": avg_processing_time
            },
            "category_performance": category_performance,
            "detailed_results": results
        }
    
    def _analyze_multi_agent_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze multi-agent evaluation results"""
        if not results:
            return {"error": "No results to analyze"}
        
        total_tests = len(results)
        avg_quality = sum(sum(r["quality_metrics"].values()) / len(r["quality_metrics"]) for r in results) / total_tests
        avg_processing_time = sum(r["workflow_result"]["processing_time"] for r in results) / total_tests
        
        # Agent usage analysis
        agent_usage = {}
        for result in results:
            agents_used = result["workflow_result"]["agents_used"]
            for agent in agents_used:
                agent_usage[agent] = agent_usage.get(agent, 0) + 1
        
        return {
            "summary": {
                "total_tests": total_tests,
                "avg_quality_score": avg_quality,
                "avg_processing_time": avg_processing_time
            },
            "agent_usage": agent_usage,
            "detailed_results": results
        }

@weave.op()
def run_comprehensive_evaluation():
    """Run comprehensive evaluation suite"""
    print("🧪 W&B Weave Agent Evaluation Suite")
    print("=" * 60)
    
    # Initialize Weave
    weave.init("weave-agent-evaluation")
    
    # Create evaluation suite
    evaluator = AgentEvaluationSuite()
    
    # Create test dataset
    test_dataset = evaluator.create_test_dataset()
    print(f" Created test dataset with {len(test_dataset)} test cases")
    
    # Evaluate single agent
    print("\n Evaluating Single Agent...")
    single_agent_results = evaluator.evaluate_single_agent(test_dataset)
    
    # Evaluate multi-agent
    print("\n Evaluating Multi-Agent Workflow...")
    multi_agent_results = evaluator.evaluate_multi_agent(test_dataset)
    
    # Display results
    print("\n Evaluation Results")
    print("=" * 40)
    
    print("\n Single Agent Performance:")
    sa_summary = single_agent_results["summary"]
    print(f"  Average Quality Score: {sa_summary['avg_quality_score']:.3f}")
    print(f"  Average Tool Selection Score: {sa_summary['avg_tool_selection_score']:.3f}")
    print(f"  Average Processing Time: {sa_summary['avg_processing_time']:.2f}s")
    
    print("\n Multi-Agent Performance:")
    ma_summary = multi_agent_results["summary"]
    print(f"  Average Quality Score: {ma_summary['avg_quality_score']:.3f}")
    print(f"  Average Processing Time: {ma_summary['avg_processing_time']:.2f}s")
    
    print("\n Category Performance (Single Agent):")
    for category, performance in single_agent_results["category_performance"].items():
        print(f"  {category}: {performance['avg_score']:.3f} ({performance['test_count']} tests)")
    
    print("\n Agent Usage (Multi-Agent):")
    for agent, usage_count in multi_agent_results["agent_usage"].items():
        print(f"  {agent}: {usage_count} times")
    
    return {
        "single_agent": single_agent_results,
        "multi_agent": multi_agent_results,
        "test_dataset": test_dataset
    }

if __name__ == "__main__":
    results = run_comprehensive_evaluation()
    print("\n Evaluation completed! Check your W&B dashboard for detailed traces.")