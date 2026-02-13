#!/usr/bin/env python3
"""
Streamlit UI for W&B Weave Agent
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
import weave
from dotenv import load_dotenv
from agent import WeaveAgent
from rl_training import RLAgent
from monitoring.monitors import MonitoringDashboard
from evaluation.evaluators import ResponseQualityEvaluator
from multi_agent.workflow import MultiAgentWorkflow
import plotly.express as px
import pandas as pd
import json
import time

load_dotenv()

@st.cache_resource
def initialize_components(use_openpipe=False):
    """Initialize agent components"""
    weave.init("weave-agent-streamlit")
    has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
    
    if use_openpipe:
        agent = RLAgent(use_openpipe=True, use_mock=not has_openai_key)
    else:
        agent = WeaveAgent(use_mock=not has_openai_key)
    dashboard = MonitoringDashboard()
    evaluator = ResponseQualityEvaluator()
    multi_agent = MultiAgentWorkflow(use_mock=not has_openai_key)
    return agent, dashboard, evaluator, multi_agent

def main():
    st.set_page_config(page_title="W&B Weave Agent", page_icon="🤖", layout="wide")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_stats" not in st.session_state:
        st.session_state.agent_stats = []
    if "use_openpipe" not in st.session_state:
        st.session_state.use_openpipe = False
    
    # Initialize components
    agent, dashboard, evaluator, multi_agent = initialize_components(st.session_state.use_openpipe)
    
    # Header
    st.title("🤖 W&B Weave Agent Interface")
    st.markdown("Interactive AI agent with Weave tracing, evaluation, and monitoring")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        agent_mode = st.selectbox(
            "Agent Mode",
            ["Single Agent", "Multi-Agent Workflow"],
            help="Choose between single agent or multi-agent coordination"
        )
        
        show_traces = st.checkbox("Show Weave Traces", value=True)
        show_metrics = st.checkbox("Show Performance Metrics", value=True)
        
        # OpenPipe Integration
        st.header("🧠 OpenPipe RL Training")
        openpipe_enabled = st.checkbox(
            "Enable Training Data Collection",
            value=st.session_state.use_openpipe,
            help="Collect interactions for model fine-tuning"
        )
        
        if openpipe_enabled != st.session_state.use_openpipe:
            st.session_state.use_openpipe = openpipe_enabled
            st.cache_resource.clear()
            st.rerun()
        
        if openpipe_enabled:
            openpipe_key = os.getenv("OPENPIPE_API_KEY")
            if openpipe_key:
                st.success("✅ OpenPipe connected")
            else:
                st.warning("⚠️ Add OPENPIPE_API_KEY to .env")
            
            # Show current training data count
            if hasattr(agent, 'training_data') and agent.training_data:
                single_count = len([d for d in agent.training_data if d.get('metadata', {}).get('mode') != 'multi_agent'])
                multi_count = len([d for d in agent.training_data if d.get('metadata', {}).get('mode') == 'multi_agent'])
                st.metric("Training Samples", f"{len(agent.training_data)}", f"Single: {single_count}, Multi: {multi_count}")
            
            if st.button("📊 Export Training Data"):
                if hasattr(agent, 'training_data') and agent.training_data:
                    # Count single vs multi-agent samples
                    single_count = len([d for d in agent.training_data if d.get('metadata', {}).get('mode') != 'multi_agent'])
                    multi_count = len([d for d in agent.training_data if d.get('metadata', {}).get('mode') == 'multi_agent'])
                    
                    data = json.dumps(agent.training_data, indent=2)
                    st.download_button(
                        "Download Training Data",
                        data,
                        "training_data.json",
                        "application/json"
                    )
                    st.success(f"💾 {len(agent.training_data)} samples ready for export")
                    st.info(f"📊 Single Agent: {single_count}, Multi-Agent: {multi_count}")
                else:
                    st.info("No training data collected yet")
        
        st.header("📊 Live Statistics")
        if st.session_state.agent_stats:
            avg_time = sum(s["time"] for s in st.session_state.agent_stats) / len(st.session_state.agent_stats)
            st.metric("Avg Response Time", f"{avg_time:.2f}s")
            st.metric("Total Queries", len(st.session_state.agent_stats))
            
            # Tools usage chart
            tools_used = []
            for stat in st.session_state.agent_stats:
                tools_used.extend(stat.get("tools", []))
            
            if tools_used:
                tools_df = pd.DataFrame({"Tool": tools_used})
                tool_counts = tools_df["Tool"].value_counts()
                fig = px.bar(x=tool_counts.index, y=tool_counts.values, title="Tool Usage")
                st.plotly_chart(fig, use_container_width=True)
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Display chat messages
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Agent:** {message['content']}")
                if "metadata" in message:
                    with st.expander(f"📊 Details {i}"):
                        st.json(message["metadata"])
            st.markdown("---")
        
        # Chat input using text_input and button
        with st.form(key="chat_form", clear_on_submit=True):
            prompt = st.text_input("Ask me anything...", key="user_input")
            submit_button = st.form_submit_button("Send")
        
        if submit_button and prompt:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.container():
                st.markdown(f"**You:** {prompt}")
            
            # Process with agent
            with st.spinner("Processing..."):
                start_time = time.time()
                
                if agent_mode == "Single Agent":
                    # Use process_with_feedback if OpenPipe is enabled
                    if st.session_state.use_openpipe and hasattr(agent, 'process_with_feedback'):
                        result = agent.process_with_feedback(prompt)
                    else:
                        result = agent.process(prompt)
                    response = result["response"]
                    metadata = {
                        "processing_time": result["processing_time"],
                        "tools_used": result["selected_tools"],
                        "reasoning": result["reasoning"]["reasoning"][:200] + "..."
                    }
                else:
                    result = multi_agent.execute_workflow(prompt)
                    response = result["final_response"]
                    metadata = {
                        "processing_time": result["processing_time"],
                        "agents_used": result["agents_used"],
                        "tools_used": result.get("tools_used", []),
                        "tool_results": result.get("tool_results", {}),
                        "task_analysis": result["task_analysis"]
                    }
                    
                    # Collect training data for multi-agent if OpenPipe enabled
                    if st.session_state.use_openpipe and hasattr(agent, 'training_data'):
                        training_entry = {
                            "messages": [
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": response}
                            ],
                            "metadata": {
                                "mode": "multi_agent",
                                "processing_time": result["processing_time"],
                                "agents_used": result["agents_used"],
                                "tools_used": result.get("tools_used", [])
                            }
                        }
                        agent.training_data.append(training_entry)
                
                end_time = time.time()
                
                # Record statistics
                st.session_state.agent_stats.append({
                    "time": end_time - start_time,
                    "tools": result.get("selected_tools", []),
                    "query": prompt
                })
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "metadata": metadata
                })
                
                # Record in dashboard
                dashboard.record_agent_interaction(result)
                
                # Display response
                st.markdown(f"**Agent:** {response}")
                
                if show_metrics:
                    st.success(f"⚡ Processed in {metadata['processing_time']:.2f}s")
                
                st.rerun()
    
    with col2:
        st.header("📈 Monitoring")
        
        # Real-time metrics
        if st.session_state.agent_stats:
            recent_times = [s["time"] for s in st.session_state.agent_stats[-10:]]
            
            # Performance chart
            fig = px.line(
                x=list(range(len(recent_times))), 
                y=recent_times,
                title="Response Time Trend",
                labels={"x": "Query #", "y": "Time (s)"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Memory stats
            memory_stats = agent.memory.get_memory_stats()
            st.subheader("🧠 Memory")
            st.json(memory_stats)
            
            # Tool stats
            tool_stats = agent.tools.get_tool_stats()
            st.subheader("🔧 Tools")
            for tool, stats in tool_stats.items():
                st.metric(
                    f"{tool}", 
                    f"{stats['usage_count']} uses",
                    f"{stats['success_rate']:.1%} success"
                )
        
        # Dashboard summary
        if st.button("🔄 Refresh Dashboard"):
            summary = dashboard.get_dashboard_summary()
            st.subheader("📊 Dashboard Summary")
            st.json(summary)
        
        # Evaluation button
        st.header("🧪 Evaluation")
        if st.button("▶️ Run Evaluation"):
            with st.spinner("Running comprehensive evaluation..."):
                try:
                    from evaluate_agent import AgentEvaluationSuite
                    evaluator = AgentEvaluationSuite()
                    test_dataset = evaluator.create_test_dataset()
                    
                    # Run single agent evaluation
                    single_results = evaluator.evaluate_single_agent(test_dataset)
                    
                    st.success("✅ Evaluation completed!")
                    st.subheader("📊 Evaluation Results")
                    st.json(single_results["summary"])
                    
                    # Category performance
                    if "category_performance" in single_results:
                        st.subheader("📈 Category Performance")
                        for category, perf in single_results["category_performance"].items():
                            st.metric(f"{category.title()}", f"{perf['avg_score']:.2f}", f"{perf['test_count']} tests")
                    
                except Exception as e:
                    st.error(f"❌ Evaluation failed: {str(e)}")
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.agent_stats = []
            st.rerun()
    
    with col2:
        if st.button("📊 Export Stats"):
            if st.session_state.agent_stats:
                df = pd.DataFrame(st.session_state.agent_stats)
                st.download_button(
                    "Download CSV",
                    df.to_csv(index=False),
                    "agent_stats.csv",
                    "text/csv"
                )
    
    with col3:
        if st.button("🧪 Quick Eval"):
            with st.spinner("Running quick evaluation..."):
                try:
                    from evaluate_agent import AgentEvaluationSuite
                    evaluator = AgentEvaluationSuite()
                    test_cases = [
                        {"query": "What's 5+5?", "expected_tools": ["calculator"], "category": "math", "difficulty": "easy"},
                        {"query": "What time is it?", "expected_tools": ["time"], "category": "tool", "difficulty": "easy"}
                    ]
                    results = evaluator.evaluate_single_agent(test_cases)
                    st.success(f"✅ Score: {results['summary']['avg_quality_score']:.2f}")
                except:
                    st.error("❌ Eval failed")
        
        st.markdown("🔗 [View in W&B Dashboard](https://wandb.ai)")

if __name__ == "__main__":
    main()