import streamlit as st
import os
from dotenv import load_dotenv
import weave
from langgraph_agent.graph_agent import LangGraphAgent
from langgraph_agent.multi_agent_graph import MultiAgentLangGraph

# Load environment variables
load_dotenv()

@st.cache_resource
def initialize_weave():
    """Initialize Weave with caching to avoid conflicts"""
    try:
        weave.init("wb-langgraph-agent")
        return True
    except Exception as e:
        st.sidebar.warning(f"⚠️ Weave disabled: {str(e)[:50]}...")
        return False

# Initialize Weave
weave_enabled = initialize_weave()

st.title("🤖 LangGraph + MCP Agent")
st.caption("Modern AI Agent with LangGraph and Model Context Protocol")

# Initialize agents
if "langgraph_agent" not in st.session_state:
    st.session_state.langgraph_agent = LangGraphAgent()
if "multi_agent" not in st.session_state:
    st.session_state.multi_agent = MultiAgentLangGraph()

# Sidebar
st.sidebar.header("Configuration")

# Agent mode selection
agent_mode = st.sidebar.selectbox(
    "Agent Mode",
    ["Single Agent", "Multi-Agent"],
    help="Choose between single agent or multi-agent coordination"
)

mode = "Mock Mode" if st.session_state.langgraph_agent.use_mock else "LLM Mode"
weave_status = "✅ Enabled" if weave_enabled else "❌ Disabled"
st.sidebar.info(f"**Current Mode:** {mode}\n**Agent Type:** {agent_mode}\n**Weave Tracing:** {weave_status}\n\nLangGraph provides graph-based routing\nMCP enables standardized tool protocols")

# Chat interface
if "langgraph_messages" not in st.session_state:
    st.session_state.langgraph_messages = []

# Display chat history
for message in st.session_state.langgraph_messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "metadata" in message:
            with st.expander("Details"):
                st.json(message["metadata"])

# Chat input
if prompt := st.chat_input("Ask the LangGraph agent..."):
    # Add user message
    st.session_state.langgraph_messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner(f"{agent_mode} LangGraph processing..."):
            if agent_mode == "Single Agent":
                result = st.session_state.langgraph_agent.process(prompt)
            else:
                result = st.session_state.multi_agent.process(prompt)
            
            st.write(result["response"])
            
            # Show metadata
            metadata = {
                "framework": result["framework"],
                "mode": result.get("mode", "unknown"),
                "agents_used": result.get("agents_used", ["single"]),
                "tools_used": result["tools_used"],
                "tool_results": result["tool_results"],
                "processing_time": f"{result['processing_time']:.2f}s",
                "weave_enabled": weave_enabled
            }
            
            if agent_mode == "Multi-Agent" and "specialist_results" in result:
                metadata["specialist_results"] = result["specialist_results"]
            
            with st.expander("LangGraph Details"):
                st.json(metadata)
    
    # Add assistant message
    st.session_state.langgraph_messages.append({
        "role": "assistant", 
        "content": result["response"],
        "metadata": metadata
    })

# Sidebar stats
st.sidebar.header("Agent Stats")
st.sidebar.metric("Messages", len(st.session_state.langgraph_messages))
st.sidebar.metric("Available MCP Tools", len(st.session_state.langgraph_agent.mcp_tools.list_tools()))

if agent_mode == "Multi-Agent":
    st.sidebar.metric("Specialist Agents", len(st.session_state.multi_agent.specialists))

# Tool info
with st.sidebar.expander("MCP Tools"):
    for tool in st.session_state.langgraph_agent.mcp_tools.list_tools():
        st.write(f"• {tool}")

# Weave info
if weave_enabled:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔗 [View Traces in W&B](https://wandb.ai)**")

# Comparison info
st.sidebar.markdown("---")
st.sidebar.markdown("**Framework Comparison:**")
st.sidebar.markdown("• Custom: Full control, simple")
st.sidebar.markdown("• LangGraph: Standardized, scalable")