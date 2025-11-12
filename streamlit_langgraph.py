import streamlit as st
import weave
import os
from langgraph_agent.graph_agent import LangGraphAgent

# Initialize Weave
weave.init("wb-langgraph-agent")

st.title("🤖 LangGraph + MCP Agent")
st.caption("Modern AI Agent with LangGraph and Model Context Protocol")

# Sidebar
st.sidebar.header("Configuration")
use_mock = st.sidebar.checkbox("Use Mock Mode", value=True)
st.sidebar.info("LangGraph provides graph-based routing\nMCP enables standardized tool protocols")

# Initialize agent
if "langgraph_agent" not in st.session_state:
    st.session_state.langgraph_agent = LangGraphAgent(use_mock=use_mock)

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
        with st.spinner("LangGraph processing..."):
            result = st.session_state.langgraph_agent.process(prompt)
            
            st.write(result["response"])
            
            # Show metadata
            metadata = {
                "framework": result["framework"],
                "tools_used": result["tools_used"],
                "tool_results": result["tool_results"],
                "processing_time": f"{result['processing_time']:.2f}s"
            }
            
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

# Tool info
with st.sidebar.expander("MCP Tools"):
    for tool in st.session_state.langgraph_agent.mcp_tools.list_tools():
        st.write(f"• {tool}")

# Comparison info
st.sidebar.markdown("---")
st.sidebar.markdown("**Framework Comparison:**")
st.sidebar.markdown("• Custom: Full control, simple")
st.sidebar.markdown("• LangGraph: Standardized, scalable")