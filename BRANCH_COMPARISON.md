# Branch Comparison: Custom vs LangGraph + MCP

## Interview Demo Commands

### Demo 1: Custom Implementation (main branch)
```bash
git checkout main
streamlit run streamlit_app.py
```
**Shows:** Custom agent, tool calling, multi-agent, RL training

### Demo 2: LangGraph + MCP (langgraph-mcp branch)  
```bash
git checkout langgraph-mcp
streamlit run streamlit_langgraph.py
```
**Shows:** Graph-based routing, MCP tools, enhanced capabilities

## Architecture Comparison

| Feature | Custom (main) | LangGraph + MCP |
|---------|---------------|-----------------|
| **Agent Core** | Custom WeaveAgent | LangGraph pipeline |
| **Tools** | ToolRegistry | MCPToolManager |
| **Routing** | Keyword-based | Graph-based |
| **State** | Manual memory | Structured state |
| **Protocol** | Custom interface | MCP standard |
| **Scalability** | Limited | High |
| **Control** | Full | Framework-guided |

## Key Differences

### Custom Implementation
- ✅ Full control over agent behavior
- ✅ Simple, direct implementation  
- ✅ Weave integration built-in
- ✅ Custom memory management
- ❌ Limited to 4 tools
- ❌ Manual state handling

### LangGraph + MCP
- ✅ Standardized tool protocol
- ✅ Graph-based execution flow
- ✅ Structured state management
- ✅ Extensible tool ecosystem
- ✅ Production-ready patterns
- ❌ Framework dependency
- ❌ Less direct control

## Both Maintain
- Weave tracing integration
- RL training capabilities  
- Multi-agent support
- Streamlit UI
- Same core functionality

## Interview Talking Points
1. **Custom shows** building from scratch skills
2. **LangGraph shows** framework adoption ability
3. **Both approaches** solve the same problems
4. **Architecture evolution** demonstrates learning
5. **Trade-offs** show decision-making skills