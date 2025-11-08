# Dedalus Labs MCP Integration

## 🎯 How Neuro-Geometric Placer Uses Dedalus Labs

### **Dedalus Labs Overview**
[Dedalus Labs](https://www.dedaluslabs.ai/) provides:
- **MCP Gateway**: Connect any LLM to any MCP server
- **MCP Hosting**: Deploy and scale MCP servers without infrastructure
- **Agent SDK**: Build multi-agent systems with standardized tools
- **Marketplace**: Discover and use community MCP servers

### **Our Dedalus Integration**

#### 1. **MCP Servers Hosted on Dedalus**
Our system deploys 3 MCP servers via Dedalus Labs:

- **PlacementScorerMCP**: `score_delta()` - Fast score computation using incremental O(k) algorithm
- **ThermalSimulatorMCP**: `generate_heatmap()` - Thermal analysis with Gaussian heat convolution
- **KiCadExporterMCP**: `export_kicad()` - CAD export with coordinate transformation

#### 2. **Agent Model Selection**
```python
# Agents use Dedalus for MCP orchestration
self.vision_agent = VisionAgent(model_provider="dedalus")  # Dedalus MCP
self.rule_agent = RuleAgent(model_provider="dedalus")     # Dedalus MCP
self.permit_agent = PermitAgent(model_provider="dedalus") # Dedalus MCP
self.simulation_agent = GeometrySimulationAgent(model_provider="xai")  # xAI for reasoning
```

#### 2. **Agent Orchestration via Dedalus**
```python
# Agents use Dedalus client for MCP calls
from backend.ai.dedalus_client import DedalusClient

client = DedalusClient()  # Uses DEDALUS_API_KEY
result = client.run_agent(prompt, tools=["placement_scorer", "thermal_simulator"])
```

#### 3. **Multi-Agent Communication**
- Agents communicate via Dedalus MCP gateway
- Standardized tool calling across all agents
- Automatic load balancing and scaling

### **Getting Started with Dedalus**

#### 1. **Get API Key**
- Visit [Dedalus Labs](https://www.dedaluslabs.ai/)
- Create account and generate API key
- Add to `.env`: `DEDALUS_API_KEY=your_key_here`

#### 2. **MCP Server Deployment**
Dedalus hosts our MCP servers automatically:
```bash
# Servers are deployed via Dedalus dashboard
# No Docker/K8s needed - just API key
```

#### 3. **Using MCP Tools**
```python
# Agents call MCP tools through Dedalus
delta = dedalus_client.call_mcp_tool(
    server="placement_scorer",
    tool="score_delta",
    placement_data=placement,
    move_data=move_params
)
```

### **Why Dedalus Labs?**

#### **Benefits:**
- **Zero Infrastructure**: No server management
- **Auto-Scaling**: Handles traffic spikes
- **Multi-Model**: Works with any LLM (Gemini, xAI, Claude)
- **Tool Marketplace**: Publish our tools for others

#### **Competitive Advantages:**
- **Vercel for AI**: Deploy MCP servers in minutes
- **Standardization**: MCP protocol ensures compatibility
- **Community**: Access to ecosystem of tools

### **Architecture with Dedalus**

```
User Request
    ↓
Intent Agent (xAI) → Weights
    ↓
Planner Agent → Strategy
    ↓
Local Placer Agent → Fast Optimization
    ↙️         ↘️
Dedalus MCP    Dedalus MCP
Placement      Thermal
Scorer         Simulator
    ↘️         ↙️
Verifier Agent → Validation
    ↓
Exporter Agent → KiCad Export
```

### **MCP Server Details**

#### **PlacementScorerMCP**
```json
{
  "name": "placement_scorer",
  "tools": ["score_delta"],
  "hosted_on": "dedalus_labs",
  "computation": "incremental_scorer"
}
```

#### **ThermalSimulatorMCP**
```json
{
  "name": "thermal_simulator",
  "tools": ["generate_heatmap"],
  "hosted_on": "dedalus_labs",
  "computation": "gaussian_heat_convolution"
}
```

#### **KiCadExporterMCP**
```json
{
  "name": "kicad_exporter",
  "tools": ["export_kicad"],
  "hosted_on": "dedalus_labs",
  "format": "kicad_pcb"
}
```

### **Testing Dedalus Integration**

```bash
# Test MCP servers
python test_mcp_servers.py

# Test Dedalus client
python -c "from backend.ai.dedalus_client import DedalusClient; DedalusClient()"

# Test full pipeline
python test_full_stack.py
```

### **Production Deployment**

1. **Deploy MCP Servers** via Dedalus dashboard
2. **Configure API Keys** in environment
3. **Scale Automatically** - Dedalus handles load
4. **Monitor** via Dedalus dashboard
5. **Publish Tools** to marketplace (optional)

### **Fallback Mode**

If Dedalus unavailable, system falls back to:
- Local MCP server execution
- Direct agent communication
- Reduced functionality but still operational

---

**Dedalus Labs enables our multi-agent, MCP-based architecture without infrastructure overhead!** 🚀
