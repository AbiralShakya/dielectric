# Dielectric Quick Reference

## ✅ What's Fixed

1. **Deterministic Optimization**: Same intent = same result (seed-based)
2. **KiCad Net Connections**: Pads properly connected to nets
3. **Large PCB Support**: Knowledge graph + hierarchical abstraction
4. **Real Constraints**: Fabrication limits integrated
5. **Knowledge Graph**: Component relationships and design patterns

## 🚀 How to Use

### Deterministic Optimization
```python
# Same intent always gives same result
result = await orchestrator.optimize_fast(placement, "minimize trace length")
```

### Knowledge Graph
```python
from src.backend.knowledge.component_graph import ComponentKnowledgeGraph

kg = ComponentKnowledgeGraph.from_placement(placement)
modules = kg.identify_modules(placement)  # Auto-identifies modules
hints = kg.get_placement_hints("U1")  # Get placement suggestions
```

### Fabrication Constraints
```python
from src.backend.constraints.pcb_fabrication import FabricationConstraints, ConstraintValidator

constraints = FabricationConstraints(
    min_trace_width=0.15,  # mm (6 mil)
    min_trace_spacing=0.15,  # mm
    min_pad_to_pad_clearance=0.2  # mm (8 mil)
)

validator = ConstraintValidator(constraints)
result = validator.validate_placement(placement, knowledge_graph=kg)
```

## 📊 Multi-Agent Workflow

1. **DesignGeneratorAgent**: Natural language → PCB design
2. **IntentAgent**: Intent + computational geometry → weights
3. **LocalPlacerAgent**: Weights → optimized placement (deterministic)
4. **VerifierAgent**: Placement → constraint validation
5. **ErrorFixerAgent**: Violations → automatic fixes
6. **ExporterAgent**: Placement → KiCad file

## 🔧 Dedalus Deployment

**Files Ready:**
- `dedalus.json` - Configuration
- `dedalus_entrypoint.py` - Entrypoint

**To Deploy:**
1. `git add dedalus.json dedalus_entrypoint.py && git commit -m "Add Dedalus" && git push`
2. Set `XAI_API_KEY` and `DEDALUS_API_KEY` in Dedalus dashboard
3. Redeploy

**Note**: System works perfectly without Dedalus (all agents run locally)

## 📝 Key Files

- `src/backend/knowledge/component_graph.py` - Knowledge graph
- `src/backend/constraints/pcb_fabrication.py` - Constraints
- `src/backend/optimization/simulated_annealing.py` - Deterministic optimizer
- `src/backend/export/kicad_exporter.py` - KiCad export with nets

## 🎯 For Large PCBs (100+ components)

1. Knowledge graph identifies modules automatically
2. Hierarchical optimization (modules → components)
3. Computational geometry validates at each level
4. Fabrication constraints enforced
5. Multi-layer KiCad export

---

**Dielectric**: Enterprise AI for PCB Design

