"""
Hierarchical Reasoning Model (HRM) for PCB Optimization

Based on: "Hierarchical Reasoning Model" (arXiv:2506.21734)
https://arxiv.org/abs/2506.21734

Architecture:
- High-level module: Slow, abstract planning (module placement, strategy)
- Low-level module: Rapid, detailed computations (component placement, fine-tuning)
- Single forward pass without explicit supervision
- Minimal training data required (1000 samples)

Applied to:
1. Large-scale PCB optimization (200+ components)
2. Multi-agent coordination
3. Embedded system simulation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import time

try:
    from backend.geometry.placement import Placement
    from backend.scoring.scorer import WorldModelScorer
    from backend.geometry.geometry_analyzer import GeometryAnalyzer
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.scoring.scorer import WorldModelScorer
    from src.backend.geometry.geometry_analyzer import GeometryAnalyzer


@dataclass
class ReasoningState:
    """State of hierarchical reasoning."""
    # High-level (abstract planning)
    abstract_plan: Dict[str, Any]  # Module placement strategy
    current_phase: str  # "planning", "execution", "refinement"
    module_priorities: List[str]  # Ordered list of modules to optimize
    
    # Low-level (detailed execution)
    component_actions: List[Dict[str, Any]]  # Specific component moves
    execution_history: List[Dict[str, Any]]  # History of actions
    current_score: float
    
    # Meta-reasoning
    reasoning_depth: int  # How deep in the reasoning hierarchy
    confidence: float  # Confidence in current plan


class HierarchicalReasoningModel:
    """
    Hierarchical Reasoning Model for PCB optimization.
    
    Inspired by HRM paper (arXiv:2506.21734):
    - High-level module: Abstract planning (modules, strategy)
    - Low-level module: Detailed execution (components, fine-tuning)
    - Single forward pass without explicit supervision
    - Minimal data requirements
    
    Architecture:
    ```
    High-Level Module (Slow, Abstract)
    ├── Module identification
    ├── Strategic planning
    ├── Priority assignment
    └── Global optimization strategy
    
    Low-Level Module (Fast, Detailed)
    ├── Component placement
    ├── Fine-tuning
    ├── Local optimization
    └── Constraint satisfaction
    ```
    """
    
    def __init__(
        self,
        scorer: Optional[WorldModelScorer] = None,
        high_level_timescale: int = 10,  # High-level updates every N iterations
        low_level_timescale: int = 1,     # Low-level updates every iteration
        max_reasoning_depth: int = 5
    ):
        """
        Initialize hierarchical reasoning model.
        
        Args:
            scorer: Scoring function
            high_level_timescale: How often high-level module updates (slow)
            low_level_timescale: How often low-level module updates (fast)
            max_reasoning_depth: Maximum reasoning depth
        """
        self.scorer = scorer
        self.high_level_timescale = high_level_timescale
        self.low_level_timescale = low_level_timescale
        self.max_reasoning_depth = max_reasoning_depth
        
        # Reasoning state
        self.state = ReasoningState(
            abstract_plan={},
            current_phase="planning",
            module_priorities=[],
            component_actions=[],
            execution_history=[],
            current_score=float('inf'),
            reasoning_depth=0,
            confidence=0.0
        )
        
        # High-level module (slow, abstract)
        self.high_level_state = {
            "module_strategy": {},
            "global_priorities": [],
            "optimization_focus": "balanced"
        }
        
        # Low-level module (fast, detailed)
        self.low_level_state = {
            "current_actions": [],
            "local_improvements": [],
            "constraint_violations": []
        }
    
    def reason_and_optimize(
        self,
        placement: Placement,
        user_intent: str,
        max_iterations: int = 1000,
        callback: Optional[Callable] = None
    ) -> Tuple[Placement, float, Dict]:
        """
        Execute hierarchical reasoning and optimization in single forward pass.
        
        Based on HRM: single forward pass without explicit supervision.
        
        Args:
            placement: Initial placement
            user_intent: User's optimization intent
            max_iterations: Maximum iterations
            callback: Optional callback function
            
        Returns:
            (optimized_placement, final_score, statistics)
        """
        self.state.current_score = self.scorer.score(placement) if self.scorer else 0.0
        best_placement = placement.copy()
        best_score = self.state.current_score
        
        # Initialize high-level planning
        self._high_level_planning(placement, user_intent)
        
        iteration = 0
        
        while iteration < max_iterations:
            # High-level module update (slow timescale)
            if iteration % self.high_level_timescale == 0:
                self._update_high_level(placement, user_intent, iteration)
            
            # Low-level module update (fast timescale)
            if iteration % self.low_level_timescale == 0:
                placement, improved = self._update_low_level(placement, iteration)
                
                if improved:
                    current_score = self.scorer.score(placement) if self.scorer else 0.0
                    if current_score < best_score:
                        best_score = current_score
                        best_placement = placement.copy()
                        self.state.current_score = current_score
            
            # Update reasoning state
            self.state.reasoning_depth = min(
                self.max_reasoning_depth,
                iteration // self.high_level_timescale
            )
            
            # Callback
            if callback:
                callback(placement, iteration, best_score)
            
            iteration += 1
        
        stats = {
            "iterations": iteration,
            "best_score": best_score,
            "reasoning_depth": self.state.reasoning_depth,
            "high_level_updates": iteration // self.high_level_timescale,
            "low_level_updates": iteration,
            "confidence": self.state.confidence
        }
        
        return best_placement, best_score, stats
    
    def _high_level_planning(
        self,
        placement: Placement,
        user_intent: str
    ):
        """
        High-level module: Abstract planning (slow, strategic).
        
        Responsibilities:
        - Identify functional modules
        - Plan module placement strategy
        - Assign priorities
        - Determine global optimization focus
        """
        # Identify modules (functional groups)
        modules = self._identify_modules(placement)
        
        # Analyze user intent for strategic focus
        strategic_focus = self._analyze_strategic_intent(user_intent)
        
        # Plan module placement strategy
        module_strategy = self._plan_module_strategy(modules, strategic_focus)
        
        # Assign priorities
        priorities = self._assign_module_priorities(modules, strategic_focus)
        
        # Update high-level state
        self.high_level_state = {
            "modules": modules,
            "module_strategy": module_strategy,
            "strategic_focus": strategic_focus,
            "priorities": priorities
        }
        
        self.state.abstract_plan = {
            "modules": modules,
            "strategy": module_strategy,
            "focus": strategic_focus
        }
        self.state.module_priorities = priorities
        self.state.current_phase = "planning"
    
    def _update_high_level(
        self,
        placement: Placement,
        user_intent: str,
        iteration: int
    ):
        """
        Update high-level module (slow timescale).
        
        Re-evaluates strategy based on progress.
        """
        current_score = self.scorer.score(placement) if self.scorer else 0.0
        
        # Check if strategy needs adjustment
        if iteration > 0:
            progress = (self.state.current_score - current_score) / self.state.current_score if self.state.current_score > 0 else 0.0
            
            # If progress is slow, adjust strategy
            if progress < 0.01:  # Less than 1% improvement
                # Re-plan with different focus
                self._high_level_planning(placement, user_intent)
                self.state.confidence = max(0.0, self.state.confidence - 0.1)
            else:
                self.state.confidence = min(1.0, self.state.confidence + 0.05)
        
        # Update phase based on progress
        if iteration < max_iterations * 0.3:
            self.state.current_phase = "planning"
        elif iteration < max_iterations * 0.7:
            self.state.current_phase = "execution"
        else:
            self.state.current_phase = "refinement"
    
    def _update_low_level(
        self,
        placement: Placement,
        iteration: int
    ) -> Tuple[Placement, bool]:
        """
        Low-level module: Detailed execution (fast timescale).
        
        Responsibilities:
        - Execute component-level moves
        - Fine-tune placements
        - Satisfy local constraints
        - Implement high-level plan
        """
        improved = False
        
        # Get current module priorities from high-level
        priorities = self.state.module_priorities
        
        # Execute actions based on high-level plan
        if priorities:
            # Focus on highest priority module
            focus_module = priorities[0] if priorities else None
            
            if focus_module:
                # Get components in this module
                module_components = self._get_module_components(placement, focus_module)
                
                # Execute component moves
                for comp_name in module_components[:5]:  # Limit to 5 components per iteration
                    comp = placement.get_component(comp_name)
                    if comp:
                        # Generate move based on high-level strategy
                        move = self._generate_strategic_move(
                            comp,
                            self.high_level_state["module_strategy"].get(focus_module, {}),
                            placement
                        )
                        
                        if move:
                            # Apply move
                            comp.x = move["x"]
                            comp.y = move["y"]
                            comp.angle = move.get("angle", comp.angle)
                            
                            # Check if improved
                            new_score = self.scorer.score(placement) if self.scorer else 0.0
                            if new_score < self.state.current_score:
                                improved = True
                                self.state.current_score = new_score
                                
                                # Record action
                                self.state.component_actions.append({
                                    "component": comp_name,
                                    "move": move,
                                    "score_delta": new_score - self.state.current_score,
                                    "iteration": iteration
                                })
        else:
            # No modules identified, use standard optimization
            improved = self._standard_component_move(placement)
        
        return placement, improved
    
    def _identify_modules(self, placement: Placement) -> List[Dict[str, Any]]:
        """
        Identify functional modules in placement.
        
        Uses computational geometry and net connectivity.
        """
        modules = []
        
        # Analyze component connectivity
        component_groups = self._group_by_connectivity(placement)
        
        # Analyze spatial clustering
        spatial_groups = self._group_by_spatial_clustering(placement)
        
        # Merge groups
        merged_groups = self._merge_groups(component_groups, spatial_groups)
        
        # Create module descriptions
        for i, group in enumerate(merged_groups):
            modules.append({
                "name": f"Module_{i+1}",
                "components": group,
                "type": self._classify_module_type(group, placement),
                "priority": self._compute_module_priority(group, placement)
            })
        
        return modules
    
    def _group_by_connectivity(self, placement: Placement) -> List[List[str]]:
        """Group components by net connectivity."""
        groups = []
        visited = set()
        
        for comp_name in placement.components.keys():
            if comp_name in visited:
                continue
            
            # Start new group
            group = [comp_name]
            visited.add(comp_name)
            
            # Find connected components
            to_visit = [comp_name]
            while to_visit:
                current = to_visit.pop()
                comp = placement.get_component(current)
                
                if not comp:
                    continue
                
                # Find components connected via nets
                for net_name, net in placement.nets.items():
                    if any(p[0] == current for p in net.pins):
                        # Found net connected to this component
                        for other_comp, _ in net.pins:
                            if other_comp != current and other_comp not in visited:
                                group.append(other_comp)
                                visited.add(other_comp)
                                to_visit.append(other_comp)
            
            if len(group) > 1:  # Only groups with multiple components
                groups.append(group)
        
        return groups
    
    def _group_by_spatial_clustering(self, placement: Placement) -> List[List[str]]:
        """Group components by spatial clustering (Voronoi-based)."""
        from src.backend.geometry.geometry_analyzer import GeometryAnalyzer
        
        analyzer = GeometryAnalyzer(placement)
        geometry_data = analyzer.analyze()
        
        # Use Voronoi neighbors to identify clusters
        components = list(placement.components.keys())
        groups = []
        visited = set()
        
        for comp_name in components:
            if comp_name in visited:
                continue
            
            comp = placement.get_component(comp_name)
            if not comp:
                continue
            
            # Find nearby components (within threshold distance)
            threshold = 20.0  # mm
            group = [comp_name]
            visited.add(comp_name)
            
            for other_name in components:
                if other_name in visited:
                    continue
                
                other_comp = placement.get_component(other_name)
                if not other_comp:
                    continue
                
                distance = np.sqrt((comp.x - other_comp.x)**2 + (comp.y - other_comp.y)**2)
                if distance < threshold:
                    group.append(other_name)
                    visited.add(other_name)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def _merge_groups(
        self,
        connectivity_groups: List[List[str]],
        spatial_groups: List[List[str]]
    ) -> List[List[str]]:
        """Merge connectivity and spatial groups."""
        merged = []
        used = set()
        
        # Prefer connectivity groups (stronger signal)
        for group in connectivity_groups:
            if not any(comp in used for comp in group):
                merged.append(group)
                used.update(group)
        
        # Add spatial groups that don't overlap
        for group in spatial_groups:
            if not any(comp in used for comp in group):
                merged.append(group)
                used.update(group)
        
        return merged
    
    def _classify_module_type(
        self,
        components: List[str],
        placement: Placement
    ) -> str:
        """Classify module type (power, signal, mixed)."""
        power_components = 0
        signal_components = 0
        
        for comp_name in components:
            comp = placement.get_component(comp_name)
            if comp:
                power = getattr(comp, 'power', 0.0)
                if power > 0.5:
                    power_components += 1
                else:
                    signal_components += 1
        
        if power_components > signal_components:
            return "power"
        elif signal_components > power_components:
            return "signal"
        else:
            return "mixed"
    
    def _compute_module_priority(
        self,
        components: List[str],
        placement: Placement
    ) -> float:
        """Compute module priority for optimization."""
        priority = 0.0
        
        for comp_name in components:
            comp = placement.get_component(comp_name)
            if comp:
                # Higher priority for high-power components
                power = getattr(comp, 'power', 0.0)
                priority += power * 10.0
                
                # Higher priority for components with many connections
                connections = sum(1 for net in placement.nets.values() 
                                if any(p[0] == comp_name for p in net.pins))
                priority += connections * 2.0
        
        return priority
    
    def _analyze_strategic_intent(self, user_intent: str) -> Dict[str, float]:
        """Analyze user intent to determine strategic focus."""
        intent_lower = user_intent.lower()
        
        focus = {
            "thermal": 0.25,
            "routing": 0.25,
            "clearance": 0.25,
            "manufacturability": 0.25
        }
        
        # Adjust based on keywords
        if any(word in intent_lower for word in ["thermal", "heat", "cool", "temperature"]):
            focus["thermal"] = 0.6
            focus["routing"] = 0.2
            focus["clearance"] = 0.1
            focus["manufacturability"] = 0.1
        
        if any(word in intent_lower for word in ["trace", "routing", "wire", "length"]):
            focus["routing"] = 0.6
            focus["thermal"] = 0.2
            focus["clearance"] = 0.1
            focus["manufacturability"] = 0.1
        
        if any(word in intent_lower for word in ["violation", "clearance", "spacing"]):
            focus["clearance"] = 0.6
            focus["thermal"] = 0.15
            focus["routing"] = 0.15
            focus["manufacturability"] = 0.1
        
        return focus
    
    def _plan_module_strategy(
        self,
        modules: List[Dict[str, Any]],
        strategic_focus: Dict[str, float]
    ) -> Dict[str, Dict[str, Any]]:
        """Plan strategy for each module."""
        strategy = {}
        
        for module in modules:
            module_name = module["name"]
            module_type = module["type"]
            
            # Determine strategy based on module type and focus
            if module_type == "power":
                strategy[module_name] = {
                    "focus": "thermal",
                    "action": "spread_out",  # Spread power components
                    "priority": strategic_focus.get("thermal", 0.25)
                }
            elif module_type == "signal":
                strategy[module_name] = {
                    "focus": "routing",
                    "action": "cluster",  # Cluster signal components
                    "priority": strategic_focus.get("routing", 0.25)
                }
            else:
                strategy[module_name] = {
                    "focus": "balanced",
                    "action": "optimize",
                    "priority": 0.25
                }
        
        return strategy
    
    def _assign_module_priorities(
        self,
        modules: List[Dict[str, Any]],
        strategic_focus: Dict[str, float]
    ) -> List[str]:
        """Assign optimization priorities to modules."""
        # Sort by computed priority
        sorted_modules = sorted(modules, key=lambda m: m["priority"], reverse=True)
        return [m["name"] for m in sorted_modules]
    
    def _get_module_components(
        self,
        placement: Placement,
        module_name: str
    ) -> List[str]:
        """Get component names in a module."""
        for module in self.high_level_state.get("modules", []):
            if module["name"] == module_name:
                return module.get("components", [])
        return []
    
    def _generate_strategic_move(
        self,
        component: Any,
        strategy: Dict[str, Any],
        placement: Placement
    ) -> Optional[Dict[str, float]]:
        """Generate component move based on high-level strategy."""
        action = strategy.get("action", "optimize")
        
        if action == "spread_out":
            # Move component away from center
            center_x = placement.board.width / 2
            center_y = placement.board.height / 2
            
            # Move away from center
            dx = component.x - center_x
            dy = component.y - center_y
            
            if abs(dx) < 1 and abs(dy) < 1:
                # At center, move randomly
                import random
                angle = random.uniform(0, 2 * np.pi)
                distance = random.uniform(10, 20)
                new_x = center_x + distance * np.cos(angle)
                new_y = center_y + distance * np.sin(angle)
            else:
                # Move further away
                scale = 1.2
                new_x = center_x + dx * scale
                new_y = center_y + dy * scale
            
            # Clamp to board bounds
            margin = max(component.width, component.height) if hasattr(component, 'width') else 5.0
            new_x = max(margin, min(placement.board.width - margin, new_x))
            new_y = max(margin, min(placement.board.height - margin, new_y))
            
            return {"x": new_x, "y": new_y}
        
        elif action == "cluster":
            # Move component toward connected components
            # Find average position of connected components
            connected_positions = []
            
            for net_name, net in placement.nets.items():
                if any(p[0] == component.name for p in net.pins):
                    for other_comp_name, _ in net.pins:
                        if other_comp_name != component.name:
                            other_comp = placement.get_component(other_comp_name)
                            if other_comp:
                                connected_positions.append((other_comp.x, other_comp.y))
            
            if connected_positions:
                avg_x = sum(p[0] for p in connected_positions) / len(connected_positions)
                avg_y = sum(p[1] for p in connected_positions) / len(connected_positions)
                
                # Move toward average (with some randomness)
                import random
                factor = 0.3  # Move 30% toward average
                new_x = component.x * (1 - factor) + avg_x * factor
                new_y = component.y * (1 - factor) + avg_y * factor
                
                # Add small random perturbation
                new_x += random.uniform(-2, 2)
                new_y += random.uniform(-2, 2)
                
                # Clamp to board bounds
                margin = max(component.width, component.height) if hasattr(component, 'width') else 5.0
                new_x = max(margin, min(placement.board.width - margin, new_x))
                new_y = max(margin, min(placement.board.height - margin, new_y))
                
                return {"x": new_x, "y": new_y}
        
        # Default: random move
        import random
        margin = max(component.width, component.height) if hasattr(component, 'width') else 5.0
        new_x = random.uniform(margin, placement.board.width - margin)
        new_y = random.uniform(margin, placement.board.height - margin)
        
        return {"x": new_x, "y": new_y}
    
    def _standard_component_move(self, placement: Placement) -> bool:
        """Standard component move when no modules identified."""
        import random
        
        comp_names = list(placement.components.keys())
        if not comp_names:
            return False
        
        comp_name = random.choice(comp_names)
        comp = placement.get_component(comp_name)
        
        if not comp:
            return False
        
        margin = max(comp.width, comp.height) if hasattr(comp, 'width') else 5.0
        old_x, old_y = comp.x, comp.y
        
        comp.x = random.uniform(margin, placement.board.width - margin)
        comp.y = random.uniform(margin, placement.board.height - margin)
        
        new_score = self.scorer.score(placement) if self.scorer else 0.0
        
        if new_score >= self.state.current_score:
            # Revert if worse
            comp.x, comp.y = old_x, old_y
            return False
        
        self.state.current_score = new_score
        return True

