"""
PCB Simulation and Testing Automation

Automates thermal simulation, signal integrity analysis, EMI testing, and DFM checks.
Uses ML techniques with xAI for intelligent test generation and result interpretation.
"""

import os
import json
from typing import Dict, List, Optional, Any
import numpy as np
try:
    from backend.ai.xai_client import XAIClient
except ImportError:
    from src.backend.ai.xai_client import XAIClient


class SimulationAutomation:
    """Automates PCB simulation and testing workflows."""
    
    def __init__(self):
        """Initialize simulation automation."""
        self.xai_client = XAIClient()
        self.simulation_results = {}
    
    def generate_test_plan(self, placement_data: Dict, design_intent: str) -> Dict:
        """
        Generate comprehensive test plan using xAI reasoning.
        
        Args:
            placement_data: PCB placement data
            design_intent: Design intent/goals
        
        Returns:
            Test plan with recommended simulations
        """
        # Analyze design characteristics
        components = placement_data.get("components", [])
        nets = placement_data.get("nets", [])
        board = placement_data.get("board", {})
        
        high_power_count = sum(1 for c in components if c.get("power", 0) > 1.0)
        high_freq_nets = len([n for n in nets if "clock" in n.get("name", "").lower() or "signal" in n.get("name", "").lower()])
        
        # Use xAI to generate test plan
        prompt = f"""
        Generate a comprehensive PCB test plan for this design:
        
        Design Intent: {design_intent}
        Board Size: {board.get('width', 0)}mm x {board.get('height', 0)}mm
        Components: {len(components)}
        High-Power Components: {high_power_count}
        High-Frequency Nets: {high_freq_nets}
        
        Recommend simulations and tests needed:
        1. Thermal analysis (if high-power components)
        2. Signal integrity (if high-frequency nets)
        3. EMI/EMC analysis
        4. Design for Manufacturing (DFM) checks
        5. Power integrity
        6. Mechanical stress
        
        Return JSON with test categories and specific tests.
        """
        
        try:
            response = self.xai_client._call_api(prompt, max_tokens=1000)
            # Parse response to extract test plan
            test_plan = self._parse_test_plan(response)
        except Exception as e:
            # Fallback test plan
            test_plan = self._generate_fallback_test_plan(components, nets, high_power_count)
        
        return test_plan
    
    def _parse_test_plan(self, xai_response: str) -> Dict:
        """Parse xAI response into structured test plan."""
        # Try to extract JSON from response
        try:
            # Look for JSON in response
            import re
            json_match = re.search(r'\{.*\}', xai_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback: structured response
        return {
            "thermal_analysis": True,
            "signal_integrity": True,
            "emi_analysis": True,
            "dfm_checks": True,
            "power_integrity": True
        }
    
    def _generate_fallback_test_plan(self, components: List, nets: List, high_power_count: int) -> Dict:
        """Generate fallback test plan based on design characteristics."""
        return {
            "thermal_analysis": high_power_count > 0,
            "signal_integrity": len(nets) > 5,
            "emi_analysis": True,
            "dfm_checks": True,
            "power_integrity": high_power_count > 0,
            "mechanical_stress": len(components) > 20
        }
    
    def run_thermal_simulation(self, placement_data: Dict) -> Dict:
        """
        Run thermal simulation (simplified model).
        
        In production, this would interface with ANSYS, COMSOL, or similar.
        """
        components = placement_data.get("components", [])
        board = placement_data.get("board", {})
        
        # Simplified thermal model
        grid_size = 50
        x_grid = np.linspace(0, board.get("width", 100), grid_size)
        y_grid = np.linspace(0, board.get("height", 100), grid_size)
        temperature_map = np.zeros((grid_size, grid_size))
        ambient_temp = 25.0  # Celsius
        
        # Calculate temperature distribution
        for comp in components:
            x, y = comp.get("x", 0), comp.get("y", 0)
            power = comp.get("power", 0)
            if power > 0:
                # Thermal resistance model
                R_thermal = 50.0  # K/W (typical for PCB)
                T_junction = ambient_temp + power * R_thermal
                
                # Gaussian distribution
                for i, gx in enumerate(x_grid):
                    for j, gy in enumerate(y_grid):
                        dist = np.sqrt((gx - x)**2 + (gy - y)**2)
                        temp_contribution = (T_junction - ambient_temp) * np.exp(-(dist**2) / (2 * 20**2))
                        temperature_map[j, i] += temp_contribution
        
        temperature_map += ambient_temp
        
        # Find hotspots
        max_temp = np.max(temperature_map)
        hotspot_threshold = ambient_temp + 30.0
        hotspots = np.where(temperature_map > hotspot_threshold)
        
        return {
            "temperature_map": temperature_map.tolist(),
            "max_temperature": float(max_temp),
            "min_temperature": float(np.min(temperature_map)),
            "hotspot_count": len(hotspots[0]),
            "hotspot_locations": [
                {"x": float(x_grid[i]), "y": float(y_grid[j]), "temp": float(temperature_map[j, i])}
                for j, i in zip(hotspots[0], hotspots[1])
            ],
            "pass": max_temp < 85.0  # Typical max operating temp
        }
    
    def run_signal_integrity_analysis(self, placement_data: Dict) -> Dict:
        """
        Run signal integrity analysis (simplified model).
        
        In production, this would interface with HyperLynx, SIwave, or similar.
        """
        nets = placement_data.get("nets", [])
        components = placement_data.get("components", [])
        
        # Simplified SI analysis
        si_issues = []
        
        for net in nets:
            net_name = net.get("name", "")
            net_pins = net.get("pins", [])
            
            if len(net_pins) < 2:
                continue
            
            # Calculate trace length (Manhattan distance)
            positions = []
            for pin_ref in net_pins:
                comp_name = pin_ref[0] if isinstance(pin_ref, list) else pin_ref
                comp = next((c for c in components if c.get("name") == comp_name), None)
                if comp:
                    positions.append([comp.get("x", 0), comp.get("y", 0)])
            
            if len(positions) >= 2:
                total_length = 0
                for i in range(len(positions) - 1):
                    x0, y0 = positions[i]
                    x1, y1 = positions[i+1]
                    length = abs(x1 - x0) + abs(y1 - y0)  # Manhattan
                    total_length += length
                
                # Check for SI issues
                if total_length > 100:  # mm
                    si_issues.append({
                        "net": net_name,
                        "length": total_length,
                        "issue": "Long trace - potential signal degradation",
                        "severity": "warning" if total_length < 150 else "error"
                    })
                
                # Check for high-frequency nets
                if "clock" in net_name.lower() or "signal" in net_name.lower():
                    if total_length > 50:
                        si_issues.append({
                            "net": net_name,
                            "length": total_length,
                            "issue": "High-frequency net too long",
                            "severity": "error"
                        })
        
        return {
            "total_nets": len(nets),
            "analyzed_nets": len([n for n in nets if len(n.get("pins", [])) >= 2]),
            "issues": si_issues,
            "issue_count": len(si_issues),
            "pass": len(si_issues) == 0
        }
    
    def run_dfm_checks(self, placement_data: Dict) -> Dict:
        """
        Run Design for Manufacturing (DFM) checks.
        
        Checks: minimum clearance, component spacing, via placement, etc.
        """
        components = placement_data.get("components", [])
        board = placement_data.get("board", {})
        clearance = board.get("clearance", 0.5)
        
        dfm_issues = []
        
        # Check component spacing
        comp_list = list(components)
        for i, c1 in enumerate(comp_list):
            for c2 in comp_list[i+1:]:
                x1, y1 = c1.get("x", 0), c1.get("y", 0)
                x2, y2 = c2.get("x", 0), c2.get("y", 0)
                dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                min_dist = (c1.get("width", 5) + c2.get("width", 5)) / 2 + clearance
                
                if dist < min_dist:
                    dfm_issues.append({
                        "type": "clearance_violation",
                        "components": [c1.get("name"), c2.get("name")],
                        "distance": dist,
                        "required": min_dist,
                        "severity": "error"
                    })
        
        # Check board edge clearance
        board_width = board.get("width", 100)
        board_height = board.get("height", 100)
        edge_clearance = 2.0  # mm
        
        for comp in components:
            x, y = comp.get("x", 0), comp.get("y", 0)
            width, height = comp.get("width", 5), comp.get("height", 5)
            
            if x - width/2 < edge_clearance or x + width/2 > board_width - edge_clearance:
                dfm_issues.append({
                    "type": "edge_clearance",
                    "component": comp.get("name"),
                    "issue": "Too close to board edge (X)",
                    "severity": "warning"
                })
            
            if y - height/2 < edge_clearance or y + height/2 > board_height - edge_clearance:
                dfm_issues.append({
                    "type": "edge_clearance",
                    "component": comp.get("name"),
                    "issue": "Too close to board edge (Y)",
                    "severity": "warning"
                })
        
        return {
            "total_checks": len(components) * (len(components) - 1) // 2 + len(components),
            "issues": dfm_issues,
            "error_count": len([i for i in dfm_issues if i.get("severity") == "error"]),
            "warning_count": len([i for i in dfm_issues if i.get("severity") == "warning"]),
            "pass": len([i for i in dfm_issues if i.get("severity") == "error"]) == 0
        }
    
    def interpret_results_with_ai(self, simulation_results: Dict, design_intent: str) -> Dict:
        """
        Use xAI to interpret simulation results and provide recommendations.
        
        Args:
            simulation_results: Results from all simulations
            design_intent: Original design intent
        
        Returns:
            AI interpretation with recommendations
        """
        prompt = f"""
        Analyze these PCB simulation results and provide recommendations:
        
        Design Intent: {design_intent}
        
        Simulation Results:
        {json.dumps(simulation_results, indent=2)}
        
        Provide:
        1. Summary of issues found
        2. Priority ranking of issues
        3. Specific recommendations for each issue
        4. Optimization suggestions
        
        Return structured JSON response.
        """
        
        try:
            response = self.xai_client._call_api(prompt, max_tokens=1500)
            interpretation = self._parse_interpretation(response)
        except Exception as e:
            interpretation = {
                "summary": "Analysis complete",
                "recommendations": ["Review simulation results manually"],
                "error": str(e)
            }
        
        return interpretation
    
    def _parse_interpretation(self, xai_response: str) -> Dict:
        """Parse xAI interpretation response."""
        try:
            import re
            json_match = re.search(r'\{.*\}', xai_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {
            "summary": xai_response[:500],
            "recommendations": []
        }
    
    def run_full_simulation_suite(self, placement_data: Dict, design_intent: str) -> Dict:
        """
        Run complete simulation suite and return comprehensive results.
        
        Args:
            placement_data: PCB placement data
            design_intent: Design intent
        
        Returns:
            Complete simulation results with AI interpretation
        """
        # Generate test plan
        test_plan = self.generate_test_plan(placement_data, design_intent)
        
        results = {
            "test_plan": test_plan,
            "simulations": {}
        }
        
        # Run simulations based on test plan
        if test_plan.get("thermal_analysis"):
            results["simulations"]["thermal"] = self.run_thermal_simulation(placement_data)
        
        if test_plan.get("signal_integrity"):
            results["simulations"]["signal_integrity"] = self.run_signal_integrity_analysis(placement_data)
        
        if test_plan.get("dfm_checks"):
            results["simulations"]["dfm"] = self.run_dfm_checks(placement_data)
        
        # AI interpretation
        results["ai_interpretation"] = self.interpret_results_with_ai(results["simulations"], design_intent)
        
        # Overall pass/fail
        all_pass = all(
            sim.get("pass", False) 
            for sim in results["simulations"].values() 
            if isinstance(sim, dict) and "pass" in sim
        )
        results["overall_pass"] = all_pass
        
        return results

