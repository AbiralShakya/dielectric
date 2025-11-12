"""
Differentiable Geometry Operations
Makes geometry operations differentiable for gradient-based optimization.

Based on:
- Loper & Black, "OpenDR: An Approximate Differentiable Renderer" (2014)
- Blinn, "A Generalization of Algebraic Surface Drawing" (1982)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial import Voronoi as ScipyVoronoi
import math


def soft_voronoi(positions: torch.Tensor, query_points: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Differentiable Voronoi approximation using softmax.
    
    Instead of hard assignment (nearest neighbor), use soft assignment.
    
    Args:
        positions: Component positions (N, 2) or (N, 3)
        query_points: Query points (M, 2) or (M, 3)
        temperature: Temperature parameter (lower = sharper, higher = softer)
        
    Returns:
        Soft Voronoi cell assignments (M, N)
    """
    # Compute distances
    distances = torch.cdist(query_points, positions)  # (M, N)
    
    # Soft assignment (temperature controls sharpness)
    # Lower temperature = sharper (closer to hard Voronoi)
    # Higher temperature = softer (more uniform)
    weights = torch.softmax(-distances / temperature, dim=-1)  # (M, N)
    
    return weights


def differentiable_mst(positions: torch.Tensor, learnable_weights: Optional[torch.Tensor] = None,
                     temperature: float = 0.1) -> torch.Tensor:
    """
    Differentiable MST using learnable edge weights.
    
    Instead of fixed distances, use learnable weights.
    
    Args:
        positions: Component positions (N, 2)
        learnable_weights: Optional learnable weights (N, N)
        temperature: Temperature for soft MST
        
    Returns:
        Soft MST edge probabilities (N, N)
    """
    # Compute distances
    distances = torch.cdist(positions, positions)  # (N, N)
    
    # Apply learnable weights if provided
    if learnable_weights is not None:
        edge_weights = distances * learnable_weights
    else:
        edge_weights = distances
    
    # Soft MST (differentiable approximation)
    # Use softmax over edges to approximate MST
    # In practice, would use more sophisticated differentiable MST algorithm
    edge_probs = torch.softmax(-edge_weights / temperature, dim=-1)
    
    # Make symmetric
    edge_probs = (edge_probs + edge_probs.t()) / 2
    
    return edge_probs


class DifferentiablePlacementOptimizer(nn.Module):
    """
    Optimize placement using gradient descent through geometry.
    
    Makes geometry operations differentiable for end-to-end optimization.
    """
    
    def __init__(self, initial_placement: Dict, target_metrics: Optional[Dict] = None):
        """
        Initialize optimizer.
        
        Args:
            initial_placement: Initial placement dictionary
            target_metrics: Target geometry metrics (optional)
        """
        super().__init__()
        
        # Extract component positions
        components = initial_placement.get("components", [])
        positions = []
        for comp in components:
            positions.append([comp.get("x", 0.0), comp.get("y", 0.0)])
        
        # Make positions learnable
        self.placement = nn.Parameter(torch.tensor(positions, dtype=torch.float32))
        
        # Store component metadata
        self.component_names = [comp.get("name", f"comp_{i}") for i, comp in enumerate(components)]
        self.board_width = initial_placement.get("board", {}).get("width", 100.0)
        self.board_height = initial_placement.get("board", {}).get("height", 100.0)
        
        # Target metrics
        self.target_metrics = target_metrics or {}
    
    def compute_voronoi_variance(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute Voronoi cell variance (measure of distribution uniformity).
        
        Args:
            positions: Component positions (N, 2)
            
        Returns:
            Voronoi variance (scalar)
        """
        # Sample query points
        num_samples = 1000
        query_x = torch.linspace(0, self.board_width, int(math.sqrt(num_samples)))
        query_y = torch.linspace(0, self.board_height, int(math.sqrt(num_samples)))
        query_points = torch.stack(torch.meshgrid(query_x, query_y, indexing='ij'), dim=-1)
        query_points = query_points.reshape(-1, 2)
        
        # Compute soft Voronoi
        voronoi_weights = soft_voronoi(positions, query_points, temperature=1.0)  # (M, N)
        
        # Compute cell areas (sum of weights per cell)
        cell_areas = torch.sum(voronoi_weights, dim=0)  # (N,)
        
        # Variance of cell areas (lower = more uniform)
        variance = torch.var(cell_areas)
        
        return variance
    
    def compute_mst_length(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute MST length.
        
        Args:
            positions: Component positions (N, 2)
            
        Returns:
            MST length (scalar)
        """
        # Compute distances
        distances = torch.cdist(positions, positions)  # (N, N)
        
        # Soft MST
        mst_probs = differentiable_mst(positions, temperature=0.1)  # (N, N)
        
        # Expected MST length (weighted sum)
        mst_length = torch.sum(mst_probs * distances) / 2  # Divide by 2 for symmetric
        
        return mst_length
    
    def compute_overlap_penalty(self, positions: torch.Tensor, component_sizes: List[Tuple[float, float]]) -> torch.Tensor:
        """
        Compute penalty for component overlaps.
        
        Args:
            positions: Component positions (N, 2)
            component_sizes: List of (width, height) for each component
            
        Returns:
            Overlap penalty (scalar)
        """
        penalty = torch.tensor(0.0, device=positions.device)
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                pos_i = positions[i]
                pos_j = positions[j]
                
                # Component sizes (simplified: assume all same size)
                size_i = component_sizes[i] if i < len(component_sizes) else (5.0, 5.0)
                size_j = component_sizes[j] if j < len(component_sizes) else (5.0, 5.0)
                
                # Distance between centers
                dist = torch.norm(pos_i - pos_j)
                
                # Minimum distance to avoid overlap
                min_dist = (size_i[0] + size_j[0]) / 2 + (size_i[1] + size_j[1]) / 2
                
                # Penalty if too close
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        return penalty
    
    def compute_boundary_penalty(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute penalty for components outside board boundaries.
        
        Args:
            positions: Component positions (N, 2)
            
        Returns:
            Boundary penalty (scalar)
        """
        penalty = torch.tensor(0.0, device=positions.device)
        margin = 5.0  # mm margin
        
        for pos in positions:
            x, y = pos[0], pos[1]
            
            # Penalty if outside bounds
            if x < margin:
                penalty += (margin - x) ** 2
            if x > self.board_width - margin:
                penalty += (x - (self.board_width - margin)) ** 2
            if y < margin:
                penalty += (margin - y) ** 2
            if y > self.board_height - margin:
                penalty += (y - (self.board_height - margin)) ** 2
        
        return penalty
    
    def forward(self) -> Dict[str, torch.Tensor]:
        """
        Compute geometry metrics for current placement.
        
        Returns:
            Dictionary with geometry metrics
        """
        positions = self.placement
        
        # Compute metrics
        voronoi_variance = self.compute_voronoi_variance(positions)
        mst_length = self.compute_mst_length(positions)
        
        # Component sizes (simplified)
        component_sizes = [(5.0, 5.0)] * len(positions)
        overlap_penalty = self.compute_overlap_penalty(positions, component_sizes)
        boundary_penalty = self.compute_boundary_penalty(positions)
        
        return {
            "voronoi_variance": voronoi_variance,
            "mst_length": mst_length,
            "overlap_penalty": overlap_penalty,
            "boundary_penalty": boundary_penalty
        }
    
    def optimize(self, num_iterations: int = 1000, lr: float = 1e-3) -> Dict:
        """
        Optimize placement using gradient descent.
        
        Args:
            num_iterations: Number of optimization iterations
            lr: Learning rate
            
        Returns:
            Optimized placement dictionary
        """
        optimizer = torch.optim.Adam([self.placement], lr=lr)
        
        for iteration in range(num_iterations):
            # Compute metrics
            metrics = self.forward()
            
            # Loss: match target metrics + penalties
            loss = torch.tensor(0.0, device=self.placement.device)
            
            # Target metrics
            if "voronoi_variance" in self.target_metrics:
                target_var = self.target_metrics["voronoi_variance"]
                loss += (metrics["voronoi_variance"] - target_var) ** 2
            
            if "mst_length" in self.target_metrics:
                target_mst = self.target_metrics["mst_length"]
                loss += (metrics["mst_length"] - target_mst) ** 2
            
            # Penalties
            loss += 10.0 * metrics["overlap_penalty"]
            loss += 10.0 * metrics["boundary_penalty"]
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Clamp positions to board bounds
            with torch.no_grad():
                self.placement.data[:, 0].clamp_(0, self.board_width)
                self.placement.data[:, 1].clamp_(0, self.board_height)
        
        # Convert back to placement dictionary
        optimized_placement = {
            "components": [],
            "board": {
                "width": self.board_width,
                "height": self.board_height
            }
        }
        
        positions_np = self.placement.detach().cpu().numpy()
        for i, (name, pos) in enumerate(zip(self.component_names, positions_np)):
            optimized_placement["components"].append({
                "name": name,
                "x": float(pos[0]),
                "y": float(pos[1]),
                "rotation": 0.0
            })
        
        return optimized_placement

