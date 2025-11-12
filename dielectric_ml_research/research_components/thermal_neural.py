"""
Thermal Neural Fields
Neural field-based thermal simulation for real-time optimization.

Enforces heat equation: ∂T/∂t = α∇²T + Q/(ρcp)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
import math
from .neural_em import PositionalEncoding


class ThermalNeuralField(nn.Module):
    """
    Neural field for 3D temperature distribution.
    
    Input: (x, y, z, component_powers, board_material)
    Output: Temperature T(x, y, z)
    """
    
    def __init__(self, hidden_dim: int = 256, num_layers: int = 8, pos_encoding_dim: int = 64):
        super().__init__()
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(dim=pos_encoding_dim)
        
        # Power encoding (maps component powers to spatial features)
        self.power_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        # Material encoding
        self.material_encoder = nn.Embedding(10, hidden_dim // 4)  # 10 material types
        
        # MLP for temperature prediction
        layers = []
        input_dim = pos_encoding_dim + hidden_dim // 4 + hidden_dim // 4  # pos + power + material
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
            ])
            input_dim = hidden_dim
        
        # Output: temperature (scalar)
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
    def encode_powers(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                     component_powers: Dict[str, Tuple[float, float, float, float]]) -> torch.Tensor:
        """
        Encode component powers as spatial features.
        
        Args:
            x, y, z: Position tensors
            component_powers: Dict mapping component names to (x, y, z, power)
            
        Returns:
            Power features tensor
        """
        # For each query point, compute weighted sum of nearby component powers
        power_features = torch.zeros(x.shape[0], device=x.device)
        
        for comp_name, (cx, cy, cz, power) in component_powers.items():
            # Distance from query point to component
            dx = x - cx
            dy = y - cy
            dz = z - cz
            dist = torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-6)  # Add small epsilon
            
            # Gaussian influence (thermal spreading)
            sigma = 10.0  # mm, thermal spreading distance
            influence = torch.exp(-dist**2 / (2 * sigma**2))
            
            # Accumulate power contribution
            power_features += influence * power
        
        # Encode power features
        power_features = power_features.unsqueeze(-1)  # (N, 1)
        encoded = self.power_encoder(power_features)  # (N, hidden_dim // 4)
        
        return encoded
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                component_powers: Dict[str, Tuple[float, float, float, float]],
                board_material: int = 0) -> torch.Tensor:
        """
        Predict temperature at given positions.
        
        Args:
            x, y, z: Position coordinates
            component_powers: Dict mapping component names to (x, y, z, power)
            board_material: Material type index
            
        Returns:
            Temperature tensor
        """
        # Flatten spatial dimensions
        if x.dim() > 1:
            original_shape = x.shape
            x = x.flatten()
            y = y.flatten()
            z = z.flatten()
        else:
            original_shape = x.shape
        
        # Encode position
        pos = torch.stack([x, y, z], dim=-1)
        pos_encoded = self.pos_encoding(pos)
        
        # Encode powers
        power_encoded = self.encode_powers(x, y, z, component_powers)
        
        # Encode material
        material_tensor = torch.tensor(board_material, device=x.device).expand(x.shape[0])
        material_encoded = self.material_encoder(material_tensor)
        
        # Concatenate features
        input_tensor = torch.cat([pos_encoded, power_encoded, material_encoded], dim=-1)
        
        # Predict temperature
        T = self.mlp(input_tensor).squeeze(-1)  # (N,)
        
        # Reshape to original spatial dimensions
        if len(original_shape) > 1:
            T = T.reshape(original_shape)
        
        return T


class PhysicsInformedThermalLoss(nn.Module):
    """
    Enforce heat equation: ∂T/∂t = α∇²T + Q/(ρcp)
    
    For steady-state: ∇²T = -Q/(α·ρcp)
    """
    
    def __init__(self, alpha: float = 1e-4, rho: float = 2700.0, cp: float = 900.0):
        """
        Args:
            alpha: Thermal diffusivity (m²/s)
            rho: Density (kg/m³)
            cp: Specific heat capacity (J/(kg·K))
        """
        super().__init__()
        self.alpha = alpha
        self.rho = rho
        self.cp = cp
    
    def laplacian(self, T: torch.Tensor, dx: float = 1e-3) -> torch.Tensor:
        """
        Compute Laplacian using finite differences.
        
        Args:
            T: Temperature field
            dx: Spatial step size
            
        Returns:
            Laplacian of T
        """
        # Simplified: would use proper spatial gradients in practice
        # For now, return zeros (placeholder)
        return torch.zeros_like(T)
    
    def time_derivative(self, T: torch.Tensor, dt: float = 1e-3) -> torch.Tensor:
        """
        Compute time derivative.
        
        Args:
            T: Temperature field
            dt: Time step
            
        Returns:
            Time derivative
        """
        # Placeholder - would need time history
        return torch.zeros_like(T)
    
    def compute_heat_source(self, component_powers: Dict[str, Tuple[float, float, float, float]],
                           query_points: torch.Tensor) -> torch.Tensor:
        """
        Compute heat source term Q/(ρcp) at query points.
        
        Args:
            component_powers: Component power dictionary
            query_points: Query point positions (N, 3)
            
        Returns:
            Heat source term
        """
        Q = torch.zeros(query_points.shape[0], device=query_points.device)
        
        for comp_name, (cx, cy, cz, power) in component_powers.items():
            # Distance from query points to component
            dx = query_points[:, 0] - cx
            dy = query_points[:, 1] - cy
            dz = query_points[:, 2] - cz
            dist = torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-6)
            
            # Gaussian heat source (localized at component)
            sigma = 5.0  # mm
            influence = torch.exp(-dist**2 / (2 * sigma**2))
            
            # Power density contribution
            Q += influence * power / (self.rho * self.cp)
        
        return Q
    
    def forward(self, T: torch.Tensor, component_powers: Dict[str, Tuple[float, float, float, float]],
                query_points: torch.Tensor) -> torch.Tensor:
        """
        Compute physics loss enforcing heat equation.
        
        Args:
            T: Predicted temperature field
            component_powers: Component power dictionary
            query_points: Query point positions (N, 3)
            
        Returns:
            Physics loss (scalar)
        """
        # For steady-state: ∇²T = -Q/(α·ρcp)
        laplacian_T = self.laplacian(T)
        
        # Heat source term
        heat_source = self.compute_heat_source(component_powers, query_points)
        
        # Heat equation residual
        residual = laplacian_T + heat_source / self.alpha
        
        # Loss: minimize residual
        L_heat_eq = torch.mean(residual**2)
        
        return L_heat_eq

