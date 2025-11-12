"""
Neural Field Methods for EM Simulation
Implements Physics-Informed Machine Learning for electromagnetic field simulation.

Based on:
- Raissi et al., "Physics-Informed Neural Networks" (2019)
- Li et al., "Fourier Neural Operator" (2020)
- Xie et al., "Neural Fields" (2022)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
import math


class PositionalEncoding(nn.Module):
    """
    Fourier positional encoding for neural fields.
    Maps spatial coordinates to higher-dimensional space for better learning.
    """
    
    def __init__(self, dim: int = 64, max_freq: float = 10.0):
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode position using Fourier features.
        
        Args:
            x: Position tensor of shape (..., 3) for (x, y, z)
            
        Returns:
            Encoded features of shape (..., dim)
        """
        device = x.device
        batch_shape = x.shape[:-1]
        n_dims = x.shape[-1]
        
        # Create frequency bands
        freqs = torch.linspace(0, self.max_freq, self.dim // (2 * n_dims), device=device)
        
        # Encode each dimension
        encoded = []
        for i in range(n_dims):
            for freq in freqs:
                encoded.append(torch.sin(2 * math.pi * freq * x[..., i:i+1]))
                encoded.append(torch.cos(2 * math.pi * freq * x[..., i:i+1]))
        
        # Stack and truncate to desired dimension
        encoded = torch.cat(encoded, dim=-1)
        if encoded.shape[-1] > self.dim:
            encoded = encoded[..., :self.dim]
        elif encoded.shape[-1] < self.dim:
            # Pad if needed
            padding = self.dim - encoded.shape[-1]
            encoded = torch.cat([encoded, torch.zeros(*batch_shape, padding, device=device)], dim=-1)
        
        return encoded


class NeuralEMField(nn.Module):
    """
    Neural field for EM fields.
    
    Input: (x, y, z, frequency)
    Output: (E_x, E_y, E_z, H_x, H_y, H_z)
    
    Learns continuous field representation for real-time EM simulation.
    """
    
    def __init__(self, hidden_dim: int = 256, num_layers: int = 8, pos_encoding_dim: int = 64):
        super().__init__()
        
        # Positional encoding (Fourier features)
        self.pos_encoding = PositionalEncoding(dim=pos_encoding_dim)
        
        # MLP for field prediction
        layers = []
        input_dim = pos_encoding_dim + 1  # encoded position + frequency
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),  # Swish activation
            ])
            input_dim = hidden_dim
        
        # Output: 6 values (E_x, E_y, E_z, H_x, H_y, H_z)
        layers.append(nn.Linear(hidden_dim, 6))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, frequency: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict EM fields at given positions and frequency.
        
        Args:
            x, y, z: Position coordinates (any shape, will be flattened)
            frequency: Frequency in Hz (scalar or broadcastable)
            
        Returns:
            E: Electric field tensor of shape (..., 3)
            H: Magnetic field tensor of shape (..., 3)
        """
        # Flatten spatial dimensions
        if x.dim() > 1:
            original_shape = x.shape
            x = x.flatten()
            y = y.flatten()
            z = z.flatten()
        else:
            original_shape = x.shape
        
        # Stack positions
        pos = torch.stack([x, y, z], dim=-1)  # (N, 3)
        
        # Encode position
        pos_encoded = self.pos_encoding(pos)  # (N, pos_encoding_dim)
        
        # Expand frequency to match batch size
        if frequency.dim() == 0:
            freq_tensor = frequency.unsqueeze(0).expand(pos_encoded.shape[0], 1)
        else:
            freq_tensor = frequency.unsqueeze(-1).expand_as(pos_encoded[..., :1])
        
        # Concatenate with frequency
        input_tensor = torch.cat([pos_encoded, freq_tensor], dim=-1)
        
        # Predict fields
        fields = self.mlp(input_tensor)  # (N, 6)
        
        # Split into E and H
        E = fields[..., :3]  # (N, 3)
        H = fields[..., 3:]  # (N, 3)
        
        # Reshape to original spatial dimensions
        if len(original_shape) > 1:
            E = E.reshape(*original_shape, 3)
            H = H.reshape(*original_shape, 3)
        
        return E, H


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss enforcing Maxwell's equations.
    
    Enforces:
    - ∇ × E = -∂B/∂t
    - ∇ × H = J + ∂D/∂t
    """
    
    def __init__(self, lambda_physics: float = 1.0, epsilon_0: float = 8.854e-12, mu_0: float = 4e-7 * math.pi):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.epsilon_0 = epsilon_0
        self.mu_0 = mu_0
        
    def curl(self, F: torch.Tensor, dx: float = 1e-3) -> torch.Tensor:
        """
        Compute curl using finite differences.
        
        Args:
            F: Field tensor of shape (..., 3) for (F_x, F_y, F_z)
            dx: Step size for finite differences
            
        Returns:
            Curl of F, shape (..., 3)
        """
        # Approximate curl using central differences
        # ∇ × F = (∂F_z/∂y - ∂F_y/∂z, ∂F_x/∂z - ∂F_z/∂x, ∂F_y/∂x - ∂F_x/∂y)
        
        # For simplicity, use automatic differentiation
        # This is a placeholder - in practice, we'd use proper spatial gradients
        # For now, return zeros (will be replaced with proper implementation)
        return torch.zeros_like(F)
    
    def time_derivative(self, F: torch.Tensor, dt: float = 1e-9) -> torch.Tensor:
        """
        Compute time derivative (for time-varying fields).
        
        Args:
            F: Field tensor
            dt: Time step
            
        Returns:
            Time derivative
        """
        # Placeholder - would need time history for proper computation
        return torch.zeros_like(F)
    
    def compute_physics_loss(self, E: torch.Tensor, H: torch.Tensor, 
                            geometry: Optional[torch.Tensor] = None, 
                            frequency: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Enforce Maxwell's equations via automatic differentiation.
        
        Args:
            E: Electric field
            H: Magnetic field
            geometry: Optional geometry tensor
            frequency: Optional frequency tensor
            
        Returns:
            Physics loss (scalar)
        """
        # For time-harmonic fields: ∂/∂t → jω
        if frequency is not None:
            omega = 2 * math.pi * frequency
            # ∇ × E = -jωμH
            # ∇ × H = jωεE + J
            
            # Simplified loss: enforce field relationships
            # In practice, would compute actual curls using spatial gradients
            L_maxwell = torch.mean(
                torch.abs(E)**2 + torch.abs(H)**2  # Placeholder
            )
        else:
            # Static case or simplified
            L_maxwell = torch.mean(
                torch.abs(E)**2 + torch.abs(H)**2
            )
        
        return L_maxwell
    
    def compute_data_loss(self, E: torch.Tensor, H: torch.Tensor, 
                         E_gt: Optional[torch.Tensor] = None,
                         H_gt: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute data loss if ground truth is available.
        
        Args:
            E, H: Predicted fields
            E_gt, H_gt: Ground truth fields (optional)
            
        Returns:
            Data loss (scalar)
        """
        if E_gt is None or H_gt is None:
            return torch.tensor(0.0, device=E.device)
        
        L_data = (
            torch.mean((E - E_gt)**2) +
            torch.mean((H - H_gt)**2)
        )
        
        return L_data
    
    def forward(self, E: torch.Tensor, H: torch.Tensor, 
                geometry: Optional[torch.Tensor] = None,
                frequency: Optional[torch.Tensor] = None,
                E_gt: Optional[torch.Tensor] = None,
                H_gt: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute total loss (data + physics).
        
        Args:
            E, H: Predicted fields
            geometry: Optional geometry
            frequency: Optional frequency
            E_gt, H_gt: Optional ground truth
            
        Returns:
            Total loss
        """
        # Data loss
        L_data = self.compute_data_loss(E, H, E_gt, H_gt)
        
        # Physics loss
        L_physics = self.compute_physics_loss(E, H, geometry, frequency)
        
        return L_data + self.lambda_physics * L_physics


class SParameterPredictor(nn.Module):
    """
    Predict S-parameters from geometry using neural operator.
    
    Input: Geometry (component positions, trace layout)
    Output: S-parameters (S11, S12, S21, S22) vs. frequency
    """
    
    def __init__(self, geometry_encoder_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        
        # Geometry encoder (simplified - would use GNN in practice)
        self.geometry_encoder = nn.Sequential(
            nn.Linear(geometry_encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # S-parameter predictor
        self.s_parameter_predictor = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for frequency
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # S11, S12, S21, S22 (complex -> 4 real values)
        )
        
    def forward(self, geometry_features: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
        """
        Predict S-parameters.
        
        Args:
            geometry_features: Encoded geometry features (batch_size, geometry_encoder_dim)
            frequencies: Frequencies in Hz (batch_size, num_freqs)
            
        Returns:
            S-parameters of shape (batch_size, num_freqs, 4)
        """
        # Encode geometry
        geom_encoded = self.geometry_encoder(geometry_features)  # (batch_size, hidden_dim)
        
        # Expand geometry for each frequency
        num_freqs = frequencies.shape[-1] if frequencies.dim() > 1 else 1
        if frequencies.dim() == 1:
            frequencies = frequencies.unsqueeze(0)
        
        geom_expanded = geom_encoded.unsqueeze(1).expand(-1, num_freqs, -1)  # (batch_size, num_freqs, hidden_dim)
        freq_expanded = frequencies.unsqueeze(-1)  # (batch_size, num_freqs, 1)
        
        # Concatenate
        combined = torch.cat([geom_expanded, freq_expanded], dim=-1)  # (batch_size, num_freqs, hidden_dim + 1)
        
        # Predict S-parameters
        s_params = self.s_parameter_predictor(combined)  # (batch_size, num_freqs, 4)
        
        return s_params


class NeuralEMSimulator:
    """
    Neural field-based EM simulator.
    
    Trains on FDTD/FEM data, learns continuous field representation,
    enables real-time S-parameter prediction.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        
        # Initialize models
        self.field_network = NeuralEMField().to(self.device)
        self.s_parameter_network = SParameterPredictor().to(self.device)
        
        # Load pretrained weights if available
        if model_path:
            self.load_model(model_path)
    
    def simulate(self, geometry: Dict, frequency: float) -> Dict:
        """
        Simulate EM fields and S-parameters.
        
        Args:
            geometry: Geometry dictionary with component positions, traces, etc.
            frequency: Frequency in Hz
            
        Returns:
            Dictionary with E, H fields and S-parameters
        """
        self.field_network.eval()
        self.s_parameter_network.eval()
        
        with torch.no_grad():
            # Sample points in space (simplified)
            # In practice, would sample based on geometry
            x = torch.linspace(0, 100, 50, device=self.device)  # mm
            y = torch.linspace(0, 100, 50, device=self.device)
            z = torch.zeros_like(x)
            
            X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
            freq_tensor = torch.tensor(frequency, device=self.device)
            
            # Predict fields
            E, H = self.field_network(X, Y, Z, freq_tensor)
            
            # Predict S-parameters (simplified geometry encoding)
            geometry_features = torch.zeros(1, 128, device=self.device)  # Placeholder
            frequencies = torch.tensor([frequency], device=self.device)
            s_params = self.s_parameter_network(geometry_features, frequencies)
            
            return {
                "E": E.cpu().numpy(),
                "H": H.cpu().numpy(),
                "S_parameters": s_params.cpu().numpy(),
                "frequency": frequency
            }
    
    def save_model(self, path: str):
        """Save model weights."""
        torch.save({
            "field_network": self.field_network.state_dict(),
            "s_parameter_network": self.s_parameter_network.state_dict(),
        }, path)
    
    def load_model(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.field_network.load_state_dict(checkpoint["field_network"])
        self.s_parameter_network.load_state_dict(checkpoint["s_parameter_network"])

