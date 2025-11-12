"""
Graph Neural Networks for Signal Integrity Prediction
Predicts impedance, crosstalk, and timing violations from net topology.

Based on:
- Kipf & Welling, "Graph Convolutional Networks" (2017)
- Veličković et al., "Graph Attention Networks" (2018)
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data, Batch
from typing import Dict, List, Tuple, Optional
import numpy as np


class NetGraph:
    """
    Represent PCB as graph for GNN.
    
    Nodes: Components (with features: position, power, package)
    Edges: Nets (with features: net name, signal type)
    """
    
    def __init__(self, placement: Dict):
        """
        Initialize graph from placement data.
        
        Args:
            placement: Dictionary with 'components' and 'nets'
        """
        self.nodes = []
        self.edges = []
        self.node_to_idx = {}
        self.edge_features = []
        
        components = placement.get("components", [])
        nets = placement.get("nets", [])
        board = placement.get("board", {})
        
        # Node features: [x, y, power, package_type]
        for idx, comp in enumerate(components):
            self.node_to_idx[comp.get("name", f"comp_{idx}")] = idx
            self.nodes.append({
                "features": [
                    comp.get("x", 0.0),
                    comp.get("y", 0.0),
                    comp.get("power", 0.0),
                    self._encode_package_type(comp.get("package", "unknown"))
                ],
                "name": comp.get("name", f"comp_{idx}")
            })
        
        # Edge features: [net_name_hash, signal_type]
        edge_index = []
        for net in nets:
            net_name = net.get("name", "")
            pins = net.get("pins", [])
            
            # Create edges between all pairs of pins in the net
            for i, pin1 in enumerate(pins):
                comp1 = pin1.get("component", "")
                if comp1 not in self.node_to_idx:
                    continue
                    
                for j, pin2 in enumerate(pins[i+1:], start=i+1):
                    comp2 = pin2.get("component", "")
                    if comp2 not in self.node_to_idx:
                        continue
                    
                    # Add edge (bidirectional)
                    idx1 = self.node_to_idx[comp1]
                    idx2 = self.node_to_idx[comp2]
                    
                    edge_index.append([idx1, idx2])
                    edge_index.append([idx2, idx1])  # Bidirectional
                    
                    # Edge features
                    signal_type = self._get_signal_type(net_name)
                    self.edge_features.append([
                        hash(net_name) % 1000 / 1000.0,  # Normalized hash
                        signal_type
                    ])
                    self.edge_features.append([
                        hash(net_name) % 1000 / 1000.0,
                        signal_type
                    ])
        
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
        self.edge_attr = torch.tensor(self.edge_features, dtype=torch.float) if self.edge_features else torch.empty((0, 2), dtype=torch.float)
    
    def _encode_package_type(self, package: str) -> float:
        """Encode package type as float."""
        package_map = {
            "SOIC-8": 0.1,
            "0805": 0.2,
            "BGA": 0.3,
            "QFN": 0.4,
            "LED": 0.5,
        }
        return package_map.get(package, 0.0)
    
    def _get_signal_type(self, net_name: str) -> float:
        """Determine signal type from net name."""
        net_lower = net_name.lower()
        if "power" in net_lower or "vdd" in net_lower or "vcc" in net_lower:
            return 1.0
        elif "ground" in net_lower or "gnd" in net_lower:
            return 0.5
        elif "clock" in net_lower or "clk" in net_lower:
            return 0.8
        elif "data" in net_lower or "d" in net_lower:
            return 0.6
        else:
            return 0.3
    
    def to_pyg_data(self) -> Data:
        """Convert to PyTorch Geometric Data object."""
        node_features = torch.tensor([node["features"] for node in self.nodes], dtype=torch.float)
        
        return Data(
            x=node_features,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr
        )


class SignalIntegrityGNN(nn.Module):
    """
    GNN for signal integrity prediction.
    
    Input: Net graph
    Output: Impedance, crosstalk, timing violations per net
    """
    
    def __init__(self, node_dim: int = 4, edge_dim: int = 2, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        
        # First layer
        self.conv_layers.append(GATConv(node_dim, hidden_dim, edge_dim=edge_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.conv_layers.append(GATConv(hidden_dim, hidden_dim, edge_dim=edge_dim))
        
        # Last layer
        self.conv_layers.append(GATConv(hidden_dim, hidden_dim, edge_dim=edge_dim))
        
        # Prediction heads
        self.impedance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.crosstalk_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.timing_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        """
        Forward pass through GNN.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Dictionary with impedance, crosstalk, timing predictions
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Graph convolutions
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index, edge_attr)
            if i < len(self.conv_layers) - 1:
                x = torch.relu(x)
        
        # Predictions
        impedance = self.impedance_head(x)  # Per-node impedance
        
        # Crosstalk: predict for pairs of nodes
        crosstalk = self.predict_crosstalk(x, edge_index)
        
        # Timing violations
        timing = self.timing_head(x)  # Per-node timing
        
        return {
            "impedance": impedance,
            "crosstalk": crosstalk,
            "timing": timing
        }
    
    def predict_crosstalk(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Predict crosstalk between connected nodes.
        
        Args:
            node_features: Node feature tensor (num_nodes, hidden_dim)
            edge_index: Edge index tensor (2, num_edges)
            
        Returns:
            Crosstalk predictions (num_edges,)
        """
        if edge_index.shape[1] == 0:
            return torch.empty(0, device=node_features.device)
        
        # Get source and target node features
        source_features = node_features[edge_index[0]]  # (num_edges, hidden_dim)
        target_features = node_features[edge_index[1]]  # (num_edges, hidden_dim)
        
        # Concatenate for crosstalk prediction
        pair_features = torch.cat([source_features, target_features], dim=-1)  # (num_edges, hidden_dim * 2)
        
        # Predict crosstalk
        crosstalk = self.crosstalk_head(pair_features).squeeze(-1)  # (num_edges,)
        
        return crosstalk
    
    def predict(self, placement: Dict) -> Dict:
        """
        Predict signal integrity metrics for a placement.
        
        Args:
            placement: Placement dictionary
            
        Returns:
            Dictionary with SI predictions
        """
        self.eval()
        
        # Convert placement to graph
        graph = NetGraph(placement)
        data = graph.to_pyg_data()
        
        with torch.no_grad():
            results = self.forward(data)
        
        # Convert to numpy for easier handling
        return {
            "impedance": results["impedance"].cpu().numpy(),
            "crosstalk": results["crosstalk"].cpu().numpy(),
            "timing": results["timing"].cpu().numpy(),
            "node_names": [node["name"] for node in graph.nodes]
        }

