"""
Graph Neural Networks for Routing Prediction
Predicts optimal routing paths before running expensive autorouters.

Based on:
- Bronstein et al., "Geometric Deep Learning" (2021)
- Kipf & Welling, "Graph Convolutional Networks" (2017)
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional
import numpy as np


class RoutingGraph:
    """
    Graph for routing prediction.
    
    Nodes: Components, vias, pads
    Edges: Potential routing paths
    """
    
    def __init__(self, placement: Dict):
        """
        Initialize routing graph from placement.
        
        Args:
            placement: Placement dictionary with components and nets
        """
        self.nodes = []
        self.edges = []
        self.node_to_idx = {}
        self.pad_to_node = {}
        
        components = placement.get("components", [])
        nets = placement.get("nets", [])
        
        node_idx = 0
        
        # Nodes: components + pads
        for comp in components:
            comp_name = comp.get("name", "")
            pads = comp.get("pads", [])
            
            # Component node
            self.node_to_idx[f"comp_{comp_name}"] = node_idx
            self.nodes.append({
                "type": "component",
                "name": comp_name,
                "position": (comp.get("x", 0.0), comp.get("y", 0.0)),
                "features": [
                    comp.get("x", 0.0),
                    comp.get("y", 0.0),
                    comp.get("power", 0.0),
                    self._encode_package_type(comp.get("package", "unknown"))
                ]
            })
            node_idx += 1
            
            # Pad nodes
            for pad in pads:
                pad_name = pad.get("name", "")
                pad_key = f"{comp_name}_{pad_name}"
                self.node_to_idx[pad_key] = node_idx
                self.pad_to_node[pad_key] = node_idx
                
                self.nodes.append({
                    "type": "pad",
                    "name": pad_name,
                    "component": comp_name,
                    "position": (
                        comp.get("x", 0.0) + pad.get("x_offset", 0.0),
                        comp.get("y", 0.0) + pad.get("y_offset", 0.0)
                    ),
                    "net": pad.get("net", ""),
                    "features": [
                        comp.get("x", 0.0) + pad.get("x_offset", 0.0),
                        comp.get("y", 0.0) + pad.get("y_offset", 0.0),
                        0.0,  # Pad-specific feature
                        0.0
                    ]
                })
                node_idx += 1
        
        # Edges: potential routing paths
        # 1. Component to its pads
        for comp in components:
            comp_name = comp.get("name", "")
            comp_idx = self.node_to_idx.get(f"comp_{comp_name}")
            if comp_idx is None:
                continue
            
            pads = comp.get("pads", [])
            for pad in pads:
                pad_name = pad.get("name", "")
                pad_key = f"{comp_name}_{pad_name}"
                pad_idx = self.pad_to_node.get(pad_key)
                if pad_idx is not None:
                    self.edges.append((comp_idx, pad_idx))
                    self.edges.append((pad_idx, comp_idx))  # Bidirectional
        
        # 2. Pads connected by nets (MST-like)
        net_pads = {}
        for net in nets:
            net_name = net.get("name", "")
            pins = net.get("pins", [])
            
            pad_indices = []
            for pin in pins:
                comp_name = pin.get("component", "")
                pad_name = pin.get("pin", "")
                pad_key = f"{comp_name}_{pad_name}"
                pad_idx = self.pad_to_node.get(pad_key)
                if pad_idx is not None:
                    pad_indices.append(pad_idx)
            
            # Connect pads in net (simplified: connect all pairs)
            for i, pad1 in enumerate(pad_indices):
                for pad2 in pad_indices[i+1:]:
                    self.edges.append((pad1, pad2))
                    self.edges.append((pad2, pad1))
    
    def _encode_package_type(self, package: str) -> float:
        """Encode package type."""
        package_map = {
            "SOIC-8": 0.1,
            "0805": 0.2,
            "BGA": 0.3,
            "QFN": 0.4,
            "LED": 0.5,
        }
        return package_map.get(package, 0.0)
    
    def to_pyg_data(self) -> Data:
        """Convert to PyTorch Geometric Data object."""
        node_features = torch.tensor([node["features"] for node in self.nodes], dtype=torch.float)
        
        if self.edges:
            edge_index = torch.tensor(self.edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        return Data(x=node_features, edge_index=edge_index)


class RoutingGNN(nn.Module):
    """
    GNN for routing path prediction.
    
    Input: Routing graph
    Output: Routing paths, via locations, layer assignments
    """
    
    def __init__(self, node_dim: int = 4, hidden_dim: int = 128, num_layers: int = 4):
        super().__init__()
        
        # Graph encoder
        self.encoder_layers = nn.ModuleList()
        self.encoder_layers.append(GATConv(node_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.encoder_layers.append(GATConv(hidden_dim, hidden_dim))
        
        self.encoder_layers.append(GATConv(hidden_dim, hidden_dim))
        
        # Path predictor (predicts routing path probability)
        self.path_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Probability
        )
        
        # Via predictor (predicts via locations)
        self.via_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Layer predictor (predicts layer assignment)
        self.layer_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4),  # 4 layers max
            nn.Softmax(dim=-1)
        )
        
    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Dictionary with routing predictions
        """
        x, edge_index = data.x, data.edge_index
        
        # Encode graph
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x, edge_index)
            if i < len(self.encoder_layers) - 1:
                x = torch.relu(x)
        
        node_features = x
        
        # Predict routing paths (for edges)
        if edge_index.shape[1] > 0:
            source_features = node_features[edge_index[0]]
            target_features = node_features[edge_index[1]]
            pair_features = torch.cat([source_features, target_features], dim=-1)
            path_probs = self.path_predictor(pair_features).squeeze(-1)
        else:
            path_probs = torch.empty(0, device=node_features.device)
        
        # Predict via locations (for nodes)
        via_probs = self.via_predictor(node_features).squeeze(-1)
        
        # Predict layer assignments (for nodes)
        layer_probs = self.layer_predictor(node_features)
        
        return {
            "paths": path_probs,
            "vias": via_probs,
            "layers": layer_probs,
            "node_features": node_features
        }
    
    def predict_routing(self, placement: Dict) -> Dict:
        """
        Predict routing for a placement.
        
        Args:
            placement: Placement dictionary
            
        Returns:
            Dictionary with routing predictions
        """
        self.eval()
        
        # Convert to graph
        graph = RoutingGraph(placement)
        data = graph.to_pyg_data()
        
        with torch.no_grad():
            results = self.forward(data)
        
        # Extract routing paths
        paths = []
        if data.edge_index.shape[1] > 0:
            edge_index = data.edge_index.cpu().numpy()
            path_probs = results["paths"].cpu().numpy()
            
            # Filter high-probability paths
            threshold = 0.5
            for i, prob in enumerate(path_probs):
                if prob > threshold:
                    source_idx = edge_index[0, i]
                    target_idx = edge_index[1, i]
                    source_node = graph.nodes[source_idx]
                    target_node = graph.nodes[target_idx]
                    
                    paths.append({
                        "source": source_node["position"],
                        "target": target_node["position"],
                        "probability": float(prob)
                    })
        
        # Extract via locations
        via_probs = results["vias"].cpu().numpy()
        vias = []
        for i, prob in enumerate(via_probs):
            if prob > 0.5:
                node = graph.nodes[i]
                vias.append({
                    "position": node["position"],
                    "probability": float(prob)
                })
        
        # Extract layer assignments
        layer_probs = results["layers"].cpu().numpy()
        layer_assignments = []
        for i, probs in enumerate(layer_probs):
            layer = int(np.argmax(probs))
            node = graph.nodes[i]
            layer_assignments.append({
                "node": node.get("name", f"node_{i}"),
                "layer": layer,
                "probabilities": probs.tolist()
            })
        
        return {
            "paths": paths,
            "vias": vias,
            "layer_assignments": layer_assignments
        }

