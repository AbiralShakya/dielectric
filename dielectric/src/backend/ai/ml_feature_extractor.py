"""
Small-Scale ML Models for Feature Extraction

Lightweight ML models that extract features/insights before/after xAI reasoning.

Models:
1. Geometry Feature Extractor - Extracts key geometric features
2. Physics Feature Extractor - Extracts physics insights
3. Optimization Feature Predictor - Predicts optimization difficulty
4. Context Summarizer - Summarizes context for xAI
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import pickle
import os
from pathlib import Path

# Make sklearn optional (for small ML models)
try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Create dummy classes for when sklearn is not available
    class StandardScaler:
        def fit(self, X): pass
        def transform(self, X): return X
    class RandomForestRegressor:
        def predict(self, X): return np.array([0.5])
    class RandomForestClassifier:
        def predict(self, X): return np.array([0])

try:
    from backend.geometry.placement import Placement
except ImportError:
    from src.backend.geometry.placement import Placement


class GeometryFeatureExtractor:
    """
    Small ML model to extract key geometric features.
    
    Trained on: Voronoi variance, MST length, convex hull, etc.
    Output: Normalized feature vector for xAI context
    """
    
    def __init__(self):
        """Initialize geometry feature extractor."""
        if SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
        else:
            self.scaler = None  # Will use simple normalization
        self.feature_names = [
            'voronoi_variance',
            'mst_length',
            'convex_hull_utilization',
            'component_density',
            'net_crossings',
            'routing_complexity',
            'thermal_hotspots',
            'overlap_risk'
        ]
        self._fitted = False
    
    def extract_features(self, geometry_data: Dict) -> np.ndarray:
        """
        Extract normalized feature vector from geometry data.
        
        Args:
            geometry_data: Dictionary with geometry metrics
            
        Returns:
            Normalized feature vector (8 features)
        """
        features = np.array([
            geometry_data.get('voronoi_variance', 0.0),
            geometry_data.get('mst_length', 0.0),
            geometry_data.get('convex_hull_utilization', 0.0),
            geometry_data.get('component_density', 0.0),
            geometry_data.get('net_crossings', 0),
            geometry_data.get('routing_complexity', 0.0),
            geometry_data.get('thermal_hotspots', 0),
            geometry_data.get('overlap_risk', 0.0)
        ])
        
        # Normalize
        if self.scaler:
            if not self._fitted:
                # Fit scaler on first use (would use training data in production)
                self.scaler.fit(features.reshape(1, -1))
                self._fitted = True
            features_normalized = self.scaler.transform(features.reshape(1, -1))[0]
        else:
            # Simple normalization without sklearn
            features_normalized = features / (np.abs(features).max() + 1e-10)
            self._fitted = True
        
        return features_normalized
    
    def interpret_features(self, features: np.ndarray) -> Dict[str, str]:
        """
        Interpret feature vector into human-readable insights.
        
        Args:
            features: Normalized feature vector
            
        Returns:
            Dictionary with interpretations
        """
        interpretations = {}
        
        feature_dict = dict(zip(self.feature_names, features))
        
        if feature_dict['voronoi_variance'] > 1.0:
            interpretations['distribution'] = "Components are clustered (high Voronoi variance) → thermal risk"
        elif feature_dict['voronoi_variance'] < -0.5:
            interpretations['distribution'] = "Components are well-distributed (low Voronoi variance) → good thermal spreading"
        else:
            interpretations['distribution'] = "Component distribution is moderate"
        
        if feature_dict['mst_length'] > 1.0:
            interpretations['routing'] = "Long trace lengths (high MST) → signal integrity concerns"
        elif feature_dict['mst_length'] < -0.5:
            interpretations['routing'] = "Short trace lengths (low MST) → good signal integrity"
        else:
            interpretations['routing'] = "Trace lengths are moderate"
        
        if feature_dict['routing_complexity'] > 1.0:
            interpretations['complexity'] = "High routing complexity → many net crossings → manufacturability risk"
        else:
            interpretations['complexity'] = "Routing complexity is manageable"
        
        return interpretations


class PhysicsFeatureExtractor:
    """
    Small ML model to extract physics insights.
    
    Trained on: Thermal data, power distribution, signal integrity metrics
    Output: Physics risk scores and insights
    """
    
    def __init__(self):
        """Initialize physics feature extractor."""
        if SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
        else:
            self.scaler = None  # Will use simple normalization
        self.feature_names = [
            'max_temperature',
            'thermal_gradient',
            'heat_flux',
            'power_density',
            'current_density',
            'ir_drop',
            'impedance_risk'
        ]
        self._fitted = False
    
    def extract_features(self, physics_data: Dict) -> np.ndarray:
        """
        Extract normalized physics features.
        
        Args:
            physics_data: Dictionary with physics metrics
            
        Returns:
            Normalized feature vector
        """
        features = np.array([
            physics_data.get('max_temperature', 25.0),
            physics_data.get('thermal_gradient', 0.0),
            physics_data.get('heat_flux_estimate', 0.0),
            physics_data.get('power_density', 0.0),
            physics_data.get('current_density_estimate', 0.0),
            physics_data.get('ir_drop_estimate', 0.0),
            physics_data.get('impedance_mismatch_risk', 0.0)
        ])
        
        # Normalize
        if self.scaler:
            if not self._fitted:
                self.scaler.fit(features.reshape(1, -1))
                self._fitted = True
            features_normalized = self.scaler.transform(features.reshape(1, -1))[0]
        else:
            # Simple normalization without sklearn
            features_normalized = features / (np.abs(features).max() + 1e-10)
            self._fitted = True
        
        return features_normalized
    
    def predict_risks(self, features: np.ndarray) -> Dict[str, float]:
        """
        Predict physics risks from features.
        
        Args:
            features: Normalized feature vector
            
        Returns:
            Dictionary with risk scores (0-1)
        """
        feature_dict = dict(zip(self.feature_names, features))
        
        # Thermal risk
        thermal_risk = min(1.0, max(0.0, 
            (feature_dict['max_temperature'] - 25.0) / 60.0 +  # Above ambient
            abs(feature_dict['thermal_gradient']) / 50.0 +  # High gradient
            feature_dict['heat_flux'] / 1000.0  # High heat flux
        ))
        
        # Power integrity risk
        power_risk = min(1.0, max(0.0,
            feature_dict['ir_drop'] / 0.1 +  # IR drop > 100mV
            feature_dict['current_density'] / 100.0  # High current density
        ))
        
        # Signal integrity risk
        si_risk = min(1.0, max(0.0,
            feature_dict['impedance_risk']  # Impedance mismatch
        ))
        
        return {
            "thermal_risk": thermal_risk,
            "power_integrity_risk": power_risk,
            "signal_integrity_risk": si_risk,
            "overall_risk": max(thermal_risk, power_risk, si_risk)
        }


class OptimizationDifficultyPredictor:
    """
    Small ML model to predict optimization difficulty.
    
    Predicts: How difficult will optimization be? How many iterations needed?
    """
    
    def __init__(self):
        """Initialize optimization difficulty predictor."""
        self.model = None
        self.scaler = StandardScaler()
        self._trained = False
    
    def predict_difficulty(self, geometry_features: np.ndarray, physics_features: np.ndarray) -> Dict:
        """
        Predict optimization difficulty.
        
        Args:
            geometry_features: Geometry feature vector
            physics_features: Physics feature vector
            
        Returns:
            Dictionary with difficulty predictions
        """
        # Combine features
        combined_features = np.concatenate([geometry_features, physics_features])
        
        if not self._trained:
            # Use simple heuristic (would train on real data)
            # High variance + high thermal = difficult
            difficulty_score = (
                abs(geometry_features[0]) +  # Voronoi variance
                abs(geometry_features[1]) +  # MST length
                abs(physics_features[0]) +  # Max temperature
                abs(physics_features[1])  # Thermal gradient
            ) / 4.0
            
            estimated_iterations = int(500 + difficulty_score * 1000)
        else:
            # Use trained model
            combined_normalized = self.scaler.transform(combined_features.reshape(1, -1))
            difficulty_score = self.model.predict(combined_normalized)[0]
            estimated_iterations = int(500 + difficulty_score * 1000)
        
        return {
            "difficulty_score": float(difficulty_score),
            "estimated_iterations": estimated_iterations,
            "optimization_strategy": self._suggest_strategy(difficulty_score)
        }
    
    def _suggest_strategy(self, difficulty: float) -> str:
        """Suggest optimization strategy based on difficulty."""
        if difficulty > 0.7:
            return "Use aggressive optimization: high initial temperature, many iterations, parallel chains"
        elif difficulty > 0.4:
            return "Use moderate optimization: standard temperature schedule, adaptive cooling"
        else:
            return "Use gentle optimization: low temperature, few iterations"


class ContextSummarizer:
    """
    Summarizes rich context into concise insights for xAI.
    
    Uses ML to identify most important features.
    """
    
    def __init__(self):
        """Initialize context summarizer."""
        self.geometry_extractor = GeometryFeatureExtractor()
        self.physics_extractor = PhysicsFeatureExtractor()
        self.optimization_predictor = OptimizationDifficultyPredictor()
    
    def summarize_for_xai(
        self,
        geometry_data: Dict,
        physics_data: Dict,
        user_intent: str
    ) -> str:
        """
        Summarize context into concise insights for xAI.
        
        Args:
            geometry_data: Geometry metrics
            physics_data: Physics metrics
            user_intent: User intent
            
        Returns:
            Concise summary string
        """
        # Extract features
        geo_features = self.geometry_extractor.extract_features(geometry_data)
        physics_features = self.physics_extractor.extract_features(physics_data)
        
        # Get interpretations
        geo_insights = self.geometry_extractor.interpret_features(geo_features)
        physics_risks = self.physics_extractor.predict_risks(physics_features)
        
        # Predict optimization difficulty
        opt_prediction = self.optimization_predictor.predict_difficulty(geo_features, physics_features)
        
        # Build summary
        summary = f"""
**KEY INSIGHTS FOR OPTIMIZATION:**

**Geometry Analysis:**
- {geo_insights.get('distribution', 'Distribution analysis pending')}
- {geo_insights.get('routing', 'Routing analysis pending')}
- {geo_insights.get('complexity', 'Complexity analysis pending')}

**Physics Risks:**
- Thermal Risk: {physics_risks['thermal_risk']:.2%} {'⚠️ HIGH' if physics_risks['thermal_risk'] > 0.7 else '✓ OK'}
- Power Integrity Risk: {physics_risks['power_integrity_risk']:.2%} {'⚠️ HIGH' if physics_risks['power_integrity_risk'] > 0.7 else '✓ OK'}
- Signal Integrity Risk: {physics_risks['signal_integrity_risk']:.2%} {'⚠️ HIGH' if physics_risks['signal_integrity_risk'] > 0.7 else '✓ OK'}

**Optimization Prediction:**
- Difficulty Score: {opt_prediction['difficulty_score']:.2f}/1.0
- Estimated Iterations: {opt_prediction['estimated_iterations']}
- Strategy: {opt_prediction['optimization_strategy']}

**User Intent:** "{user_intent}"

**Recommended Focus Areas:**
"""
        
        # Add recommendations based on risks
        if physics_risks['thermal_risk'] > 0.7:
            summary += "- 🔥 PRIORITY: Thermal management (distribute high-power components)\n"
        if physics_risks['power_integrity_risk'] > 0.7:
            summary += "- ⚡ PRIORITY: Power integrity (add decoupling caps, reduce IR drop)\n"
        if physics_risks['signal_integrity_risk'] > 0.7:
            summary += "- 📡 PRIORITY: Signal integrity (minimize trace length, maintain impedance)\n"
        if geo_features[0] > 1.0:  # High Voronoi variance
            summary += "- 📐 PRIORITY: Component distribution (reduce clustering)\n"
        if geo_features[1] > 1.0:  # Long MST
            summary += "- 🔌 PRIORITY: Trace length minimization\n"
        
        return summary

