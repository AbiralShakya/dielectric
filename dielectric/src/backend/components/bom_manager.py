"""
BOM (Bill of Materials) Manager

Features:
- Automatic BOM generation
- Component sourcing (JLCPCB, DigiKey, Mouser)
- Cost estimation
- Availability checking
- Alternative part suggestions
- BOM export (CSV, Excel, XML)
"""

import logging
import csv
import json
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

try:
    from backend.geometry.placement import Placement
    from backend.geometry.component import Component
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.geometry.component import Component

logger = logging.getLogger(__name__)


class BOMManager:
    """
    Bill of Materials Manager.
    
    Features:
    - BOM generation
    - Component sourcing
    - Cost estimation
    - Availability checking
    """
    
    def __init__(self):
        """Initialize BOM manager."""
        self.components: List[Dict] = []
    
    def generate_bom(
        self,
        placement: Placement,
        include_pricing: bool = True
    ) -> Dict:
        """
        Generate Bill of Materials from placement.
        
        Args:
            placement: Placement with components
            include_pricing: Whether to include pricing information
        
        Returns:
            BOM dictionary
        """
        bom_items = []
        component_counts = {}
        
        # Count components
        for comp_name, comp in placement.components.items():
            # Group by value/package
            key = f"{comp.value or 'UNKNOWN'}_{comp.package or 'UNKNOWN'}"
            
            if key not in component_counts:
                component_counts[key] = {
                    "designators": [],
                    "value": comp.value,
                    "package": comp.package,
                    "footprint": comp.package,
                    "quantity": 0
                }
            
            component_counts[key]["designators"].append(comp_name)
            component_counts[key]["quantity"] += 1
        
        # Create BOM items
        for key, data in component_counts.items():
            item = {
                "part_number": data.get("value", "UNKNOWN"),
                "designator": ",".join(data["designators"]),
                "value": data["value"],
                "package": data["package"],
                "footprint": data["footprint"],
                "quantity": data["quantity"],
                "description": f"{data['value']} {data['package']}"
            }
            
            # Add sourcing information if available
            if include_pricing:
                sourcing = self._get_sourcing_info(item)
                item.update(sourcing)
            
            bom_items.append(item)
        
        # Calculate totals
        total_cost = sum(item.get("unit_price", 0) * item["quantity"] for item in bom_items)
        total_quantity = sum(item["quantity"] for item in bom_items)
        
        bom = {
            "board_name": placement.board.name or "board",
            "date": datetime.now().isoformat(),
            "items": bom_items,
            "summary": {
                "total_components": len(bom_items),
                "total_quantity": total_quantity,
                "total_cost": total_cost,
                "currency": "USD"
            }
        }
        
        logger.info(f"Generated BOM with {len(bom_items)} unique components")
        return bom
    
    def export_csv(
        self,
        bom: Dict,
        output_path: str
    ) -> str:
        """
        Export BOM to CSV.
        
        Args:
            bom: BOM dictionary
            output_path: Output file path
        
        Returns:
            Path to exported file
        """
        filepath = Path(output_path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "Designator",
                "Part Number",
                "Value",
                "Package",
                "Quantity",
                "Unit Price",
                "Total Price",
                "Supplier",
                "LCSC Part #"
            ])
            
            # Items
            for item in bom["items"]:
                writer.writerow([
                    item["designator"],
                    item.get("part_number", ""),
                    item.get("value", ""),
                    item.get("package", ""),
                    item["quantity"],
                    item.get("unit_price", ""),
                    item.get("unit_price", 0) * item["quantity"],
                    item.get("supplier", ""),
                    item.get("lcsc", "")
                ])
        
        logger.info(f"Exported BOM to CSV: {filepath}")
        return str(filepath)
    
    def export_json(self, bom: Dict, output_path: str) -> str:
        """Export BOM to JSON."""
        filepath = Path(output_path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(json.dumps(bom, indent=2))
        
        logger.info(f"Exported BOM to JSON: {filepath}")
        return str(filepath)
    
    def _get_sourcing_info(self, item: Dict) -> Dict:
        """
        Get component sourcing information.
        
        Args:
            item: BOM item
        
        Returns:
            Sourcing information dictionary
        """
        # Try to get from JLCPCB
        try:
            from src.backend.integrations.jlcpcb_parts import JLCPCBPartsManager
            
            parts_manager = JLCPCBPartsManager()
            
            # Search for component
            query = item.get("value") or item.get("part_number", "")
            results = parts_manager.search_parts(query=query, limit=1)
            
            if results:
                part = results[0]
                return {
                    "supplier": "JLCPCB",
                    "lcsc": part.get("lcsc", ""),
                    "unit_price": self._extract_price(part.get("prices", [])),
                    "stock": part.get("stock", 0),
                    "available": part.get("stock", 0) > 0,
                    "library_type": part.get("library_type", "Extended")
                }
        except Exception as e:
            logger.debug(f"JLCPCB lookup failed: {e}")
        
        # Fallback
        return {
            "supplier": "Unknown",
            "unit_price": 0.0,
            "available": False
        }
    
    def _extract_price(self, prices: List) -> float:
        """Extract price from pricing data."""
        if not prices:
            return 0.0
        
        # Try to get first price
        if isinstance(prices, list) and len(prices) > 0:
            price_data = prices[0]
            if isinstance(price_data, dict):
                return float(price_data.get("price", 0.0))
            elif isinstance(price_data, (int, float)):
                return float(price_data)
        
        return 0.0
    
    def check_availability(self, bom: Dict) -> Dict:
        """
        Check component availability.
        
        Args:
            bom: BOM dictionary
        
        Returns:
            Availability report
        """
        unavailable_items = []
        low_stock_items = []
        
        for item in bom["items"]:
            available = item.get("available", False)
            stock = item.get("stock", 0)
            quantity = item["quantity"]
            
            if not available:
                unavailable_items.append(item)
            elif stock < quantity:
                low_stock_items.append(item)
        
        return {
            "all_available": len(unavailable_items) == 0 and len(low_stock_items) == 0,
            "unavailable_items": unavailable_items,
            "low_stock_items": low_stock_items,
            "total_unavailable": len(unavailable_items),
            "total_low_stock": len(low_stock_items)
        }
    
    def find_alternatives(
        self,
        item: Dict
    ) -> List[Dict]:
        """
        Find alternative parts for a component.
        
        Args:
            item: BOM item
        
        Returns:
            List of alternative parts
        """
        alternatives = []
        
        # Try to find alternatives from JLCPCB
        try:
            from src.backend.integrations.jlcpcb_parts import JLCPCBPartsManager
            
            parts_manager = JLCPCBPartsManager()
            
            # Search for similar components
            query = item.get("value") or item.get("part_number", "")
            results = parts_manager.search_parts(query=query, limit=5)
            
            for part in results:
                alternatives.append({
                    "part_number": part.get("lcsc", ""),
                    "description": part.get("description", ""),
                    "price": self._extract_price(part.get("prices", [])),
                    "stock": part.get("stock", 0),
                    "supplier": "JLCPCB"
                })
        except Exception as e:
            logger.debug(f"Alternative search failed: {e}")
        
        return alternatives

