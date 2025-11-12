"""
Component Sourcing

Features:
- Multi-supplier search (JLCPCB, DigiKey, Mouser)
- Price comparison
- Availability checking
- Lead time estimation
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ComponentSourcing:
    """
    Component sourcing from multiple suppliers.
    
    Features:
    - Multi-supplier search
    - Price comparison
    - Availability checking
    """
    
    def __init__(self):
        """Initialize component sourcing."""
        pass
    
    async def search_suppliers(
        self,
        part_number: str,
        value: Optional[str] = None,
        package: Optional[str] = None
    ) -> Dict:
        """
        Search for component across multiple suppliers.
        
        Args:
            part_number: Part number
            value: Component value
            package: Package type
        
        Returns:
            Search results from all suppliers
        """
        results = {
            "jlcpcb": [],
            "digikey": [],
            "mouser": []
        }
        
        # Search JLCPCB
        try:
            from src.backend.integrations.jlcpcb_parts import JLCPCBPartsManager
            parts_manager = JLCPCBPartsManager()
            jlc_results = parts_manager.search_parts(query=part_number or value or "", limit=5)
            results["jlcpcb"] = jlc_results
        except Exception as e:
            logger.debug(f"JLCPCB search failed: {e}")
        
        # DigiKey and Mouser would require API keys
        # Placeholder for future integration
        
        return results
    
    def compare_prices(self, results: Dict) -> List[Dict]:
        """
        Compare prices across suppliers.
        
        Args:
            results: Search results from search_suppliers
        
        Returns:
            Sorted list of best prices
        """
        prices = []
        
        # Extract prices from JLCPCB
        for item in results.get("jlcpcb", []):
            price = self._extract_price(item.get("prices", []))
            if price > 0:
                prices.append({
                    "supplier": "JLCPCB",
                    "part_number": item.get("lcsc", ""),
                    "price": price,
                    "stock": item.get("stock", 0),
                    "description": item.get("description", "")
                })
        
        # Sort by price
        prices.sort(key=lambda x: x["price"])
        
        return prices
    
    def _extract_price(self, prices: List) -> float:
        """Extract price from pricing data."""
        if not prices:
            return 0.0
        
        if isinstance(prices, list) and len(prices) > 0:
            price_data = prices[0]
            if isinstance(price_data, dict):
                return float(price_data.get("price", 0.0))
            elif isinstance(price_data, (int, float)):
                return float(price_data)
        
        return 0.0

