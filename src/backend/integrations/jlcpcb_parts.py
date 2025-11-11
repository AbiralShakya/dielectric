"""
JLCPCB Parts Database Manager

Manages local SQLite database of JLCPCB parts for fast searching.
"""

import sqlite3
import json
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class JLCPCBPartsManager:
    """
    Manager for JLCPCB parts database.
    
    Features:
    - Fast parametric search
    - Filter by package, category, library type
    - Map packages to KiCad footprints
    - Get part details
    """
    
    def __init__(self, db_path: str = "data/jlcpcb_parts.db"):
        """
        Initialize parts manager.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        
        # Initialize database if needed
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema if needed."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS components (
                lcsc TEXT PRIMARY KEY,
                category TEXT,
                subcategory TEXT,
                mfr_part TEXT,
                package TEXT,
                solder_joints INTEGER,
                manufacturer TEXT,
                library_type TEXT,
                description TEXT,
                datasheet TEXT,
                stock INTEGER,
                price_json TEXT,
                last_updated INTEGER
            )
        """)
        
        # Create indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_category ON components(category, subcategory)",
            "CREATE INDEX IF NOT EXISTS idx_package ON components(package)",
            "CREATE INDEX IF NOT EXISTS idx_manufacturer ON components(manufacturer)",
            "CREATE INDEX IF NOT EXISTS idx_library_type ON components(library_type)",
            "CREATE INDEX IF NOT EXISTS idx_stock ON components(stock)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        self.conn.commit()
    
    def search_parts(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        package: Optional[str] = None,
        library_type: Optional[str] = None,
        manufacturer: Optional[str] = None,
        in_stock: bool = True,
        limit: int = 20
    ) -> List[Dict]:
        """
        Search parts with filters.
        
        Args:
            query: Free-text search in description/mfr_part
            category: Category name (e.g., "Resistors")
            package: Package name (e.g., "0603")
            library_type: "Basic" or "Extended"
            manufacturer: Manufacturer name
            in_stock: Only parts with stock > 0
            limit: Maximum results
        
        Returns:
            List of part dictionaries
        """
        cursor = self.conn.cursor()
        
        conditions = []
        params = []
        
        if query:
            conditions.append("(description LIKE ? OR mfr_part LIKE ?)")
            params.extend([f"%{query}%", f"%{query}%"])
        
        if category:
            conditions.append("category = ?")
            params.append(category)
        
        if package:
            conditions.append("package = ?")
            params.append(package)
        
        if library_type:
            conditions.append("library_type = ?")
            params.append(library_type)
        
        if manufacturer:
            conditions.append("manufacturer = ?")
            params.append(manufacturer)
        
        if in_stock:
            conditions.append("stock > 0")
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        sql = f"""
            SELECT * FROM components
            WHERE {where_clause}
            ORDER BY 
                CASE WHEN library_type = 'Basic' THEN 0 ELSE 1 END,
                stock DESC
            LIMIT ?
        """
        
        params.append(limit)
        
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    def get_part_info(self, lcsc_number: str) -> Optional[Dict]:
        """
        Get detailed info for specific part.
        
        Args:
            lcsc_number: LCSC part number
        
        Returns:
            Part dictionary or None
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM components WHERE lcsc = ?", (lcsc_number,))
        row = cursor.fetchone()
        
        if row:
            part = dict(row)
            # Parse price JSON
            if part.get("price_json"):
                try:
                    part["prices"] = json.loads(part["price_json"])
                except:
                    part["prices"] = []
            return part
        
        return None
    
    def map_package_to_footprint(self, package: str, component_type: Optional[str] = None) -> List[str]:
        """
        Map JLCPCB package name to KiCad footprint(s).
        
        Args:
            package: Package name (e.g., "0603")
            component_type: Optional component type hint ("resistor", "capacitor", etc.)
        
        Returns:
            List of KiCad footprint names
        """
        # Common package mappings
        mappings = {
            "0603": {
                "resistor": ["Resistor_SMD:R_0603_1608Metric"],
                "capacitor": ["Capacitor_SMD:C_0603_1608Metric"],
                "inductor": ["Inductor_SMD:L_0603_1608Metric"]
            },
            "0805": {
                "resistor": ["Resistor_SMD:R_0805_2012Metric"],
                "capacitor": ["Capacitor_SMD:C_0805_2012Metric"],
                "inductor": ["Inductor_SMD:L_0805_2012Metric"]
            },
            "SOIC-8": {
                "ic": ["Package_SO:SOIC-8_3.9x4.9mm_P1.27mm"]
            },
            "QFN-16": {
                "ic": ["Package_DFN_QFN:QFN-16-1EP_3x3mm_P0.5mm_EP1.7x1.7mm"]
            }
        }
        
        if package in mappings:
            if component_type:
                component_type_lower = component_type.lower()
                for key, footprints in mappings[package].items():
                    if key in component_type_lower:
                        return footprints
            # Return all possible footprints for this package
            all_footprints = []
            for footprints in mappings[package].values():
                all_footprints.extend(footprints)
            return list(set(all_footprints))
        
        return []
    
    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM components")
        total_parts = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM components WHERE library_type = 'Basic'")
        basic_parts = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM components WHERE stock > 0")
        in_stock = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT package) FROM components")
        unique_packages = cursor.fetchone()[0]
        
        return {
            "total_parts": total_parts,
            "basic_parts": basic_parts,
            "in_stock": in_stock,
            "unique_packages": unique_packages
        }

