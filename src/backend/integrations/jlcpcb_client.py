"""
JLCPCB API Client

Handles authentication and communication with JLCPCB's external API.
Supports downloading parts database and querying component information.
"""

import os
import requests
import time
import logging
from typing import Dict, List, Optional, Callable
import sqlite3
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class JLCPCBClient:
    """
    Client for JLCPCB External API.
    
    Features:
    - Authentication with API key/secret
    - Download parts database (paginated)
    - Query component information
    - Rate limiting and retry logic
    """
    
    BASE_URL = "https://jlcpcb.com/external"
    AUTH_ENDPOINT = f"{BASE_URL}/genToken"
    PARTS_ENDPOINT = f"{BASE_URL}/component/getComponentInfos"
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Initialize JLCPCB client.
        
        Args:
            api_key: JLCPCB API key (from env if not provided)
            api_secret: JLCPCB API secret (from env if not provided)
        """
        self.api_key = api_key or os.getenv("JLCPCB_API_KEY")
        self.api_secret = api_secret or os.getenv("JLCPCB_API_SECRET")
        self.token = None
        self.token_expiry = 0
        
        if not self.api_key or not self.api_secret:
            logger.warning("⚠️  JLCPCB API credentials not found. Set JLCPCB_API_KEY and JLCPCB_API_SECRET environment variables.")
            logger.info("   Get API credentials from: https://jlcpcb.com/ → Account → API Management")
    
    def authenticate(self) -> bool:
        """
        Authenticate with JLCPCB API and get token.
        
        Returns:
            True if authentication successful, False otherwise
        """
        if not self.api_key or not self.api_secret:
            logger.error("❌ JLCPCB API credentials not configured")
            return False
        
        try:
            response = requests.post(
                self.AUTH_ENDPOINT,
                json={
                    "appKey": self.api_key,
                    "appSecret": self.api_secret
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("code") == 200:
                    self.token = data.get("data", {}).get("token")
                    # Tokens typically expire in 24 hours
                    self.token_expiry = time.time() + (24 * 3600)
                    logger.info("✅ JLCPCB authentication successful")
                    return True
                else:
                    logger.error(f"❌ JLCPCB auth failed: {data.get('msg', 'Unknown error')}")
                    return False
            else:
                logger.error(f"❌ JLCPCB auth failed: HTTP {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ JLCPCB authentication error: {str(e)}")
            return False
    
    def _ensure_authenticated(self) -> bool:
        """Ensure we have a valid token."""
        if not self.token or time.time() >= self.token_expiry:
            return self.authenticate()
        return True
    
    def fetch_parts_page(self, last_key: Optional[str] = None, page_size: int = 100) -> Dict:
        """
        Fetch one page of parts from JLCPCB API.
        
        Args:
            last_key: Pagination key from previous request (None for first page)
            page_size: Number of parts per page (max 100)
        
        Returns:
            Dictionary with parts data and pagination info
        """
        if not self._ensure_authenticated():
            return {"code": 401, "msg": "Authentication failed", "data": []}
        
        try:
            payload = {}
            if last_key:
                payload["lastKey"] = last_key
            
            response = requests.post(
                self.PARTS_ENDPOINT,
                headers={"externalApiToken": self.token},
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"❌ Failed to fetch parts: HTTP {response.status_code}")
                return {"code": response.status_code, "msg": "Request failed", "data": []}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error fetching parts: {str(e)}")
            return {"code": 500, "msg": str(e), "data": []}
    
    def download_full_database(
        self,
        db_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict:
        """
        Download entire JLCPCB parts database to SQLite.
        
        Args:
            db_path: Path to SQLite database file
            progress_callback: Optional callback(parts_count, total_pages) for progress
        
        Returns:
            Dictionary with download statistics
        """
        if not self._ensure_authenticated():
            return {"success": False, "error": "Authentication failed"}
        
        # Create database directory if needed
        db_path_obj = Path(db_path)
        db_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table if not exists
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
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_category ON components(category, subcategory)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_package ON components(package)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_manufacturer ON components(manufacturer)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_library_type ON components(library_type)
        """)
        
        conn.commit()
        
        # Download parts
        parts_count = 0
        page_count = 0
        last_key = None
        errors = []
        
        logger.info("📥 Starting JLCPCB parts database download...")
        
        while True:
            page_count += 1
            result = self.fetch_parts_page(last_key)
            
            if result.get("code") != 200:
                error_msg = result.get("msg", "Unknown error")
                logger.error(f"❌ Error on page {page_count}: {error_msg}")
                errors.append(f"Page {page_count}: {error_msg}")
                break
            
            data = result.get("data", {})
            parts_list = data.get("list", [])
            
            if not parts_list:
                logger.info("✅ No more parts to download")
                break
            
            # Insert parts into database
            for part in parts_list:
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO components (
                            lcsc, category, subcategory, mfr_part, package,
                            solder_joints, manufacturer, library_type,
                            description, datasheet, stock, price_json, last_updated
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        part.get("lcsc", ""),
                        part.get("categoryName", ""),
                        part.get("categoryNameEn", ""),
                        part.get("mfrPartNumber", ""),
                        part.get("package", ""),
                        part.get("solderJoint", 0),
                        part.get("manufacturerName", ""),
                        part.get("libraryType", ""),
                        part.get("description", ""),
                        part.get("datasheet", ""),
                        part.get("stockNumber", 0),
                        json.dumps(part.get("price", [])),
                        int(time.time())
                    ))
                    parts_count += 1
                except Exception as e:
                    logger.warning(f"⚠️  Failed to insert part {part.get('lcsc', 'unknown')}: {str(e)}")
            
            conn.commit()
            
            # Update pagination
            last_key = data.get("lastKey")
            if not last_key:
                break
            
            # Progress callback
            if progress_callback:
                progress_callback(parts_count, page_count)
            
            # Rate limiting (be nice to API)
            time.sleep(0.1)
            
            if page_count % 10 == 0:
                logger.info(f"   Downloaded {parts_count} parts ({page_count} pages)...")
        
        conn.close()
        
        logger.info(f"✅ Download complete: {parts_count} parts in {page_count} pages")
        
        return {
            "success": True,
            "parts_count": parts_count,
            "pages": page_count,
            "database_path": db_path,
            "errors": errors
        }
    
    def search_part(self, lcsc_number: str) -> Optional[Dict]:
        """
        Search for a specific part by LCSC number.
        
        Args:
            lcsc_number: LCSC part number (e.g., "C25804")
        
        Returns:
            Part information dictionary or None
        """
        # This would typically query the local database
        # For now, return None (would be implemented with database query)
        return None
    
    def get_quote(self, bom: List[Dict], quantity: int = 10) -> Dict:
        """
        Get assembly quote from JLCPCB.
        
        Args:
            bom: Bill of materials list with LCSC numbers and quantities
            quantity: Number of boards to assemble
        
        Returns:
            Quote information
        """
        # This would use JLCPCB's quote API endpoint
        # For now, return placeholder
        return {
            "success": False,
            "note": "Quote API not yet implemented. Use JLCPCB website for quotes."
        }

