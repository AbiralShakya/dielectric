"""
JLCPCB Uploader

Uploads PCB manufacturing files to JLCPCB:
- Gerber files
- Drill files
- BOM
- CPL (component placement)
- Quote generation
- Order placement
"""

import logging
import requests
import zipfile
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class JLCPCBUploader:
    """
    Upload manufacturing files to JLCPCB.
    
    Features:
    - Gerber file upload
    - Drill file upload
    - BOM upload
    - CPL (component placement) upload
    - Quote generation
    - Order placement
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize JLCPCB uploader.
        
        Args:
            api_key: Optional JLCPCB API key
        """
        self.api_key = api_key
        self.base_url = "https://jlcpcb.com/api"
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def create_zip_package(
        self,
        gerber_files: Dict[str, str],
        drill_file: Optional[str] = None,
        bom_file: Optional[str] = None,
        cpl_file: Optional[str] = None,
        output_path: str = "jlcpcb_package.zip"
    ) -> str:
        """
        Create ZIP package for JLCPCB upload.
        
        Args:
            gerber_files: Dictionary of layer names to file paths
            drill_file: Drill file path
            bom_file: BOM file path
            cpl_file: CPL (component placement) file path
            output_path: Output ZIP file path
        
        Returns:
            Path to created ZIP file
        """
        zip_path = Path(output_path)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add Gerber files
            for layer_name, file_path in gerber_files.items():
                zipf.write(file_path, f"gerber/{Path(file_path).name}")
            
            # Add drill file
            if drill_file:
                zipf.write(drill_file, f"drill/{Path(drill_file).name}")
            
            # Add BOM
            if bom_file:
                zipf.write(bom_file, f"bom/{Path(bom_file).name}")
            
            # Add CPL
            if cpl_file:
                zipf.write(cpl_file, f"cpl/{Path(cpl_file).name}")
        
        logger.info(f"Created JLCPCB package: {zip_path}")
        return str(zip_path)
    
    async def upload_for_quote(
        self,
        zip_file: str,
        board_parameters: Optional[Dict] = None
    ) -> Dict:
        """
        Upload files to JLCPCB for quote.
        
        Args:
            zip_file: Path to ZIP package
            board_parameters: Optional board parameters (thickness, color, etc.)
        
        Returns:
            Quote information
        """
        # JLCPCB upload endpoint (simplified - actual API may differ)
        upload_url = f"{self.base_url}/pcb/upload"
        
        board_params = board_parameters or {
            "thickness": 1.6,  # mm
            "color": "green",
            "quantity": 5,
            "surface_finish": "HASL"
        }
        
        try:
            with open(zip_file, 'rb') as f:
                files = {'file': f}
                data = board_params
                
                response = self.session.post(
                    upload_url,
                    files=files,
                    data=data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    quote = response.json()
                    logger.info(f"Quote received: ${quote.get('price', 'N/A')}")
                    return {
                        "success": True,
                        "quote": quote,
                        "order_id": quote.get("order_id")
                    }
                else:
                    logger.error(f"Upload failed: {response.status_code} - {response.text}")
                    return {
                        "success": False,
                        "error": f"HTTP {response.status_code}",
                        "details": response.text
                    }
        
        except Exception as e:
            logger.error(f"Upload exception: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def place_order(
        self,
        quote_id: str,
        shipping_info: Dict
    ) -> Dict:
        """
        Place order with JLCPCB.
        
        Args:
            quote_id: Quote ID from upload_for_quote
            shipping_info: Shipping information
        
        Returns:
            Order confirmation
        """
        order_url = f"{self.base_url}/pcb/order"
        
        data = {
            "quote_id": quote_id,
            **shipping_info
        }
        
        try:
            response = self.session.post(
                order_url,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                order = response.json()
                logger.info(f"Order placed: {order.get('order_id')}")
                return {
                    "success": True,
                    "order": order
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "details": response.text
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_order_status(self, order_id: str) -> Dict:
        """
        Get order status from JLCPCB.
        
        Args:
            order_id: Order ID
        
        Returns:
            Order status information
        """
        status_url = f"{self.base_url}/pcb/order/{order_id}/status"
        
        try:
            response = self.session.get(status_url, timeout=30)
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "status": response.json()
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}"
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

