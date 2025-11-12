# JLCPCB Integration Implementation

## Overview

JLCPCB integration has been implemented to enable real-time component availability checking and pricing from JLCPCB's parts database.

## Files Created

1. **`src/backend/integrations/jlcpcb_client.py`**
   - `JLCPCBClient` class for API authentication and communication
   - Handles token management and rate limiting
   - Downloads full parts database to SQLite
   - Supports paginated API requests

2. **`src/backend/integrations/jlcpcb_parts.py`**
   - `JLCPCBPartsManager` class for local database management
   - Fast parametric search with filters
   - Package to KiCad footprint mapping
   - Part information retrieval

## Integration Points

### Design Generator Agent
- Updated `check_jlcpcb_availability()` to use real database
- Falls back to cached parts if database unavailable
- Returns stock, pricing, and library type information

## Setup Instructions

### 1. Get JLCPCB API Credentials

1. Visit https://jlcpcb.com/
2. Log in to your account
3. Go to: Account → API Management
4. Create API Key
5. Save your `appKey` and `appSecret`

### 2. Set Environment Variables

```bash
export JLCPCB_API_KEY="your_app_key_here"
export JLCPCB_API_SECRET="your_app_secret_here"
```

### 3. Download Parts Database (Optional)

The system will automatically use the database if available. To download:

```python
from src.backend.integrations.jlcpcb_client import JLCPCBClient

client = JLCPCBClient()
result = client.download_full_database("data/jlcpcb_parts.db")
print(f"Downloaded {result['parts_count']} parts")
```

## Usage

### Check Component Availability

```python
from src.backend.agents.design_generator_agent import ComponentLibrary

library = ComponentLibrary()
part_info = library.check_jlcpcb_availability("C25804")  # LCSC number
# or
part_info = library.check_jlcpcb_availability("0805")  # Package name

if part_info:
    print(f"Stock: {part_info['stock']}")
    print(f"Price: ${part_info['price']}")
    print(f"Type: {part_info['library_type']}")
```

### Search Parts

```python
from src.backend.integrations.jlcpcb_parts import JLCPCBPartsManager

manager = JLCPCBPartsManager()
results = manager.search_parts(
    query="10k resistor",
    package="0603",
    library_type="Basic",
    in_stock=True,
    limit=10
)

for part in results:
    print(f"{part['lcsc']}: {part['description']} - ${part['price']}")
```

## Features

- ✅ API authentication with token management
- ✅ Full database download (100k+ parts)
- ✅ Fast local SQLite database
- ✅ Parametric search with filters
- ✅ Package to footprint mapping
- ✅ Stock and pricing information
- ✅ Basic/Extended library type filtering
- ✅ Automatic fallback if database unavailable

## Test Zip File

Created `NFCREAD-001-RevA.zip` in project root for testing folder upload functionality.

## Next Steps

1. Set up JLCPCB API credentials
2. Download parts database (optional, improves performance)
3. Test component availability checking
4. Integrate with BOM generation
5. Add quote generation API (future enhancement)

