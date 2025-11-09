# Dedalus Labs Deployment Guide

## What You Need to Do

### 1. Create `dedalus.json` Configuration
✅ **DONE** - Created `dedalus.json` in project root

### 2. Ensure Entrypoint Exists
The entrypoint should be: `src/backend/mcp_servers/ngp_server.py`

Let me check if this file exists and is properly configured.

### 3. Set Environment Variables in Dedalus Dashboard
In your Dedalus dashboard:
1. Go to your server settings
2. Add environment variables:
   - `XAI_API_KEY` = your xAI API key
   - `DEDALUS_API_KEY` = your Dedalus API key

### 4. Ensure Requirements File
Make sure `requirements.txt` includes all dependencies.

### 5. Project Structure
Dedalus needs to detect the project type. Ensure:
- `dedalus.json` in root (✅ done)
- `requirements.txt` in root (should exist)
- Python entrypoint file exists

## Troubleshooting

### "Unable to detect project type"
**Fix**: Ensure `dedalus.json` is in the root directory and properly formatted.

### Build fails
**Check**:
1. All dependencies in `requirements.txt`
2. Entrypoint file exists and is executable
3. Environment variables set in Dedalus dashboard

### Server won't start
**Check**:
1. API keys are set correctly
2. Entrypoint file has correct imports
3. All dependencies installed

## Next Steps

1. **Commit and push** `dedalus.json` to your GitHub repo
2. **Redeploy** in Dedalus dashboard
3. **Check logs** for any errors
4. **Test** MCP tools once deployed

---

**Note**: If Dedalus still fails, we can run agents locally without Dedalus (current setup works fine).

