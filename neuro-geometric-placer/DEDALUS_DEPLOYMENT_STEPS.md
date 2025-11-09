# Dedalus Labs Deployment: Step-by-Step Guide

## What You Need to Do

### Step 1: Commit Files to GitHub

Make sure these files are in your GitHub repo:
- ✅ `dedalus.json` (project root)
- ✅ `dedalus_entrypoint.py` (project root)
- ✅ `requirements.txt` (project root)
- ✅ `src/backend/mcp_servers/ngp_server.py` (MCP server)

```bash
git add dedalus.json dedalus_entrypoint.py
git commit -m "Add Dedalus deployment configuration"
git push
```

### Step 2: Configure Environment Variables in Dedalus

In your Dedalus dashboard:
1. Go to your server: `hackprincetonfall2025`
2. Click on "Settings" or "Environment Variables"
3. Add these variables:
   - `XAI_API_KEY` = `your_xai_api_key_here`
   - `DEDALUS_API_KEY` = `your_dedalus_api_key_here`

### Step 3: Update dedalus.json (if needed)

The `dedalus.json` should have:
```json
{
  "name": "dielectric-mcp-server",
  "type": "mcp-server",
  "entrypoint": "dedalus_entrypoint.py",
  "runtime": {
    "type": "python",
    "version": "3.12"
  }
}
```

### Step 4: Redeploy in Dedalus

1. Go to your server in Dedalus dashboard
2. Click "Redeploy" or "Deploy"
3. Wait for build to complete
4. Check logs for errors

### Step 5: Verify Deployment

Once deployed, check:
- Server status: Should be "Running"
- Logs: Should show "Starting Dielectric MCP Server"
- Tools: Should list available tools

## Troubleshooting

### "Unable to detect project type"
**Fix**: 
- Ensure `dedalus.json` is in root directory
- Check JSON syntax is valid
- Make sure `type: "mcp-server"` is set

### "Build failed"
**Check**:
- All dependencies in `requirements.txt`
- `dedalus_entrypoint.py` exists and is executable
- Python version matches (3.12)

### "Import errors"
**Fix**:
- Check `dedalus_entrypoint.py` has correct imports
- Ensure all dependencies are in `requirements.txt`
- Verify project structure

### "Server won't start"
**Check**:
- Environment variables set correctly
- API keys are valid
- Entrypoint file is executable (`chmod +x dedalus_entrypoint.py`)

## Alternative: Run Without Dedalus

If Dedalus continues to fail, the system works perfectly without it:
- All agents run locally
- Full functionality available
- No Dedalus required for HackPrinceton demo

**Current setup works great without Dedalus!**

## Quick Test

After deployment, test the MCP server:
```bash
# Should show available tools
python dedalus_entrypoint.py
```

---

**Note**: If Dedalus setup is complex, focus on the core product - it's already impressive without Dedalus!

