# Quick Test Guide

## 🚀 Fastest Way to Test Everything

### 1. Start Backend (Terminal 1)
```bash
cd /Users/abiralshakya/Documents/hackprinceton2025/neuro-geometric-placer
python deploy_simple.py
```

Wait for: `Uvicorn running on http://0.0.0.0:8000`

### 2. Run Quick Test (Terminal 2)
```bash
cd /Users/abiralshakya/Documents/hackprinceton2025/neuro-geometric-placer
python quick_test.py
```

**Expected output:**
```
✅ Backend is running
✅ KiCAD export successful!
✅ All tests passed!
```

### 3. Test Frontend (Optional)
```bash
# In Terminal 3
streamlit run frontend/app_dielectric.py
```

Then:
1. Open http://localhost:8501
2. Click "Generate Design" tab
3. Enter: "Design a simple LED circuit"
4. Click "Generate Design"
5. Click "Optimize Design"
6. Click "Export to KiCad"

## ⚡ What Gets Tested

- ✅ Backend API health
- ✅ KiCAD export functionality
- ✅ File generation
- ✅ Error handling

## 🐛 If Tests Fail

### Backend not running
```bash
# Make sure backend is running
python deploy_simple.py
```

### KiCAD export fails
- This is OK! System falls back to manual exporter
- Check console for: "KiCAD Python API not available, using manual exporter"
- Export will still work, just lower quality

### Port already in use
```bash
# Kill existing process
lsof -ti:8000 | xargs kill -9
```

## 📝 Full Test (Takes Longer)
```bash
python quick_test.py --full
```

This also tests the optimization endpoint.

