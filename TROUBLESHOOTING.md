# MLX Fine-Tuning GUI - Troubleshooting Guide

## ✅ Installation Issues RESOLVED

### NPM Permission Issues ✅ FIXED
- **Issue**: `EACCES` permission errors with npm cache
- **Solution**: Using `--cache /tmp/npm-cache` flag bypasses permission issues
- **Status**: ✅ All dependencies installed successfully

## 🚀 Quick Start

### Option 1: Use the Startup Script (Recommended)
```bash
cd mlx-finetune-gui
./start.sh
```

### Option 2: Manual Startup
```bash
cd mlx-finetune-gui
npm run dev
```

## 🔍 Common Issues & Solutions

### 1. Backend Server Won't Start
**Symptoms**: Python errors, FastAPI not found
**Solution**:
```bash
source "/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/.venv/bin/activate"
pip install -r backend/requirements.txt
```

### 2. Frontend Build Errors
**Symptoms**: TypeScript or Vite errors
**Solution**:
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install --cache /tmp/npm-cache --no-optional
```

### 3. Electron Won't Launch
**Symptoms**: Electron app doesn't open
**Solution**:
```bash
npm run build
npm run electron
```

### 4. WebSocket Connection Issues
**Symptoms**: Real-time updates not working
**Check**:
- Backend server running on http://localhost:8000
- Frontend connecting to correct WebSocket endpoint
- No firewall blocking port 8000

### 5. Model Discovery Issues
**Symptoms**: No models showing in dropdown
**Check**:
- MLX models are in the correct directory
- Path configuration in backend matches your setup
- Models are in Hugging Face format

## 🧪 Testing

### Backend Health Check
```bash
curl http://localhost:8000/health
# Should return: {"status": "healthy"}
```

### Frontend Development Server
- Should be accessible at: http://localhost:5173
- Check browser console for any React errors

### Electron App
- Desktop application should launch automatically
- Menu bar should show "MLX Fine-Tune GUI"

## 📊 Performance Tips

### Memory Optimization
- Close unused applications before training
- Monitor Activity Monitor during fine-tuning
- Adjust batch size if memory issues occur

### Training Performance
- Use MPS acceleration (enabled by default)
- Monitor GPU usage in Activity Monitor
- Adjust sequence length if stability issues

## 🔧 Development Mode

### Hot Reload
- Frontend changes reload automatically
- Backend changes require server restart
- Electron main process changes require full restart

### Debug Mode
```bash
# Backend with debug logging
cd backend
python main.py --debug

# Frontend with debug info
cd frontend
npm run dev -- --debug
```

## 📁 File Structure Verification

Ensure all files are present:
```
mlx-finetune-gui/
├── ✅ package.json
├── ✅ src/main.ts
├── ✅ backend/
│   ├── ✅ main.py
│   └── ✅ requirements.txt
└── ✅ frontend/
    ├── ✅ package.json
    ├── ✅ src/App.tsx
    └── ✅ public/index.html
```

## 🆘 Still Having Issues?

1. **Check logs**: Look for error messages in terminal output
2. **Verify paths**: Ensure all file paths are correct for your system
3. **Check permissions**: Verify you have read/write access to project directory
4. **Restart**: Sometimes a simple restart resolves connection issues

## 🎯 Success Indicators

When everything is working correctly:
- ✅ Backend server starts without errors
- ✅ Frontend builds and serves without warnings
- ✅ Electron app launches as desktop application
- ✅ WebSocket connection established (check browser dev tools)
- ✅ Models appear in setup dropdown
- ✅ Real-time training updates work

## 📞 Support

If issues persist:
1. Check the implementation logs in the `logs/` directory
2. Review the backend terminal output for Python errors
3. Check browser developer tools for JavaScript errors
4. Verify MLX and model files are accessible