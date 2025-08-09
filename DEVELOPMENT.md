# Development Guide

This document provides detailed information for developers working on the MLX Fine-Tuning GUI project.

## 🧭 Project Status

### Current Development State (as of creation)

**✅ Completed Components:**
- Project structure and build configuration
- Backend FastAPI server with WebSocket support
- Redux store architecture with all slices
- Electron main process with backend integration
- WebSocket hook for real-time communication
- Training process management and monitoring
- Model discovery and configuration system

**🚧 In Progress:**
- React components and UI implementation
- Training configuration forms
- Real-time metrics visualization

**📋 Next Priority Items:**
1. Complete React component implementation
2. Implement training configuration UI
3. Create real-time chart components
4. Add model comparison interface
5. Implement error handling and recovery

## 🏗️ Architecture Deep Dive

### Data Flow Architecture

```
User Interaction (React)
    ↓
Redux Store (State Management)
    ↓
API Layer (Axios)
    ↓
FastAPI Backend
    ↓
MLX Training Process
    ↓
WebSocket Updates
    ↓
Redux Store Updates
    ↓
React Component Re-renders
```

### File Structure Explained

```
mlx-finetune-gui/
├── backend/
│   ├── main.py                 # FastAPI app with training management
│   └── requirements.txt        # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/         # Reusable UI components
│   │   │   ├── Layout.tsx      # Main app layout
│   │   │   ├── Sidebar.tsx     # Navigation sidebar
│   │   │   └── ...            # Other components (TO DO)
│   │   ├── pages/             # Full page components
│   │   │   ├── SetupPage.tsx   # Model/dataset setup (TO DO)
│   │   │   ├── TrainingPage.tsx # Real-time training monitor (TO DO)
│   │   │   ├── ResultsPage.tsx # Training results/history (TO DO)
│   │   │   └── ComparePage.tsx # Model comparison (TO DO)
│   │   ├── store/             # Redux state management
│   │   │   ├── store.ts       # ✅ Store configuration
│   │   │   └── slices/        # State slices
│   │   │       ├── trainingSlice.ts  # ✅ Training state
│   │   │       ├── modelsSlice.ts    # ✅ Models state
│   │   │       └── uiSlice.ts        # ✅ UI state
│   │   ├── hooks/             # Custom React hooks
│   │   │   └── useWebSocket.ts # ✅ WebSocket communication
│   │   ├── services/          # API communication (TO DO)
│   │   ├── types/             # TypeScript type definitions (TO DO)
│   │   └── App.tsx            # ✅ Main app component
│   ├── public/                # Static assets (TO DO)
│   └── package.json           # ✅ Frontend dependencies
├── src/
│   └── main.ts                # ✅ Electron main process
└── package.json               # ✅ Root configuration
```

## 🔧 Development Workflow

### Setting Up Development Environment

1. **Clone and Install Dependencies:**
```bash
cd mlx-finetune-gui
npm install
cd frontend && npm install
```

2. **Install Backend Dependencies:**
```bash
cd backend
/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun\ LLM\ Writing/local_qwen/.venv/bin/pip install -r requirements.txt
```

3. **Start Development Mode:**
```bash
npm run dev
```

### Development Scripts Explained

```bash
npm run dev              # Start full dev environment (backend + frontend + electron)
npm run dev:backend      # Start only FastAPI backend server
npm run dev:frontend     # Start only Electron with React dev server
npm run build:frontend   # Build React app for production
npm run build:main       # Compile Electron main process TypeScript
npm run build           # Build everything for production
npm run start           # Start Electron with built files
npm run dist            # Package for distribution (macOS)
```

## 🏃‍♂️ Next Development Steps

### Phase 1: Complete Core UI (2-3 days)

**Priority 1: Basic Layout and Navigation**
```typescript
// TO DO: Create these components
frontend/src/components/
├── Layout.tsx          # Main app shell with sidebar
├── Sidebar.tsx         # Navigation with icons
├── Header.tsx          # Top bar with theme toggle
└── StatusBar.tsx       # Bottom status bar
```

**Priority 2: Setup Page**
```typescript
// TO DO: Create setup page with:
frontend/src/pages/SetupPage.tsx
- ModelSelector component (dropdown)
- DatasetUpload component (file picker)
- ConfigurationPanel component (hyperparameters)
```

**Priority 3: Training Page**
```typescript
// TO DO: Create training monitoring:
frontend/src/pages/TrainingPage.tsx
- TrainingControls component (start/stop buttons)
- MetricsChart component (real-time loss curves)
- LogViewer component (scrollable training logs)
- ProgressIndicator component (progress bar with ETA)
```

### Phase 2: Advanced Features (3-4 days)

**Results and Analytics**
```typescript
// TO DO: Create results analysis:
frontend/src/pages/ResultsPage.tsx
- TrainingHistory component
- MetricsComparison component
- ExportResults component
```

**Model Comparison Interface**
```typescript
// TO DO: Create comparison tools:
frontend/src/pages/ComparePage.tsx
- ModelSelector component (side-by-side)
- PromptInput component
- ResponseComparison component
- ScoringInterface component
```

### Phase 3: Polish and Testing (1-2 days)

**Error Handling and Recovery**
- Connection loss recovery
- Training failure handling
- File validation and error messages
- Graceful degradation

**UI Polish**
- Loading states and animations
- Responsive design improvements
- Keyboard shortcuts
- Accessibility improvements

## 🧩 Component Design Patterns

### State Management Pattern
```typescript
// Use typed hooks for Redux
import { useAppDispatch, useAppSelector } from '../store/hooks';

const MyComponent = () => {
  const dispatch = useAppDispatch();
  const trainingState = useAppSelector(state => state.training.state);
  
  // Component logic here
};
```

### API Communication Pattern
```typescript
// Create service layer for API calls
// frontend/src/services/api.ts (TO DO)
import axios from 'axios';

export const trainingApi = {
  start: (config: TrainingConfig) => axios.post('/training/start', config),
  stop: () => axios.post('/training/stop'),
  getStatus: () => axios.get('/training/status'),
};
```

### WebSocket Integration Pattern
```typescript
// Use the WebSocket hook in components
import { useWebSocket } from '../hooks/useWebSocket';

const TrainingPage = () => {
  const { isConnected } = useWebSocket();
  
  // Real-time updates automatically handled via Redux
};
```

## 🔍 Debugging and Testing

### Backend Debugging
```bash
# Start backend manually with debugging
cd backend
/Users/.../local_qwen/.venv/bin/python -m uvicorn main:app --reload --log-level debug
```

### Frontend Debugging
- Electron DevTools automatically open in development mode
- Redux DevTools extension supported
- React DevTools available

### Common Issues and Solutions

**Backend Won't Start:**
- Check Python venv path in `src/main.ts`
- Verify all dependencies installed
- Check port 8000 not already in use

**Models Not Listed:**
- Verify model directory exists and has proper permissions
- Check model directories contain `config.json` files
- Review backend logs for errors

**WebSocket Connection Issues:**
- Check backend server is running
- Verify CORS configuration
- Review browser console for connection errors

**Training Process Fails:**
- Check MLX environment setup
- Verify training data format (JSONL with proper structure)
- Review training logs for specific error messages

## 📚 Key Dependencies

### Backend Dependencies
```python
fastapi==0.104.1        # Web framework
uvicorn[standard]==0.24.0  # ASGI server
websockets==12.0        # WebSocket support
pydantic==2.5.0         # Data validation
PyYAML==6.0.1           # Configuration files
```

### Frontend Dependencies
```json
{
  "react": "^18.2.0",           // UI framework
  "react-redux": "^9.0.4",     // Redux integration
  "@reduxjs/toolkit": "^2.0.1", // State management
  "socket.io-client": "^4.7.4", // WebSocket client
  "axios": "^1.6.2",           // HTTP client
  "chart.js": "^4.4.0",        // Charts (TO DO: implement)
  "tailwindcss": "^3.3.6"      // Styling (TO DO: configure)
}
```

### Electron Dependencies
```json
{
  "electron": "^28.0.0",        // Desktop app framework
  "typescript": "^5.3.0",      // TypeScript support
  "concurrently": "^8.2.2"     // Run multiple processes
}
```

## 🚀 Deployment Considerations

### Build Process
```bash
npm run build      # Build frontend and main process
npm run dist       # Create macOS installer (.dmg)
```

### Distribution
- Target: macOS 10.15+ (Apple Silicon optimized)
- Package format: DMG installer
- Code signing: Configure in `build.mac` section of `package.json`

### Environment Variables
Consider adding these for production:
- `NODE_ENV=production`
- `BACKEND_PORT=8000`
- `MODEL_BASE_PATH=/path/to/models`

This development guide provides everything needed to continue building the MLX Fine-Tuning GUI in future sessions. The architecture is solid, core systems are in place, and the next steps are clearly defined with specific file paths and component requirements.