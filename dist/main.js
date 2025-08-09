"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const electron_1 = require("electron");
const path = __importStar(require("path"));
const fs = __importStar(require("fs"));
const child_process_1 = require("child_process");
const axios_1 = __importDefault(require("axios"));
// Keep a global reference of the window object
let mainWindow = null;
let backendProcess = null;
const BACKEND_PORT = 8000;
const createWindow = () => {
    // Create the browser window
    mainWindow = new electron_1.BrowserWindow({
        width: 1400,
        height: 900,
        minWidth: 1024,
        minHeight: 768,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.js')
        },
        titleBarStyle: 'hiddenInset',
        show: false
    });
    // Load the frontend
    const isDev = process.env.NODE_ENV === 'development';
    if (isDev) {
        // In development, load from the built frontend dist folder
        const frontendPath = `file://${path.join(__dirname, '../frontend/dist/index.html')}`;
        console.log('Loading frontend from:', frontendPath);
        mainWindow.loadURL(frontendPath);
    }
    else {
        // In production, load from the packaged resources
        const frontendPath = `file://${path.join(__dirname, '../frontend/dist/index.html')}`;
        mainWindow.loadURL(frontendPath);
    }
    // Show window when ready
    mainWindow.once('ready-to-show', () => {
        if (mainWindow) {
            mainWindow.show();
        }
    });
    // Open DevTools in development
    if (isDev) {
        mainWindow.webContents.openDevTools();
    }
    // Handle window closed
    mainWindow.on('closed', () => {
        mainWindow = null;
    });
};
const startBackendServer = async () => {
    return new Promise((resolve, reject) => {
        const backendPath = path.join(__dirname, '../backend');
        const pythonPath = '/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/.venv/bin/python';
        // Check if python path exists
        if (!fs.existsSync(pythonPath)) {
            reject(new Error(`Python venv not found at: ${pythonPath}`));
            return;
        }
        backendProcess = (0, child_process_1.spawn)(pythonPath, ['-m', 'uvicorn', 'main:app', '--host', '0.0.0.0', '--port', BACKEND_PORT.toString()], {
            cwd: backendPath,
            stdio: ['pipe', 'pipe', 'pipe']
        });
        backendProcess.stdout?.on('data', (data) => {
            console.log(`Backend: ${data}`);
        });
        backendProcess.stderr?.on('data', (data) => {
            console.error(`Backend Error: ${data}`);
        });
        backendProcess.on('error', (error) => {
            console.error('Failed to start backend:', error);
            reject(error);
        });
        // Wait for server to be ready
        const checkServer = async (attempts = 0) => {
            if (attempts > 30) {
                reject(new Error('Backend server failed to start within 30 seconds'));
                return;
            }
            try {
                await axios_1.default.get(`http://localhost:${BACKEND_PORT}/health`);
                resolve();
            }
            catch (error) {
                setTimeout(() => checkServer(attempts + 1), 1000);
            }
        };
        setTimeout(() => checkServer(), 2000);
    });
};
const stopBackendServer = () => {
    if (backendProcess) {
        backendProcess.kill('SIGTERM');
        backendProcess = null;
    }
};
// App event handlers
electron_1.app.whenReady().then(async () => {
    try {
        console.log('Starting backend server...');
        await startBackendServer();
        console.log('Backend server started successfully');
        createWindow();
        // macOS specific behavior
        electron_1.app.on('activate', () => {
            if (electron_1.BrowserWindow.getAllWindows().length === 0) {
                createWindow();
            }
        });
    }
    catch (error) {
        console.error('Failed to start application:', error);
        electron_1.dialog.showErrorBox('Startup Error', `Failed to start backend server: ${error}`);
        electron_1.app.quit();
    }
});
electron_1.app.on('window-all-closed', () => {
    stopBackendServer();
    if (process.platform !== 'darwin') {
        electron_1.app.quit();
    }
});
electron_1.app.on('before-quit', () => {
    stopBackendServer();
});
// IPC handlers for file dialogs
electron_1.ipcMain.handle('show-open-dialog', async (event, options) => {
    const result = await electron_1.dialog.showOpenDialog(mainWindow, options);
    return result;
});
electron_1.ipcMain.handle('show-save-dialog', async (event, options) => {
    const result = await electron_1.dialog.showSaveDialog(mainWindow, options);
    return result;
});
// Menu setup
const createMenu = () => {
    const template = [
        {
            label: 'MLX Fine-Tune GUI',
            submenu: [
                { role: 'about' },
                { type: 'separator' },
                { role: 'services' },
                { type: 'separator' },
                { role: 'hide' },
                { role: 'hideOthers' },
                { role: 'unhide' },
                { type: 'separator' },
                { role: 'quit' }
            ]
        },
        {
            label: 'File',
            submenu: [
                {
                    label: 'Open Training Data...',
                    accelerator: 'CmdOrCtrl+O',
                    click: () => {
                        mainWindow?.webContents.send('menu-open-training-data');
                    }
                },
                { type: 'separator' },
                { role: 'close' }
            ]
        },
        {
            label: 'View',
            submenu: [
                { role: 'reload' },
                { role: 'forceReload' },
                { role: 'toggleDevTools' },
                { type: 'separator' },
                { role: 'resetZoom' },
                { role: 'zoomIn' },
                { role: 'zoomOut' },
                { type: 'separator' },
                { role: 'togglefullscreen' }
            ]
        },
        {
            label: 'Training',
            submenu: [
                {
                    label: 'Start Training',
                    accelerator: 'CmdOrCtrl+R',
                    click: () => {
                        mainWindow?.webContents.send('menu-start-training');
                    }
                },
                {
                    label: 'Stop Training',
                    accelerator: 'CmdOrCtrl+.',
                    click: () => {
                        mainWindow?.webContents.send('menu-stop-training');
                    }
                }
            ]
        }
    ];
    const menu = electron_1.Menu.buildFromTemplate(template);
    electron_1.Menu.setApplicationMenu(menu);
};
electron_1.app.whenReady().then(() => {
    createMenu();
});
