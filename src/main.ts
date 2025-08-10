import { app, BrowserWindow, ipcMain, dialog, Menu } from 'electron';
import * as path from 'path';
import * as fs from 'fs';
import { spawn, ChildProcess } from 'child_process';
import axios from 'axios';

// Keep a global reference of the window object
let mainWindow: BrowserWindow | null = null;
let backendProcess: ChildProcess | null = null;
const BACKEND_PORT = 8000;

const createWindow = (): void => {
  console.log('🪟 Creating main window...');
  
  // Create the browser window
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1024,
    minHeight: 768,
    x: 100,  // Position window at visible location
    y: 100,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    titleBarStyle: 'hiddenInset',
    show: true,  // Show immediately
    alwaysOnTop: true,  // Force on top initially
    title: 'MLX Fine-tuning GUI',  // Set clear title
    skipTaskbar: false,
    resizable: true,
    center: true
  });

  console.log('✅ BrowserWindow created');
  console.log('📍 Window position:', mainWindow.getPosition());
  console.log('📏 Window size:', mainWindow.getSize());
  console.log('👀 Window visible:', mainWindow.isVisible());

  // Load the frontend
  const isDev = process.env.NODE_ENV === 'development';
  
  // Always load from the built frontend dist folder
  const frontendPath = `file://${path.join(__dirname, '../frontend/dist/index.html')}`;
  console.log('🔗 Loading frontend from:', frontendPath);
  
  // Check if the file exists
  if (require('fs').existsSync(path.join(__dirname, '../frontend/dist/index.html'))) {
    console.log('✅ Frontend index.html exists');
  } else {
    console.error('❌ Frontend index.html NOT FOUND!');
  }
  
  mainWindow.loadURL(frontendPath);
  console.log('📄 Frontend loading started');

  // Aggressively show the window when ready
  mainWindow.once('ready-to-show', () => {
    console.log('Window ready-to-show event fired');
    if (mainWindow) {
      mainWindow.show();
      mainWindow.focus();
      mainWindow.moveTop();
      console.log('Window shown and focused');
    }
  });
  
  // Remove alwaysOnTop after 3 seconds to allow normal window behavior  
  setTimeout(() => {
    if (mainWindow) {
      mainWindow.setAlwaysOnTop(false);
      mainWindow.focus();
      mainWindow.moveTop();
    }
  }, 3000);
  
  // Force show window after 5 seconds if ready-to-show doesn't fire
  setTimeout(() => {
    if (mainWindow && !mainWindow.isVisible()) {
      console.log('Force showing window after timeout');
      mainWindow.show();
      mainWindow.focus();
    }
  }, 5000);

  // Open DevTools in development
  if (isDev) {
    mainWindow.webContents.openDevTools();
  }

  // Handle window closed
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
};

const startBackendServer = async (): Promise<void> => {
  return new Promise((resolve, reject) => {
    const backendPath = path.join(__dirname, '../backend');
    const pythonPath = 'python3';

    backendProcess = spawn(pythonPath, ['-m', 'uvicorn', 'main:app', '--host', '0.0.0.0', '--port', BACKEND_PORT.toString()], {
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
    const checkServer = async (attempts = 0): Promise<void> => {
      if (attempts > 60) {
        reject(new Error('Backend server failed to start within 60 seconds'));
        return;
      }

      try {
        // Use IPv4 explicitly to match backend binding
        const response = await axios.get(`http://127.0.0.1:${BACKEND_PORT}/training/status`, { timeout: 5000 });
        if (response.status === 200) {
          console.log(`Backend health check passed after ${attempts + 1} attempts`);
          resolve();
          return;
        }
        throw new Error(`Unexpected status: ${response.status}`);
      } catch (error) {
        console.log(`Backend health check attempt ${attempts + 1} failed:`, error.message);
        setTimeout(() => checkServer(attempts + 1), 1000);
      }
    };

    setTimeout(() => checkServer(), 2000);
  });
};

const stopBackendServer = (): void => {
  if (backendProcess) {
    backendProcess.kill('SIGTERM');
    backendProcess = null;
  }
};

// App event handlers
app.whenReady().then(async () => {
  try {
    console.log('Starting backend server...');
    await startBackendServer();
    console.log('Backend server started successfully');
    
    createWindow();
    
    // macOS specific behavior
    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
      }
    });
  } catch (error) {
    console.error('Failed to start application:', error);
    dialog.showErrorBox(
      'Startup Error', 
      `Failed to start backend server: ${error}`
    );
    app.quit();
  }
});

app.on('window-all-closed', () => {
  stopBackendServer();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  stopBackendServer();
});

// IPC handlers for file dialogs
ipcMain.handle('show-open-dialog', async (event, options) => {
  const result = await dialog.showOpenDialog(mainWindow!, options);
  return result;
});

ipcMain.handle('show-save-dialog', async (event, options) => {
  const result = await dialog.showSaveDialog(mainWindow!, options);
  return result;
});

// Menu setup
const createMenu = (): void => {
  const template: Electron.MenuItemConstructorOptions[] = [
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

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
};

app.whenReady().then(() => {
  createMenu();
});