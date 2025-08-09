"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const electron_1 = require("electron");
// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
electron_1.contextBridge.exposeInMainWorld('electronAPI', {
    // File dialogs
    showOpenDialog: (options) => electron_1.ipcRenderer.invoke('show-open-dialog', options),
    showSaveDialog: (options) => electron_1.ipcRenderer.invoke('show-save-dialog', options),
    // Menu events
    onMenuOpenTrainingData: (callback) => electron_1.ipcRenderer.on('menu-open-training-data', callback),
    onMenuStartTraining: (callback) => electron_1.ipcRenderer.on('menu-start-training', callback),
    onMenuStopTraining: (callback) => electron_1.ipcRenderer.on('menu-stop-training', callback),
    // Platform info
    platform: process.platform,
});
