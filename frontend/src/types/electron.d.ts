export interface ElectronAPI {
  showOpenDialog: (options: {
    title?: string;
    defaultPath?: string;
    buttonLabel?: string;
    filters?: Array<{ name: string; extensions: string[] }>;
    properties?: ('openFile' | 'openDirectory' | 'multiSelections' | 'showHiddenFiles')[];
  }) => Promise<{
    canceled: boolean;
    filePaths: string[];
  }>;
  
  showSaveDialog: (options: {
    title?: string;
    defaultPath?: string;
    buttonLabel?: string;
    filters?: Array<{ name: string; extensions: string[] }>;
  }) => Promise<{
    canceled: boolean;
    filePath?: string;
  }>;
  
  onMenuOpenTrainingData: (callback: () => void) => void;
  onMenuStartTraining: (callback: () => void) => void;
  onMenuStopTraining: (callback: () => void) => void;
  
  platform: string;
}

declare global {
  interface Window {
    electronAPI?: ElectronAPI;
  }
}

export {};