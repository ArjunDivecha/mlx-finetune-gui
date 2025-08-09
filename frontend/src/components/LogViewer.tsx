import React, { useEffect, useRef } from 'react';
import { Search, Download } from 'lucide-react';

interface LogViewerProps {
  logs: string[];
}

export const LogViewer: React.FC<LogViewerProps> = ({ logs }) => {
  const logContainerRef = useRef<HTMLDivElement>(null);
  const [searchTerm, setSearchTerm] = React.useState('');
  const [autoScroll, setAutoScroll] = React.useState(true);

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  const handleScroll = () => {
    if (logContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = logContainerRef.current;
      const isAtBottom = scrollTop + clientHeight >= scrollHeight - 10;
      setAutoScroll(isAtBottom);
    }
  };

  const getLogLineClass = (line: string) => {
    const lowerLine = line.toLowerCase();
    if (lowerLine.includes('error') || lowerLine.includes('failed') || lowerLine.includes('exception')) {
      return 'log-line log-error';
    } else if (lowerLine.includes('warning') || lowerLine.includes('warn')) {
      return 'log-line log-warning';
    } else if (lowerLine.includes('info') || lowerLine.includes('iter ') || lowerLine.includes('val loss')) {
      return 'log-line log-info';
    }
    return 'log-line';
  };

  const filteredLogs = searchTerm 
    ? logs.filter(log => log.toLowerCase().includes(searchTerm.toLowerCase()))
    : logs;

  const exportLogs = () => {
    const logContent = logs.join('\n');
    const blob = new Blob([logContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `training-logs-${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="h-96 flex flex-col">
      {/* Log Controls */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-700 space-y-3">
        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search logs..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
        </div>

        {/* Controls */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={autoScroll}
                onChange={(e) => setAutoScroll(e.target.checked)}
                className="rounded border-gray-300"
              />
              <span className="text-sm text-gray-600 dark:text-gray-400">Auto-scroll</span>
            </label>
            
            <div className="text-sm text-gray-500 dark:text-gray-400">
              {filteredLogs.length} / {logs.length} lines
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={exportLogs}
              disabled={logs.length === 0}
              className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 disabled:opacity-50 disabled:cursor-not-allowed"
              title="Export logs"
            >
              <Download className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Log Content */}
      <div 
        ref={logContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto bg-gray-50 dark:bg-gray-900 log-viewer scrollbar-thin"
      >
        {filteredLogs.length === 0 ? (
          <div className="p-8 text-center text-gray-500 dark:text-gray-400">
            {logs.length === 0 ? (
              <div>
                <div className="w-12 h-12 border-2 border-gray-300 dark:border-gray-600 border-dashed rounded-lg flex items-center justify-center mx-auto mb-2">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
                <p>No logs available</p>
                <p className="text-sm">Training logs will appear here in real-time</p>
              </div>
            ) : (
              <div>
                <p>No logs match your search</p>
                <p className="text-sm">Try a different search term</p>
              </div>
            )}
          </div>
        ) : (
          <div className="font-mono text-xs leading-relaxed">
            {filteredLogs.map((line, index) => (
              <div
                key={index}
                className={getLogLineClass(line)}
              >
                <span className="text-gray-400 dark:text-gray-500 select-none mr-3">
                  {(index + 1).toString().padStart(4, ' ')}
                </span>
                {searchTerm ? (
                  <span
                    dangerouslySetInnerHTML={{
                      __html: line.replace(
                        new RegExp(`(${searchTerm})`, 'gi'),
                        '<mark class="bg-yellow-200 dark:bg-yellow-800 px-1">$1</mark>'
                      )
                    }}
                  />
                ) : (
                  <span>{line}</span>
                )}
              </div>
            ))}
          </div>
        )}
        
        {/* Auto-scroll indicator */}
        {!autoScroll && logs.length > 0 && (
          <div className="sticky bottom-2 right-2 float-right">
            <button
              onClick={() => {
                if (logContainerRef.current) {
                  logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
                  setAutoScroll(true);
                }
              }}
              className="px-3 py-1 bg-primary-600 text-white text-xs rounded-full shadow-lg hover:bg-primary-700 transition-colors"
            >
              Scroll to bottom
            </button>
          </div>
        )}
      </div>
    </div>
  );
};