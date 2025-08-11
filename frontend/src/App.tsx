import React, { useEffect } from 'react';
import { HashRouter as Router, Routes, Route } from 'react-router-dom';
import { Provider } from 'react-redux';
import { store } from './store/store';
import { Layout } from './components/Layout';
import { SetupPage } from './pages/SetupPage';
import { TrainingPage } from './pages/TrainingPage';
import { ResultsPage } from './pages/ResultsPage';
import { ComparePage } from './pages/ComparePage';
import { useWebSocket } from './hooks/useWebSocket';
import './App.css';

const AppContent: React.FC = () => {
  const { connect, disconnect } = useWebSocket();

  useEffect(() => {
    // Connect to WebSocket when app starts
    connect();

    // Cleanup on unmount
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<SetupPage />} />
          <Route path="/setup" element={<SetupPage />} />
          <Route path="/training" element={<TrainingPage />} />
          <Route path="/results" element={<ResultsPage />} />
          <Route path="/compare" element={<ComparePage />} />
        </Routes>
      </Layout>
    </Router>
  );
};

const App: React.FC = () => {
  return (
    <Provider store={store}>
      <div className="App">
        <AppContent />
      </div>
    </Provider>
  );
};

export default App;