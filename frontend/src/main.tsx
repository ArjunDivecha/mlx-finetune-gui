import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.tsx';
import './App.css';

// Show the body once React is ready
document.addEventListener('DOMContentLoaded', () => {
  document.body.style.visibility = 'visible';
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);