
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import ChatInterface from './components/ChatInterface';
import HealthPage from './components/HealthPage';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<ChatInterface />} />
          <Route path="/health" element={<HealthPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
