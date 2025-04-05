import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import './App.css';
import DataUploadForm from './components/DataUploadForm';
import PersonalizedModeConfig from './components/PersonalizedModeConfig';
import ColdStartModeConfig from './components/ColdStartModeConfig';
import ModelList from './components/ModelList';
import TestRecommendations from './components/TestRecommendations';

const API_BASE_URL = 'https://amalragh-reccomend.hf.space';

function App() {
  const [uploadedFiles, setUploadedFiles] = useState({
    interaction: null,
    product_metadata: null,
    user_data: null,
    image_data: null,
  });
  
  const [activeModel, setActiveModel] = useState(null);
  
  const handleFileUpload = (dataType, filepath) => {
    setUploadedFiles(prev => ({
      ...prev,
      [dataType]: filepath
    }));
  };
  
  const handleSetActiveModel = (model) => {
    setActiveModel(model);
  };

  return (
    <Router>
      <div className="app">
        <header className="app-header">
          <h1>E-commerce Recommendation System</h1>
          <nav>
            <ul className="nav-menu">
              <li><Link to="/">Data Upload</Link></li>
              <li><Link to="/personalized">Personalized Mode</Link></li>
              <li><Link to="/cold-start">Cold Start Mode</Link></li>
              <li><Link to="/models">Model Management</Link></li>
              <li><Link to="/test">Test Recommendations</Link></li>
            </ul>
          </nav>
        </header>

        <main className="app-main">
          <Routes>
            <Route path="/" element={
              <DataUploadForm 
                onFileUpload={handleFileUpload} 
                uploadedFiles={uploadedFiles}
              />
            } />
            <Route path="/personalized" element={
              <PersonalizedModeConfig 
                uploadedFiles={uploadedFiles}
              />
            } />
            <Route path="/cold-start" element={
              <ColdStartModeConfig
                uploadedFiles={uploadedFiles}
              />
            } />
            <Route path="/models" element={
              <ModelList
                onSetActiveModel={handleSetActiveModel}
                activeModel={activeModel}
              />
            } />
            <Route path="/test" element={
              <TestRecommendations
                uploadedFiles={uploadedFiles}
                activeModel={activeModel}
              />
            } />
          </Routes>
        </main>
        
        <footer className="app-footer">
          <p>E-commerce Recommendation System © 2025</p>
        </footer>
      </div>
    </Router>
  );
}

export default App;