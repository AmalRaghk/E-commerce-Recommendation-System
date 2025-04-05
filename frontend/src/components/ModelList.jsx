import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

function ModelList({ onSetActiveModel, activeModel }) {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  
  useEffect(() => {
    fetchModels();
  }, []);
  
  const fetchModels = async () => {
    setLoading(true);
    
    try {
      const response = await axios.get(`${API_BASE_URL}/get_models`);
      
      if (response.data.success) {
        setModels(response.data.models);
      } else {
        setMessage(`Error: ${response.data.error}`);
      }
    } catch (error) {
      setMessage(`Error: ${error.response?.data?.error || error.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  const handleActivateModel = async (modelFilepath) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/set_active_model`, {
        modelFilepath
      });
      
      if (response.data.success) {
        onSetActiveModel(response.data.config);
        setMessage('Active model set successfully!');
      } else {
        setMessage(`Error: ${response.data.error}`);
      }
    } catch (error) {
      setMessage(`Error: ${error.response?.data?.error || error.message}`);
    }
  };
  
  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };
  
  return (
    <div className="model-list-container">
      <h2>Model Management</h2>
      
      <div className="active-model-info">
        <h3>Active Model</h3>
        {activeModel ? (
          <div className="active-model">
            <p><strong>Mode:</strong> {activeModel.mode}</p>
            {activeModel.mode === 'personalized' && (
              <p><strong>Algorithm:</strong> {activeModel.algorithm}</p>
            )}
            {activeModel.mode === 'cold_start' && (
              <p><strong>Sub-mode:</strong> {activeModel.sub_mode}</p>
            )}
          </div>
        ) : (
          <p>No active model set</p>
        )}
      </div>
      
      <div className="model-list">
        <h3>Available Models</h3>
        <button onClick={fetchModels} disabled={loading}>
          {loading ? 'Loading...' : 'Refresh Models'}
        </button>
        
        {models.length > 0 ? (
          <table className="models-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Type</th>
                <th>Details</th>
                <th>Created</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {models.map((model, index) => (
                <tr key={index}>
                  <td>{model.filename}</td>
                  <td>{model.config.mode}</td>
                  <td>
                    {model.config.mode === 'personalized' && (
                      <span>Algorithm: {model.config.algorithm}</span>
                    )}
                    {model.config.mode === 'cold_start' && (
                      <span>Sub-mode: {model.config.sub_mode}</span>
                    )}
                  </td>
                  <td>{model.timestamp}</td>
                  <td>
                    <button onClick={() => handleActivateModel(model.filepath)}>
                      Set Active
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p>No models available</p>
        )}
      </div>
      
      {message && <p className="message">{message}</p>}
    </div>
  );
}

export default ModelList;