import React, { useState } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

function PersonalizedModeConfig({ uploadedFiles }) {
  const [algorithm, setAlgorithm] = useState('svd');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [modelPath, setModelPath] = useState(null);
  
  const algorithms = [
    { value: 'user_user_cf', label: 'User-User Collaborative Filtering' },
    { value: 'item_item_cf', label: 'Item-Item Collaborative Filtering' },
    { value: 'svd', label: 'Singular Value Decomposition (SVD)' },
    { value: 'nmf', label: 'Non-negative Matrix Factorization (NMF)' },
    { value: 'neural_cf', label: 'Neural Collaborative Filtering' }
  ];
  
  const handleTrainModel = async (e) => {
    e.preventDefault();
    
    if (!uploadedFiles.interaction) {
      setMessage('Interaction data is required for personalized mode');
      return;
    }
    
    setLoading(true);
    setMessage('');
    
    try {
      const response = await axios.post(`${API_BASE_URL}/train_model`, {
        mode: 'personalized',
        algorithm,
        interactionDataPath: uploadedFiles.interaction,
        productMetadataPath: uploadedFiles.product_metadata,
        userDataPath: uploadedFiles.user_data
      });
      
      if (response.data.success) {
        setMessage(`Model trained successfully using ${algorithm}!`);
        setModelPath(response.data.model_path);
      } else {
        setMessage(`Error: ${response.data.error}`);
      }
    } catch (error) {
      setMessage(`Error: ${error.response?.data?.error || error.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="mode-config-container">
      <h2>Personalized Recommendation Mode</h2>
      
      <div className="data-requirements">
        <h3>Data Requirements</h3>
        <ul>
          <li>Interaction Data: {uploadedFiles.interaction ? <span className="available">✓ Available</span> : <span className="required">Required</span>}</li>
          <li>Product Metadata: {uploadedFiles.product_metadata ? <span className="available">✓ Available</span> : <span className="optional">Optional</span>}</li>
          <li>User Data: {uploadedFiles.user_data ? <span className="available">✓ Available</span> : <span className="optional">Optional</span>}</li>
        </ul>
      </div>
      
      <form onSubmit={handleTrainModel}>
        <div className="form-group">
          <label htmlFor="algorithm">Select Algorithm:</label>
          <select 
            id="algorithm" 
            value={algorithm} 
            onChange={(e) => setAlgorithm(e.target.value)}
          >
            {algorithms.map(alg => (
              <option key={alg.value} value={alg.value}>{alg.label}</option>
            ))}
          </select>
        </div>
        
        <button type="submit" disabled={loading || !uploadedFiles.interaction}>
          {loading ? 'Training Model...' : 'Train Model'}
        </button>
        
        {message && <p className="message">{message}</p>}
        {modelPath && (
          <div className="model-info">
            <p>Model trained and saved at: {modelPath}</p>
            <p>You can view and activate this model in the Model Management section.</p>
          </div>
        )}
      </form>
    </div>
  );
}

export default PersonalizedModeConfig;
