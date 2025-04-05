import React, { useState } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

function ColdStartModeConfig({ uploadedFiles }) {
  const [subMode, setSubMode] = useState('similarity_based');
  const [nRecommendations, setNRecommendations] = useState(5);
  const [weights, setWeights] = useState({
    CATEGORY_L1: 0.5,
    PRICE: 0.3,
    PRODUCT_DESCRIPTION: 0.2
  });
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [modelPath, setModelPath] = useState(null);
  const [testItemSku, setTestItemSku] = useState('');
  const [testResults, setTestResults] = useState(null);
  
  const handleWeightChange = (feature, value) => {
    setWeights(prev => ({
      ...prev,
      [feature]: parseFloat(value)
    }));
  };
  
  const handleTrainModel = async (e) => {
    e.preventDefault();
    
    if (!uploadedFiles.product_metadata) {
      setMessage('Product metadata is required for cold start mode');
      return;
    }
    
    setLoading(true);
    setMessage('');
    
    try {
      const response = await axios.post(`${API_BASE_URL}/train_model`, {
        mode: 'cold_start',
        subMode,
        productMetadataPath: uploadedFiles.product_metadata,
        imageDataPath: uploadedFiles.image_data,
        nRecommendations: parseInt(nRecommendations),
        weights
      });
      
      if (response.data.success) {
        setMessage(`Model trained successfully using ${subMode}!`);
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
  
  const handleTestRecommendation = async (e) => {
    e.preventDefault();
    
    if (!testItemSku) {
      setMessage('Please enter an Item SKU to test');
      return;
    }
    
    try {
      const response = await axios.post(`${API_BASE_URL}/try_recommendation`, {
        mode: 'cold_start',
        subMode,
        productMetadataPath: uploadedFiles.product_metadata,
        item_sku: testItemSku,
        weights
      });
      
      if (response.data.success) {
        setTestResults(response.data);
        setMessage('Test recommendations generated successfully!');
      } else {
        setMessage(`Error: ${response.data.error}`);
      }
    } catch (error) {
      setMessage(`Error: ${error.response?.data?.error || error.message}`);
    }
  };
  
  // Get metadata columns from the first few rows for the weights form
  const [metadataColumns, setMetadataColumns] = useState([
    'CATEGORY_L1', 'CATEGORY_L2', 'CATEGORY_L3', 'PRICE', 'PRODUCT_DESCRIPTION', 
    'AGE_GROUP', 'GENDER', 'ADULT'
  ]);
  
  return (
    <div className="mode-config-container">
      <h2>Cold Start Recommendation Mode</h2>
      
      <div className="data-requirements">
        <h3>Data Requirements</h3>
        <ul>
          <li>Product Metadata: {uploadedFiles.product_metadata ? <span className="available">✓ Available</span> : <span className="required">Required</span>}</li>
          <li>Image Data: {uploadedFiles.image_data ? <span className="available">✓ Available</span> : <span className="optional">Optional</span>}</li>
        </ul>
      </div>
      
      <div className="submode-selector">
        <h3>Select Sub-mode</h3>
        <div className="radio-group">
          <label>
            <input 
              type="radio" 
              value="similarity_based" 
              checked={subMode === 'similarity_based'} 
              onChange={(e) => setSubMode(e.target.value)}
            />
            Similarity-Based (Amount-Set)
          </label>
          <label>
            <input 
              type="radio" 
              value="metadata_weighted" 
              checked={subMode === 'metadata_weighted'} 
              onChange={(e) => setSubMode(e.target.value)}
            />
            Metadata-Weighted
          </label>
        </div>
      </div>
      
      {subMode === 'similarity_based' && (
        <div className="similarity-config">
          <h3>Similarity-Based Configuration</h3>
          <div className="form-group">
            <label htmlFor="nRecommendations">Number of Recommendations:</label>
            <input 
              type="number" 
              id="nRecommendations" 
              min="1" 
              max="20" 
              value={nRecommendations} 
              onChange={(e) => setNRecommendations(e.target.value)}
            />
          </div>
        </div>
      )}
      
      {subMode === 'metadata_weighted' && (
        <div className="weights-config">
          <h3>Metadata Weights Configuration</h3>
          <p>Assign weights to different metadata features (sum should equal 1.0)</p>
          
          {metadataColumns.map(feature => (
            <div className="form-group weight-slider" key={feature}>
              <label htmlFor={`weight-${feature}`}>{feature}:</label>
              <input 
                type="range" 
                id={`weight-${feature}`} 
                min="0" 
                max="1" 
                step="0.1" 
                value={weights[feature] || 0}
                onChange={(e) => handleWeightChange(feature, e.target.value)}
              />
              <span>{weights[feature] || 0}</span>
            </div>
          ))}
          
          <p>Total weight: {Object.values(weights).reduce((sum, w) => sum + w, 0).toFixed(1)}</p>
        </div>
      )}
      
      <form onSubmit={handleTrainModel}>
        <button type="submit" disabled={loading || !uploadedFiles.product_metadata}>
          {loading ? 'Training Model...' : 'Train Model'}
        </button>
      </form>
      
      <div className="test-recommendations">
        <h3>Try it Out</h3>
        <form onSubmit={handleTestRecommendation}>
          <div className="form-group">
            <label htmlFor="testItemSku">Enter Item SKU:</label>
            <input 
              type="text" 
              id="testItemSku" 
              value={testItemSku}
              onChange={(e) => setTestItemSku(e.target.value)}
              placeholder="e.g. product_456"
            />
          </div>
          
          <button type="submit" disabled={!uploadedFiles.product_metadata}>
            Get Recommendations
          </button>
        </form>
        
        {testResults && (
          <div className="test-results">
            <h4>Recommendations for {testResults.item_sku}</h4>
            <ul>
              {testResults.recommendations.map((item, index) => (
                <li key={index}>{item}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
      
      {message && <p className="message">{message}</p>}
    </div>
  );
}

export default ColdStartModeConfig;
