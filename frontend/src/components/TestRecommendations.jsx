import React, { useState } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

function TestRecommendations({ activeModel }) {
  const [itemSku, setItemSku] = useState('');
  const [userId, setUserId] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [recommendations, setRecommendations] = useState(null);
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!activeModel) {
      setMessage('No active model set. Please set an active model in Model Management.');
      return;
    }
    
    if (!itemSku) {
      setMessage('Please enter an Item SKU');
      return;
    }
    
    if (activeModel.mode === 'personalized' && !userId) {
      setMessage('User ID is required for personalized recommendations');
      return;
    }
    
    setLoading(true);
    setMessage('');
    
    try {
      const response = await axios.post(`${API_BASE_URL}/recommend_sku`, {
        item_sku: itemSku,
        user_id: userId
      });
      
      if (response.data.success) {
        setRecommendations(response.data);
        setMessage('Recommendations generated successfully!');
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
    <div className="test-recommendations-container">
      <h2>Test Recommendations API</h2>
      
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
      
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="itemSku">Item SKU:</label>
          <input
            type="text"
            id="itemSku"
            value={itemSku}
            onChange={(e) => setItemSku(e.target.value)}
            placeholder="e.g. product_456"
          />
        </div>
        
        {activeModel && activeModel.mode === 'personalized' && (
          <div className="form-group">
            <label htmlFor="userId">User ID:</label>
            <input
              type="text"
              id="userId"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              placeholder="e.g. user_123"
            />
          </div>
        )}
        
        <button type="submit" disabled={loading || !activeModel}>
          {loading ? 'Getting Recommendations...' : 'Get Recommendations'}
        </button>
      </form>
      
      {recommendations && (
        <div className="recommendations-results">
          <h3>Recommendations</h3>
          <p>Recommendations for item {recommendations.item_sku}:</p>
          <ol>
            {recommendations.recommendations.map((item, index) => (
              <li key={index}>{item}</li>
            ))}
          </ol>
        </div>
      )}
      
      {message && <p className="message">{message}</p>}
      
      <div className="api-docs">
        <h3>API Documentation</h3>
        <p>To use the recommendations API in your e-commerce application:</p>
        <pre>
          {`POST ${API_BASE_URL}/recommend_sku
Content-Type: application/json

{
  "item_sku": "product_id",
  "user_id": "user_id"  // only required for personalized mode
}
          `}
        </pre>
      </div>
    </div>
  );
}

export default TestRecommendations;