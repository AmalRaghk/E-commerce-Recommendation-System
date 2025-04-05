import React, { useState } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

function DataUploadForm({ onFileUpload, uploadedFiles }) {
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const dataType = formData.get('dataType');
    const file = formData.get('file');
    
    if (!file) {
      setMessage('Please select a file');
      return;
    }
    
    setLoading(true);
    setMessage('');
    
    try {
      const response = await axios.post(
        `${API_BASE_URL}/upload_data`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        }
      );
      
      if (response.data.success) {
        setMessage(`${dataType} data uploaded successfully!`);
        onFileUpload(dataType, response.data.filepath);
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
    <div className="upload-form-container">
      <h2>Data Upload</h2>
      
      <div className="upload-status">
        <h3>Upload Status</h3>
        <ul>
          <li>Interaction Data:{uploadedFiles.interaction ? <span className="uploaded">✓ Uploaded</span> : <span className="pending">Pending</span>}
          </li>
          <li>Product Metadata: 
            {uploadedFiles.product_metadata ? <span className="uploaded">✓ Uploaded</span> : <span className="pending">Pending</span>}
          </li>
          <li>User Data: 
            {uploadedFiles.user_data ? <span className="uploaded">✓ Uploaded</span> : <span className="pending">Pending</span>}
          </li>
          <li>Image Data: 
            {uploadedFiles.image_data ? <span className="uploaded">✓ Uploaded</span> : <span className="pending">Pending</span>}
          </li>
        </ul>
      </div>
      
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="dataType">Data Type:</label>
          <select name="dataType" id="dataType" required>
            <option value="interaction">Interaction Data</option>
            <option value="product_metadata">Product Metadata</option>
            <option value="user_data">User Data</option>
            <option value="image_data">Image Data</option>
          </select>
        </div>
        
        <div className="form-group">
          <label htmlFor="file">Select File:</label>
          <input type="file" name="file" id="file" required />
        </div>
        
        <button type="submit" disabled={loading}>
          {loading ? 'Uploading...' : 'Upload'}
        </button>
        
        {message && <p className="message">{message}</p>}
      </form>
    </div>
  );
}

export default DataUploadForm;