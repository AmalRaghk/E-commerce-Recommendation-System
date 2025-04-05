import os
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import datetime
import time


from flask_cors import CORS
app = Flask(__name__)
CORS(app)

 # Enable Cross-Origin Resource Sharing

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
IMAGE_FOLDER = 'images'
ALLOWED_EXTENSIONS = {'csv', 'json', 'jpg', 'jpeg', 'png'}

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, MODEL_FOLDER, IMAGE_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER

# Initialize global variables to store active model and data
active_model = None
active_model_config = None
product_metadata = None
similarity_matrix = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Helper function to save a model
def save_model(model, config, model_name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.pkl"
    filepath = os.path.join(app.config['MODEL_FOLDER'], filename)

    with open(filepath, 'wb') as f:
        pickle.dump({
            'model': model,
            'config': config,
            'timestamp': timestamp
        }, f)
    return filepath


# Helper function to load a model
def load_model(filepath):
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
        return model_data['model'], model_data['config']


@app.route('/api/upload_data', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    data_type = request.form.get('dataType')

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{data_type}_{filename}")
        file.save(filepath)

        # Process the file based on data_type
        try:
            if data_type == 'interaction':
                df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_json(filepath)
                # Validate required columns
                required_cols = ['USER_ID', 'ITEM_ID', 'TIMESTAMP', 'EVENT_TYPE']
                if not all(col in df.columns for col in required_cols):
                    return jsonify({'error': f'Missing required columns in {data_type} data'}), 400

            elif data_type == 'product_metadata':
                df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_json(filepath)
                # Validate required columns
                required_cols = ['ITEM_ID', 'PRICE', 'CATEGORY_L1']
                if not all(col in df.columns for col in required_cols):
                    return jsonify({'error': f'Missing required columns in {data_type} data'}), 400
                # Store product metadata globally for cold start recommendations
                global product_metadata
                product_metadata = df

            elif data_type == 'user_data':
                df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_json(filepath)
                # Validate required columns
                if 'USER_ID' not in df.columns:
                    return jsonify({'error': f'Missing USER_ID column in {data_type} data'}), 400

            elif data_type == 'image_data':
                # For image_data we would expect a directory upload, but for simplicity,
                # let's just check if the file is an image
                if not filename.endswith(('.jpg', '.jpeg', '.png')):
                    return jsonify({'error': 'Invalid image format'}), 400

            return jsonify({
                'success': True,
                'message': f'{data_type} data uploaded successfully',
                'filepath': filepath
            })
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    return jsonify({'error': 'Invalid file format'}), 400


@app.route('/api/train_model', methods=['POST'])
def train_model():
    data = request.json
    mode = data.get('mode')

    if mode == 'personalized':
        algorithm = data.get('algorithm')
        interaction_data_path = data.get('interactionDataPath')
        product_metadata_path = data.get('productMetadataPath')
        user_data_path = data.get('userDataPath')

        try:
            # Load interaction data
            if interaction_data_path.endswith('.csv'):
                interactions_df = pd.read_csv(interaction_data_path)
            else:
                interactions_df = pd.read_json(interaction_data_path)

            # Train model based on selected algorithm
            if algorithm == 'user_user_cf':
                model = train_user_user_cf(interactions_df)
            elif algorithm == 'item_item_cf':
                model = train_item_item_cf(interactions_df)
            elif algorithm == 'svd':
                model = train_svd(interactions_df)
            elif algorithm == 'nmf':
                model = train_nmf(interactions_df)
            elif algorithm == 'neural_cf':
                model = train_neural_cf(interactions_df, product_metadata_path)
            else:
                return jsonify({'error': f'Unsupported algorithm: {algorithm}'}), 400

            # Save model
            config = {
                'mode': mode,
                'algorithm': algorithm,
                'interaction_data_path': interaction_data_path,
                'product_metadata_path': product_metadata_path,
                'user_data_path': user_data_path
            }

            model_path = save_model(model, config, f"personalized_{algorithm}")

            return jsonify({
                'success': True,
                'message': f'Model trained successfully using {algorithm}',
                'model_path': model_path
            })

        except Exception as e:
            return jsonify({'error': f'Error training model: {str(e)}'}), 500

    elif mode == 'cold_start':
        sub_mode = data.get('subMode')
        product_metadata_path = data.get('productMetadataPath')
        image_data_path = data.get('imageDataPath')

        try:
            # Load product metadata
            if product_metadata_path.endswith('.csv'):
                metadata_df = pd.read_csv(product_metadata_path)
            else:
                metadata_df = pd.read_json(product_metadata_path)

            if sub_mode == 'similarity_based':
                n_recommendations = data.get('nRecommendations', 5)
                model = train_similarity_based(metadata_df, n_recommendations)

                # Calculate similarity matrix for quick recommendations
                global similarity_matrix
                similarity_matrix = calculate_similarity_matrix(metadata_df)

            elif sub_mode == 'metadata_weighted':
                weights = data.get('weights', {})
                model = train_metadata_weighted(metadata_df, weights)

            else:
                return jsonify({'error': f'Unsupported sub-mode: {sub_mode}'}), 400

            # Save model
            config = {
                'mode': mode,
                'sub_mode': sub_mode,
                'product_metadata_path': product_metadata_path,
                'image_data_path': image_data_path
            }

            if sub_mode == 'similarity_based':
                config['n_recommendations'] = n_recommendations
            elif sub_mode == 'metadata_weighted':
                config['weights'] = weights

            model_path = save_model(model, config, f"cold_start_{sub_mode}")

            return jsonify({
                'success': True,
                'message': f'Model trained successfully using {sub_mode}',
                'model_path': model_path
            })

        except Exception as e:
            return jsonify({'error': f'Error training model: {str(e)}'}), 500
    else:
        return jsonify({'error': f'Unsupported mode: {mode}'}), 400


def train_user_user_cf(interactions_df):
    # User-User CF implementation using adjusted cosine similarity
    user_item_matrix = interactions_df.pivot_table(index='USER_ID', columns='ITEM_ID',
                                                 values='EVENT_TYPE', aggfunc='count').fillna(0)

    # Calculate mean ratings for each user
    user_mean_ratings = user_item_matrix.mean(axis=1)

    # Center user-item matrix by subtracting mean ratings
    user_item_matrix_centered = user_item_matrix.subtract(user_mean_ratings, axis='rows')

    # Calculate cosine similarity between users
    user_similarity = cosine_similarity(user_item_matrix_centered)
    return user_similarity

def train_item_item_cf(interactions_df):
    # Item-Item CF implementation using cosine similarity
    user_item_matrix = interactions_df.pivot_table(index='USER_ID', columns='ITEM_ID',
                                                 values='EVENT_TYPE', aggfunc='count').fillna(0)
    item_item_similarity = cosine_similarity(user_item_matrix.T)
    return item_item_similarity


def train_svd(interactions_df):
    # SVD implementation
    user_item_matrix = interactions_df.pivot_table(index='USER_ID', columns='ITEM_ID',
                                                 values='EVENT_TYPE', aggfunc='count').fillna(0)
    U, S, Vt = np.linalg.svd(user_item_matrix, full_matrices=False)  # Using full_matrices=False ensures compatible dimensions
    k = min(100, min(U.shape[1], Vt.shape[0]))  # Ensure k is not larger than our matrices
    U_k = U[:, :k]
    S_k = np.diag(S[:k])  # Create a diagonal matrix
    Vt_k = Vt[:k, :]
    return U_k, S_k, Vt_k


def train_nmf(interactions_df):
    # NMF implementation
    user_item_matrix = interactions_df.pivot_table(index='USER_ID', columns='ITEM_ID',
                                                 values='EVENT_TYPE', aggfunc='count').fillna(0)
    from sklearn.decomposition import NMF
    nmf_model = NMF(n_components=50, random_state=42, max_iter=200)
    W = nmf_model.fit_transform(user_item_matrix)
    H = nmf_model.components_
    return W, H


def train_neural_cf(interactions_df, product_metadata_path=None):
    # Placeholder for Neural CF. Returns a simple model for now.
    return "neural_cf_placeholder_model"


def calculate_similarity_matrix(metadata_df):
    # Preprocess metadata for similarity calculation
    # Handle categorical features with one-hot encoding
    categorical_cols = ['CATEGORY_L1', 'CATEGORY_L2', 'CATEGORY_L3', 'AGE_GROUP', 'GENDER', 'ADULT']
    categorical_cols = [col for col in categorical_cols if col in metadata_df.columns]

    # Preprocess text descriptions if available
    if 'PRODUCT_DESCRIPTION' in metadata_df.columns:
        # Fill missing descriptions
        metadata_df['PRODUCT_DESCRIPTION'] = metadata_df['PRODUCT_DESCRIPTION'].fillna('')
        # Create TF-IDF vectors
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(metadata_df['PRODUCT_DESCRIPTION'])
        desc_similarity = cosine_similarity(tfidf_matrix)
    else:
        desc_similarity = np.zeros((len(metadata_df), len(metadata_df)))

    # One-hot encode categorical features
    encoded_data = pd.get_dummies(metadata_df[categorical_cols], drop_first=False)

    # Normalize numerical features
    numerical_cols = ['PRICE']
    if 'CREATION_TIMESTAMP' in metadata_df.columns:
        numerical_cols.append('CREATION_TIMESTAMP')
    numerical_cols = [col for col in numerical_cols if col in metadata_df.columns]
    if numerical_cols:
        scaler = StandardScaler()
        scaled_numerical = pd.DataFrame(
            scaler.fit_transform(metadata_df[numerical_cols]),
            columns=numerical_cols
        )
        # Combine all features
        all_features = pd.concat([encoded_data, scaled_numerical], axis=1)
    else:
        all_features = encoded_data

    # Calculate cosine similarity between items based on all features
    feature_similarity = cosine_similarity(all_features)

    # Combine similarities with equal weights for now
    # In a more advanced implementation, you would allow weights to be configured
    combined_similarity = 0.7 * feature_similarity + 0.3 * desc_similarity if 'PRODUCT_DESCRIPTION' in metadata_df.columns else feature_similarity
    return combined_similarity



def train_similarity_based(metadata_df, n_recommendations=5):
    # Simple similarity-based model that stores metadata and parameters
    model = {
        'metadata_df': metadata_df,
        'n_recommendations': n_recommendations
    }
    return model


def train_metadata_weighted(metadata_df, weights):
    # Metadata-weighted model that stores metadata and weights
    model = {
        'metadata_df': metadata_df,
        'weights': weights
    }
    return model


@app.route('/api/get_models', methods=['GET'])
def get_models():
    models = []
    for filename in os.listdir(app.config['MODEL_FOLDER']):
        if filename.endswith('.pkl'):
            try:
                filepath = os.path.join(app.config['MODEL_FOLDER'], filename)
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                model_info = {
                    'filename': filename,
                    'filepath': filepath,
                    'config': model_data['config'],
                    'timestamp': model_data['timestamp']
                }
                models.append(model_info)
            except Exception as e:
                print(f"Error loading model {filename}: {str(e)}")
    return jsonify({
        'success': True,
        'models': models
    })



@app.route('/api/set_active_model', methods=['POST'])
def set_active_model():
    data = request.json
    model_filepath = data.get('modelFilepath')

    try:
        global active_model, active_model_config
        active_model, active_model_config = load_model(model_filepath)
        return jsonify({
            'success': True,
            'message': 'Active model set successfully',
            'config': active_model_config
        })
    except Exception as e:
        return jsonify({'error': f'Error setting active model: {str(e)}'}), 500



@app.route('/api/recommend_sku', methods=['POST'])
def recommend_sku():
    data = request.json
    item_sku = data.get('item_sku')
    user_id = data.get('user_id', None)

    if not item_sku:
        return jsonify({'error': 'Item SKU is required'}), 400

    if active_model is None:  # Changed from "if not active_model:"
        return jsonify({'error': 'No active model set'}), 400

    try:
        recommendations = []
        if active_model_config['mode'] == 'personalized':
            # For personalized recommendations, we need user_id
            if not user_id:
                return jsonify({'error': 'User ID is required for personalized recommendations'}), 400

            algorithm = active_model_config['algorithm']

            if algorithm == 'user_user_cf':
                user_similarity = active_model
                interactions_df = pd.read_csv(active_model_config['interaction_data_path']) # added to get the original df
                user_item_matrix = interactions_df.pivot_table(index='USER_ID', columns='ITEM_ID',
                                                             values='EVENT_TYPE', aggfunc='count').fillna(0)

                if user_id not in user_item_matrix.index:
                    return jsonify({'error': f'User ID {user_id} not found in data.'}), 400

                # Get the target user's similarity scores
                target_user_index = user_item_matrix.index.get_loc(user_id)
                similar_users = sorted(list(enumerate(user_similarity[target_user_index])), key=lambda x: x[1],
                                               reverse=True)[1:11]  # Top 10 similar users excluding the target user.

                similar_user_ids = [user_item_matrix.index[user_index] for user_index, _ in similar_users]

                # Get items interacted with by similar users
                similar_user_items = user_item_matrix.loc[similar_user_ids]

                # Get items the target user has interacted with
                target_user_items = user_item_matrix.loc[user_id]

                # Calculate weighted average of similar users' ratings for unseen items
                weighted_item_ratings = {}
                for item_id in user_item_matrix.columns:
                    if target_user_items[item_id] == 0:  # Consider only unseen items
                        weighted_sum = 0
                        similarity_sum = 0
                        for i, similar_user_id in enumerate(similar_user_ids):
                            similarity = similar_users[i][1]
                            weighted_sum += similar_user_items[item_id][similar_user_id] * similarity
                            similarity_sum += similarity
                        weighted_item_ratings[item_id] = weighted_sum / similarity_sum if similarity_sum > 0 else 0

                # Sort items by predicted rating
                recommended_items = sorted(weighted_item_ratings.items(), key=lambda x: x[1], reverse=True)[:10]
                recommendations = [item_id for item_id, _ in recommended_items]


            elif algorithm == 'item_item_cf':
                item_similarity = active_model
                interactions_df = pd.read_csv(active_model_config['interaction_data_path'])  # added to get the original df
                user_item_matrix = interactions_df.pivot_table(index='USER_ID', columns='ITEM_ID',
                                                             values='EVENT_TYPE',
                                                             aggfunc='count').fillna(0)
                if item_sku not in user_item_matrix.columns:
                    return jsonify({'error': f'Item SKU {item_sku} not found in data.'}), 400

                item_index = user_item_matrix.columns.get_loc(item_sku)
                similar_items = sorted(list(enumerate(item_similarity[item_index])), key=lambda x: x[1],
                                               reverse=True)[1:11]  # Top 10 similar items

                recommendations = [user_item_matrix.columns[i] for i, _ in similar_items]

            
            elif algorithm in ['svd', 'nmf']:
                # SVD and NMF
                interactions_df = pd.read_csv(active_model_config['interaction_data_path'])
                user_item_matrix = interactions_df.pivot_table(index='USER_ID', columns='ITEM_ID',
                                                             values='EVENT_TYPE',
                                                             aggfunc='count').fillna(0)
                if user_id not in user_item_matrix.index:
                    return jsonify({'error': f'User ID {user_id} not found in data.'}), 400
                user_index = user_item_matrix.index.get_loc(user_id)
                
                if algorithm == 'svd':
                    U_k, S_k, Vt_k = active_model
                    
                    # Debug shapes
                    print(f"U_k shape: {U_k.shape}, S_k shape: {S_k.shape}, Vt_k shape: {Vt_k.shape}")
                    
                    user_vector = U_k[user_index]  # Get the user's vector
                    print(f"user_vector shape: {user_vector.shape}")
                    
                    # Explicitly handle the dimension mismatch
                    # If U_k features (k) don't match S_k rows
                    if U_k.shape[1] != S_k.shape[0]:
                        # Option 1: Truncate to the smaller dimension
                        min_k = min(U_k.shape[1], S_k.shape[0], Vt_k.shape[0])
                        user_vector = user_vector[:min_k]
                        S_k_adjusted = np.zeros((min_k, min_k))
                        for i in range(min_k):
                            if i < len(np.diag(S_k)):
                                S_k_adjusted[i, i] = np.diag(S_k)[i]  # Make sure to get diagonal elements correctly
                        Vt_k_adjusted = Vt_k[:min_k, :]
                        
                        # Calculate ratings with adjusted matrices
                        predicted_ratings = user_vector @ S_k_adjusted @ Vt_k_adjusted
                    
                    # If S_k columns don't match Vt_k rows
                    elif S_k.shape[1] != Vt_k.shape[0]:
                        # Option 2: Try a different multiplication approach
                        # First multiply user vector with S_k diagonal directly
                        weighted_user_vector = user_vector * np.diag(S_k)
                        # Then multiply with Vt_k
                        predicted_ratings = weighted_user_vector @ Vt_k
                    
                    else:
                        # Standard approach if dimensions match
                        if len(user_vector.shape) == 1:
                            user_vector = user_vector.reshape(1, -1)
                        
                        # Try direct multiplication
                        predicted_ratings = user_vector @ S_k @ Vt_k
                        
                        if len(predicted_ratings.shape) > 1:
                            predicted_ratings = predicted_ratings[0]
                        
                else:  # nmf
                    W, H = active_model
                    user_vector = W[user_index]
                    
                    # For NMF, the matrix multiplication is simpler
                    # Ensure proper dimensions
                    if len(user_vector.shape) == 1:
                        user_vector = user_vector.reshape(1, -1)
                    
                    # Calculate predicted ratings
                    predicted_ratings = user_vector @ H
                    
                    # Ensure predicted_ratings is a 1D array for sorting
                    if len(predicted_ratings.shape) > 1:
                        predicted_ratings = predicted_ratings[0]
                
                # Get the indices of the top 10 items
                top_item_indices = predicted_ratings.argsort()[-10:][::-1]
                # Get the item IDs corresponding to the indices
                recommendations = [user_item_matrix.columns[i] for i in top_item_indices]

            elif algorithm == 'neural_cf':
                recommendations = [f"neural_cf_item_{i}" for i in range(1, 11)]

        elif active_model_config['mode'] == 'cold_start':
            sub_mode = active_model_config['sub_mode']
            if sub_mode == 'similarity_based':
                # Use pre-calculated similarity matrix
                if similarity_matrix is not None:
                    metadata_df = active_model['metadata_df']
                    n_recommendations = active_model['n_recommendations']

                    # Find index of item_sku in metadata_df
                    try:
                        item_index = metadata_df[metadata_df['ITEM_ID'] == item_sku].index[0]
                    except IndexError:
                        return jsonify({'error': f'Item SKU {item_sku} not found in metadata'}), 400

                    # Get similarity scores for this item
                    item_similarities = similarity_matrix[item_index]

                    # Get indices of top N similar items (excluding self)
                    similar_indices = item_similarities.argsort()[::-1][1:n_recommendations + 1]

                    # Convert indices to item IDs
                    recommendations = metadata_df.iloc[similar_indices]['ITEM_ID'].tolist()
            elif sub_mode == 'metadata_weighted':
                metadata_df = active_model['metadata_df']
                weights = active_model['weights']

                # Find the item in metadata
                try:
                    item_data = metadata_df[metadata_df['ITEM_ID'] == item_sku].iloc[0]
                except IndexError:
                    return jsonify({'error': f'Item SKU {item_sku} not found in metadata'}), 400

                # Calculate weighted similarity to all items
                similarity_scores = []
                for i, row in metadata_df.iterrows():
                    if row['ITEM_ID'] == item_sku:
                        continue  # Skip the item itself

                    score = 0
                    # Process each feature according to weights
                    for feature, weight in weights.items():
                        if feature not in metadata_df.columns:
                            continue
                        weight = float(weight)
                        # Skip if weight is 0
                        if weight == 0:
                            continue
                        # Different similarity calculation based on feature type
                        if feature == 'PRICE':
                            # Inverse of price difference, normalized
                            price_diff = abs(item_data[feature] - row[feature])
                            max_price = metadata_df[feature].max()
                            sim = 1 - (price_diff / max_price) if max_price > 0 else 0
                            score += weight * sim
                        elif feature == 'PRODUCT_DESCRIPTION' and feature in metadata_df.columns:
                            # TF-IDF similarity would be more appropriate here
                            # For simplicity, using 1 if both descriptions are non-empty, 0 otherwise
                            has_desc1 = isinstance(item_data[feature], str) and len(item_data[feature].strip()) > 0
                            has_desc2 = isinstance(row[feature], str) and len(row[feature].strip()) > 0
                            sim = 1 if has_desc1 and has_desc2 else 0
                            score += weight * sim
                        elif feature in ['CATEGORY_L1', 'CATEGORY_L2', 'CATEGORY_L3', 'AGE_GROUP', 'GENDER', 'ADULT']:
                            # Categorical feature: 1 if matching, 0 if not
                            sim = 1 if item_data[feature] == row[feature] else 0
                            score += weight * sim
                    similarity_scores.append((row['ITEM_ID'], score))
                # Sort by similarity score and get top 10
                similarity_scores.sort(key=lambda x: x[1], reverse=True)
                recommendations = [item for item, _ in similarity_scores[:10]]
        return jsonify({
            'success': True,
            'item_sku': item_sku,
            'user_id': user_id,
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'error': f'Error generating recommendations: {str(e)}'}), 500



@app.route('/api/try_recommendation', methods=['POST'])
def try_recommendation():
    data = request.json
    mode = data.get('mode')
    item_sku= data.get('item_sku')
    user_id = data.get('user_id')

    if mode == 'cold_start':
        sub_mode = data.get('subMode')
        product_metadata_path = data.get('productMetadataPath')

        try:
            # Load product metadata
            if product_metadata_path.endswith('.csv'):
                metadata_df = pd.read_csv(product_metadata_path)
            else:
                metadata_df = pd.read_json(product_metadata_path)

            # Find the item in metadata
            try:
                item_data = metadata_df[metadata_df['ITEM_ID'] == item_sku].iloc[0]
            except IndexError:
                return jsonify({'error': f'Item SKU {item_sku} not found in metadata'}), 400

            if sub_mode == 'similarity_based':
                # Calculate similarity to all items
                similarity_scores = []
                # Quick implementation for testing:
                # Just recommend items in the same category
                same_category_items = metadata_df[
                    (metadata_df['CATEGORY_L1'] == item_data['CATEGORY_L1']) &
                    (metadata_df['ITEM_ID'] != item_sku)
                    ]
                recommendations = same_category_items['ITEM_ID'].tolist()[:10]
            elif sub_mode == 'metadata_weighted':
                weights = data.get('weights', {})
                # Filter items by same category for simplicity
                # (a more sophisticated approach would calculate weighted similarity)
                same_category_items = metadata_df[
                    (metadata_df['CATEGORY_L1'] == item_data['CATEGORY_L1']) &
                    (metadata_df['ITEM_ID'] != item_sku)
                    ]
                recommendations = same_category_items['ITEM_ID'].tolist()[:10]
            else:
                return jsonify({'error': f'Unsupported sub-mode: {sub_mode}'}), 400

            return jsonify({
                'success': True,
                'item_sku': item_sku,
                'recommendations': recommendations
            })
        except Exception as e:
            return jsonify({'error': f'Error in try recommendation: {str(e)}'}), 500
    elif mode == 'personalized':
        # This would require loading and using your trained model
        # For testing, just return a sample response
        return jsonify({
            'success': True,
            'item_sku': item_sku,
            'user_id': user_id,
            'recommendations': [f"sample_item_{i}" for i in range(1, 11)]
        })
    else:
        return jsonify({'error': f'Unsupported mode: {mode}'}), 400


if __name__ == '__main__':
    app.run(debug=True)
