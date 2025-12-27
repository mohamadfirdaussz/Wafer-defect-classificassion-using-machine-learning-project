import os
import sys
import numpy as np
import pandas as pd
import joblib
import base64
import io
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from pathlib import Path
from collections import Counter

# Add ml_flow to path to import components
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'ml_flow'))

try:
    from config import CLEANED_DATA_FILE, MODEL_ARTIFACTS_DIR, PREPROCESSING_DIR, FEATURE_SELECTION_DIR
    from feature_engineering import process_single_wafer
    from feature_combination import generate_math_combinations
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import confusion_matrix, classification_report
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Get dashboard directory for serving static files
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=DASHBOARD_DIR)
CORS(app)

# ============ LOAD ASSETS ============
print("Loading model and preprocessing assets...")
BEST_MODEL_PATH = os.path.join(MODEL_ARTIFACTS_DIR, "best_model_optimized.joblib")
SCALER_PATH = os.path.join(PREPROCESSING_DIR, "standard_scaler.joblib")
DATA_4B_PATH = os.path.join(FEATURE_SELECTION_DIR, "data_track_4B_RFE.npz")
FEATURE_IMPORTANCE_PATH = os.path.join(FEATURE_SELECTION_DIR, "RF_Feature_Importance_Ranking.csv")

if not os.path.exists(BEST_MODEL_PATH):
    print(f"Model not found at {BEST_MODEL_PATH}")
    sys.exit(1)

model = joblib.load(BEST_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
data_4b = np.load(DATA_4B_PATH, allow_pickle=True)
selected_feature_names = data_4b['feature_names'].tolist()

# Load cleaned data for random samples
data_raw = np.load(CLEANED_DATA_FILE, allow_pickle=True)
wafer_maps = data_raw['waferMap']
labels_gt = data_raw['labels']

# Load track data for computing confusion matrix
X_test_4b = data_4b['X_test']
y_test_4b = data_4b['y_test']
y_train_4b = data_4b['y_train']

MAPPING_TYPE = {
    0: "Center", 1: "Donut", 2: "Edge-Loc", 3: "Edge-Ring",
    4: "Loc", 5: "Random", 6: "Scratch", 7: "none"
}

CLASS_NAMES = [MAPPING_TYPE[i] for i in range(8)]

# ============ HELPER FUNCTIONS ============

def get_base_feature_names():
    return (
        [f"density_{i+1}" for i in range(13)] +
        [f"radon_mean_{i+1}" for i in range(20)] +
        [f"radon_std_{i+1}" for i in range(20)] +
        ["geom_area", "geom_perimeter", "geom_major_axis", "geom_minor_axis", 
         "geom_eccentricity", "geom_solidity", "geom_num_regions"] +
        ["stat_mean", "stat_std", "stat_var", "stat_skew", "stat_kurt", "stat_median"]
    )

BASE_FEATURE_NAMES = get_base_feature_names()

def expand_for_inference(X_scaled_base):
    """Replicates Stage 3.5 for a single sample."""
    X_math, names_math = generate_math_combinations(X_scaled_base, BASE_FEATURE_NAMES)
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    X_poly = poly.fit_transform(X_scaled_base)
    names_poly = poly.get_feature_names_out(input_features=BASE_FEATURE_NAMES).tolist()
    X_expanded = np.column_stack([X_scaled_base, X_math, X_poly])
    expanded_names = BASE_FEATURE_NAMES + names_math + names_poly
    return X_expanded, expanded_names

# Precompute confusion matrix on test set
print("Computing confusion matrix...")
y_pred_test = model.predict(X_test_4b)
conf_matrix = confusion_matrix(y_test_4b, y_pred_test)
class_report = classification_report(y_test_4b, y_pred_test, target_names=CLASS_NAMES, output_dict=True)

# ============ STATIC FILE SERVING ============

@app.route('/')
def serve_index():
    return send_from_directory(DASHBOARD_DIR, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(DASHBOARD_DIR, path)

# ============ DATA APIS ============

@app.route('/api/dataset_stats', methods=['GET'])
def get_dataset_stats():
    """Return dataset statistics"""
    total_samples = len(labels_gt)
    train_size = len(y_train_4b)
    test_size = len(y_test_4b)
    
    class_counts = Counter(labels_gt)
    class_distribution = [{"class": MAPPING_TYPE[i], "count": int(class_counts.get(i, 0))} for i in range(8)]
    
    return jsonify({
        'total_samples': total_samples,
        'train_size': train_size,
        'test_size': test_size,
        'train_ratio': round(train_size / (train_size + test_size) * 100, 1),
        'num_classes': 8,
        'num_features_base': 66,
        'num_features_expanded': 8500,
        'num_features_selected': len(selected_feature_names),
        'class_distribution': class_distribution
    })

@app.route('/api/confusion_matrix', methods=['GET'])
def get_confusion_matrix():
    """Return confusion matrix data"""
    cm_list = conf_matrix.tolist()
    
    # Normalize for percentage view
    cm_normalized = (conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100).tolist()
    
    return jsonify({
        'matrix': cm_list,
        'matrix_normalized': cm_normalized,
        'labels': CLASS_NAMES
    })

@app.route('/api/class_metrics', methods=['GET'])
def get_class_metrics():
    """Return per-class metrics (F1, Precision, Recall)"""
    metrics = []
    for class_name in CLASS_NAMES:
        if class_name in class_report:
            metrics.append({
                'class': class_name,
                'precision': round(class_report[class_name]['precision'] * 100, 1),
                'recall': round(class_report[class_name]['recall'] * 100, 1),
                'f1': round(class_report[class_name]['f1-score'] * 100, 1),
                'support': class_report[class_name]['support']
            })
    
    # Add macro/weighted averages
    metrics.append({
        'class': 'Macro Avg',
        'precision': round(class_report['macro avg']['precision'] * 100, 1),
        'recall': round(class_report['macro avg']['recall'] * 100, 1),
        'f1': round(class_report['macro avg']['f1-score'] * 100, 1),
        'support': class_report['macro avg']['support']
    })
    
    return jsonify(metrics)

@app.route('/api/feature_importance', methods=['GET'])
def get_feature_importance():
    """Return top N feature importances"""
    top_n = request.args.get('n', 15, type=int)
    
    try:
        df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
        top_features = df.head(top_n).to_dict('records')
        return jsonify(top_features)
    except Exception as e:
        # Fallback: use model's feature_importances_ if available
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            features = [{'Feature': selected_feature_names[i], 'Importance': float(importances[i])} for i in indices]
            return jsonify(features)
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_comparison', methods=['GET'])
def get_model_comparison():
    """Return model comparison data for radar chart"""
    csv_path = os.path.join(MODEL_ARTIFACTS_DIR, "master_model_comparison.csv")
    try:
        df = pd.read_csv(csv_path)
        # Get top model per track
        tracks = df['Track'].unique()
        comparison = []
        for track in tracks:
            track_df = df[df['Track'] == track].sort_values('Test_F1_Macro', ascending=False).head(1)
            if len(track_df) > 0:
                row = track_df.iloc[0]
                comparison.append({
                    'track': row['Track'],
                    'model': row['Model'],
                    'f1': round(row.get('Test_F1_Macro', 0) * 100, 1),
                    'recall': round(row.get('Test_Recall_Macro', 0) * 100, 1),
                    'precision': round(row.get('Test_Precision_Macro', 0) * 100, 1) if 'Test_Precision_Macro' in row else 0,
                    'overfit_gap': round(row.get('Overfit_Gap', 0) * 100, 1)
                })
        return jsonify(comparison)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/misclassifications', methods=['GET'])
def get_misclassifications():
    """Return examples of misclassified wafers"""
    limit = request.args.get('limit', 10, type=int)
    
    # Find misclassified indices in test set
    misclassified_mask = y_pred_test != y_test_4b
    misclassified_indices = np.where(misclassified_mask)[0]
    
    # Sample some examples
    sample_size = min(limit, len(misclassified_indices))
    if sample_size == 0:
        return jsonify([])
    
    sample_indices = np.random.choice(misclassified_indices, sample_size, replace=False)
    
    examples = []
    for idx in sample_indices:
        examples.append({
            'index': int(idx),
            'true_label': MAPPING_TYPE[int(y_test_4b[idx])],
            'predicted_label': MAPPING_TYPE[int(y_pred_test[idx])]
        })
    
    return jsonify(examples)

@app.route('/api/random_wafer', methods=['GET'])
def get_random_wafer():
    """Get a random wafer with optional class filter"""
    target_type = request.args.get('type')
    
    if target_type and target_type != 'all':
        try:
            target_label = int(target_type)
            indices = np.where(labels_gt == target_label)[0]
            if len(indices) == 0:
                return jsonify({'error': 'No wafers found for this class'}), 404
            idx = int(np.random.choice(indices))
        except ValueError:
            idx = np.random.randint(0, len(wafer_maps))
    else:
        idx = np.random.randint(0, len(wafer_maps))

    wafer = wafer_maps[idx]
    label = int(labels_gt[idx])
    
    # Convert wafer to image
    plt.figure(figsize=(4, 4))
    plt.imshow(wafer, cmap='viridis')
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, facecolor='black')
    plt.close()
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return jsonify({
        'index': idx,
        'label': MAPPING_TYPE.get(label, "Unknown"),
        'image': img_b64
    })

@app.route('/api/classify', methods=['POST'])
def classify():
    """Classify a wafer by index"""
    data = request.json
    idx = data.get('index')
    threshold = data.get('threshold', 0.0)  # Optional confidence threshold
    
    if idx is None or idx >= len(wafer_maps):
        return jsonify({'error': 'Invalid index'}), 400
    
    wafer = wafer_maps[idx]
    
    # Feature Extraction -> Scaling -> Expansion
    base_features = process_single_wafer(wafer).reshape(1, -1)
    base_features_df = pd.DataFrame(base_features, columns=BASE_FEATURE_NAMES)
    base_features_scaled = scaler.transform(base_features_df)
    X_expanded, expanded_names = expand_for_inference(base_features_scaled)
    
    df_expanded = pd.DataFrame(X_expanded, columns=expanded_names)
    df_expanded = df_expanded.loc[:, ~df_expanded.columns.duplicated()]
    X_final = df_expanded.reindex(columns=selected_feature_names, fill_value=0).values
    
    # Predict
    pred_idx = model.predict(X_final)[0]
    probs = model.predict_proba(X_final)[0]
    max_prob = float(probs.max())
    
    # Apply threshold
    prediction = MAPPING_TYPE[int(pred_idx)] if max_prob >= threshold else "Uncertain"
    
    prob_list = [{"label": MAPPING_TYPE.get(i, f"Class {i}"), "score": float(p)} for i, p in enumerate(probs)]
    
    return jsonify({
        'prediction': prediction,
        'confidence': round(max_prob * 100, 1),
        'probabilities': prob_list
    })

@app.route('/api/batch_classify', methods=['POST'])
def batch_classify():
    """Classify multiple wafers at once"""
    data = request.json
    indices = data.get('indices', [])
    
    if not indices:
        return jsonify({'error': 'No indices provided'}), 400
    
    results = []
    for idx in indices[:50]:  # Limit to 50 at a time
        if idx >= len(wafer_maps):
            continue
            
        wafer = wafer_maps[idx]
        base_features = process_single_wafer(wafer).reshape(1, -1)
        base_features_df = pd.DataFrame(base_features, columns=BASE_FEATURE_NAMES)
        base_features_scaled = scaler.transform(base_features_df)
        X_expanded, expanded_names = expand_for_inference(base_features_scaled)
        
        df_expanded = pd.DataFrame(X_expanded, columns=expanded_names)
        df_expanded = df_expanded.loc[:, ~df_expanded.columns.duplicated()]
        X_final = df_expanded.reindex(columns=selected_feature_names, fill_value=0).values
        
        pred_idx = model.predict(X_final)[0]
        probs = model.predict_proba(X_final)[0]
        
        results.append({
            'index': idx,
            'ground_truth': MAPPING_TYPE.get(int(labels_gt[idx]), "Unknown"),
            'prediction': MAPPING_TYPE[int(pred_idx)],
            'confidence': round(float(probs.max()) * 100, 1),
            'correct': MAPPING_TYPE[int(pred_idx)] == MAPPING_TYPE.get(int(labels_gt[idx]), "")
        })
    
    correct_count = sum(1 for r in results if r['correct'])
    
    return jsonify({
        'results': results,
        'accuracy': round(correct_count / len(results) * 100, 1) if results else 0,
        'total': len(results),
        'correct': correct_count
    })

print("Dashboard Server loaded successfully!")
print(f"Serving static files from: {DASHBOARD_DIR}")

if __name__ == '__main__':
    print("Dashboard Server starting on http://localhost:5000")
    app.run(port=5000, debug=False, threaded=True)