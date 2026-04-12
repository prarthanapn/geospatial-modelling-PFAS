import logging
import pickle
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import shap

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "outputs" / "models"
KD_DIR = ROOT / "outputs" / "kdtrees"
EARTH_RADIUS_KM = 6371.0

class PFASPredictor:
    def __init__(self):
        with open(MODELS_DIR / "lgbm_exceedance_v1.pkl", "rb") as f: self.clf = pickle.load(f)
        with open(MODELS_DIR / "lgbm_regression_v1.pkl", "rb") as f: self.reg = pickle.load(f)
        with open(MODELS_DIR / "feature_schema.json", "r") as f: self.schema = json.load(f)
        with open(KD_DIR / "training_points.pkl", "rb") as f: self.tree_train = pickle.load(f)
        with open(KD_DIR / "airports.pkl", "rb") as f: self.tree_air = pickle.load(f)
        self.train_vals = np.load(KD_DIR / "training_log_values.npy")
        
        # Initialize SHAP explainer for XAI
        self.explainer = shap.TreeExplainer(self.clf)
        
    def predict(self, lat, lon, substance="PFOS"):
        pt = np.deg2rad([[lat, lon]])
        d_tr, _ = self.tree_train.query(pt, k=1)
        d_air, _ = self.tree_air.query(pt, k=1)
        
        feat = {f: -1.0 for f in self.schema}
        feat['substance_ord'] = 6.0 # PFOS default
        feat['nearest_training_point_km'] = d_tr[0] * EARTH_RADIUS_KM
        feat['dist_to_airport_km'] = d_air[0] * EARTH_RADIUS_KM
        
        # Search for recent local lag
        idx_neighbors = self.tree_train.query_ball_point(pt[0], r=50.0/EARTH_RADIUS_KM)
        if idx_neighbors:
            feat['mean_log_value_50km'] = np.mean(self.train_vals[idx_neighbors])
            feat['spatial_density_50km'] = len(idx_neighbors)
        
        X = pd.DataFrame([feat])[self.schema]
        prob = self.clf.predict_proba(X)[0, 1]
        val = self.reg.predict(X)[0]
        
        # Calculate local SHAP values for XAI
        shap_values = self.explainer.shap_values(X)
        # For binary classification, shap_values might be a list or array
        if isinstance(shap_values, list): # Older SHAP versions
            local_shap = shap_values[1][0] 
        else:
            local_shap = shap_values[0]
            
        return {
            "exceedance_prob": prob,
            "predicted_value_ngl": np.expm1(val),
            "dist_to_airport_km": d_air[0] * EARTH_RADIUS_KM,
            "dist_to_nearest_sample_km": d_tr[0] * EARTH_RADIUS_KM,
            "shap_values": dict(zip(self.schema, local_shap.tolist()))
        }

if __name__ == "__main__":
    p = PFASPredictor()
    print(p.predict(51.5, -0.1))
