"""
src/modeling/multicompound.py
==============================
Module 4 — Multi-Compound Predictive Modeling

Predicts exceedance probabilities for multiple PFAS substances simultaneously.
Uses ClassifierChain to model inter-substance correlations.

Substances: PFOS, PFOA, PFHxS, PFNA, PFDA, PFHpA, PFBS.
"""

import logging
import pickle
import json
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = ROOT / "data" / "features" / "pfas_features_train.parquet"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

SUBSTANCES = ["PFOS", "PFOA", "PFHxS", "PFNA", "PFDA", "PFHpA", "PFBS"]
FEATURE_COLS = [
    "is_long_chain", "carbon_chain_length", "is_sulfonyl",
    "media_ord", "is_aquatic", "is_soil_based", "is_wastewater",
    "year_normalized", "is_post_2018", "month",
    "spatial_density_50km", "mean_log_value_50km", "nearest_training_point_km",
    "elevation_m", "land_use_class", "annual_precip_mm",
    "soil_permeability", "population_density_log",
    "dist_to_airport_km", "dist_to_military_km", "dist_to_wastewater_km",
]

def load_multicompound_data():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError("Run feature engineering first.")
    
    df = pd.read_parquet(FEATURES_PATH)
    
    # We need a 'wide' format where each row is a sample point and columns are substances
    # However, the current feature matrix is 'long' (one row per substance per sample)
    # We need to pivot or rebuild the data to have one row per location/time/media
    
    # Identify unique samples by location (rounded), year, and media
    df['sample_key'] = (
        df['lat'].round(4).astype(str) + "_" + 
        df['lon'].round(4).astype(str) + "_" + 
        df['year'].astype(str) + "_" + 
        df['media_ord'].astype(str)
    )
    
    # Create targets for each substance
    pivoted = df.pivot_table(
        index='sample_key',
        columns='substance',
        values='above_100_ng_l',
        aggfunc='max'
    ).reindex(columns=SUBSTANCES).dropna() # Drop rows where ANY substance is missing for clean training
    
    # Get features for each sample (first occurrence in each sample group)
    feat_df = df.groupby('sample_key')[FEATURE_COLS + ['spatial_block_id']].first()
    
    combined = pd.merge(feat_df, pivoted, left_index=True, right_index=True)
    return combined

def train_multicompound():
    log.info("Loading multicompound data...")
    data = load_multicompound_data()
    X = data[FEATURE_COLS].fillna(-1).values
    Y = data[SUBSTANCES].values
    groups = data["spatial_block_id"].values
    
    log.info(f"Multicompound dataset: {len(data)} samples, {Y.sum()} total exceedances across {len(SUBSTANCES)} compounds")
    
    base_lr = LGBMClassifier(n_estimators=100, num_leaves=31, learning_rate=0.05, n_jobs=-1, verbose=-1)
    chain = ClassifierChain(base_lr, order='random', random_state=42)
    
    cv = GroupKFold(n_splits=5)
    aucs = []
    
    log.info("Starting Spatial Block CV for Multi-Compound Model...")
    for i, (tr, vl) in enumerate(cv.split(X, Y, groups)):
        chain.fit(X[tr], Y[tr])
        y_prob = chain.predict_proba(X[vl])
        
        fold_aucs = []
        for j, sub in enumerate(SUBSTANCES):
            if len(np.unique(Y[vl][:, j])) > 1:
                auc = roc_auc_score(Y[vl][:, j], y_prob[:, j])
                fold_aucs.append(auc)
        
        mean_auc = np.mean(fold_aucs)
        aucs.append(mean_auc)
        log.info(f"Fold {i+1} Mean ROC-AUC: {mean_auc:.4f}")
        
    log.info(f"Overall Multi-Compound Mean ROC-AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    
    log.info("Training final Multi-Compound model on full data...")
    chain.fit(X, Y)
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "multicompound_chain_v1.pkl", "wb") as f:
        pickle.dump(chain, f)
    
    with open(MODELS_DIR / "multicompound_substances.json", "w") as f:
        json.dump(SUBSTANCES, f)
        
    log.info("Saved multicompound_chain_v1.pkl")

if __name__ == "__main__":
    train_multicompound()
