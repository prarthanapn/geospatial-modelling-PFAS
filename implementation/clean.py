"""
implementation/clean.py
======================
Full Data Pipeline: Ingestion -> Cleaning -> Feature Engineering

1. Cleans pfas_raw.csv -> pfas_cleaned.parquet
2. Builds KD-trees for airports and training points
3. Scales features -> pfas_features.parquet
"""

import logging
import pickle
import json
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import KDTree
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
OUTPUTS_DIR = ROOT / "outputs"
RAW_CSV = DATASET_DIR / "pfas_raw.csv"
AIRPORTS_CSV = DATASET_DIR / "airports.csv"

# Output Paths
CLEANED_PARQUET = DATASET_DIR / "pfas_cleaned.parquet"
FEATURES_PARQUET = DATASET_DIR / "pfas_features.parquet"
KD_DIR = OUTPUTS_DIR / "kdtrees"
ENCODER_DIR = OUTPUTS_DIR / "encoders"

EARTH_RADIUS_KM = 6371.0

# Substance parameters
SUBSTANCE_ORD = {"PFBS": 0, "PFHpA": 1, "PFHxS": 2, "PFNA": 3, "PFDA": 4, "PFOA": 5, "PFOS": 6}
CARBON_CHAIN = {"PFBS": 4, "PFHpA": 7, "PFHxS": 6, "PFNA": 9, "PFDA": 10, "PFOA": 8, "PFOS": 8}
LONG_CHAIN = {"PFOS", "PFOA", "PFNA", "PFDA"}
SULFONYL = {"PFOS", "PFHxS", "PFBS"}
AQUATIC_MEDIA = {"Surface water", "Groundwater", "Sea water", "Drinking water"}
SOIL_MEDIA = {"Soil", "Sediment"}
WASTEWATER_MEDIA = {"Wastewater", "Leachate"}

def clean_ingestion():
    log.info("Step 1: Ingestion & Cleaning...")
    if not RAW_CSV.exists(): raise FileNotFoundError(f"Missing {RAW_CSV}")
    
    df = pd.read_csv(RAW_CSV, low_memory=False)
    
    # NEW: Merge with PDH CNRS Data if exists
    pdh_path = ROOT / "outputs" / "ingestion" / "pdh" / "pdh_long.parquet"
    if pdh_path.exists():
        log.info(f"Merging PDH CNRS data from {pdh_path}...")
        pdh_df = pd.read_parquet(pdh_path)
        pdh_df = pdh_df.rename(columns={"matrix": "type", "unit": "measurement_units"})
        df = pd.concat([df, pdh_df], ignore_index=True)
        log.info(f"Merged {len(pdh_df)} records from PDH.")

    # NEW: Merge with pfas_contamination.shp if exists
    shp_path = DATASET_DIR / "pfas_contamination.shp"
    if shp_path.exists():
        log.info(f"Merging Shapefile data from {shp_path}...")
        shp_gdf = gpd.read_file(shp_path)
        # Convert geometry to lat/lon if needed
        if shp_gdf.crs != "EPSG:4326": shp_gdf = shp_gdf.to_crs("EPSG:4326")
        shp_gdf['lat'] = shp_gdf.geometry.y
        shp_gdf['lon'] = shp_gdf.geometry.x
        
        # Melt wide-format compounds into long format to match our pipeline
        compounds = ["pfos", "pfoa", "pfna", "pfbs", "pfhxs", "pfhxa"]
        id_vars = [c for c in shp_gdf.columns if c not in compounds and c != 'geometry']
        shp_long = shp_gdf.melt(id_vars=id_vars, value_vars=compounds, var_name="substance", value_name="value")
        shp_long["substance"] = shp_long["substance"].str.upper()
        shp_long["measurement_units"] = "ng/L" # Likely given the values in such maps
        
        df = pd.concat([df, shp_long], ignore_index=True)
        log.info(f"Merged {len(shp_long)} records from Shapefile.")

    # Standardize types to prevent PyArrow schema errors (e.g. mixed float/string)
    for col in ["country", "city", "type", "substance", "measurement_units", "measurement_location_type"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    # Basic cleaning
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "value"])
    df = df[df["value"] >= 0.0001]
    
    # Target creation
    ng_l = df["measurement_units"].str.lower() == "ng/l"
    df["log_value"] = np.log1p(df["value"])
    df["above_100_ng_l"] = np.where(ng_l, (df["value"] >= 100).astype(int), np.nan)
    
    # Features
    df["substance_ord"] = df["substance"].map(SUBSTANCE_ORD).fillna(-1).astype(int)
    df["is_long_chain"] = df["substance"].isin(LONG_CHAIN).astype(int)
    df["carbon_chain_length"] = df["substance"].map(CARBON_CHAIN).fillna(-1).astype(int)
    df["is_sulfonyl"] = df["substance"].isin(SULFONYL).astype(int)
    
    media_col = "measurement_location_type" if "measurement_location_type" in df.columns else "type"
    df["is_aquatic"] = df[media_col].isin(AQUATIC_MEDIA).astype(int)
    df["is_soil_based"] = df[media_col].isin(SOIL_MEDIA).astype(int)
    df["is_wastewater"] = df[media_col].isin(WASTEWATER_MEDIA).astype(int)
    
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["year_normalized"] = (df["year"] - 2000) / 25.0
    df["month"] = pd.to_datetime(df["date"], errors='coerce').dt.month.fillna(-1).astype(int)
    
    # Spatial blocks (1.0 degree)
    df["spatial_block_id"] = (df["lat"].round(0).astype(str) + "_" + df["lon"].round(0).astype(str))
    
    df.to_parquet(CLEANED_PARQUET, index=False)
    log.info(f"Cleaned data saved to {CLEANED_PARQUET}")
    return df

def build_kd_trees(df):
    log.info("Step 2: Building KD-Trees...")
    KD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Training points KD-tree
    coords = np.deg2rad(df[["lat", "lon"]].values)
    tree_train = KDTree(coords)
    with open(KD_DIR / "training_points.pkl", "wb") as f:
        pickle.dump(tree_train, f)
    np.save(KD_DIR / "training_log_values.npy", df["log_value"].values)
    
    # Airport KD-tree
    if AIRPORTS_CSV.exists():
        airports = pd.read_csv(AIRPORTS_CSV)
        airports = airports[airports['type'].isin(['large_airport', 'medium_airport'])]
        air_coords = np.deg2rad(airports[['lat', 'lon']].values)
        tree_air = KDTree(air_coords)
        with open(KD_DIR / "airports.pkl", "wb") as f:
            pickle.dump(tree_air, f)
        log.info("Airport KD-tree built.")
    else:
        log.warning("AIRPORTS_CSV not found — skipping airport proximity.")

def build_features(df):
    log.info("Step 3: Feature Engineering (Proximity)...")
    KD_DIR.mkdir(parents=True, exist_ok=True)
    ENCODER_DIR.mkdir(parents=True, exist_ok=True)
    
    # Proximity features (from earlier logic)
    coords_rad = np.deg2rad(df[["lat", "lon"]].values)
    
    # Load trees
    with open(KD_DIR / "training_points.pkl", "rb") as f: tree_train = pickle.load(f)
    train_vals = np.load(KD_DIR / "training_log_values.npy")
    
    # k-NN Spatial Lag (50km)
    k = 50
    radius_rad = 50.0 / EARTH_RADIUS_KM
    dists, idxs = tree_train.query(coords_rad, k=k+1) # +1 to exclude self
    
    # Fast vectorized lag
    lag_means = []
    lag_counts = []
    
    log.info("Calculating spatial lags for 370k points...")
    for i in tqdm(range(len(df)), desc="Spatial Engineering"):
        d = dists[i, 1:] # exclude self
        ix = idxs[i, 1:]
        mask = d <= radius_rad
        valid_ix = ix[mask]
        if len(valid_ix) > 0:
            lag_means.append(np.mean(train_vals[valid_ix]))
            lag_counts.append(len(valid_ix))
        else:
            lag_means.append(0)
            lag_counts.append(0)
            
    df["mean_log_value_50km"] = lag_means
    df["spatial_density_50km"] = lag_counts
    
    # Dist to airport
    if (KD_DIR / "airports.pkl").exists():
        with open(KD_DIR / "airports.pkl", "rb") as f: tree_air = pickle.load(f)
        d_air, _ = tree_air.query(coords_rad, k=1)
        df["dist_to_airport_km"] = d_air * EARTH_RADIUS_KM
    else:
        df["dist_to_airport_km"] = -1
        
    # Placeholder for rasters
    for c in ["elevation_m", "land_use_class", "annual_precip_mm", "soil_permeability", "population_density_log", "dist_to_military_km", "dist_to_wastewater_km"]:
        df[c] = -1.0
        
    df.to_parquet(FEATURES_PARQUET, index=False)
    log.info(f"Feature matrix saved to {FEATURES_PARQUET}")

if __name__ == "__main__":
    df = clean_ingestion()
    build_kd_trees(df)
    build_features(df)
    log.info("All data processing complete.")
