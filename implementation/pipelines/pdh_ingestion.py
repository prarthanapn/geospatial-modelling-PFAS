import logging
from typing import List, Optional
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import time
from implementation.data_sources.pdh_client import PDHClient
from implementation.data_sources.pdh_schema import PDHPoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# List of major contributors to the PFAS Data Hub for broader coverage
GLOBAL_COUNTRY_LIST = [
    "France", "Germany", "Netherlands", "Belgium", "Italy", "Spain", "Sweden", 
    "Denmark", "Norway", "United Kingdom", "USA", "Canada", "Australia"
]

class PDHPipeline:
    """Production ingestion pipeline for CNRS PDH data with multi-country batch support."""
    
    def __init__(self, outputs_dir: Path = Path("outputs/ingestion/pdh")):
        self.outputs_dir = outputs_dir
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.client = PDHClient()

    def normalize_points(self, points: List[PDHPoint], censored_strategy: str = "half_threshold") -> pd.DataFrame:
        rows = []
        for pt in points:
            base_info = pt.dict(exclude={"pfas_values"})
            if not pt.pfas_values:
                row = base_info.copy()
                row["substance"] = "Total PFAS"
                row["value"] = pt.pfas_sum
                row["is_censored"] = False
                rows.append(row)
                continue
                
            for pv in pt.pfas_values:
                row = base_info.copy()
                row.update(pv.dict())
                if pv.less_than is not None:
                    row["is_censored"] = True
                    row["value"] = pv.less_than / 2.0 if censored_strategy == "half_threshold" else pv.less_than
                else:
                    row["is_censored"] = False
                rows.append(row)
        
        return pd.DataFrame(rows)

    def run(self, country: Optional[str] = None):
        """Ingets a single country or batches through the global list if no country is specified."""
        countries = [country] if country else GLOBAL_COUNTRY_LIST
        all_dfs = []
        
        log.info(f"Starting ingestion for {len(countries)} countries...")
        for c in countries:
            try:
                log.info(f"Ingesting: {c}")
                raw_points = self.client.fetch_export(params={"country": c})
                if raw_points:
                    all_dfs.append(self.normalize_points(raw_points))
                time.sleep(1) # Defensive rate limit
            except Exception as e:
                log.error(f"Failed to ingest {c}: {e}")
                continue
        
        if not all_dfs:
            log.warning("No data retrieved from any source.")
            return None
            
        df_long = pd.concat(all_dfs, ignore_index=True)
        
        # Georeferencing
        df_long = df_long.dropna(subset=["lat", "lon"])
        geometry = [Point(xy) for xy in zip(df_long.lon, df_long.lat)]
        gdf = gpd.GeoDataFrame(df_long, geometry=geometry, crs="EPSG:4326")
        
        # Save
        parquet_path = self.outputs_dir / "pdh_long.parquet"
        gdf.to_parquet(parquet_path)
        log.info(f"Saved merged global dataset ({len(gdf)} records) to {parquet_path}")
        return gdf

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", help="Specific country")
    args = parser.parse_args()
    
    PDHPipeline().run(country=args.country)
