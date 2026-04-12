"""
main.py
=======
Root orchestration script for the PFAS Geospatial System.
Runs the entire pipeline from cleaning to visualization.
"""

import subprocess
import sys
from pathlib import Path

def run_module(module_name, args=None, optional=False):
    print(f"\n>>> Running Module: {module_name} ...")
    cmd = [sys.executable, "-m", module_name]
    if args: cmd.extend(args)
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent) + os.pathsep + env.get("PYTHONPATH", "")
    
    result = subprocess.run(cmd, env=env, capture_output=False)
    if result.returncode != 0:
        if optional:
            print(f"S Warning: Optional module {module_name} failed. Proceeding with local data.")
            return
        print(f"!!! Module {module_name} FAILED !!!")
        sys.exit(1)
    print(f"--- Module {module_name} COMPLETED ---")

def main():
    # 0. PDH CNRS Ingestion (Global Pull)
    run_module("implementation.pipelines.pdh_ingestion", optional=True)
    
    # 1. Pipeline (Merges local Raw + Local Shapefile + PDH if it exists)
    run_module("implementation.clean")
    
    # 2. Training
    run_module("implementation.train")
    
    # 3. Spatial & Indexing
    run_module("implementation.hotspot")
    run_module("implementation.cci")
    
    print("\n✅ ALL STEPS FINISHED SUCCESSFULLY.")
    print("\nTo launch the dashboard:")
    print("  streamlit run implementation/dashboard.py")

if __name__ == "__main__":
    main()
