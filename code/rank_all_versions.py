import os
import sys
from rank_models_by_kge import rank_models

def main():
    versions = [
        "output_L6_OWUS_MultiPET_v3.2.1.1",
        "output_L6_OWUS_SinglePET_v12.3.1.1_DynamicBeta",
        "output_L6_OWUS_SinglePET_v13.3.1.1_DynamicBeta"
    ]
    
    base_path = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/ptjpl_ptjplsm"
    
    print("="*60)
    print("      MASTER MODEL PERFORMANCE RANKING (v3, v12, v13)")
    print("="*60)
    
    for version in versions:
        full_path = os.path.join(base_path, version)
        if os.path.exists(full_path):
            print(f"\n>>> PROCESSING VERSION: {version}")
            rank_models(full_path)
            print("-" * 40)
        else:
            print(f"\n[!] SKIPPING: {version} (Directory not found)")

if __name__ == "__main__":
    main()
