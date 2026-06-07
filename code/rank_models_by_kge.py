import os
import re
import pandas as pd
import numpy as np
import sys

SITE_MAP = {
    "adelaideriver": "AU-Adr",
    "alpinepeatland": "AU-Alp",
    "capetribulation": "AU-Ctr",
    "dalypasture": "AU-DaP",
    "emerald": "AU-Eme",
    "foggdam": "AU-Fog",
    "gatumpasture": "AU-Gat",
    "greatwesternwoodlands": "AU-Gre",
    "litchfield": "AU-Lit",
    "reddirtmelonfarm": "AU-RDF",
    "ridgefield": "AU-Rgf",
    "silverplains": "AU-SiP",
    "sturtplains": "AU-Stp",
    "wallabycreek": "AU-Wal",
    "whroo": "AU-Whr",
    "yarramundiirrigated": "AU-YarI",
    "yanco": "AU-Ync"
}

def parse_metrics(file_path, site_folder):
    if not os.path.exists(file_path):
        return None
    
    # Filter by whitelist
    site_key = site_folder.lower()
    if site_key not in SITE_MAP:
        return None

    with open(file_path, 'r') as f:
        lines = f.readlines()
    data = []
    current_condition = None
    for line in lines:
        line = line.strip()
        if not line: continue
        m = re.match(r'===\s*(.+?)\s*===', line)
        if m:
            current_condition = m.group(1).split('(')[0].strip()
            continue
        if '|' in line and current_condition == "ALL DAYS":
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 6:
                try:
                    data.append({
                        'Model': parts[0],
                        'KGE': float(parts[5]),
                        'R2': float(parts[3]),
                        'RMSE': float(parts[1])
                    })
                except ValueError: continue
    return pd.DataFrame(data)

def rank_models(version_dir):
    print(f"\nAnalyzing Ranking for: {version_dir}")
    all_data = []
    
    if not os.path.exists(version_dir):
        print(f"Error: Directory {version_dir} does not exist.")
        return

    for site_folder in sorted(os.listdir(version_dir)):
        site_path = os.path.join(version_dir, site_folder)
        if not os.path.isdir(site_path) or site_folder == 'logs': continue
        
        m_files = [f for f in os.listdir(site_path) if f.endswith('_metrics.txt')]
        for mf in m_files:
            df = parse_metrics(os.path.join(site_path, mf), site_folder)
            if df is not None and not df.empty:
                site_id = SITE_MAP[site_folder.lower()]
                
                # Export site-specific ranking
                site_rank = df.sort_values(by='KGE', ascending=False)
                site_csv_path = os.path.join(site_path, f"{site_id}_performance_ranking.csv")
                site_rank.to_csv(site_csv_path, index=False)
                
                df['Site'] = site_id
                all_data.append(df)
    
    if not all_data:
        print("No metrics data found for the whitelisted sites.")
        return

    full_df = pd.concat(all_data, ignore_index=True)
    
    # Calculate median metrics per model
    stats = full_df.groupby('Model').agg({
        'KGE': ['median', 'count'],
        'R2': 'median',
        'RMSE': 'median'
    })
    
    # Flatten columns
    stats.columns = ['KGE_median', 'Site_Count', 'R2_median', 'RMSE_median']
    stats = stats.reset_index()
    
    # Sort all by median KGE (Descending)
    full_ranking = stats.sort_values(by='KGE_median', ascending=False)
    
    # Save full ranking to CSV in the version folder
    csv_path = os.path.join(version_dir, "model_performance_ranking.csv")
    full_ranking.to_csv(csv_path, index=False)
    print(f"\n[SUCCESS] Full ranking saved to: {csv_path}")
    
    # Take TOP 3 for summary display
    top_3 = full_ranking.head(3)
    
    print("\n--- TOP 3 CONFIGURATIONS (Summary) ---")
    print(top_3[['Model', 'KGE_median', 'R2_median', 'RMSE_median', 'Site_Count']].to_string(index=False))
    
    return full_ranking

if __name__ == "__main__":
    if len(sys.argv) > 1:
        rank_models(sys.argv[1])
    else:
        # Default to the most recent version if none provided
        default_dir = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/ptjpl_ptjplsm/output_L6_OWUS_SinglePET_v13.3.1.1_DynamicBeta"
        rank_models(default_dir)
