import os
import re
import pandas as pd

def parse_metrics(filepath):
    """Parses the metrics file and returns a dictionary of metrics."""
    if not os.path.exists(filepath):
        return None
    
    metrics = {}
    current_section = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Identify section
            if line.startswith('===') and '===' in line:
                current_section = line.replace('===', '').strip()
                metrics[current_section] = {}
                continue
            
            # Identify model data
            if '|' in line and (current_section is not None):
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 9 and parts[0] != 'Model' and not parts[0].startswith('---'):
                    model_name = parts[0]
                    try:
                        metrics[current_section][model_name] = {
                            'RMSE': float(parts[1]),
                            'Bias': float(parts[2]),
                            'R2':   float(parts[3]),
                            'NSE':  float(parts[4]),
                            'KGE':  float(parts[5]),
                            'r':    float(parts[6]),
                            'a':    float(parts[7]),
                            'b':    float(parts[8])
                        }
                    except ValueError:
                        continue
    return metrics

sites = {
    "AU-Adr": "adelaideriver",
    "AU-Alp": "alpinepeatland",
    "AU-Ctr": "capetribulation",
    "AU-DaP": "dalypasture",
    "AU-Eme": "emerald",
    "AU-Fog": "foggdam",
    "AU-Gat": "gatumpasture",
    "AU-Gre": "greatwesternwoodlands",
    "AU-Lit": "litchfield",
    "AU-RDF": "reddirtmelonfarm",
    "AU-Rgf": "ridgefield",
    "AU-SiP": "silverplains",
    "AU-Stp": "sturtplains",
    "AU-Wal": "wallabycreek",
    "AU-Whr": "whroo",
    "AU-YarI": "yarramundiirrigated",
    "AU-Ync": "yanco"
}

base_path_old = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/ptjpl_ptjplsm/output_L6_OWUS_MultiPET_v3.2.1"
base_path_new = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/ptjpl_ptjplsm/output_L6_OWUS_MultiPET_v3.2.1.1"

summary_data = []

for site_id, folder_name in sites.items():
    # Construct expected filenames based on observation
    # AdelaideRiver_v3.2.1_metrics.txt
    # Convert folder_name to TitleCase if necessary? 
    # Let's check the dir again or just search for the file
    
    file_old = None
    file_new = None
    
    dir_old = os.path.join(base_path_old, folder_name)
    dir_new = os.path.join(base_path_new, folder_name)
    
    if os.path.isdir(dir_old):
        for f in os.listdir(dir_old):
            if f.endswith('_v3.2.1_metrics.txt'):
                file_old = os.path.join(dir_old, f)
                break
    
    if os.path.isdir(dir_new):
        for f in os.listdir(dir_new):
            if f.endswith('_v3.2.1.1_metrics.txt'):
                file_new = os.path.join(dir_new, f)
                break
                
    if not file_old or not file_new:
        print(f"Warning: Could not find both files for {site_id} ({folder_name})")
        continue
    
    m_old = parse_metrics(file_old)
    m_new = parse_metrics(file_new)
    
    if not m_old or not m_new:
        continue
        
    # Compare SM_BF in ALL DAYS section
    section = "ALL DAYS (n=586)" # This n might vary per site
    # Let's find the section that starts with "ALL DAYS"
    s_old = next((s for s in m_old if s.startswith("ALL DAYS")), None)
    s_new = next((s for s in m_new if s.startswith("ALL DAYS")), None)
    
    if s_old and s_new:
        bf_old = m_old[s_old].get('SM_BF')
        bf_new = m_new[s_new].get('SM_BF')
        
        if bf_old and bf_new:
            summary_data.append({
                'SiteID': site_id,
                'v321_KGE': bf_old['KGE'],
                'v3211_KGE': bf_new['KGE'],
                'KGE_Diff': bf_new['KGE'] - bf_old['KGE'],
                'v321_RMSE': bf_old['RMSE'],
                'v3211_RMSE': bf_new['RMSE'],
                'RMSE_Diff': bf_new['RMSE'] - bf_old['RMSE']
            })

df = pd.DataFrame(summary_data)
if not df.empty:
    print(df.to_markdown(index=False))
else:
    print("No comparison data generated.")
