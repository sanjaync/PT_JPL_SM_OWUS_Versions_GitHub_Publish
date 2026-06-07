import os
import glob
import pandas as pd

dir_v32 = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/ptjpl_ptjplsm/output_L6_OWUS_MultiPET_v3.2"
dir_v321 = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/ptjpl_ptjplsm/output_L6_OWUS_MultiPET_v3.2.1"

def parse_metrics(filepath):
    metrics = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    in_all_days = False
    for line in lines:
        if line.startswith("=== ALL DAYS"):
            in_all_days = True
            continue
        elif in_all_days and line.startswith("==="):
            break
            
        if in_all_days and "|" in line and "Model" not in line and "---" not in line:
            parts = line.split("|")
            model = parts[0].strip()
            try:
                rmse = float(parts[1].strip())
                bias = float(parts[2].strip())
                r2 = float(parts[3].strip())
                nse = float(parts[4].strip())
                kge = float(parts[5].strip())
                metrics[model] = {'RMSE': rmse, 'Bias': bias, 'R2': r2, 'NSE': nse, 'KGE': kge}
            except ValueError:
                pass
            
    return metrics

sites = [d for d in os.listdir(dir_v321) if os.path.isdir(os.path.join(dir_v321, d))]
sites.sort()

results = []

for site in sites:
    file_v32 = glob.glob(os.path.join(dir_v32, site, f"*_v3.2_metrics.txt"))
    file_v321 = glob.glob(os.path.join(dir_v321, site, f"*_v3.2.1_metrics.txt"))
    
    if file_v32 and file_v321:
        m_v32 = parse_metrics(file_v32[0])
        m_v321 = parse_metrics(file_v321[0])
        
        for model in ['SM_OPT', 'SM_BF', 'PTJPL_Base', 'PTJPL_SM']:
            if model in m_v32 and model in m_v321:
                kge_diff = m_v321[model]['KGE'] - m_v32[model]['KGE']
                rmse_diff = m_v321[model]['RMSE'] - m_v32[model]['RMSE']
                results.append({
                    'site': site,
                    'model': model,
                    'kge_v32': m_v32[model]['KGE'],
                    'kge_v321': m_v321[model]['KGE'],
                    'kge_diff': kge_diff,
                    'rmse_v32': m_v32[model]['RMSE'],
                    'rmse_v321': m_v321[model]['RMSE'],
                    'rmse_diff': rmse_diff
                })

df = pd.DataFrame(results)

print("=== AVERAGE METRICS ACROSS ALL SITES (ALL DAYS) ===")
print("KGE:")
print(df.groupby('model')[['kge_v32', 'kge_v321', 'kge_diff']].mean().round(3))
print("\nRMSE:")
print(df.groupby('model')[['rmse_v32', 'rmse_v321', 'rmse_diff']].mean().round(3))

print("\n=== TOP 5 SITES: KGE IMPROVEMENTS FOR SM_BF ===")
print(df[df['model']=='SM_BF'].sort_values('kge_diff', ascending=False).head(5)[['site', 'kge_v32', 'kge_v321', 'kge_diff']].to_string(index=False))

print("\n=== TOP 5 SITES: KGE DEGRADATIONS FOR SM_BF ===")
print(df[df['model']=='SM_BF'].sort_values('kge_diff', ascending=True).head(5)[['site', 'kge_v32', 'kge_v321', 'kge_diff']].to_string(index=False))

print("\n=== TOP 5 SITES: KGE IMPROVEMENTS FOR SM_OPT ===")
print(df[df['model']=='SM_OPT'].sort_values('kge_diff', ascending=False).head(5)[['site', 'kge_v32', 'kge_v321', 'kge_diff']].to_string(index=False))

print("\n=== TOP 5 SITES: KGE DEGRADATIONS FOR SM_OPT ===")
print(df[df['model']=='SM_OPT'].sort_values('kge_diff', ascending=True).head(5)[['site', 'kge_v32', 'kge_v321', 'kge_diff']].to_string(index=False))
