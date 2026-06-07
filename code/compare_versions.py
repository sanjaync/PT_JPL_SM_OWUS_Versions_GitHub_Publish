import os
import pandas as pd
import re

def parse_metrics_file(filepath):
    """Parses a metrics.txt file and returns a dictionary of metrics for SM_BF (ALL DAYS)."""
    if not os.path.exists(filepath):
        return None
    
    metrics = {}
    with open(filepath, 'r') as f:
        content = f.read()
        
    # Find the ALL DAYS section
    all_days_match = re.search(r"=== ALL DAYS \(n=\d+\) ===(.*?)(?====|\Z)", content, re.DOTALL)
    if not all_days_match:
        return None
    
    all_days_text = all_days_match.group(1)
    
    # Extract SM_BF line
    sm_bf_match = re.search(r"SM_BF\s+\|\s+([\d.-]+)\s+\|\s+([\d.-]+)\s+\|\s+([\d.-]+)\s+\|\s+([\d.-]+)\s+\|\s+([\d.-]+)", all_days_text)
    if sm_bf_match:
        metrics['SM_BF_RMSE'] = float(sm_bf_match.group(1))
        metrics['SM_BF_Bias'] = float(sm_bf_match.group(2))
        metrics['SM_BF_R2'] = float(sm_bf_match.group(3))
        metrics['SM_BF_NSE'] = float(sm_bf_match.group(4))
        metrics['SM_BF_KGE'] = float(sm_bf_match.group(5))
        
    # Extract SM_OPT for comparison
    sm_opt_match = re.search(r"SM_OPT\s+\|\s+([\d.-]+)\s+\|\s+([\d.-]+)\s+\|\s+([\d.-]+)\s+\|\s+([\d.-]+)\s+\|\s+([\d.-]+)", all_days_text)
    if sm_opt_match:
        metrics['SM_OPT_RMSE'] = float(sm_opt_match.group(1))
        metrics['SM_OPT_Bias'] = float(sm_opt_match.group(2))
        metrics['SM_OPT_R2'] = float(sm_opt_match.group(3))
        
    return metrics

def compare_versions(dir_v1, dir_v2):
    sites = [d for d in os.listdir(dir_v1) if os.path.isdir(os.path.join(dir_v1, d)) and d != 'logs']
    results = []
    
    for site in sorted(sites):
        # Construct filenames
        file_v1 = os.path.join(dir_v1, site, f"{site.capitalize()}_v3.2.1_metrics.txt")
        file_v2 = os.path.join(dir_v2, site, f"{site.capitalize()}_v3.2.1.1_metrics.txt")
        
        # Handle some capitalization variations if any (AdelaideRiver vs adelaideriver)
        # The file system shows 'adelaideriver' folder but 'AdelaideRiver_v3.2.1_metrics.txt'
        # Let's try to find the file correctly
        if not os.path.exists(file_v1):
            # Try lowercase site name in filename or folder
            files_in_site_v1 = [f for f in os.listdir(os.path.join(dir_v1, site)) if f.endswith('.txt')]
            if files_in_site_v1:
                file_v1 = os.path.join(dir_v1, site, files_in_site_v1[0])
        
        if not os.path.exists(os.path.join(dir_v2, site)):
            results.append({'Site': site, 'Status': 'Missing in v3.2.1.1'})
            continue
            
        if not os.path.exists(file_v2):
            files_in_site_v2 = [f for f in os.listdir(os.path.join(dir_v2, site)) if f.endswith('.txt')]
            if files_in_site_v2:
                file_v2 = os.path.join(dir_v2, site, files_in_site_v2[0])
            else:
                results.append({'Site': site, 'Status': 'No metrics file in v3.2.1.1'})
                continue

        metrics_v1 = parse_metrics_file(file_v1)
        metrics_v2 = parse_metrics_file(file_v2)
        
        if metrics_v1 and metrics_v2:
            res = {'Site': site}
            res['RMSE (v3.2.1)'] = metrics_v1.get('SM_BF_RMSE')
            res['RMSE (v3.2.1.1)'] = metrics_v2.get('SM_BF_RMSE')
            res['RMSE_Diff'] = res['RMSE (v3.2.1.1)'] - res['RMSE (v3.2.1)']
            
            res['Bias (v3.2.1)'] = metrics_v1.get('SM_BF_Bias')
            res['Bias (v3.2.1.1)'] = metrics_v2.get('SM_BF_Bias')
            res['Bias_Diff'] = res['Bias (v3.2.1.1)'] - res['Bias (v3.2.1)']
            
            res['R2 (v3.2.1)'] = metrics_v1.get('SM_BF_R2')
            res['R2 (v3.2.1.1)'] = metrics_v2.get('SM_BF_R2')
            res['R2_Diff'] = res['R2 (v3.2.1.1)'] - res['R2 (v3.2.1)']
            
            res['NSE (v3.2.1)'] = metrics_v1.get('SM_BF_NSE')
            res['NSE (v3.2.1.1)'] = metrics_v2.get('SM_BF_NSE')
            res['NSE_Diff'] = res['NSE (v3.2.1.1)'] - res['NSE (v3.2.1)']
            
            results.append(res)
        else:
            results.append({'Site': site, 'Status': 'Parsing failed'})

    df = pd.DataFrame(results)
    return df

if __name__ == "__main__":
    dir_v1 = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/ptjpl_ptjplsm/output_L6_OWUS_MultiPET_v3.2.1"
    dir_v2 = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/ptjpl_ptjplsm/output_L6_OWUS_MultiPET_v3.2.1.1"
    
    df = compare_versions(dir_v1, dir_v2)
    
    # Save to CSV
    df.to_csv("version_comparison_metrics.csv", index=False)
    
    # Create a nice markdown summary
    with open("version_comparison_summary.md", "w") as f:
        f.write("# PT-JPL Version Comparison Summary (SM_BF Model)\n\n")
        f.write("Comparing `v3.2.1` with `v3.2.1.1` (Analytical OWUS implementation).\n\n")
        
        # Only include successful comparisons in the table
        df_valid = df[df['Status'].isna()]
        if not df_valid.empty:
            f.write("## Metrics Improvement (Negative RMSE Diff is Better)\n\n")
            f.write(df_valid[['Site', 'RMSE (v3.2.1)', 'RMSE (v3.2.1.1)', 'RMSE_Diff', 'R2 (v3.2.1)', 'R2 (v3.2.1.1)', 'R2_Diff']].to_markdown(index=False))
            f.write("\n\n")
        
        # List missing sites
        missing = df[df['Status'].notna()]
        if not missing.empty:
            f.write("## Issues / Missing Data\n\n")
            f.write(missing[['Site', 'Status']].to_markdown(index=False))
            f.write("\n")
            
    print("Comparison complete. Check 'version_comparison_summary.md' and 'version_comparison_metrics.csv'.")
