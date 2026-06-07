import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os
import glob
import sys
import argparse

def parse_metrics(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    data = []
    current_condition = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for condition header like "=== ALL DAYS (n=586) ==="
        m = re.match(r'===\s*(.+?)\s*===', line)
        if m:
            condition_full = m.group(1)
            
            # Extract the raw condition name and the "n=xxx" part
            raw_condition = re.sub(r'\s*\(n=\d+\)', '', condition_full).strip()
            n_match = re.search(r'\(n=(\d+)\)', condition_full)
            n_val = f"\n(n={n_match.group(1)})" if n_match else ""
            
            # Map condition names to shorter, horizontal-friendly labels
            mapping = {
                'ALL DAYS': 'All Days',
                'DRY DAYS (precip)': 'Dry\n(Precip)',
                'WET DAYS (precip)': 'Wet\n(Precip)',
                'VERY DRY (SM regime)': 'Very Dry\n(SM)',
                'TRANSITION (SM regime)': 'Transition\n(SM)',
                'WET (SM regime)': 'Wet\n(SM)'
            }
            mapped_name = mapping.get(raw_condition, raw_condition)
            
            # Append the n= value to the final label
            current_condition = f"{mapped_name}{n_val}"
            continue
            
        # Skip header lines
        if line.startswith('Model') or line.startswith('---') or (line.startswith('Site:') and '|' not in line[5:]) or line.startswith('Total') or line.startswith('SM regime'):
            continue
            
        # Parse data lines
        if '|' in line and current_condition:
            parts = [p.strip() for p in line.split('|')]
            # Handle the Site line which has | but is not data
            if parts[0].startswith('Site:'):
                continue
                
            if len(parts) >= 8:
                try:
                    data.append({
                        'Condition': current_condition,
                        'Model': parts[0],
                        'RMSE': float(parts[1]),
                        'Bias': float(parts[2]),
                        'R2': float(parts[3]),
                        'NSE': float(parts[4]),
                        'KGE': float(parts[5]),
                        'r': float(parts[6]),
                        'a': float(parts[7]),
                        'b': float(parts[8]) if len(parts) > 8 else np.nan
                    })
                except ValueError as e:
                    pass # Silently skip header lines that look like data
                    
    return pd.DataFrame(data)

def generate_plots(df, site_name, version, out_path_png, out_path_svg):
    # Use seaborn context for professional scientific plots - "ticks" gives a clean look
    sns.set_theme(style="ticks", context="paper", font_scale=1.4)
    
    # Colorblind safe categorical palette
    unique_models = df['Model'].unique()
    colors = sns.color_palette("colorblind", n_colors=len(unique_models))
    
    # Specific layout/ordering requested for publication
    # Top Row: KGE, R2. Bottom Row: RMSE, Bias.
    metrics = ['KGE', 'R2', 'RMSE', 'Bias']
    ylabels = ['KGE (unitless)', r'R² (unitless)', 'RMSE (W m⁻²)', 'Mean Bias (W m⁻²)']
    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Plot bars
        sns.barplot(
            data=df, 
            x='Condition', 
            y=metric, 
            hue='Model', 
            ax=ax,
            palette=colors,
            alpha=0.9,
            edgecolor='black',
            linewidth=1.2
        )
        
        # Format axes and labels
        ax.set_title('')  # Remove inner subplot titles
        ax.set_ylabel(ylabels[i], fontweight='bold')
        ax.set_xlabel('')
        
        # Add (a, b, c, d) panel label to top left, positioned OUTSIDE the plot area
        ax.set_title(panel_labels[i], loc='left', fontsize=16, fontweight='bold', pad=10)
        
        # Gridlines configuration: subtle dashed lines on y-axis only, printed behind bars
        ax.grid(axis='y', linestyle='--', color='gray', alpha=0.3)
        ax.set_axisbelow(True)
        
        # X-axis label formatting
        if i < 2:
            # Top row: completely disable x-tick labels and remove bottom tick marks to avoid clutter
            ax.set_xticklabels([])
            ax.tick_params(axis='x', bottom=False)
        else:
            # Bottom row: horizontal, shortened labels
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
        
        # Reference lines
        if metric in ['Bias', 'NSE', 'KGE']:
            ax.axhline(0, color='darkgray', linewidth=1.5, linestyle='-')
        elif metric in ['R2']:
            ax.set_ylim(0, 1.0)
            
        # Legend handling - remove individual axis legends
        if ax.get_legend() is not None:
            ax.get_legend().remove()
            
    # Clean up right/top border lines globally
    sns.despine()
                 
    # Add a minimal title back in just to show the site name, without cluttering
    formatted_site_name = site_name.replace('_', ' ').title()
    plt.suptitle(f'Site: {formatted_site_name}', fontsize=18, fontweight='bold', y=1.03)

    plt.tight_layout()
    
    # Central horizontal legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='', # No title needed for clean look
               loc='lower center', ncol=len(unique_models), bbox_to_anchor=(0.5, -0.05), frameon=False)
    
    # Save figures
    fig.savefig(out_path_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_path_svg, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready plots from PT-JPL metrics files")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing site folders")
    parser.add_argument("--version", type=str, default="v3.2.1", help="Model version string (e.g., v3.2.1)")
    parser.add_argument("--site", type=str, help="Specific site to process (if omitted, processes all sites in base_dir)")
    
    args = parser.parse_args()
    
    base_dir = args.base_dir
    version = args.version
    
    # Determine which sites to process
    if args.site:
        sites = [args.site]
    else:
        # List all subdirectories in the base directory
        sites = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        
    print(f"Found {len(sites)} sites to process.")
    
    success_count = 0
    failure_count = 0
    
    for site in sites:
        site_dir = os.path.join(base_dir, site)
        
        # Try a few common naming conventions for the metrics file
        metrics_file = None
        
        # 1. Exact match pattern based on site name and version
        expected_name = f"{site}_{version}_metrics.txt"
        path1 = os.path.join(site_dir, expected_name)
        
        # 2. Case-insensitive search
        for pattern in [f"*{version}_metrics.txt", "*_metrics.txt"]:
            matches = glob.glob(os.path.join(site_dir, pattern))
            if matches:
                metrics_file = matches[0]
                break
                
        if not metrics_file:
            print(f"[{site}] Skipped: No metrics file found in {site_dir}")
            failure_count += 1
            continue
            
        try:
            # Extract actual site name from filename if possible
            basename = os.path.basename(metrics_file)
            site_display_name = basename.split(f"_{version}")[0] if f"_{version}" in basename else site
            
            # Setup output path
            out_basename = f"{site_display_name}_performance_analytics"
            out_png = os.path.join(site_dir, f"{out_basename}.png")
            out_svg = os.path.join(site_dir, f"{out_basename}.svg")
            
            # Parse and plot
            df = parse_metrics(metrics_file)
            if len(df) > 0:
                generate_plots(df, site_display_name, version, out_png, out_svg)
                print(f"[{site}] Success: Generated {out_basename} (.png and .svg)")
                success_count += 1
            else:
                print(f"[{site}] Failed: No valid data parsed from {metrics_file}")
                failure_count += 1
                
        except Exception as e:
            print(f"[{site}] Error processing {metrics_file}: {e}")
            import traceback
            traceback.print_exc()
            failure_count += 1
            
    print(f"\nProcessing Complete: {success_count} successful, {failure_count} failures.")

if __name__ == "__main__":
    main()
