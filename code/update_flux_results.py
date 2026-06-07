import xarray as xr
import pandas as pd
import numpy as np
import os
import shutil

# 1. Define Directories & Files
out_dir = '/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/beta1_beta2_HB_updated_ET'
os.makedirs(out_dir, exist_ok=True)

param_file = '/home/sanjays/et97_scratch2/oldscratch/Bayesian_merging_APS_3/Final_corrected_data/SMAP/2022-01-01_merged.nc'
tower_file = '/fs04/scratch2/et97/oldscratch/Ozflux_data_full/L6/Yanco_L6.nc'

# 2. Load static parameters and Flux Tower data
print(f"Loading static soil parameters...")
try:
    ds_params = xr.open_dataset(param_file, group='parameters')
    wltsmc = ds_params['WLTSMC']
    maxsmc = ds_params['MAXSMC']
except Exception as e:
    print(f"Error loading parameters: {e}")
    exit(1)

print(f"Loading Flux Tower data from {os.path.basename(tower_file)}...")
try:
    ds_tower = xr.open_dataset(tower_file, decode_times=True)
except Exception as e:
    print(f"Error loading tower data: {e}")
    exit(1)

# 3. Complete list of all files
all_files = [
    '/home/sanjays/et97_scratch2/oldscratch/Bayesian_merging_APS_3/Final_corrected_data/PLMR/PLMR_simulated_tb_HB/2022-01-01_correlatedP_distArea_TBhTBv_Applex_1_updated_sm.nc',
    '/home/sanjays/et97_scratch2/oldscratch/Bayesian_merging_APS_3/Final_corrected_data/PLMR/PLMR_simulated_tb_HB/2022-01-01_correlatedP_distArea_TBhTBv_Applex_2_updated_sm.nc',
    '/home/sanjays/et97_scratch2/oldscratch/Bayesian_merging_APS_3/Final_corrected_data/PPMR/PPMR_simulated_tb_HB/2022-01-01_distArea_corr_Applex1_PPMR_updated_sm.nc',
    '/home/sanjays/et97_scratch2/oldscratch/Bayesian_merging_APS_3/Final_corrected_data/PPMR/PPMR_simulated_tb_HB/2022-01-01_distArea_corr_Applex2_PPMR_updated_sm.nc',
    '/home/sanjays/et97_scratch2/oldscratch/Bayesian_merging_APS_3/Final_corrected_data/SMAP/2022-01-01_merged.nc',
    '/home/sanjays/et97_scratch2/oldscratch/Bayesian_merging_APS_3/Final_corrected_data/SMAP/2021-01-01_merged.nc',
    '/home/sanjays/et97_scratch2/oldscratch/Bayesian_merging_APS_3/Final_corrected_data/SMAP/2020-01-01_merged.nc',
    '/home/sanjays/et97_scratch2/oldscratch/Bayesian_merging_APS_3/Final_corrected_data/SMAP/2019-01-01_merged.nc',
    '/home/sanjays/et97_scratch2/oldscratch/Bayesian_merging_APS_3/Final_corrected_data/SMAP/2018-01-01_merged.nc',
    '/home/sanjays/et97_scratch2/oldscratch/Bayesian_merging_APS_3/Final_corrected_data/SMAP/2017-01-01_merged.nc',
    '/home/sanjays/et97_scratch2/oldscratch/Bayesian_merging_APS_3/Final_corrected_data/SMAP/2016-01-01_merged.nc',
    '/home/sanjays/et97_scratch2/oldscratch/Bayesian_merging_APS_3/Final_corrected_data/SMAP/2015-01-01_merged.nc'
]

for file_path in all_files:
    if not os.path.exists(file_path):
        print(f"  Skipping missing file: {file_path}")
        continue
        
    fname = os.path.basename(file_path)
    print(f"\nProcessing: {fname}...")
    
    out_nc = os.path.join(out_dir, fname.replace('.nc', '_updated.nc'))
    out_csv = os.path.join(out_dir, fname.replace('.nc', '_daily_summary.csv'))
    
    shutil.copy2(file_path, out_nc)
    
    try:
        ds_data = xr.open_dataset(file_path, group='data')
        ds_meta = xr.open_dataset(file_path, group='metadata', decode_times=True) 
        
        smc1 = ds_data['smc1']
        updated_sm = ds_data['updated_sm']
        lh = ds_data['lh']
        dates = ds_meta['date'].values
        
        # --- FLUX TOWER DATA ALIGNMENT & SPATIAL ISOLATION ---
        # 1. Pull the 1D time series and flatten it safely
        fe_matched = ds_tower['Fe'].reindex(time=dates, method='nearest', tolerance='3H').squeeze().values
        if fe_matched.ndim > 1:
            fe_matched = fe_matched.flatten()
            
        # 2. Create an empty 2D array of NaNs shaped (Time, HRU)
        num_times = len(dates)
        num_hrus = len(ds_data['hru'])
        fe_2d = np.full((num_times, num_hrus), np.nan)
        
        # 3. Inject the tower data strictly into HRU index 11
        target_hru = 11
        fe_2d[:, target_hru] = fe_matched
        
        # --- THE PHYSICAL FIX ---
        beta1 = ((smc1 - wltsmc) / (maxsmc - wltsmc)).clip(0, 1)
        beta2 = ((updated_sm - wltsmc) / (maxsmc - wltsmc)).clip(0, 1)
        
        safe_beta1 = beta1.clip(min=0.05)
        potential_lh = (lh / safe_beta1).clip(max=600.0)
        HB_updated_latent_heat = potential_lh * beta2
        # ------------------------
        
        # Add variables. Notice fluxtower_latent_heat is now ('time', 'hru')
        ds_new_vars = xr.Dataset({
            'beta1': beta1,
            'beta2': beta2,
            'HB_updated_latent_heat': HB_updated_latent_heat,
            'fluxtower_latent_heat': (('time', 'hru'), fe_2d)
        })
        
        ds_new_vars['beta1'].attrs = {'long_name': 'Water stress factor (model)'}
        ds_new_vars['beta2'].attrs = {'long_name': 'Water stress factor (updated)'}
        ds_new_vars['HB_updated_latent_heat'].attrs = {'long_name': 'Updated Latent Heat Flux', 'units': 'W/m2'}
        ds_new_vars['fluxtower_latent_heat'].attrs = {'long_name': 'Tower Observed Latent Heat Flux (Fe)', 'units': 'W/m2', 'description': 'Only valid for HRU 11'}
        
        # Append to the NetCDF file
        ds_new_vars.to_netcdf(out_nc, group='data', mode='a', engine='netcdf4')
        print(f"  -> Saved updated NetCDF: {os.path.basename(out_nc)}")

        # ---------------------------------------------------------
        # STEP B: Create the Daily CSV (PRESERVING HRUs)
        # ---------------------------------------------------------
        ds_csv = xr.Dataset({
            'soil_moisture': smc1,
            'lh': lh,
            'beta1': beta1,
            'updated_sm': updated_sm,
            'beta2': beta2,
            'HB_updated_latent_heat': HB_updated_latent_heat,
            'fluxtower_latent_heat': (('time', 'hru'), fe_2d)
        })
        
        ds_csv = ds_csv.assign_coords(date=("time", dates))
        ds_csv = ds_csv.swap_dims({"time": "date"})
        
        ds_daily = ds_csv.resample(date='1D').mean(skipna=True)
        df_daily = ds_daily.to_dataframe().reset_index()
        
        # Drop rows where updated_sm is empty (meaning no flight/smap data)
        df_daily_clean = df_daily.dropna(subset=['updated_sm'])
        
        df_daily_clean.to_csv(out_csv, index=False)
        print(f"  -> Saved daily CSV: {os.path.basename(out_csv)} ({len(df_daily_clean)} rows)")
        
        ds_data.close()
        ds_meta.close()
        ds_new_vars.close()
    except Exception as e:
        print(f"Error processing {fname}: {e}")

ds_params.close()
ds_tower.close()
print("\nSuccess! All files processed with physical flux limits and mapped Flux Tower data.")