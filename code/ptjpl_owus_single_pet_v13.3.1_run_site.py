#!/bin/env python3
# -*- coding: utf-8 -*-

"""
ptjpl_owus_single_pet_v13.3.1_run_site.py
-------------------------------------------------------------------
Version 13.3.1: SM-Only Grid Search + VOD Integration + Single PET (Defense-Proofed)
- Incorporates VODCA CXKu-Band and L-Band data as moisture proxies.
- Normalizes VOD using historical site min/max to interface with OWUS thresholds.
- Removed multiple PET inputs; relies exclusively on base PT-JPL PET.
- Retains time-adaptive K-Means and non-linear shapes (Exp/Sig).
- Fixed cf conversion factor (LV / 86400, not LV * RHO_W / 86400).
- CHANGE vs v13.3: "Base" removed from soil_opts grid search. Because all three
  proxy types (SM, VOD_C, VOD_L) in the OWUS architecture use explicit external
  observations mapped through OWUS thresholds (s_wilt, s_star, beta_ww), applying
  this machinery to the implicit PT-JPL soil evaporation (RH^VPD scalar) is
  physically inconsistent. "Base" total LE is still saved as LE_PTJPL_Base.
  Grid search: 432 → 216 variants (3 proxies × 1 soil × 2 OWUS × 1 PET × 2 int
  × 3 shape × 2 dyn).
"""

import os
import sys
import glob
import re
import types
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================
# 1. SETUP OFFLINE ENVIRONMENT
# =====================================================================
os.environ["PTJPLSM_OFFLINE"] = "1"
os.environ["PTJPL_OFFLINE"] = "1"
os.environ["NO_DOWNLOAD"] = "1"

try:
    from PTJPLSM.process_PTJPLSM_table import process_PTJPLSM_table
    from PTJPL.process_PTJPL_table import process_PTJPL_table
except ImportError:
    try:
        from PTJPLSM import PTJPLSM  # noqa: F401
    except ImportError:
        print("[ERROR] Could not import PTJPLSM.")
        sys.exit(1)

# =====================================================================
# 2. CONFIGURATION
# =====================================================================
UNIFIED_CSV = (
    "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/ptjpl_ptjplsm/"
    "OzFluxStations_NDVI_Soil_MetaUnified_withCanopy_LAI_FPAR_"
    "2003_2024_DAILY_FIXED.csv"
)
L6_DATA_DIR = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/L6"
OWUS_BF_CSV = (
    "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/"
    "OWUS_australia_agent_3_scientific/output_corrected/combined/"
    "results_bf__ozflux_1.csv"
)
OWUS_OPT_CSV = (
    "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/"
    "OWUS_australia_agent_3_scientific/output_corrected/combined/"
    "results_opt__ozflux_1.csv"
)

META_CSV = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/OWUS_australia/sanjay data creation/ozflux_metadata.csv"
KG_CSV = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/OWUS_australia/sanjay data creation/ozflux_Köppen_climate_classification.csv"

# --- VOD SOURCES (v13) ---
VOD_CXKU_CSV = "/fs04/scratch2/et97/oldscratch/Ozflux_data_full/vod/ozflux_VODCA_CXKu_DAILY_INTERPOLATED_2003_2021.csv"
VOD_L_CSV = "/fs04/scratch2/et97/oldscratch/Ozflux_data_full/vod/ozflux_VODCA_L_DAILY_INTERPOLATED_2010_2021.csv"

OUTPUT_DIR = (
    "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/ptjpl_ptjplsm/"
    "output_L6_OWUS_SinglePET_v13.3.1_DynamicBeta"
)

SIGMA = 5.670374419e-8
TOPT_FALLBACK = 30.0
LV = 2.45e6
RHO_W = 1000
FE_VALID_MIN = -100.0
FE_VALID_MAX = 1000.0

GASH_S, GASH_P, GASH_PT, GASH_ER = 1.0, 0.2, 0.02, 0.15
USE_FPAR_FOR_COVER = True
LAI_K_Beer = 0.5
COVER_MIN, COVER_MAX = 0.05, 0.98

SM_REGIME_THRESHOLD_SET = "OPT"

SITE_MAPPING = {
    "gatumpasture": "AU-Gat", "otway": "AU-Otw", "silverplains": "AU-SiP", 
    "emerald": "AU-Eme", "alpinepeatland": "AU-Alp", "riggscreek": "AU-Rig",
    "yarramundiirrigated": "AU-YarI", "yarramundicontrol": "AU-YarC",
    "yanco": "AU-Ync", "dalypasture": "AU-DaP", "foggdam": "AU-Fog", 
    "sturtplains": "AU-Stp", "wallabycreek": "AU-Wal", "collie": "AU-Col",
    "boyagin": "AU-Boy", "adelaideriver": "AU-Adr", "dalyuncleared": "AU-DaS", 
    "litchfield": "AU-Lit", "greatwesternwoodlands": "AU-Gre", "ridgefield": "AU-Rgf",
    "capetribulation": "AU-Ctr", "cowbay": "AU-Cow", "whroo": "AU-Whr", 
    "tumbarumba": "AU-Tum", "loxton": "AU-Lox", "warra": "AU-War",
    "gingin": "AU-Gin", "reddirtmelonfarm": "AU-RDF", "howardsprings": "AU-How", 
    "cumberlandplain": "AU-Cum", "robsoncreek": "AU-Rob", "alicespringsmulga": "AU-ASM",
    "titreeeast": "AU-TTE", "digbyplantation": "AU-Dig", "wombatstateforest": "AU-Wom",
}

# =====================================================================
# 3. UTILITY FUNCTIONS
# =====================================================================
def norm_site_string(s):
    if not s: return ""
    return re.sub(r"[\s_]+", "", str(s).strip().lower())

def strip_trailing_digits(s):
    return re.sub(r"\d+$", "", s)

def get_au_season(month):
    if month in [12, 1, 2]: return "SUMMER"
    if month in [3, 4, 5]: return "AUTUMN"
    if month in [6, 7, 8]: return "WINTER"
    return "SPRING"

def find_nc_files(site_name, search_dir):
    target = strip_trailing_digits(norm_site_string(site_name))
    files = glob.glob(os.path.join(search_dir, "*_L6.nc"))
    return sorted([f for f in files if strip_trailing_digits(norm_site_string(os.path.basename(f).replace("_L6.nc", "").replace(".nc", ""))) == target])

def calculate_stats(obs, model):
    df = pd.DataFrame({"o": obs, "m": model}).dropna()
    nans = {k: np.nan for k in ["RMSE", "Bias", "R2", "ubRMSE", "MAE", "NSE", "KGE", "r", "alpha", "beta"]}
    if len(df) < 10: return nans
    o, m = df["o"].astype(float), df["m"].astype(float)
    rmse = np.sqrt(((m - o) ** 2).mean())
    bias = (m - o).mean()
    mae = np.abs(m - o).mean()
    ubrmse = np.sqrt(np.maximum(rmse**2 - bias**2, 0))
    std_o, std_m = o.std(), m.std()
    mean_o, mean_m = o.mean(), m.mean()
    r = np.corrcoef(o, m)[0, 1] if std_o > 0 and std_m > 0 else np.nan
    r2 = (r ** 2) if np.isfinite(r) else np.nan
    denom_nse = ((o - mean_o) ** 2).sum()
    nse = (1 - (((m - o) ** 2).sum() / denom_nse) if denom_nse > 0 else np.nan)
    alpha = (std_m / std_o) if std_o > 0 else np.nan
    beta = (mean_m / mean_o) if mean_o != 0 else np.nan
    kge = (1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2) if np.isfinite(r) and np.isfinite(alpha) and np.isfinite(beta) else np.nan)
    return {"RMSE": rmse, "Bias": bias, "R2": r2, "ubRMSE": ubrmse, "MAE": mae, "NSE": nse, "KGE": kge, "r": r, "alpha": alpha, "beta": beta}

def derive_topt(df):
    gpp = df.get("GPP_SOLO", df.get("GPP"))
    ta = df.get("Ta")
    if gpp is None or ta is None: return TOPT_FALLBACK
    df_t = pd.DataFrame({"Ta": ta, "GPP": gpp}).dropna()
    df_t = df_t[(df_t["Ta"] > -5) & (df_t["Ta"] < 50) & (df_t["GPP"] > 0)]
    if df_t.empty: return TOPT_FALLBACK
    gpp_bin = df_t.groupby(df_t["Ta"].round(0))["GPP"].mean()
    return float(gpp_bin.idxmax()) if not gpp_bin.empty else TOPT_FALLBACK

def canopy_cover_proxy(daily_df):
    if USE_FPAR_FOR_COVER and "FPAR" in daily_df.columns and daily_df["FPAR"].notna().any():
        c = daily_df["FPAR"].astype(float)
    elif "LAI" in daily_df.columns and daily_df["LAI"].notna().any():
        lai = daily_df["LAI"].astype(float).clip(lower=0.0)
        c = 1.0 - np.exp(-LAI_K_Beer * lai)
        c = pd.Series(c, index=daily_df.index)
    else:
        c = pd.Series(np.nan, index=daily_df.index)
    c = c.clip(COVER_MIN, COVER_MAX).fillna(c.median() if c.notna().any() else 0.5)
    return c

def compute_gash_daily_dynamic(p_hh, c_daily, S=GASH_S, p0=GASH_P, pt0=GASH_PT, ER0=GASH_ER):
    s = p_hh.dropna()
    daily_idx = p_hh.resample("1D").sum().index
    zeros = pd.Series(0.0, index=daily_idx)
    if s.empty: return zeros
    rain_t = s[s > 0].index
    if len(rain_t) == 0: return zeros
    ev_ids = (rain_t.to_series().diff() >= pd.Timedelta(hours=6)).cumsum()
    events = s[s > 0].groupby(ev_ids).sum()
    ev_day = s[s > 0].groupby(ev_ids).apply(lambda x: x.index.min()).dt.floor("D")
    c_event = ev_day.map(c_daily.reindex(daily_idx).fillna(method="ffill").fillna(method="bfill")).astype(float).values
    p_event, pt_event, ER_event = np.clip(1.0 - c_event, 0.0, 0.999), np.clip(pt0 * c_event, 0.0, 0.2), np.clip(ER0 * c_event, 1e-6, 0.95)
    denom_log = 1.0 - p_event - pt_event
    PG_prime = np.full(len(events), S, dtype=float)
    ok = (denom_log > 1e-6) & (ER_event < denom_log)
    PG_prime[ok] = -(S / ER_event[ok]) * np.log(1.0 - ER_event[ok] / denom_log[ok])
    R_ev = events.values.astype(float)
    I_ev = np.where(R_ev < PG_prime, (1.0 - p_event - pt_event) * R_ev, (1.0 - p_event - pt_event) * PG_prime + ER_event * (R_ev - PG_prime))
    df_rain = pd.DataFrame({"R": s[s > 0], "eid": ev_ids})
    frac = (df_rain["R"] / df_rain.groupby("eid")["R"].transform("sum").replace(0, np.nan)).astype(float)
    I_hh = pd.Series(0.0, index=p_hh.index)
    I_hh.update((frac * df_rain["eid"].map(pd.Series(I_ev, index=events.index))).dropna())
    return I_hh.resample("1D").sum().reindex(daily_idx, fill_value=0.0)

def simple_kmeans(X, n_clusters=2, max_iter=100):
    np.random.seed(42)
    idx = np.random.choice(len(X), n_clusters, replace=False)
    centers = X[idx]
    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = np.array([X[labels == j].mean(axis=0) if np.sum(labels == j) > 0 else centers[j] for j in range(n_clusters)])
        if np.allclose(centers, new_centers): break
        centers = new_centers
    return labels

# =====================================================================
# 4. CORE PROCESSING
# =====================================================================
def process_single_nc(site_name, nc_path, df_owus_bf, df_owus_opt, meta_all, df_science_meta, df_vod_c, df_vod_l):
    site_suffix = os.path.basename(nc_path).replace("_L6.nc", "")
    site_label = f"{site_name}::{site_suffix}"
    site_dir = os.path.join(OUTPUT_DIR, norm_site_string(site_suffix))
    os.makedirs(site_dir, exist_ok=True)
    
    out_csv = os.path.join(site_dir, f"{site_suffix}_v13.3.1_results.csv")
    metrics_txt = os.path.join(site_dir, f"{site_suffix}_v13.3.1_metrics.txt")
    manifest_csv = os.path.join(site_dir, f"{site_suffix}_v13.3.1_manifest.csv")

    if os.path.exists(out_csv):
        print(f"\n=== SKIP: {site_label} (exists) ===")
        return

    print(f"\n=== PROCESSING v13.3.1: {site_label} ===")

    sn_low = norm_site_string(site_suffix)
    sc_row = df_science_meta[df_science_meta["site_match"] == sn_low]
    if sc_row.empty:
        sc_row = df_science_meta[df_science_meta["site_match"] == strip_trailing_digits(sn_low)]
        
    kg_label = sc_row["KG_Label"].iloc[0] if not sc_row.empty else "Unknown"
    kg_code = sc_row["KG_Code"].iloc[0] if not sc_row.empty else -1
    veg_type = sc_row["Vegetation Type"].iloc[0] if not sc_row.empty else "Unknown"
    h_meta = sc_row["Canopy_Height_m"].iloc[0] if not sc_row.empty else 0.5
    
    print(f"  [INFO] Metadata: KG={kg_label}, Veg={veg_type}, H={h_meta}m")

    ds = None
    for eng in ["h5netcdf", None, "scipy"]:
        try: ds = xr.open_dataset(nc_path, engine=eng); break
        except Exception: continue
    if ds is None:
        print(f"  [ERROR] Cannot open {nc_path}"); return

    lat_val = float(ds.attrs.get("latitude", ds.get("latitude", np.array(0.0)).values.flat[0]))
    lon_val = float(ds.attrs.get("longitude", ds.get("longitude", np.array(0.0)).values.flat[0]))
    ds = ds.squeeze(drop=True)

    def get_owus(df_owus, label=""):
        if df_owus.empty: return 0.65, 0.20, 0.95
        cn = strip_trailing_digits(norm_site_string(site_suffix))
        tid = SITE_MAPPING.get(cn, site_suffix)
        m = df_owus[df_owus["siteID"].str.lower() == tid.lower()]
        if not m.empty: return float(m.iloc[0]["s_star_median"]), float(m.iloc[0]["s_wilt_median"]), float(m.iloc[0]["beta_ww_median"])
        dist = np.sqrt((df_owus["lon"] - lon_val)**2 + (df_owus["lat"] - lat_val)**2)
        b = df_owus.loc[dist.idxmin()]
        return float(b["s_star_median"]), float(b["s_wilt_median"]), float(b["beta_ww_median"])

    bf_ss, bf_sw, bf_bw = get_owus(df_owus_bf, "BF")
    opt_ss, opt_sw, opt_bw = get_owus(df_owus_opt, "OPT")

    vn = ["Fsd", "Fsu", "Fld", "Flu", "Fn", "Fg", "Ta", "RH", "Sws", "Fe", "Precip", "GPP_SOLO", "GPP"]
    df_raw = ds[[v for v in vn if v in ds]].to_dataframe()
    if "time" in df_raw.columns: df_raw = df_raw.set_index("time")
    p_hh = (ds["Precip"].to_series() if "Precip" in ds else pd.Series(0.0, index=df_raw.index))
    if "time" not in p_hh.index.names: p_hh.index = df_raw.index

    df_d = df_raw.resample("1D").mean(numeric_only=True)
    if "Fe" in df_d:
        df_d.loc[(df_d["Fe"] < FE_VALID_MIN) | (df_d["Fe"] > FE_VALID_MAX), "Fe"] = np.nan
    df_d["Precip"] = p_hh.resample("1D").sum()
    df_d["lat"], df_d["lon"] = lat_val, lon_val

    if {"Fsd", "Fsu"}.issubset(df_d.columns):
        alb_c = (df_d["Fsu"] / df_d["Fsd"]).replace([np.inf, -np.inf], np.nan).dropna()
        alb_c = alb_c[(alb_c > 0) & (alb_c < 1)]
        df_d["albedo"] = (df_d["Fsu"] / df_d["Fsd"]).clip(float(alb_c.quantile(0.01)) if len(alb_c) > 10 else 0.05, float(alb_c.quantile(0.99)) if len(alb_c) > 10 else 0.80)
    if "Flu" in df_d: df_d["ST_C"] = ((df_d["Flu"] / (0.98 * SIGMA)) ** 0.25) - 273.15
    if "Fn" in df_d: df_d["Rn"] = df_d["Fn"]
    elif {"Fsd", "Fsu"}.issubset(df_d.columns): df_d["Rn"] = (df_d["Fsd"] - df_d["Fsu"] + (df_d["Fld"] - df_d["Flu"] if {"Fld", "Flu"}.issubset(df_d) else 0))
    else: df_d["Rn"] = np.nan
    if "RH" in df_d and df_d["RH"].max() > 1.5: df_d["RH"] /= 100.0

    sn, sb = norm_site_string(site_name), strip_trailing_digits(norm_site_string(site_name))
    ms = meta_all[meta_all["site_norm"] == sn].copy()
    if ms.empty: ms = meta_all[meta_all["site_base"] == sb].copy()
    if ms.empty: ms = meta_all[meta_all["site_norm"].str.contains(sb)]
    
    if not ms.empty:
        ms = ms.copy()
        ms["date"] = pd.to_datetime(ms["date"])
        ma = ms.groupby("date").mean(numeric_only=True).sort_index().reindex(df_d.index).interpolate(limit_direction="both")
        
        df_d["NDVI"] = ma.get("ndvi", np.nan).clip(0, 1)
        df_d["LAI"] = ma.get("lai", np.nan)
        df_d["FPAR"] = ma.get("fpar", np.nan).clip(0, 1)
        df_d["canopy_height_meters"] = h_meta if h_meta > 0.5 else ma.get("canopy_height", 0.5)
        df_d["field_capacity"] = ma.get("fc_000_005" if "fc_000_005" in ma.columns and ma["fc_000_005"].notna().any() else "fc_0_30cm", 0.30)
        df_d["wilting_point"] = ma.get("wp_000_005" if "wp_000_005" in ma.columns and ma["wp_000_005"].notna().any() else "wp_0_30cm", 0.10)
    else:
        df_d["NDVI"], df_d["canopy_height_meters"], df_d["field_capacity"], df_d["wilting_point"] = 0.5, h_meta, 0.3, 0.1

    if "Sws" in df_d: df_d["SM"] = df_d["Sws"].clip(lower=df_d["wilting_point"], upper=df_d["field_capacity"])
    else: df_d["SM"] = np.nan
    
    # -----------------------------------------------------------------
    # INTEGRATE VOD DATA (v13)
    # -----------------------------------------------------------------
    if not df_vod_c.empty:
        vc = df_vod_c[df_vod_c['site_match'] == sn_low].copy()
        if vc.empty: vc = df_vod_c[df_vod_c['site_match'] == sb].copy()
        if not vc.empty:
            vc = vc.groupby('date').mean(numeric_only=True)
            vod_col = 'VOD_CXKu' if 'VOD_CXKu' in vc.columns else 'VOD_C'
            if vod_col in vc.columns: df_d['VOD_CXKu'] = vc[vod_col]

    if not df_vod_l.empty:
        vl = df_vod_l[df_vod_l['site_match'] == sn_low].copy()
        if vl.empty: vl = df_vod_l[df_vod_l['site_match'] == sb].copy()
        if not vl.empty:
            vl = vl.groupby('date').mean(numeric_only=True)
            if 'VOD_L' in vl.columns: df_d['VOD_L'] = vl['VOD_L']

    df_d["fAPARmax"] = df_d["FPAR"].quantile(0.95) if "FPAR" in df_d and df_d["FPAR"].notna().any() else (df_d["NDVI"].quantile(0.95) if "NDVI" in df_d else 0.7)
    df_d["Topt_C"] = derive_topt(df_d)

    ns_nd = df_d["NDVI"].dropna()
    if not ns_nd.empty:
        nmin, nmax = (float(ns_nd.quantile(0.05)), float(ns_nd.quantile(0.95))) if len(ns_nd) > 20 else (0.10, 0.85)
        df_d["emissivity"] = 0.97 * (1 - ((df_d["NDVI"] - nmin) / (nmax - nmin)).clip(0, 1) ** 2) + 0.99 * ((df_d["NDVI"] - nmin) / (nmax - nmin)).clip(0, 1) ** 2
    else: df_d["emissivity"] = 0.98

    inp = df_d.copy()
    if "Ta" in inp: inp.rename(columns={"Ta": "Ta_C"}, inplace=True)
    req = ["NDVI", "ST_C", "albedo", "Ta_C", "RH", "SM", "Rn", "fAPARmax", "lat", "lon"]
    
    for r in req:
        if r not in inp.columns: inp[r] = np.nan
            
    df_run = inp[inp[req].notna().all(axis=1)].copy()
    if df_run.empty: print("  [ERROR] No valid rows"); return

    prec = df_d.reindex(df_run.index)["Precip"]
    dry_mask = ((prec == 0) & (prec.shift(1) == 0)).fillna(False)
    wet_mask = ((prec > 0) | (prec.shift(1) > 0)).fillna(False)

    mg, mw = types.ModuleType("gedi_canopy_height"), types.ModuleType("soil_capacity_wilting")
    mg.load_canopy_height = lambda **k: np.full(len(df_run), df_run["canopy_height_meters"].iloc[0])
    mw.load_field_capacity = lambda **k: np.full(len(df_run), df_run["field_capacity"].iloc[0])
    mw.load_wilting_point = lambda **k: np.full(len(df_run), df_run["wilting_point"].iloc[0])
    sys.modules["gedi_canopy_height"], sys.modules["soil_capacity_wilting"] = mg, mw

    res_sm, res_base = process_PTJPLSM_table(df_run), process_PTJPL_table(df_run)

    N = len(df_run)
    LE_can_orig = res_base[next((c for c in ["LE_canopy", "LE_canopy_Wm2"] if c in res_base.columns), None)].values
    LE_soil_base = res_base[next((c for c in ["LE_soil", "LE_soil_Wm2"] if c in res_base.columns), None)].values if next((c for c in ["LE_soil", "LE_soil_Wm2"] if c in res_base.columns), None) else np.zeros(N)
    LE_int_base = res_base[next((c for c in ["LE_interception", "LE_interception_Wm2"] if c in res_base.columns), None)].values if next((c for c in ["LE_interception", "LE_interception_Wm2"] if c in res_base.columns), None) else np.zeros(N)
    LE_tot_base = res_base[next((c for c in ["LE", "LE_Wm2", "evapotranspiration"] if c in res_base.columns), None)].values if next((c for c in ["LE", "LE_Wm2", "evapotranspiration"] if c in res_base.columns), None) else np.zeros(N)
    LE_soil_sm = res_sm[next((c for c in ["LE_soil", "LE_soil_Wm2"] if c in res_sm.columns), None)].values if next((c for c in ["LE_soil", "LE_soil_Wm2"] if c in res_sm.columns), None) else np.zeros(N)
    LE_int_sm = res_sm[next((c for c in ["LE_interception", "LE_interception_Wm2"] if c in res_sm.columns), None)].values if next((c for c in ["LE_interception", "LE_interception_Wm2"] if c in res_sm.columns), None) else np.zeros(N)

    # Conversion factor from mm/day to W/m2: 1 mm/day = 1 kg/m²/day → multiply by LV / 86400
    cf = LV / (24 * 3600)
    Ei_mm = compute_gash_daily_dynamic(p_hh, canopy_cover_proxy(df_d.reindex(df_run.index)).reindex(p_hh.resample("1D").sum().index)).reindex(df_run.index).fillna(0).values
    LE_int_gash = Ei_mm * cf

    # Base PET from PTJPL (single PET - no multi-PET)
    pet_pt = res_base["PET"].values if "PET" in res_base.columns else np.full(N, 1000.0)

    wp, fc = df_run["wilting_point"].values, df_run["field_capacity"].values
    
    # Generate 3 Proxies: Normalized SM, VOD_C, VOD_L
    s_arr = np.clip((df_run["SM"].values - wp) / np.maximum(fc - wp, 1e-6), 0.0, 1.0)
    
    vod_c_vals = df_run.get("VOD_CXKu", pd.Series(np.nan, index=df_run.index)).values
    vod_l_vals = df_run.get("VOD_L", pd.Series(np.nan, index=df_run.index)).values
    
    with np.errstate(invalid='ignore'):
        vc_min, vc_max = np.nanmin(vod_c_vals), np.nanmax(vod_c_vals)
        vl_min, vl_max = np.nanmin(vod_l_vals), np.nanmax(vod_l_vals)
        
    vc_den = np.maximum(vc_max - vc_min, 1e-6) if not np.isnan(vc_min) else 1e-6
    vl_den = np.maximum(vl_max - vl_min, 1e-6) if not np.isnan(vl_min) else 1e-6
    
    vod_c_arr = np.clip((vod_c_vals - vc_min) / vc_den, 0.0, 1.0) if not np.isnan(vc_min) else np.full(N, np.nan)
    vod_l_arr = np.clip((vod_l_vals - vl_min) / vl_den, 0.0, 1.0) if not np.isnan(vl_min) else np.full(N, np.nan)
    
    cluster_feat = pd.DataFrame(index=df_run.index)
    cluster_feat['LAI_roll'] = df_run['LAI'].rolling(window=30, min_periods=1).mean()
    cluster_feat['PET_roll'] = pd.Series(pet_pt, index=df_run.index).rolling(window=30, min_periods=1).mean()
    cluster_feat = cluster_feat.fillna(cluster_feat.median()).values
    
    X_scaled = (cluster_feat - np.mean(cluster_feat, axis=0)) / (np.std(cluster_feat, axis=0) + 1e-6)
    regimes = simple_kmeans(X_scaled, n_clusters=2)
    
    dyn_mult_ss = np.where(regimes == 0, 0.9, 1.1)
    dyn_mult_sw = np.where(regimes == 0, 0.9, 1.1)
    dyn_mult_bw = np.where(regimes == 0, 1.05, 0.95)

    fM = np.clip(df_run["FPAR"].values / df_run["fAPARmax"].values, 0.0, 1.0)
    weight = df_run["RH"].values ** (4.0 * (1.0 - s_arr) * (1.0 - df_run["RH"].values))

    owus_results = {}; manifest_records = []

    def calc_le_owus(proxy_arr, s_star_arr, s_wilt_arr, beta_ww_arr, soil_comp, int_comp, pet_limit, bid_base, shape_t='lin', shape_p=0.0):
        # Calculate stress factor based on whichever proxy is passed (SM, VOD_C, VOD_L)
        s_norm = np.clip((proxy_arr - s_wilt_arr) / np.maximum(s_star_arr - s_wilt_arr, 1e-6), 0.0, 1.0)
        
        if shape_t == 'lin':
            f_OWUS_val = s_norm
        elif shape_t == 'exp':
            den_exp = 1.0 - np.exp(shape_p)
            if abs(den_exp) < 1e-6:
                f_OWUS_val = s_norm
            else:
                f_OWUS_val = (1.0 - np.exp(shape_p * s_norm)) / den_exp
        elif shape_t == 'sig':
            f_OWUS_val = (1.0 - 2.0 ** (- ((s_norm + 1e-6) / 0.5) ** shape_p)) / (1.0 - 2.0 ** (- (1.0 / 0.5) ** shape_p))
            f_OWUS_val = np.clip(f_OWUS_val, 0.0, 1.0)
        else:
            f_OWUS_val = s_norm

        f_OWUS = np.where(proxy_arr > s_star_arr, beta_ww_arr, beta_ww_arr * f_OWUS_val)
        f_OWUS = np.where(proxy_arr <= s_wilt_arr, 0.0, f_OWUS)
        
        f_TRM = (1.0 - weight) * fM + weight * f_OWUS
        fM_safe = np.maximum(fM, 0.05)
        LE_can = np.divide(LE_can_orig * f_TRM, fM_safe, out=np.zeros_like(LE_can_orig), where=(fM_safe > 0))
        
        PET_rem = np.maximum(pet_limit - int_comp, 0.0)
        LE_sc_raw = soil_comp + LE_can
        
        LE_tc = int_comp + np.clip(LE_sc_raw, 0, PET_rem)
        LE_cc = int_comp + soil_comp + np.maximum(LE_can - np.maximum(LE_sc_raw - PET_rem, 0.0), 0.0)
        LE_sc = int_comp + soil_comp + LE_can * np.clip(np.where(LE_sc_raw > PET_rem, np.divide(PET_rem - soil_comp, LE_can, out=np.ones_like(LE_can), where=LE_can > 0), 1.0), 0, 1)
        
        return LE_tc, LE_cc, LE_sc

    # =================================================================
    # REDUCED GRID SEARCH: 216 Variants (SM-only soil; Base removed)
    # -----------------------------------------------------------------
    # Rationale: All three proxy types (SM, VOD_C, VOD_L) use explicit
    # external observations fed through OWUS thresholds (s_wilt, s_star,
    # beta_ww). The PT-JPL "Base" soil evaporation instead uses an
    # *implicit* RH^VPD scalar — there is no explicit soil moisture
    # signal to pair with these thresholds and the dynamic-beta machinery.
    # Applying s_arr thresholds calibrated for explicit SM to the implicit
    # Base soil term breaks the physical logic for all three proxies.
    # "Base" total LE is still preserved as LE_PTJPL_Base below.
    # Grid search: 432 → 216 variants (3 × 1 × 2 × 1 × 2 × 3 × 2).
    # =================================================================
    proxy_opts = {"SM": s_arr, "VODC": vod_c_arr, "VODL": vod_l_arr}
    soil_opts = {"SM": (LE_soil_sm, LE_int_sm)}   # Base removed — SM only
    owus_opts = {"BF": (bf_ss, bf_sw, bf_bw), "OPT": (opt_ss, opt_sw, opt_bw)}
    int_opts = {"PTJPL": None, "GASH": LE_int_gash}
    shape_opts = {"Lin": ("lin", 0.0), "Exp": ("exp", 2.0), "Sig": ("sig", 2.0)}
    dyn_opts = {"Stat": False, "Dyn": True}

    for proxy_name, proxy_v in proxy_opts.items():
        for s_name, (soil_v, int_ptjpl) in soil_opts.items():
            for o_name, (ss, sw, bw) in owus_opts.items():
                for i_name, i_val in int_opts.items():
                    for sh_name, (sh_t, sh_p) in shape_opts.items():
                        for d_name, is_dyn in dyn_opts.items():
                            act_int = int_ptjpl if i_name == "PTJPL" else i_val
                            plim = pet_pt
                            p_name = "PT"  # Single PET for downstream column compatibility
                            
                            if is_dyn:
                                ss_arr = ss * dyn_mult_ss
                                sw_arr = sw * dyn_mult_sw
                                ss_arr = np.maximum(ss_arr, sw_arr + 1e-4) 
                                bw_arr = np.clip(bw * dyn_mult_bw, 0.0, 1.0)
                            else:
                                ss_arr = np.full(N, ss)
                                sw_arr = np.full(N, sw)
                                ss_arr = np.maximum(ss_arr, sw_arr + 1e-4)
                                bw_arr = np.full(N, bw)
                                
                            bid = f"{proxy_name}_{s_name}_{o_name}_{p_name}_{i_name}_{sh_name}_{d_name}"
                            tc, cc, sc = calc_le_owus(proxy_v, ss_arr, sw_arr, bw_arr, soil_v, act_int, plim, bid, sh_t, sh_p)
                            
                            owus_results[f"{bid}_TC"], owus_results[f"{bid}_CC"], owus_results[f"{bid}_SC"] = tc, cc, sc
                            for cap in ["TC", "CC", "SC"]:
                                manifest_records.append({
                                    "Model_ID": f"{bid}_{cap}", "Proxy": proxy_name, "Soil": s_name, "OWUS": o_name, 
                                    "PET": p_name, "Cap": cap, "Int": i_name, "Shape": sh_name, "Dynamic": d_name,
                                    "KG_Label": kg_label, "Veg": veg_type
                                })

    # Final Output Assemble
    output = pd.DataFrame(index=df_run.index)
    output["LE_Obs"] = df_d.reindex(df_run.index)["Fe"]
    output["LE_PTJPL_Base"] = LE_tot_base   # Unadulterated base model — preserved as reference
    output["LE_PTJPL_SM"] = res_sm["LE"]
    
    for k, v in owus_results.items(): output[k] = v

    output["Precip"] = prec
    output["dry_day"] = dry_mask.astype(int)
    output["wet_day"] = wet_mask.astype(int)
    
    output["VOD_CXKu"] = vod_c_vals
    output["VOD_L"] = vod_l_vals
    output["SM_PAW"] = s_arr
    output["SM_regime_threshold_set"] = "OPT"
    output["SM_s_wilt_reg"] = float(opt_sw)
    output["SM_s_star_reg"] = float(opt_ss)

    output["canopy_cover_c"] = canopy_cover_proxy(df_d.reindex(df_run.index)).values

    output["KG_Label"] = kg_label
    output["Vegetation_Type"] = veg_type
    output["Year"] = output.index.year
    output["Season"] = output.index.month.map(get_au_season)
    output["Regime_Code"] = regimes 

    regime_sm = np.full(s_arr.shape, "nan", dtype=object)
    regime_sm_code = np.full(s_arr.shape, -1, dtype=int)
    m_valid = np.isfinite(s_arr)
    m_vdry = m_valid & (s_arr <= opt_sw)
    m_trans = m_valid & (s_arr > opt_sw) & (s_arr <= opt_ss)
    m_wet = m_valid & (s_arr > opt_ss)
    regime_sm[m_vdry] = "very_dry"
    regime_sm[m_trans] = "transition"
    regime_sm[m_wet] = "wet"
    regime_sm_code[m_vdry] = 0
    regime_sm_code[m_trans] = 1
    regime_sm_code[m_wet] = 2

    output["regime_sm"] = regime_sm
    output["regime_sm_code"] = regime_sm_code

    output.to_csv(out_csv, float_format="%.4f")
    pd.DataFrame(manifest_records).to_csv(manifest_csv, index=False)

    # Metrics File
    all_model_cols = ["LE_PTJPL_Base", "LE_PTJPL_SM"] + list(owus_results.keys())
    
    sm_vdry_mask = (regime_sm_code == 0)
    sm_trans_mask = (regime_sm_code == 1)
    sm_wet_mask = (regime_sm_code == 2)
    
    with open(metrics_txt, "w") as f:
        f.write(f"Site: {site_name} | L6: {site_suffix} | KG: {kg_label} | Veg: {veg_type}\n")
        f.write(f"Rows: {len(output)} | Dry(precip): {int(dry_mask.sum())} | Wet(precip): {int(wet_mask.sum())}\n")
        f.write(f"SM regime threshold set: OPT | s_wilt={opt_sw:.3f} | s_star={opt_ss:.3f}\n")
        f.write(f"Total models: {len(all_model_cols)}\n")
        
        hdr = (f"{'Model':<60s} | RMSE | Bias |   R2 |  NSE |  KGE |    r |    a |    b")
        sep = "-" * 115

        def wb(f, label, mask, models):
            if mask is not None:
                o_data = output["LE_Obs"][mask]
                p_data = output.loc[mask, models]
                n = int(mask.sum())
            else:
                o_data = output["LE_Obs"]
                p_data = output[models]
                n = len(output)

            f.write(f"\n=== {label} (n={n}) ===\n{hdr}\n{sep}\n")
            if n < 5:
                f.write("  [NOT ENOUGH DATA]\n")
                return

            for m in models:
                s = calculate_stats(o_data, p_data[m])
                f.write(f"{m.replace('LE_',''):<60s} | {s['RMSE']:4.1f} | {s['Bias']:4.1f}"
                        f" | {s['R2']:4.2f} | {s['NSE']:4.2f} | {s['KGE']:4.2f}"
                        f" | {s['r']:.2f} | {s['alpha']:.2f}"
                        f" | {s['beta']:.2f}\n")

        wb(f, "ALL DAYS", None, all_model_cols)
        wb(f, "DRY DAYS (precip)", dry_mask, all_model_cols)
        wb(f, "WET DAYS (precip)", wet_mask, all_model_cols)
        wb(f, "VERY DRY (SM regime)", sm_vdry_mask, all_model_cols)
        wb(f, "TRANSITION (SM regime)", sm_trans_mask, all_model_cols)
        wb(f, "WET (SM regime)", sm_wet_mask, all_model_cols)

    print(f"  [OK] Saved results ({len(all_model_cols)} variants), manifest, and metrics for {site_suffix}.")

def process_site(site_name):
    print(f"=== BATCH PROCESSING: {site_name} (v13.3.1) ===")
    df_bf = pd.read_csv(OWUS_BF_CSV) if os.path.exists(OWUS_BF_CSV) else pd.DataFrame()
    df_opt = pd.read_csv(OWUS_OPT_CSV) if os.path.exists(OWUS_OPT_CSV) else pd.DataFrame()
    meta_all = pd.read_csv(UNIFIED_CSV, low_memory=False)
    meta_all.columns = [c.strip().lower() for c in meta_all.columns]
    meta_all["site_norm"] = meta_all["site"].apply(norm_site_string)
    meta_all["site_base"] = meta_all["site_norm"].apply(strip_trailing_digits)

    df_sci_meta = pd.read_csv(META_CSV)
    df_kg = pd.read_csv(KG_CSV)
    df_sci_meta = pd.merge(df_sci_meta, df_kg[["original_site", "KG_Code", "KG_Label"]], on="original_site", how="left")
    df_sci_meta["site_match"] = df_sci_meta["original_site"].apply(norm_site_string)

    # Load VOD Data
    df_vod_c = pd.DataFrame()
    if os.path.exists(VOD_CXKU_CSV):
        df_vod_c = pd.read_csv(VOD_CXKU_CSV)
        if 'date' in df_vod_c.columns:
            df_vod_c['date'] = pd.to_datetime(df_vod_c['date'], dayfirst=True, errors='coerce')
        if 'site_name' in df_vod_c.columns:
            df_vod_c['site_match'] = df_vod_c['site_name'].apply(norm_site_string)
        elif 'site_id' in df_vod_c.columns:
            df_vod_c['site_match'] = df_vod_c['site_id'].apply(norm_site_string)

    df_vod_l = pd.DataFrame()
    if os.path.exists(VOD_L_CSV):
        df_vod_l = pd.read_csv(VOD_L_CSV)
        if 'date' in df_vod_l.columns:
            df_vod_l['date'] = pd.to_datetime(df_vod_l['date'], dayfirst=True, errors='coerce')
        if 'site_name' in df_vod_l.columns:
            df_vod_l['site_match'] = df_vod_l['site_name'].apply(norm_site_string)
        elif 'site_id' in df_vod_l.columns:
            df_vod_l['site_match'] = df_vod_l['site_id'].apply(norm_site_string)

    for nc in find_nc_files(site_name, L6_DATA_DIR):
        process_single_nc(site_name, nc, df_bf, df_opt, meta_all, df_sci_meta, df_vod_c, df_vod_l)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ptjpl_owus_single_pet_v13.3.1_run_site.py <site_name>")
        sys.exit(1)
    process_site(sys.argv[1])
