# Parallelised: Combined Script to run Sensitivity Analysis for Tracks, DU, and Drop Distances (Raw Data to Pareto Values)
# ******* Ensure the kernel used to run the code has the below libraries installed as well as Grandin et al's (2024) flux calculator tool pre-requisite libraries installed too. 
# The required installation for Grandin's tool can be found here https://git.icare.univ-lille.fr/icare-public/so2-flux-calculator/-/blob/main/README.md
# Ensure the ECMWF CDS API within the tool is set up as well *******

import os, json, numpy as np, pandas as pd
from netCDF4 import Dataset
from glob import glob
from collections import defaultdict
from datetime import datetime
from scipy.spatial import ConvexHull, cKDTree
from scipy.interpolate import griddata
from matplotlib.path import Path
import subprocess
import itertools
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
import concurrent.futures

##**********THINGS TO CHANGE:********
# Modify the download and output directories in the run_phase1 + run_phase2 functions and optimisation loop, the final sensitivity analysis output file,
# and the config dates within run_phase1
# Ensure there is an appropriate wind file as well if using manual data
# ==========================================
# 1. DEFINING SEARCH SPACE
# ==========================================
TRACKS_GRID = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
DU_GRID = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

# Format these exactly as the CLI expects them (e.g., comma-separated strings without spaces)
# An empty string "" means no dropped distances.
# The list of valid distances to drop are{25,50,75,100,125,150,175,200,250,300,350,400,500,1000}
DROP_DISTANCES_GRID = ["", "500", "400 500", "300 400 500", "250 300 400 500"] 

# ==========================================
# 2. ENCAPSULATING SCRIPTS INTO FUNCTIONS
# ==========================================

# Phase 1 conducts the pre-processing and mass integration step to obtain integrated SO2 Mass Files and Cloud Fraction Files
def run_phase1_extraction(tracks_val, du_val):
    # --- CONFIG & PATHS ---
    # ---------------------------------------------------------------- INSERT PATH FOR RAW SATELLITE DOWNLOAD & OUTPUT FOLDERS HERE --------------------------------------------------------------
    DOWNLOAD_DIR = r"XXXX"
    OUTPUT_DIR = r"XXXX" # Best to create a folder for output since each parameter combination outputs one SO2 mass file and one cloud fraction file
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialising Config dictionary
    # Remember to change the dates to the correct ranges of interest! Other values can be left as is as the loop overrides them
    CONFIG = {
        "volcano": "Merapi",
        "latitude": -7.54, "longitude": 110.446,
        "radii": [500, 400, 300, 250, 200, 150, 100, 75, 50, 25],
        "filters": {"tracks": 5, "du": 0, "qa": 0, "sza": 70},
        "start": "2023-06-01", "end": "2023-12-31",
        "grid_res": 0.05, "max_gap_deg": 1.0 
    }

    # Modify CONFIG dynamically based on the input arguments
    CONFIG['filters']['tracks'] = tracks_val
    CONFIG['filters']['du'] = du_val

    # Grandin's exact Kappa for assumed Area A = 25km2
    KAPPA = 0.0007 

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        p1, p2, dp, dl = np.radians(lat1), np.radians(lat2), np.radians(lat2-lat1), np.radians(lon2-lon1)
        a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
        return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    def get_safe_overhead_time(p_group, t_lat, t_lon, fname):
        # Extracts the start time directly from the TROPOMI filename.
        try:
            # fname format: S5P_OFFL_L2__SO2____20230601T045051_...
            # Splits at '____' and grabs the first 15 characters: '20230601T045051'
            dt_str = fname.split('____')[1][:15]
            
            # Convert to a datetime object, then format as desired
            dt = datetime.strptime(dt_str, "%Y%m%dT%H%M%S")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
            
        except Exception as e:
            print(f"Warning: Could not parse time from filename {fname}. Error: {e}")
            return "Unknown Time"

    def extract_nc_combined(path, t_lat, t_lon):
        with Dataset(path, 'r') as root:
            p = root.groups['PRODUCT']
            res = p.groups['SUPPORT_DATA'].groups['DETAILED_RESULTS']
            input_data = p.groups['SUPPORT_DATA'].groups['INPUT_DATA']
            geo = p.groups['SUPPORT_DATA'].groups['GEOLOCATIONS']
            
            v_so2_raw = res.variables['sulfurdioxide_total_vertical_column_7km']
            conv = v_so2_raw.getncattr('multiplication_factor_to_convert_to_DU')
            cf_raw = input_data.variables['cloud_fraction_crb']
            
            lat_full, lon_full = p.variables['latitude'][:], p.variables['longitude'][:]
            crop = (lat_full > t_lat - 6) & (lat_full < t_lat + 6) & (lon_full > t_lon - 6) & (lon_full < t_lon + 6)
            if not np.any(crop): return None, None
            
            so2 = np.ma.filled(v_so2_raw[:].astype(float), np.nan)[crop] * conv
            cf = np.ma.filled(cf_raw[:].astype(float), np.nan)[crop]
            
            df = pd.DataFrame({
                'lat': lat_full[crop], 'lon': lon_full[crop],
                'qa': p.variables['qa_value'][:][crop],
                'sza': geo.variables['solar_zenith_angle'][:][crop],
                'row': np.tile(np.arange(p.dimensions['ground_pixel'].size), p.dimensions['scanline'].size).reshape(lat_full.shape)[crop],
                'so2': so2, 'cf': cf
            })
            return df, get_safe_overhead_time(p, t_lat, t_lon, os.path.basename(path))

    # --- EXECUTION ---
    files = glob(os.path.join(DOWNLOAD_DIR, "*.nc"))
    start_dt, end_dt = datetime.strptime(CONFIG['start'], "%Y-%m-%d"), datetime.strptime(CONFIG['end'], "%Y-%m-%d")
    groups = defaultdict(list)
    for f in files:
        d_str = os.path.basename(f).split('____')[1][:8]
        if start_dt <= datetime.strptime(d_str, "%Y%m%d") <= end_dt:
            groups[d_str].append(f)

    so2_results, cf_results = [], []

    for date_key in sorted(groups.keys()):
        paths = groups[date_key]
        paths.sort(key=lambda x: int(os.path.basename(x).split('_')[12]))
        
        daily_data, final_time = [], ""
        for p_path in paths:
            df, t = extract_nc_combined(p_path, CONFIG['latitude'], CONFIG['longitude'])
            if df is not None:
                daily_data.append(df); final_time = t

        if not daily_data: continue
        master = pd.concat(daily_data, ignore_index=True)
        master['dist'] = haversine(CONFIG['latitude'], CONFIG['longitude'], master['lat'], master['lon'])
        
        m_e = (master['row'] < CONFIG['filters']['tracks']) | (master['row'] > (449 - CONFIG['filters']['tracks']))
        m_q, m_s = (master['qa'] < CONFIG['filters']['qa']), (master['sza'] > CONFIG['filters']['sza'])
        
        soundings = master[~(m_e | m_q | m_s | master['so2'].isna())].copy()
        if len(soundings) < 4: continue

        lons = np.arange(CONFIG['longitude']-5.5, CONFIG['longitude']+5.5, CONFIG['grid_res'])
        lats = np.arange(CONFIG['latitude']-5.5, CONFIG['latitude']+5.5, CONFIG['grid_res'])
        grid_lon, grid_lat = np.meshgrid(lons, lats)
        grid_pts = np.column_stack([grid_lon.ravel(), grid_lat.ravel()])
        
        pts = soundings[['lon', 'lat']].values
        hull_path = Path(pts[ConvexHull(pts).vertices])
        
        grid_so2 = griddata(pts, soundings['so2'].values, grid_pts, method='linear')
        grid_cf = griddata(pts, soundings['cf'].values, grid_pts, method='linear')
        dists, _ = cKDTree(pts).query(grid_pts)
        
        mask = hull_path.contains_points(grid_pts) & (dists <= CONFIG['max_gap_deg'])
        
        # Creating a mask to filter out interpolated values beyond the max gap degree
        mask = hull_path.contains_points(grid_pts) & (dists <= CONFIG['max_gap_deg'])
        
        # Apply the spatial mask (region outside Convex Hull to exactly 0.0)
        spatially_masked_so2 = np.where(mask, np.nan_to_num(grid_so2, nan=0.0), 0.0)
        # Apply the DU filter (keep the value if it's >= the DU threshold, otherwise force to 0.0)
        grid_so2_final = np.where(spatially_masked_so2 >= CONFIG['filters']['du'], spatially_masked_so2, 0.0)
        
        grid_cf_final = np.where(mask, np.nan_to_num(grid_cf, nan=0.0), np.nan)
        
        g_df = pd.DataFrame({'lon': grid_pts[:,0], 'lat': grid_pts[:,1], 'so2': grid_so2_final, 'cf': grid_cf_final})
        g_df['dist'] = haversine(CONFIG['latitude'], CONFIG['longitude'], g_df['lat'], g_df['lon'])

        for r in CONFIG['radii']:
            # For tracking the original raw metrics
            r_orig = master[master['dist'] <= r]
            if r_orig.empty: continue
            
            f_e = (r_orig['row'] < CONFIG['filters']['tracks']) | (r_orig['row'] > (449 - CONFIG['filters']['tracks']))
            f_q, f_s = r_orig['qa'] < CONFIG['filters']['qa'], r_orig['sza'] > CONFIG['filters']['sza']

            # Set up the mathematical grid for this radius
            r_grid = g_df[g_df['dist'] <= r]
            total_grid_points = len(r_grid) # e.g., 25,619 for 500km
            
            # Strict Coverage Check using fully FILTERED raw data
            if total_grid_points > 0:
                # Isolate the surviving, high-quality satellite pixels for this radius
                r_soundings = soundings[soundings['dist'] <= r]
                
                if r_soundings.empty:
                    proportion_missing = 100.0 # 100% missing if no valid data survived
                else:
                    # 1. Build a strict KD-Tree using ONLY the valid, filtered raw pixels
                    valid_tree = cKDTree(r_soundings[['lon', 'lat']].values)
                    grid_coords = r_grid[['lon', 'lat']].values
                    
                    # 2. Measure distance from every mathematical grid square to the nearest valid pixel
                    dists_to_valid, _ = valid_tree.query(grid_coords)
                    
                    # 3. A grid point is valid ONLY if a valid pixel is within 0.05 degrees
                    valid_grid_points = (dists_to_valid <= CONFIG['grid_res']).sum()
                    
                    proportion_missing = (total_grid_points - valid_grid_points) / total_grid_points * 100
            else:
                proportion_missing = np.nan

            # Extract the INTERPOLATED grid metrics for mass calculation 
            so2_sum = (r_grid['so2'] * KAPPA).sum()
            npoints = (r_grid['so2'] > 0).sum()
            
            if not r_grid.empty and r_grid['so2'].sum() > 0:
                c_lat = np.average(r_grid['lat'], weights=r_grid['so2'])
                c_lon = np.average(r_grid['lon'], weights=r_grid['so2'])
                cf_mean = r_grid['cf'].dropna().mean()
            else:
                c_lat, c_lon, cf_mean = CONFIG['latitude'], CONFIG['longitude'], r_grid['cf'].dropna().mean() if not r_grid['cf'].dropna().empty else 0.0

            # --- Compile Metrics ---
            other_metrics = {
                "radius": r, 
                "npoints": npoints, 
                "proportion_missing": proportion_missing, 
                "centroid_lon": c_lon, 
                "centroid_lat": c_lat,
                "fraction_filtered_edge_tracks": f_e.mean()*100,
                "fraction_filtered_quality": f_q.mean()*100,
                "fraction_filtered_sza": f_s.mean()*100,
                "fraction_filtered_total": (f_e | f_q | f_s).mean()*100
            }
            
            so2_results.append({"Time": final_time, "TROPOMI_SO2-7km_mass": so2_sum, **other_metrics})
            cf_results.append({"Time": final_time, "TROPOMI_CloudFraction_mean": cf_mean, **other_metrics})

        print(f"Processed: {date_key}")
    # --- DYNAMIC FILENAMES ---
    # Construct radii string: "25-50-75-100-150-200-250-300-400-500km" for file naming
    rad_str = "-".join(map(str, sorted(CONFIG['radii']))) + "km"

    # SO2 filename
    fn_so2 = (f"TROPOMI_SO2-7km_mass_{rad_str}_{CONFIG['volcano']}_"
            f"tracks-{CONFIG['filters']['tracks']}_du-{CONFIG['filters']['du']}_"
            f"qa-{CONFIG['filters']['qa']}_sza-{CONFIG['filters']['sza']}_"
            f"{CONFIG['start']}_{CONFIG['end']}.csv")

    # Cloud Fraction filename 
    fn_cf = (f"TROPOMI_CloudFraction_mean_{rad_str}_{CONFIG['volcano']}_"
            f"tracks-{CONFIG['filters']['tracks']}_"
            f"qa-{CONFIG['filters']['qa']}_sza-{CONFIG['filters']['sza']}_"
            f"{CONFIG['start']}_{CONFIG['end']}.csv")

    # --- SAVE WITH METADATA ---
    # 1. Mass Report
    with open(os.path.join(OUTPUT_DIR, fn_so2), 'w', newline='') as f:
        so2_meta = CONFIG.copy()
        so2_meta = {"variable": "TROPOMI_SO2-7km_mass", **so2_meta} # Ensure variable is first
        f.write(f"# Computation info: {json.dumps(so2_meta)}\n")
        pd.DataFrame(so2_results).to_csv(f, index=False, lineterminator='\n')

    # 2. Cloud Report
    with open(os.path.join(OUTPUT_DIR, fn_cf), 'w', newline='') as f:
        cf_meta = CONFIG.copy()
        cf_meta = {"variable": "TROPOMI_CloudFraction_mean", **cf_meta} # Ensure variable is first
        f.write(f"# Computation info: {json.dumps(cf_meta)}\n")
        pd.DataFrame(cf_results).to_csv(f, index=False, lineterminator='\n')

    print(f"Done! Reports saved to {OUTPUT_DIR}")
        
    # Return the generated filenames so Phase 2 can find them
    return fn_so2, fn_cf

# Phase 2 conducts the inversion step using the API tool 
def run_phase2_inversion(so2_csv, cf_csv, drop_dists_val):
    # ----------------------------------------------------------------- INSERT PATH TO SAME OUTPUT FOLDER FROM PHASE 1 HERE -------------------------------------------------------------
    output_dir = r"XXXX"
    os.makedirs(output_dir, exist_ok=True)
    # ----------------------------------------------------------------------- INSERT APPROPRIATE PRESSURE VALUE HERE  -------------------------------------------------------------------
    # Set appropriate pressure value (in hPa)based on the assumed plume altitude 
    # Possible pressure_vals accepted by Grandin's Tool: {1,2,3,5,7,10,20,30,50,70,100,125,150,175,200,225,250,300,350,400,450,500,550,600,650,700,750,775,800,825,850,875,900,925,950,975,1000}
    pressure_val = "550"

    # Extract the settings label (tracks_du_qa_sza) from the SO2 filename
    filename = os.path.basename(so2_csv)
    parts = filename.split("_")
    date_info = "_".join(parts[-2:]).replace(".csv", "")
    
    tracks = [p for p in parts if 'tracks-' in p][0]
    qa = [p for p in parts if 'qa-' in p][0]
    sza = [p for p in parts if 'sza-' in p][0]
    du = [p for p in parts if 'du-' in p][0]
    settings_label = f"{tracks}_{du}_{qa}_{sza}"
    
    # Create a unique output filename based on all parameters, including drop distances
    if drop_dists_val:
        # Replaces commas with spaces, splits by space, and joins with hyphens. 
        safe_drops = "-".join(drop_dists_val.replace(',', ' ').split())
    else:
        safe_drops = "none"
    
    output_filename = f"Merapi_Flux_7km_{date_info}_{settings_label}_drops-{safe_drops}_P{pressure_val}"

    # Runs the command in the format required by Grandin et al.'s flux calculator format
    command = [
        "so2_flux_calculator",
        so2_csv.replace("\\", "/"),
        cf_csv.replace("\\", "/"),
        "--method", "curvefit",
        "--pressurelevel", pressure_val,
        "--output_file", output_filename,
        "--output_directory", output_dir.replace("\\", "/")
    ]
    
    # Conditionally add the drop_distances flag if it's not empty
    if drop_dists_val:
        command.append("--drop_distances")
        # Replace commas with spaces, split into a list of strings: ['400', '500']
        dists_list = drop_dists_val.replace(',', ' ').split()
        command.extend(dists_list)
        
    print(f"\n>>> Running Inversion: {output_filename}")
    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        if result.returncode != 0:
            print(f"!!! Error !!! STDERR: {result.stderr}")
    except Exception as e:
        print(f"Subprocess failed to launch: {e}")
    
    # The calculator adds .csv automatically, so we append it for the return path
    return os.path.join(output_dir, output_filename + ".csv")

# ==========================================
# 3. THE OPTIMIZATION LOOP (PARALLELIZED)
# ==========================================

def process_parameter_combo(tracks, du, drop_distances_grid, ground_data_path):
    """
    Worker function to run extraction, inversion, and calculate filtered correlation.
    """
    results = []
    
    # Load ground truth INSIDE the worker to avoid Windows memory sharing issues
    ground_df = pd.read_csv(ground_data_path, parse_dates=['date'])
    
    print(f"[Worker Started] Tracks: {tracks} | DU: {du}")
    
    # 1. Run Phase 1 Extraction
    so2_file, cf_file = run_phase1_extraction(tracks, du)
    # ----------------------------------------------------------------- INSERT PATH TO SAME OUTPUT FOLDER FROM PHASE 1 HERE -------------------------------------------------------------
    so2_path = os.path.join(r"XXX", so2_file)
    cf_path = os.path.join(r"XXX", cf_file)
    
    # --- Extract the Coverage Proportions from Phase 1 ---
    # We read the CSV (ignoring the # metadata header) to get the missing proportions
    p1_df = pd.read_csv(so2_path, comment='#')
    p1_df['Time'] = pd.to_datetime(p1_df['Time']).dt.normalize() # Strip to pure date
    
    # Isolate the 25km and 500km coverage metrics
    cov_25 = p1_df[p1_df['radius'] == 25][['Time', 'proportion_missing']].rename(columns={'proportion_missing': 'missing_25km'})
    cov_500 = p1_df[p1_df['radius'] == 500][['Time', 'proportion_missing']].rename(columns={'proportion_missing': 'missing_500km'})
    
    # Merge them into a single coverage lookup table
    coverage_df = pd.merge(cov_25, cov_500, on='Time', how='outer')
    
    # 2. Loop through Phase 2 Inversions for this specific extraction
    for drops in drop_distances_grid:
        ts_path = run_phase2_inversion(so2_path, cf_path, drops)
        
        # Load the newly generated time series
        tropomi_df = pd.read_csv(ts_path, comment='#', parse_dates=['time'])
        tropomi_df['time'] = tropomi_df['time'].dt.tz_localize(None).dt.normalize()
        
        # Merge Phase 2 Flux with Ground Truth
        merged = pd.merge(tropomi_df, ground_df, left_on='time', right_on='date', how='inner')
        
        # Merge in the Coverage Data from Phase 1
        merged = pd.merge(merged, coverage_df, left_on='time', right_on='Time', how='left')

        # Drop rows with NaN math columns
        clean_merged = merged.dropna(subset=['observed_x2.4_ton', 'flux', 'sigma_flux']).copy()
        
        # Apply the coverage filters
        # If coverage is NaN (meaning absolutely 0 raw data survived to build a grid), treat as 100% missing (1.0)
        clean_merged['missing_25km'] = clean_merged['missing_25km'].fillna(1.0)
        clean_merged['missing_500km'] = clean_merged['missing_500km'].fillna(1.0)
        
        # The Filter: Keep days where 25km is <= 50% missing AND 500km is <= 20% missing
        filtered_df = clean_merged[
            (clean_merged['missing_25km'] <= 50.0) & 
            (clean_merged['missing_500km'] <= 20.0)
        ]
        
        # Calculate Objectives using the STRICTLY FILTERED dataframe
        if not filtered_df.empty and len(filtered_df) > 1:
            mae = mean_absolute_error(filtered_df['observed_x2.4_ton'], filtered_df['flux'])
            mean_unc = filtered_df['sigma_flux'].mean()
            corr, p_val = spearmanr(filtered_df['observed_x2.4_ton'], filtered_df['flux'])
        else:
            mae, mean_unc, corr, p_val = np.nan, np.nan, np.nan, np.nan
        
        # Store result dictionary
        results.append({
            'Tracks': tracks,
            'DU': du,
            'Drops': drops,
            'Spearman_Corr': corr,                 
            'P_Value': p_val,                      
            'MAE_Filtered': mae,
            'Mean_Uncertainty': mean_unc,
            'Raw_Days': len(clean_merged),         # Total days overlapping with ground data before filtering out unreliable days
            'Filtered_Days': len(filtered_df)      # Actual sample size (N) used for the correlation
        })
        
    print(f"[Worker Finished] Tracks: {tracks} | DU: {du}")
    return results

# ==========================================
# 4. EXECUTION AND SAVING PARETO FRONTIER
# ==========================================

if __name__ == '__main__':
    # ----------------------------------------------------------- ------ INSERT PATH TO CSV OF GROUND-MEASURED NOVAC SO2 MEASUREMENTS  ------------------------------------------------------
    ground_df_path = r"XXX.csv"
    
    # Create a list of all combinations of Tracks and DU
    all_combinations = list(itertools.product(TRACKS_GRID, DU_GRID))
    total_tasks = len(all_combinations)
    
    print(f"Starting Parallel Grid Search for {total_tasks} total parameter combinations...")
    
    final_results_log = []
    
    # Launch the parallel pool. By default, this uses all available CPU cores.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all tasks to the executor
        futures = [
            executor.submit(process_parameter_combo, t, d, DROP_DISTANCES_GRID, ground_df_path) 
            for t, d in all_combinations
        ]
        
        # Gather results as they finish
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                # Extend our master log with the list of results returned by the worker
                final_results_log.extend(future.result())
                print(f"--- Progress: {i}/{total_tasks} extractions completed ---")
            except Exception as e:
                print(f"A worker crashed with error: {e}")

    # Save everything
    results_df = pd.DataFrame(final_results_log)
    results_df.to_csv(r"XXX.csv", index=False)
    print("\nOptimization Complete! All Pareto Results saved.")
