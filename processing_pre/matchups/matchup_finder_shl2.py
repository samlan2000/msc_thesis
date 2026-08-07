import os
import pandas as pd
import numpy as np
import xarray as xr
# Import from sencast
from dias_apis.coah import coah

# -----------------------
# Adjust paths
# -----------------------
out_basepath = r"C:\MA_data\insitu\chla\SHL2"
l2_folder = r"C:\MA_data\insitu\chla\SHL2\data_partners\L2"

def mean_chla_at_depth(ds, depth_value):
    if depth_value in ds.depth.values:
        return round(ds["chlorophyll_a"].sel(depth=depth_value).mean().item(), 2)
    else:
        return np.nan

def get_Sentinel_matchups(OUT_BASEPATH, l2_folder):
        
    # Provide a log file path
    env = {"General": {"log": "search.log"}}
    auth = None
    sensor = "OLCI"
    resolution = 300

    # SHL2 coords
    coords = "46.453457, 6.5942335"
    lat, lon = map(float, coords.split(","))  # split the string into numbers
    wkt = f"POINT ({lon} {lat})"

    df = pd.DataFrame()
    df["insitu_date"] = np.nan
    df["product_sensing_start"] = np.nan
    df["product_name"] = np.nan
    df["product_uuid"] = np.nan
    df["product_satellite"] = np.nan
    df["product_sensor"] = np.nan
    df["L2_file"] = np.nan
    df["station"] = np.nan
    df["lat"] = np.nan
    df["lon"] = np.nan
    df["chla_0"] = np.nan
    df["chla_1"] = np.nan
    df["chla_2p5"] = np.nan
    df["chla_3p5"] = np.nan
    df["chla_5"] = np.nan
    df["chla_7p5"] = np.nan
    df["chla_10"] = np.nan
    df["chla_15"] = np.nan
    df["chla_20"] = np.nan
    df["chla_30"] = np.nan
    
    for file in os.listdir(l2_folder):
        
        year = file[-11:-7]
        if "Geneva" not in file or "chlorophyll_a" not in file or int(year) < 2016:
            continue
        
        path = os.path.join(l2_folder, file)
        ds = xr.open_dataset(path)
        
        # Only select high quality data (spectrophotometry)
        if "methodology" not in ds.attrs:
            sub_ds = ds.sel(methodology="Spectrophotometry ISO10260:1992")
        elif ds.attrs["methodology"] != "Spectrophotometry ISO10260:1992":
            continue
        else:
            sub_ds = ds.copy()
            
        chla_0   = mean_chla_at_depth(sub_ds, 0)
        chla_1   = mean_chla_at_depth(sub_ds, 1)
        chla_2p5   = mean_chla_at_depth(sub_ds, 2.5)
        chla_3p5 = mean_chla_at_depth(sub_ds, 3.5)
        chla_5   = mean_chla_at_depth(sub_ds, 5)
        chla_7p5 = mean_chla_at_depth(sub_ds, 7.5)
        chla_10  = mean_chla_at_depth(sub_ds, 10)
        chla_15  = mean_chla_at_depth(sub_ds, 15)
        chla_20  = mean_chla_at_depth(sub_ds, 20)
        chla_30  = mean_chla_at_depth(sub_ds, 30)
        
        date = ds.date
        d,m,Y = date.split(" ")[-1].split("/")
        date_str = f"{Y}-{m}-{d}"
        start_date_str = date_str + "T00:00:00"
        end_date_str = date_str + "T23:59:59"
        
        print("")
        print(f"Search date: {date_str}")
        products = coah.get_download_requests(auth, start_date_str, end_date_str, sensor, resolution, wkt, env)
        for product in products:
            print(f"Matched product {product['name']}")
            row = {
                "insitu_date": date_str,
                "product_sensing_start": product["sensing_start"],
                "product_name": product["name"],
                "product_uuid": product["uuid"],
                "product_satellite": product["satellite"],
                "product_sensor": sensor,
                "L2_file": file,
                "station": "SHL2",
                "lat": lat,
                "lon": lon,
                "chla_0": chla_0,
                "chla_1": chla_1,
                "chla_2p5": chla_2p5,
                "chla_3p5": chla_3p5,
                "chla_5": chla_5,
                "chla_7p5": chla_7p5,
                "chla_10": chla_10,
                "chla_15": chla_15,
                "chla_20": chla_20,
                "chla_30": chla_30
            }
            df.loc[len(df)] = row
            df.to_csv(os.path.join(OUT_BASEPATH, "temp", "matchups_shl2_v3.csv"), sep=";", encoding="utf-8-sig", index=False)
        print("")
            
    df.to_csv(os.path.join(OUT_BASEPATH, "matchups_shl2_v3_full.csv"), sep=";", encoding="utf-8-sig", index=False)

          
get_Sentinel_matchups(out_basepath, l2_folder)    