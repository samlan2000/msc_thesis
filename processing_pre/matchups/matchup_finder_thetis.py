import os
import pandas as pd
import xarray as xr
import numpy as np
import openeo
# Import from sencast
from dias_apis.coah import coah

# -----------------------
# Adjust paths
# -----------------------
out_basepath = r"C:\MSc_thesis_samuel\processing_pre\matchups"
L2_folder_path = r"C:\MSc_thesis_data\insitu\thetis"

# Lake Geneva (water only)
aoi_geojson = {
  "type": "Polygon",
  "coordinates": [
    [
      [
        6.155063,
        46.207208
      ],
      [
        6.173621,
        46.217187
      ],
      [
        6.18943,
        46.238562
      ],
      [
        6.188742,
        46.256606
      ],
      [
        6.201114,
        46.268949
      ],
      [
        6.213486,
        46.268949
      ],
      [
        6.24098,
        46.307857
      ],
      [
        6.249228,
        46.308806
      ],
      [
        6.262287,
        46.337257
      ],
      [
        6.290375,
        46.363325
      ],
      [
        6.326117,
        46.375644
      ],
      [
        6.350861,
        46.372801
      ],
      [
        6.363921,
        46.351952
      ],
      [
        6.387977,
        46.34342
      ],
      [
        6.398975,
        46.352899
      ],
      [
        6.414784,
        46.363799
      ],
      [
        6.432654,
        46.363325
      ],
      [
        6.476644,
        46.377065
      ],
      [
        6.478706,
        46.394116
      ],
      [
        6.513719,
        46.411636
      ],
      [
        6.541212,
        46.400746
      ],
      [
        6.651646,
        46.409742
      ],
      [
        6.742375,
        46.409742
      ],
      [
        6.79269,
        46.396768
      ],
      [
        6.854551,
        46.392032
      ],
      [
        6.874388,
        46.398236
      ],
      [
        6.899132,
        46.40013
      ],
      [
        6.918034,
        46.398236
      ],
      [
        6.923533,
        46.415754
      ],
      [
        6.898789,
        46.43516
      ],
      [
        6.869577,
        46.44486
      ],
      [
        6.854112,
        46.444623
      ],
      [
        6.850331,
        46.454085
      ],
      [
        6.832804,
        46.456923
      ],
      [
        6.830766,
        46.465437
      ],
      [
        6.793994,
        46.470876
      ],
      [
        6.76499,
        46.473619
      ],
      [
        6.740246,
        46.487331
      ],
      [
        6.718251,
        46.481657
      ],
      [
        6.670825,
        46.501039
      ],
      [
        6.632334,
        46.501512
      ],
      [
        6.59728,
        46.509074
      ],
      [
        6.58422,
        46.514272
      ],
      [
        6.564382,
        46.498959
      ],
      [
        6.512145,
        46.506522
      ],
      [
        6.479152,
        46.484778
      ],
      [
        6.444786,
        46.464917
      ],
      [
        6.427044,
        46.460187
      ],
      [
        6.403674,
        46.450726
      ],
      [
        6.37893,
        46.459241
      ],
      [
        6.348687,
        46.453565
      ],
      [
        6.300668,
        46.419494
      ],
      [
        6.282914,
        46.387296
      ],
      [
        6.259545,
        46.384454
      ],
      [
        6.227927,
        46.359818
      ],
      [
        6.21693,
        46.34086
      ],
      [
        6.195602,
        46.311177
      ],
      [
        6.174294,
        46.295049
      ],
      [
        6.175669,
        46.273221
      ],
      [
        6.166734,
        46.258505
      ],
      [
        6.158486,
        46.251859
      ],
      [
        6.152987,
        46.228588
      ],
      [
        6.160548,
        46.217187
      ],
      [
        6.155063,
        46.207208
      ]
    ]
  ]
}



def get_s3_cloud_fraction_openeo(sensing_time, con):

    cube = con.load_collection(
        "SENTINEL3_OLCI_L1B",
        temporal_extent=[sensing_time, sensing_time],
        spatial_extent=aoi_geojson,
        bands=["B17"]
    )

    # ---- Cloud mask (boolean cube)
    # Clouds are bright in NIR over water
    cloud_mask = cube > 0.03

    # ---- Zonal mean = cloud fraction
    zonal = cloud_mask.aggregate_spatial(
        geometries=aoi_geojson,
        reducer="mean"
    )

    # ---- Execute
    result = zonal.execute()

    # Single polygon → single scalar
    cloud_fraction = list(result.values())[0][0][0]

    return float(cloud_fraction)


def get_Sentinel_matchups(OUT_BASEPATH, L2_FOLDER_PATH):

    # Authentication for cdse
    con = openeo.connect("https://openeo.dataspace.copernicus.eu")
    con.authenticate_oidc()

    # -----------------------
    # Helper to get matchups with L1B products
    # -----------------------
    def process_file(ds):

        nonlocal df

        product_list = []
        thetis_results = {}
        cloud_results = {}

        unique_dates = np.unique(ds.time.dt.date.values)
        for date in unique_dates:
            print("")
            print(f"Processing date {date}...")
            date_str = str(date)
            ds_daily = ds.sel(time=date_str, drop=True)
            ds_daily_surface = ds_daily.where(ds_daily.depth <= 7.5, drop=True)
            avg_chla = np.nanmean(ds_daily_surface["chla"].values)
            std_chla = np.nanstd(ds_daily_surface["chla"].values)
            start_date = f"{date}T00:00:00"
            end_date = f"{date}T23:59:59"

            products = None
            products = coah.get_download_requests(auth, start_date, end_date, sensor, resolution, wkt, env)
            if not products:
                continue
            [product_list.append(product) for product in products]

            for product in products:
                thetis_results[product["uuid"]] = [date_str, avg_chla, std_chla]
                cloud_results[product["uuid"]] = get_s3_cloud_fraction_openeo(product["sensing_start"], con)
    
        # No matchup
        if len(product_list) == 0:
            return
    
        # -----------------------
        # Append products to dataframe
        # -----------------------
        print("")
        for product in product_list:
            cloud_fraction = cloud_results.get(product["uuid"], np.nan)
            print(f"Cloud fraction for {product['name']}: {cloud_fraction}")
            date_str = thetis_results.get(product["uuid"], np.nan)[0]
            avg_chla = thetis_results.get(product["uuid"], np.nan)[1]
            std_chla = thetis_results.get(product["uuid"], np.nan)[2]
    
            row = {
                "station": "LXP",
                "date": date_str,
                "L2_file": file,
                "lat": lat,
                "lon": lon,
                "product_name": product["name"],
                "product_uuid": product["uuid"],
                "product_satellite": product["satellite"],
                "product_sensor": sensor,
                "avg_surface_chla": avg_chla,
                "std_surface_chla": std_chla,
                "CV_surface_chla": std_chla / avg_chla,
                "cloud_fraction": cloud_fraction,
                "nan_mask": 1 if avg_chla < 0 else 0
            }
    
            df.loc[len(df)] = row
        

    # Provide a log file path
    env = {"General": {"log": "search.log"}}
    auth = None
    sensor = "OLCI"
    resolution = 300

    # LXP coords
    coords = "46.500283, 6.660889"
    lat, lon = map(float, coords.split(","))  # split the string into numbers
    wkt = f"POINT ({lon} {lat})"

    df = pd.DataFrame()
    df["product_name"] = np.nan
    df["product_uuid"] = np.nan
    df["product_satellite"] = np.nan
    df["product_sensor"] = np.nan
    df["L2_file"] = np.nan
    df["avg_surface_chla"] = np.nan
    df["std_surface_chla"] = np.nan
    df["CV_surface_chla"] = np.nan
    df["station"] = np.nan
    df["date"] = np.nan
    df["lat"] = np.nan
    df["lon"] = np.nan
    df["cloud_fraction"] = np.nan
    df["nan_mask"] = np.nan
    
    count = 0
    tot = len(os.listdir(L2_FOLDER_PATH))
    
    for file in os.listdir(L2_FOLDER_PATH):
        print("="*80)
        count += 1
        print(f"Processing file {file}... [file {count} out of {tot}]")
        if ".nc" not in file:
            continue
        print("")
        path = os.path.join(L2_FOLDER_PATH, file)
        ds = xr.open_dataset(path)
        if "chla" not in ds.variables:
            print("Skipped file since it contains no chla data.")
            continue
        process_file(ds)
        df.to_csv(os.path.join(OUT_BASEPATH, "thetis_matchups_temp.csv"), index=False, sep=";", encoding="utf-8-sig")
        print("="*80)
        print("")
        print("")

    df.to_csv(os.path.join(OUT_BASEPATH, "thetis_matchups.csv"), index=False, sep=";", encoding="utf-8-sig")
          
get_Sentinel_matchups(out_basepath, L2_folder_path)    