import pandas as pd
import configparser
import os

def clone_and_update_ini(input_file, output_file, start, end):
    config = configparser.ConfigParser()
    config.read(input_file)

    # Edit only start/end inside [General]
    config["General"]["start"] = start
    config["General"]["end"] = end

    # Write to a new ini file
    with open(output_file, "w") as f:
        config.write(f)

    print(f"Saved modified INI → {output_file}")

### EXAMPLE FOR THREE CAMPAIGN IMAGES
list_uniquedates=["2021.05.20", "2021.06.11", "2021.06.16"]

### We can easily do the same using a column of dates, e.g. for thetis:
# path_thetis = "C:\MSc_thesis_samuel\processing_pre\matchups\matchups_thetis.csv"
# list_uniquedates = df.drop_duplicates(subset="date")["date"].tolist()

input_ini  = r"C:\MSc_thesis_samuel\processing_pre\sencast_parameter_files\parameters_thetis\S3_download_atcor_20181018.ini"
directory = r"C:\MSc_thesis_samuel\processing_pre\sencast_parameter_files\parameters_campaigns"

for date in list_uniquedates:
    Y,m,d = date.split(".")
    start_time = f"{Y}-{m}-{d}T00:00:00.000Z"
    end_time = f"{Y}-{m}-{d}T23:59:59.999Z"
    output_ini = os.path.join(directory, f"S3_download_atcor_{Y}{m}{d}.ini")
    clone_and_update_ini(input_ini, output_ini, start_time, end_time)