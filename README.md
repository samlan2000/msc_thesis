# MSc_thesis_samuel — processing chain

Processing chain for the MSc thesis work on lake water-quality retrieval from
Sentinel-3 OLCI, covering three independent data sources / validation
targets:

- **Thetis** — long-term automated profiler platform on Lake Geneva.
- **Campaigns** — multi-station field campaigns (HPLC Chl-a, TSM, CDOM).
- **SHL2** — fixed station, Chl-a and phytoplankton community match-ups.

Each data source has its own orchestrator script, `main_thetis.py`,
`main_campaigns.py`, and `main_shl2.py`, at the repo root. Everything else
lives under `processing_pre/` (download, atmospheric correction, format
conversion, QA/filtering) and `processing_main/` (WASI inversion, match-up,
plotting).

## Repository structure

```
MSc_thesis_samuel/
├── main_thetis.py            orchestrator: Thetis chain (5 stages)
├── main_campaigns.py         orchestrator: campaigns chain (4 stages)
├── main_shl2.py               orchestrator: SHL2 chain (3 stages)
│
├── processing_pre/           pre-processing stage code
│   ├── run_sencast.py         driver for the download_ac (sencast/Docker) stage
│   ├── bsqConverterPolymer.py .nc -> band-restricted .bsq conversion
│   ├── valid_images_thetis.py filters the combined Thetis .bsq archive
│   ├── matchups/               in-situ/satellite matchup finder scripts + csvs
│   ├── chl_npq_correction/     NPQ correction notebook for Thetis Chl-a
│   └── parameters/             pre-generated sencast .ini files (see below)
│       ├── campaigns/, shl2/, thetis/   one .ini per acquisition date
│       ├── generate_ini.py     generates new .ini files from a date list
│       ├── INFO.txt            sencast setup + usage notes
│       └── powershell_command_example.txt
│
├── processing_main/          core inversion / matchup / plotting code
│   ├── MiniWASIsafe.py         WASI bio-optical model
│   ├── PixelProcessor.py       single-pixel (per-station) inversion
│   ├── ImageProcessor.py       whole-image (per-pixel cube) inversion
│   ├── processing_thetis.py, plotting_thetis.py, hyperspectral_rrs_inversion.py
│   ├── processing_shl2.py
│   ├── processing_campaigns.py, image_processor_campaigns.py
│   ├── resampling.py, rrs_qa.py
│   └── data/                   WASI model lookup tables / spectral libraries
│
├── LUTs/                      date -> in-situ profile file lookup tables (.pkl)
├── outputs_intermediate/      intermediate results (e.g. in-situ Rrs inversion)
└── outputs_L3/                final plots, match-up databases, processed images
```

## Requirements

- Python 3.10 with `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`,
  `statsmodels`, `xarray`, `rasterio`, `spectral`, `tqdm`.
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) — only
  needed for the `download_ac` stage (see below). All other stages run with
  plain Python and no Docker dependency.
- A local [sencast](https://sencast.readthedocs.io/en/latest/) installation
  — only needed for the `download_ac` stage.

## Running a chain

Each `main_x.py` is a standalone script split into independently toggleable
stages, controlled by `RUN_*` switches near the top of the file:

| Script              | Stages (in order)                                                              |
|---------------------|---------------------------------------------------------------------------------|
| `main_thetis.py`    | `download_ac` → `pre_processing` → `insitu_inversion` → `processing_thetis` → `plotting_thetis` |
| `main_campaigns.py` | `download_ac` → `pre_processing` → `processing_campaigns` → `image_processing` |
| `main_shl2.py`      | `download_ac` → `pre_processing` → `processing_shl2`                          |

To (re-)run only part of a chain, set the corresponding `RUN_*` flags to
`True`/`False` and run the script directly, e.g. `python main_thetis.py`.
Stages read/write their outputs to disk, so a later stage can be re-run on
its own as long as the earlier stages' outputs already exist.

Paths are split into two sections at the top of each script:

- **EXTERNAL INPUT PATHS** — absolute paths to anything *outside* this repo
  (raw satellite/in-situ data, the sencast install). Edit these to match
  your machine before running.
- **INTERNAL PATHS** — computed relative to the script's own location via
  `BASE_DIR`, so they follow the repo's folder structure automatically no
  matter where the repo itself is checked out.

### The `download_ac` stage — read this before enabling it

`download_ac` drives [sencast](https://sencast.readthedocs.io/en/latest/) in
Docker to download raw Sentinel-3 products and run atmospheric correction,
once per `.ini` file found in the chain's parameters folder
(`processing_pre/run_sencast.py`). It defaults to **off**
(`RUN_DOWNLOAD_AC = False`) in all three scripts, and should stay off unless
you specifically need to (re-)download and atmospherically correct raw
imagery.

**This stage is slow.** Downloading + atmospherically correcting the full
image archive can take on the order of **10 days** for the Thetis chain
(hundreds of acquisition dates, one Docker run each) — see
`processing_pre/parameters/INFO.txt`. The campaigns and SHL2 chains involve
far fewer dates and are correspondingly quicker, but still not instant.
Before turning it on, make sure you actually need fresh imagery rather than
re-running a later stage on data that's already been downloaded.

Pre-generated `.ini` parameter files for all three chains already ship in
this repo, under `processing_pre/parameters/{campaigns,shl2,thetis}/`. Per
`processing_pre/parameters/INFO.txt`, sencast expects the `parameters`
folder to live inside your local sencast install; `SENCAST_DIR`,
`*_DIAS_TEMP_DIR`, and `*_SENCAST_PARAMS_DIR` near the top of each
`main_x.py` control where sencast, its scratch/download space, and its
parameters folder are found — update these to match your machine (and,
if needed, point `*_SENCAST_PARAMS_DIR` at this repo's
`processing_pre/parameters/<chain>/` folder directly rather than a copy).
`processing_pre/parameters/generate_ini.py` shows how to generate new `.ini`
files for additional acquisition dates.

## Data availability

Raw satellite and in-situ input data are not included in this repository —
they're too large to version, and referenced by absolute paths under
`C:\MSc_thesis_data\...` that need to be adapted per machine (see the
EXTERNAL INPUT PATHS section of each `main_x.py`). Availability of the
in-situ input data differs by chain:

- **Thetis** — the processed Thetis in-situ data used by this repo
  (`insitu/thetis_L2`: `Level2/`, `Level2_orig/`, and
  `df_thetis_chla_cor.csv`) is openly available on Zenodo:
  [Processed Thetis vertical-profiler data (Lake Geneva, 2018–2025) used for
  Sentinel-3 OLCI water-quality validation](https://doi.org/10.5281/zenodo.22203433).
- **SHL2** — the SHL2 in-situ data cannot be redistributed here under the
  data provider's user agreement. It must be requested directly from
  [SOERE OLA (INRAE)](https://si-ola.inrae.fr/).
- **Campaigns** — publishing of the Eawag field campaign in-situ data is
  underway; this README will be updated with a DOI/link once available.

## Outputs

- `outputs_intermediate/` — intermediate stage results (currently: in-situ
  hyperspectral Rrs → WASI inversion results for Thetis).
- `outputs_L3/` — final outputs: match-up plots per chain
  (`plots_thetis/`, `plots_campaigns/`, `plots_shl2/`), the Thetis match-up
  database (`db_thetis.pkl`), and processed campaign image cubes +
  quicklooks (`images_campaigns/`).

Large generated binary outputs (e.g. `.bsq`/`.img`/`.hdr` image cubes under
`outputs_L3/images_campaigns/`) are excluded from git via `.gitignore` and
are regenerated by re-running the relevant `main_x.py` stage.
The same applies to all output images. Currently, `main_x.py` processing 
includes only the relevant reference pixels. The full images can be reprocessed 
using the `image_processor_all.py` script.
