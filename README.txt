================================================================================
IW-FusionFM: Ocean Internal Wave Detection from Sentinel-1 SAR and SWOT
================================================================================

Anonymous release by the IW-FusionFM Team (prepared for double-blind review).

IW-FusionFM is a research pipeline for detecting oceanic internal solitary
waves (ISWs) in Sentinel-1 IW GRD SAR imagery, with SWOT KaRIn sea-surface
height as an auxiliary modality, plus physics-based amplitude inversion and
a curated event database for the northern South China Sea.

Pipeline overview
-----------------
1. SAR segmentation: Swin-UNet family models (with optional FiLM-based SWOT
   feature fusion) detect ISW bright/dark stripe signatures in Sentinel-1
   tiles.
2. Two-stage verification: a context verifier cascade re-scores candidate
   patches from the segmenter to suppress false alarms (rain cells, wind
   streaks, speckle), with scene-level leakage-safe evaluation splits.
3. Visibility correction: an empirical soft visibility function v(w) of the
   ERA5 10 m wind speed is fitted by maximum likelihood, so raw occurrence
   rates can be corrected for the wind-dependent imaging window
   (evaluation/visibility_correction.py).
4. KdV/eKdV inversion: event geometry (wavelength, direction, location) is
   combined with SRTM15+ bathymetry and a seasonal stratification envelope
   to invert ISW amplitude and phase speed as sensitivity intervals
   (inversion/).
5. Event database: crest-line extraction, wave-packet clustering and ERA5
   wind collocation produce a quality-flagged database of 2315 verified
   ISW events (event_database/, see event_database/DATASHEET.txt).

Repository layout
-----------------
    code/
      configs/                 model / training / data YAML configs
      data_preprocessing/
        download/              Sentinel-1 (CDSE/ASF), SWOT (PO.DAAC), ERA5
        sar_preprocess/        SAFE reading, calibration, incidence
                               normalization, tiling (SNAP optional)
        swot_preprocess/       SWOT quality control, reprojection, SSHA
        matchup/               S1 x SWOT spatiotemporal pairing
        labeling/              SAM-assisted labeling utilities
      models/                  backbones, FiLM fusion, segmentation,
                               context verifier
      training/                datasets, losses, trainer, train scripts
      inference/               tiled prediction, crest extraction, wind
                               filter, event database builder
      evaluation/              metrics, cascade evaluation, visibility
                               correction, verification analysis
      inversion/               KdV/eKdV physics, stratification scenarios,
                               batch inversion
      tests/                   dataset builders, scene inference, shell
                               pipelines, diagnostic scripts
      utils/                   config, geo, logging, seed helpers
    event_database/            released event database (CC-BY-4.0)
      events_verified.csv      verified events with review outcome
      events.csv               full candidate event table
      inversion_results.csv    per-event KdV/eKdV inversion results
      pattern_summary.json     regional pattern statistics
      DATASHEET.txt            data card (sources, fields, limitations)
    environment.yml            conda environment (Python 3.11, PyTorch cu128)
    LICENSE                    MIT license (code only)
    CITATION.cff               citation metadata

Setup
-----
    conda env create -f environment.yml
    conda activate iw

The environment targets PyTorch with CUDA 12.8 (cu128). On CPU-only machines,
install a CPU build of torch/torchvision instead.

Before running anything, edit code/configs/data.yaml:

  - project_root: set to the absolute path of your local project root.
  - paths.*: relative data locations (raw downloads, processed tiles,
    dataset levels L1/L2/L3). The defaults assume the layout
    <project_root>/data/raw, <project_root>/data/processed,
    <project_root>/data/datasets.

Data-download scripts need your own credentials (never commit them):
  - Copernicus Data Space (Sentinel-1): environment variables
    CDSE_USER / CDSE_PASS, or .secrets/credentials.json {"cdse": {...}}
  - NASA Earthdata (SWOT, ASF): EARTHDATA_USER / EARTHDATA_PASS
    or .secrets/credentials.json {"earthdata": {...}}

Quick start
-----------
All commands assume the conda environment is active.

1. Download Sentinel-1 IW GRD scenes (Copernicus Data Space):

    python code/data_preprocessing/download/download_sentinel1.py \
        --source cdse --aoi "109,18,114,23" \
        --start 2023-07-01 --end 2023-10-31 \
        --out data/raw/sentinel1

   SWOT L2 KaRIn SSH (NASA PO.DAAC):

    python code/data_preprocessing/download/download_swot.py \
        --bbox "109,18,114,23" --start 2023-04-01 --end 2023-06-30 \
        --out data/raw/swot

   ERA5 10 m wind:

    python code/data_preprocessing/download/download_era5.py --help

2. Preprocess a SAFE product into georeferenced training tiles
   (pure-Python path; the ESA SNAP GPT path is also provided via
   data_preprocessing/sar_preprocess/run_snap.py):

    python code/data_preprocessing/sar_preprocess/preprocess_safe.py \
        --safe data/raw/sentinel1/<scene>.SAFE \
        --out data/processed/sar_tiles

3. Train a segmentation model (from the code/ directory):

    cd code
    python training/scripts/train_baseline.py --config configs/train_baseline.yaml

   Other configs (configs/train_swin*.yaml) cover the Swin-UNet variants,
   FiLM fusion and negmix experiments. Train the context verifier with:

    python training/train_verifier.py --split winter_holdout --mode context

4. Run scene-level inference (tiled prediction + probability mosaic):

    python tests/run_scene_inference.py \
        --tiles-dir ../data/processed/sar_tiles/<scene>.SAFE \
        --ckpt <path/to/best.pth> \
        --model-config configs/model_swin_strip.yaml \
        --out ../results/scene_eval/<scene>

5. Build the event database from scene probability maps (crest extraction,
   wave-packet clustering, ERA5 wind collocation):

    python inference/build_event_database.py

6. Physics inversion (KdV/eKdV amplitude and phase-speed intervals):

    python inversion/batch_invert.py --quality high,medium

7. Evaluate a checkpoint on a labeled dataset:

    python evaluation/evaluate.py \
        --ckpt <path/to/best.pth> \
        --model-config configs/model_swin_strip.yaml \
        --data <dataset/dir>

Event database
--------------
event_database/events_verified.csv — one row per candidate event reviewed by
the verification cascade and manual interpretation. keep_verified=True marks
the 2315 verified events (quality high/medium/low = 389/530/1396).

Main fields (see event_database/DATASHEET.txt for the full dictionary):

    event_id          unique event identifier
    scene             source Sentinel-1 scene name
    time_utc          scene acquisition time (UTC)
    lon, lat          event centroid (WGS84 degrees)
    n_crest           number of extracted crest lines
    crest_length_km   total crest length (km)
    direction_deg     propagation direction (geographic azimuth, degrees)
    wavelength_m      median inter-crest spacing (m)
    mean_prob         mean segmentation probability over the event
    wind_ms           collocated ERA5 10 m wind speed (m/s)
    wind_flag         wind-window flag (in_window = 2-10 m/s)
    bbox_lon/bbox_lat event bounding box (lon/lat ranges)
    area_km2          event area (km^2)
    quality           high / medium / low
    dist_coast_km     distance to coastline (km)
    keep_verified     True = verified event, False = rejected by review
    drop_reason       rejection reason when keep_verified is False

The event database is released under CC-BY-4.0 (see below and
event_database/DATASHEET.txt).

License
-------
Code: MIT License (see LICENSE).
Event database (event_database/): Creative Commons Attribution 4.0
International (CC-BY-4.0). Please credit "IW-FusionFM Team" and cite the
DOI below when using the database.

Citation
--------
If you use this code or the event database, please cite:

    IW-FusionFM Team. IW-FusionFM: Ocean Internal Wave Detection from
    Sentinel-1 SAR and SWOT. Zenodo.
    DOI: 10.5281/zenodo.XXXXXXX   (placeholder - to be filled in after
                                   the Zenodo DOI is assigned)

Machine-readable citation metadata is in CITATION.cff.

Notes and limitations
---------------------
  - Raw satellite imagery, trained checkpoints and intermediate products are
    NOT included in this release; reproduce them with the download and
    preprocessing scripts above.
  - The released event database mainly covers July-October 2023; winter
    recall is not measurable with the current census (see DATASHEET.txt).
    Treat low-quality events with caution.
  - Some scripts under code/tests/ are research/diagnostic utilities and
    assume the project directory layout described in code/configs/data.yaml.
