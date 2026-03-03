# Biodiversity impacts modeling pipeline

This repository contains the data processing and modeling pipeline used in the manuscript: **_Actionable biodiversity monitoring hinges on representative data and model design_.** (Nyström et al, 2026). Preprint DOI: https://doi.org/10.32942/X2507T.

The study evaluates how well pressure-response biodiversity models generalize from observed data to new sites and studies. Using 25,987 species inventories from 681 studies, the pipeline compares two different model structures, implemented as:

- a generalized linear mixed model (GLMM; `glmmTMB` backend in R), and
- a biogeographic-taxonomic Bayesian hierarchical model (PyMC).

A central result is that effect-size inference and out-of-sample prediction can diverge strongly, especially under distribution shifts and limited data coverage.

## How to reproduce results

This section is written for reviewers who want to reproduce model runs and manuscript outputs.

### 1. Clone the repository

```bash
git clone https://github.com/j-nystrom/biodiversity-impacts.git
cd biodiversity-impacts
```

### 2. Create the conda environment

```bash
conda env create -f env.yaml
conda activate bio_impact
```

### 3. Configure `PYTHONPATH`

The code expects imports from the repository root (for example `core.*`).

macOS/Linux:

```bash
./setup_conda_path.sh
conda deactivate
conda activate bio_impact
```

Windows (Command Prompt):

```bat
setup_conda_path.bat
conda deactivate
conda activate bio_impact
```

### 4. Install required R packages (for GLMM runs)

GLMM execution uses `Rscript` with `core/model/r_backend/glmm_tmb_runner.R`, requiring:

- `glmmTMB`
- `arrow`
- `jsonlite`

Install in R:

```r
install.packages(c("glmmTMB", "arrow", "jsonlite"))
```

## Data layout and path assumptions

Current configs use paths relative to a `data/` directory located next to the repo directory:

```text
<project_parent>/
  biodiversity-impacts/        # this repo
  data/
    PREDICTS/
    GPW/
    gROADS/
    WorldClim/
    EarthEnv/
    output/
    runs/
```

Path definitions are in:

- `core/data/data_configs.yaml`
- `core/features/feature_configs.yaml`
- `core/utils/util_configs.yaml`

If your data is elsewhere, update those config files before running DAGs.

## Pipeline structure (DAGs and tasks)

Entry point: `core/dags/dags.py`

Run from `core/dags`:

```bash
cd core/dags
python dags.py <dag_name>
```

Available DAGs:

- `predicts`: merge PREDICTS releases and build site buffers
- `population`: raster extraction for population density
- `roads`: road-density extraction
- `bioclimatic`: climate covariate extraction
- `topographic`: topographic covariate extraction
- `features`: combine all inputs and generate model features
- `alpha`: alpha-diversity dataset generation
- `beta`: beta-diversity dataset generation
- `model_data`: model-ready data/folds only
- `training`: model-data prep + model fit + in-sample evaluation
- `crossval`: model-data prep + k-fold evaluation

## Running manuscript experiments

Use the experiment runner to create reproducible run folders with resolved config snapshots.

Single experiment file:

```bash
python experiments/run_experiment.py \
  --experiments-file experiments/bhm/bhm_alpha_train.yaml
```

Multiple experiments from one YAML:

```bash
python experiments/run_experiment.py \
  --experiments-file <yaml_file> \
  --parallel 2
```

Manuscript experiment suites are stored in:

- `experiments/bhm/`
- `experiments/glmm/`
- `experiments/glmm_25/`

## Model configuration and overrides

Base config: `core/model/model_configs.yaml`

Experiment files override subsets of base config, mainly:

- `data_scope`: diversity type, filtering, subsampling, taxonomic resolution
- `run_settings.model_type`: `bayesian` or `glmm`
- `run_settings.model_variables`: variable set (`bayesian_alpha`, `glmm_beta`, etc.)
- `run_settings.bayesian` / `run_settings.glmm`: model-specific settings
- `cv_settings`: fold count, split level (`site`/`study`), stratification columns

## Run outputs

Each run creates a timestamped folder in `data/runs/`:

- direct DAG runs: `run_folder_<timestamp>`
- experiment runner: `run_folder_<timestamp>_<experiment_name>`

Typical contents:

- `config.yaml` (resolved config used for that run; experiment runner)
- `experiment.log` (captured stdout/stderr)
- `model_configs.yaml` (config snapshot)
- `scope_manifest.parquet`, `final_scope_manifest.parquet`
- `training_data.parquet` or `train_fold_*.parquet` + `test_fold_*.parquet`
- `key_output/`: predictions and metrics for downstream analysis
- `additional_output/`: large/optional artifacts (model objects, traces, predictive distributions)

## Notebook analysis

Manuscript result analysis is done in notebooks under `notebooks/`, primarily:

- `notebooks/1_manuscript_main.ipynb`
- `notebooks/2_extended_data.ipynb`

`notebooks/1_manuscript_module.py` is generated from the notebook workflow and should not be edited directly.

## Minimal reproduction path

If processed diversity datasets already exist under `data/output/`, the shortest path to core model results is:

1. activate environment and configure `PYTHONPATH`
2. run one training YAML and one cross-validation YAML from `experiments/bhm/` or `experiments/glmm/`
3. inspect `data/runs/<run_folder>/key_output/`

This reproduces core predictive outputs and metrics without rerunning the full raw geodata processing pipeline.

## Full directory tree

The tree below reflects the paths currently referenced in:

- `core/data/data_configs.yaml`
- `core/features/feature_configs.yaml`
- `core/utils/util_configs.yaml`

```text
<project_parent>/
├── biodiversity-impacts/
│   ├── core/
│   ├── experiments/
│   ├── notebooks/
│   ├── env.yaml
│   ├── setup_conda_path.sh
│   └── setup_conda_path.bat
└── data/
    ├── PREDICTS/
    │   ├── PREDICTS_2016/
    │   │   └── data.csv
    │   └── PREDICTS_2022/
    │       └── data.csv
    ├── GPW/
    │   ├── gpw_v4_2000_30_sec.tif
    │   ├── gpw_v4_2005_30_sec.tif
    │   ├── gpw_v4_2010_30_sec.tif
    │   ├── gpw_v4_2015_30_sec.tif
    │   └── gpw_v4_2020_30_sec.tif
    ├── gROADS/
    │   ├── africa/groads-v1-africa.shp
    │   ├── americas/groads-v1-americas.shp
    │   ├── asia/groads-v1-asia.shp
    │   ├── europe/groads-v1-europe.shp
    │   └── oceania/groads-v1-oceania.shp (Note: Source consists of two files that need merging)
    ├── WorldClim/
    │   └── Bioclimatic/
    │       ├── wc2.1_30s_bio_1.tif
    │       ├── wc2.1_30s_bio_4.tif
    │       ├── wc2.1_30s_bio_5.tif
    │       ├── wc2.1_30s_bio_6.tif
    │       ├── wc2.1_30s_bio_12.tif
    │       ├── wc2.1_30s_bio_13.tif
    │       ├── wc2.1_30s_bio_14.tif
    │       └── wc2.1_30s_bio_15.tif
    ├── EarthEnv/
    │   └── topography/
    │       ├── elevation_1KMmn_GMTEDmd.tif
    │       ├── slope_1KMmn_GMTEDmd.tif
    │       ├── roughness_1KMmn_GMTEDmd.tif
    │       └── tri_1KMmn_GMTEDmd.tif
    ├── output/
    │   ├── predicts/
    │   │   └── all_predicts.parquet
    │   ├── site_coords/
    │   │   └── all_site_coords.shp
    │   ├── buff_polygons/
    │   │   ├── glob_buff_polygons_1km.shp
    │   │   ├── glob_buff_polygons_5km.shp
    │   │   ├── glob_buff_polygons_10km.shp
    │   │   ├── glob_buff_polygons_50km.shp
    │   │   ├── utm_buff_polygons_1km.shp
    │   │   ├── utm_buff_polygons_5km.shp
    │   │   ├── utm_buff_polygons_10km.shp
    │   │   └── utm_buff_polygons_50km.shp
    │   ├── pop_density/
    │   │   ├── pop_density_1km.parquet
    │   │   ├── pop_density_10km.parquet
    │   │   └── pop_density_50km.parquet
    │   ├── road_density/
    │   │   ├── road_density_oceania.parquet
    │   │   ├── road_density_europe.parquet
    │   │   ├── road_density_africa.parquet
    │   │   ├── road_density_asia.parquet
    │   │   └── road_density_americas.parquet
    │   ├── environment/
    │   │   ├── bioclimatic_1km.parquet
    │   │   ├── bioclimatic_10km.parquet
    │   │   ├── topography_1km.parquet
    │   │   └── topography_10km.parquet
    │   ├── combined/
    │   │   └── combined_data.parquet
    │   ├── features/
    │   │   └── feature_data.parquet
    │   ├── alpha_diversity/
    │   │   ├── alpha_all_species.parquet
    │   │   └── alpha_custom.parquet
    │   ├── beta_diversity/
    │   │   ├── beta_all_species.parquet
    │   │   └── beta_custom.parquet
    └── runs/
```
