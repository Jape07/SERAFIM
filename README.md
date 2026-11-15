# SERAFIM

SERAFIM shows how SPHEREx images of the sky change over time. A Python pipeline turns public
SPHEREx images from IRSA into small Parquet tables and PNG files ("gold"), and a static site on
GitHub Pages reads them directly in the browser with DuckDB-WASM. There is no server.

Site: https://jape07.github.io/SERAFIM/

## Architecture

Medallion layers in a single Python repo, GitHub Actions as the engine and a static site on
GitHub Pages. Everything heavy happens before publishing; the browser only reads gold.

```text
IRSA (SPHEREx) ──> bronze ──> silver ──> gold ──> web (GitHub Pages)
                                  │         ▲
                                  └─> AI & validation (catalogs, status, CNN)
```

| Layer | What it stores | Format | Published? |
| --- | --- | --- | --- |
| Bronze | FITS cutouts exactly as they come from IRSA, plus `manifest.parquet` | FITS, Parquet | No |
| Silver | Clean, aligned images, difference images and detections | NumPy (`.npz`), Parquet | No |
| Gold | Final tables and PNG files for the website | Parquet, PNG, JSON | Yes |

Each layer only reads the previous one, bronze is never modified and every step is idempotent.

### Watched regions (`config/regions.yaml`)

1. **pole**: North Ecliptic Pole deep field (RA 270°, Dec +66.56°). Longest time series.
2. **asteroid**: field near the ecliptic crossed by a bright known asteroid (chosen in task A2).
3. **variable_star**: field with a known variable star that does not saturate (chosen in task A2).

## Repository layout

```text
SERAFIM/
├─ serafim/                 Python package of the pipeline
│  ├─ bronze/               download from IRSA (data engineering)
│  ├─ silver/               cleaning, alignment, subtraction and detection (data engineering)
│  ├─ gold/                 tables, PNG and export (data engineering)
│  ├─ validation/           targets.py, catalogs.py and status.py (AI & validation)
│  ├─ ai/                   predict.py, CNN inference with ONNX (AI & validation)
│  ├─ contract.py           gold schemas (whole team)
│  └─ cli.py                python -m serafim
├─ training/                dataset, training, Optuna, experiments.csv, RESULTS.md (AI & validation)
├─ web/                     static site (web)
│  ├─ index.html, viewer.html, catalog.html, about.html
│  ├─ css/styles.css
│  ├─ js/data.js, js/app.js
│  ├─ img/                  SPHEREx reference images (NASA/JPL-Caltech)
│  ├─ texts/examples.json   guided example texts (written by AI & validation)
│  └─ gold/                 local copy of gold (git-ignored)
├─ config/targets.yaml      the 5 challenge targets (python -m serafim.validation.targets)
├─ config/regions.yaml      the old 3 watched regions; removed when the pipeline moves to targets
├─ models/                  final ONNX model (AI & validation)
├─ tests/                   tests and a tiny FITS fixture
├─ data/                    local bronze, silver, gold, dataset and catalog cache (git-ignored)
└─ .github/workflows/       ci.yml and publish.yml
```

## Gold contract

The source of truth is [`serafim/contract.py`](serafim/contract.py). Agreed at H0; any change
needs approval from all three roles.

| File | One row per | Columns |
| --- | --- | --- |
| `regions.parquet` | region | region_id, name, ra, dec, size_deg, wcs_json |
| `frames.parquet` | image at one date | frame_id, region_id, obs_id, date_utc, wavelength_um, bandwidth_um, png, png_diff |
| `detections.parquet` | change found in a difference image | detection_id, frame_id, object_id, ra, dec, x_px, y_px, snr, sign, p_real |
| `objects.parquet` | grouped object | object_id, region_id, name, type, status, reason, catalog, url, n_detections, first_date_utc, last_date_utc |
| `meta.json` | pipeline run | run_date_utc, commit, n_frames |

`status` can only be `Known`, `Unidentified candidate` or `Discarded`. Never call a candidate a
"new object".

## Spanish → English names

The role documents are in Spanish; everything in this repo is in English.

| In the documents | In the repo |
| --- | --- |
| Ingeniería de datos / IA y validación / Desarrollo web | data engineering / AI & validation / web |
| `datos/`, `bronce/`, `plata/`, `oro/` | `data/`, `bronze/`, `silver/`, `gold/` |
| `contrato.py`, `validar_oro()` | `contract.py`, `validate_gold()` |
| `validacion/catalogos.py`, `estados.py`, `clasificar()` | `validation/catalogs.py`, `status.py`, `classify()` |
| `ia/predecir.py`, `cnn_real_artefacto.onnx` | `ai/predict.py`, `cnn_real_artifact.onnx` |
| `entrenamiento/` `dataset.py`, `entrenar.py`, `buscar.py`, `evaluar.py` | `training/` `dataset.py`, `train.py`, `search.py`, `evaluate.py` |
| `experimentos.csv`, `RESULTADOS.md` | `experiments.csv`, `RESULTS.md` |
| `modelos/`, `config/regiones.yaml` | `models/`, `config/regions.yaml` |
| `restar()`, `recorte()` | `subtract()`, `cutout()` |
| `visor.html`, `catalogo.html`, `acerca.html` | `viewer.html`, `catalog.html`, `about.html` |
| `css/estilos.css`, `js/datos.js`, `js/visor.js`, `js/inicio.js`, `js/catalogo.js` | `css/styles.css`, `js/data.js`, `js/app.js` |
| `textos/ejemplos.json` | `texts/examples.json` |
| `publicar.yml` | `publish.yml` |
| `polo`, `asteroide`, `estrella variable` | `pole`, `asteroid`, `variable_star` |
| `regiones`, `cuadros`, `detecciones`, `objetos` | `regions`, `frames`, `detections`, `objects` |
| `nombre`, `tipo`, `estado`, `motivo`, `catalogo`, `enlace` | `name`, `type`, `status`, `reason`, `catalog`, `url` |
| `fecha_utc`, `longitud_onda_um`, `ancho_banda_um`, `png_resta`, `signo` | `date_utc`, `wavelength_um`, `bandwidth_um`, `png_diff`, `sign` |
| `tamano_grados`, `primera_fecha`, `ultima_fecha` | `size_deg`, `first_date_utc`, `last_date_utc` |
| `fecha_ejecucion`, `n_cuadros`, `n_detecciones` | `run_date_utc`, `n_frames`, `n_detections` |
| Conocido / Candidato sin identificar / Descartado | Known / Unidentified candidate / Discarded |
| `python -m serafim mock`, `run --desde plata` | `python -m serafim mock`, `run --from silver` |

## Getting started

Use Python 3.11 (the same version as GitHub Actions).

```bash
git clone https://github.com/Jape07/SERAFIM.git
cd SERAFIM
py -3.11 -m venv .venv          # macOS/Linux: python3.11 -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"         # everyone
pip install -e ".[dev,ai]"      # AI & validation: adds PyTorch, Optuna and ONNX
ruff check .
pytest
```

Run the site locally with simulated gold:

```bash
python -m serafim mock          # simulated gold in data/gold/
cp -r data/gold web/gold        # web/gold/ is git-ignored
python -m http.server 8000 -d web
```

## How we work on GitHub

- One task = one branch = one small PR. Nobody pushes to `main`; `main` must always be publishable.
- Branch prefixes: `data/`, `ai/`, `web/` (for example `ai/A1-catalogs`).
- Commit messages: `type(area): what it does`, with type `feat`, `fix`, `test`, `docs` or `chore`.
- Every PR says what it does, `Closes #<issue>` and how it was verified. Web PRs need before and
  after screenshots.
- Reviews: data engineering and AI & validation review each other; web PRs are reviewed by
  whoever is free first. Goal: under 15 minutes.
- Merge with "Squash and merge" and delete the branch.
- If `main` moved: `git pull --rebase origin main` and `git push --force-with-lease`, only on
  your own branch.
- Labels: `data`, `web`, `ai`, `blocker`.
- Tags: `v0.1-mvp` at H14 and `v1.0` at delivery.

## Credits

Data from NASA's SPHEREx mission, served by IRSA at IPAC (Caltech).
