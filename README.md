# NAMT — Nuisance-Aware Muon Tomography

Cosmic-muon material tomography posed directly on the detector hits (the
measurement domain). The same hits are reconstructed in two channels:

| channel | instrument (tier) | observable | output field |
|---|---|---|---|
| **NAMT-S** (scattering) | 4 RPC planes, no magnets (**A**) | spread of the hits about the straight track | radiation length `λ = 1/X₀` |
| **NAMT-E** (energy) | 6 RPC planes + 2×3 T magnets (**B**) | momentum loss from the magnetic bend (Bethe–Bloch line integral) | stopping power `s = ρZ/A` |

NAMT-S runs on a 3-plane tracker (one fewer plane than the baselines need) and a
4-plane tracker; NAMT-E uses the 6-plane spectrometer. Baselines: PoCA, ASR, and
MLS-EM, each also run with the spectrometer's measured momentum; and an
energy-loss statistic (per-muon momentum-corrected loss binned by per-column
median, no calibration) as a derived-observable baseline for the energy channel.

This repository reproduces every number in the paper's result tables. It does not
generate the figures.

## Reproduce

The benchmark data is produced by a Geant4 + CRY simulation of the detector. The
simulation executable is not bundled; `sim/` fully specifies the geometry,
materials, and run configuration (see `sim/README.md`). Steps 2–3 run on the
produced `data/`.

```bash
pip install -r requirements.txt          # numpy + torch (CUDA GPU recommended)

# 1. (optional) regenerate the benchmark hits with Geant4 + CRY -> data/
#    needs the detector program (not bundled); sim/ specifies geometry + run config
cd sim && bash campaign.sh && bash convert_all.sh && cd ..

# 2. reconstruct every cell across the listed GPUs -> results/cells/
python experiments/dispatch.py 0,1,2,3

# 3. score -> results/tables/ and stdout
python experiments/metrics.py            # per-scene AUC + exposure ladder
bash experiments/robust_all.sh           # position-error robustness
```

A single cell: `python experiments/run_cell.py --method rht_s --scene u_pb_sio2 --cap 0`.

Calibration is fixed on the object-free blank scans (cached in `cal/`); nothing is
tuned on the object. The single-point per-muon statistics (PoCA, energy-loss)
fall below the default per-column coverage at the lowest exposure, so their 15k
entries use `min_cov=10`.

## Layout

```
.
├── namt/                     the method (importable package)
│   ├── field.py             VoxelField3D — 3-D voxel grid + column projection
│   ├── contrast.py          4-plane REML scattering contrast + Student-t loss
│   ├── physics.py           single-muon transport, Bethe–Bloch, Fermi–Eyges
│   ├── trackfit.py          Levenberg–Marquardt per-muon track + momentum fit
│   ├── inversion.py         shared MAP-inversion pieces (Huber, coverage, axes)
│   ├── calib.py             blank-scan calibration (flat field, tails)
│   ├── scenes.py            benchmark scenes + truth masks (scoring only)
│   ├── data.py              load / subsample / smear the hit datasets
│   └── methods/             plugins (base.py + registry in __init__.py)
│       ├── baselines.py     PoCA, ASR, MLS-EM (4-plane tracker)
│       ├── asr_p.py         ASR with the spectrometer's measured momentum
│       ├── poca_p.py        PoCA with the spectrometer's measured momentum
│       ├── mlsem_p.py       MLS-EM with the spectrometer's measured momentum
│       ├── eloss.py         energy-loss statistic (energy-channel baseline)
│       ├── rht_3p.py        NAMT-S on 3 planes (bottom plane dropped)
│       ├── rht_s.py         NAMT-S  (4-plane scattering → λ)
│       └── rht_6p.py        NAMT-E (energy → s) from the 6-plane spectrometer
├── experiments/             run_cell.py · matrix.py · dispatch.py (multi-GPU) ·
│                            metrics.py (AUC) · robustness.py + robust_table.py
├── cal/                     cached blank-scan calibration (per method × background)
├── sim/                     simulation spec + conversion (defines data/)
│   ├── g4/config/           detector geometry, materials, layout (yaml + u.poly)
│   ├── g4/Cry.mac           CRY cosmic-muon generator settings
│   ├── campaign.sh          per-scene settings (one centered U; 3 mat × 2 bg + blanks)
│   ├── run_one.sh           how one simulation job is configured
│   └── convert.py, convert_all.sh   ROOT → data/{A,B}_*.npz
└── requirements.txt
```

`dispatch.py` / `metrics.py` write into `results/`; `sim/` regenerates `data/`.

## Data flow

```
hits .npz ──► per-muon fit          ──► 3-D voxel field          ──► column map ──► AUC
(data/)      NAMT-S: REML contrast       penalised MLE on a grid      (top-down)     (metrics.py,
             NAMT-E: momentum fit        (VoxelField3D) + TV prior                    truth only here)
```

## How to modify

- **Add a method:** add `namt/methods/your_method.py` subclassing `Method`
  (`name`, `tier`, `calibrate()`, `reconstruct()`), import it in
  `namt/methods/__init__.py`, add its `name` to `ALL_METHODS` in
  `experiments/matrix.py` and a row to `ROWS` in `experiments/metrics.py`.
- **Add a scene / material:** edit `namt/scenes.py` (`SCENES`, `MAT`) and the
  matching Geant4 config under `sim/`.
- **Grid / exposure / priors:** `SCORE` in `run_cell.py` (grid), `CAPS` in
  `matrix.py` (exposures), or the `reconstruct()` keyword args.

## Benchmark

Two backgrounds (fused silica SiO₂, concrete); per background, one scene per
material holding a single **U-shaped block** (150 mm, 45 mm arms, 60 mm gap)
centered at half-depth — 3 materials (water, Pb, Al) × 2 backgrounds = 6 scenes.
Cosmic muons (CRY + Geant4). Exposure ladder full (~125k) / 50k / 15k; robustness
vs hit-position error σ_pos. Scoring: ROC-AUC on a common 6 mm grid.
