# Simulation

The benchmark hit data is produced with Geant4 (electromagnetic physics) and the
CRY cosmic-ray generator. The detector executable is not included; this directory
ships the geometry, materials, and run configuration that fully specify how the
benchmark was generated.

Two instruments:
- 4-plane tracker (tier A): `g4/config/newlayout4_L400_sio2.yaml` -- no magnets
- 6-plane spectrometer (tier B): `g4/config/newlayout6_L400_sio2.yaml` -- two 3 T gaps (`MUPOS_BFIELD=3`)

Eight scenes -- 6 objects (one centered U-shaped block, 3 materials x 2
backgrounds) plus 2 object-free background blanks (one per background, for
calibration) -- each simulated on both instruments. Per-scene settings are in
`campaign.sh`; `run_one.sh` shows how a single job is configured. ~150k triggered
muons per object scene, ~330k per blank.

`convert.py` turns the per-event ROOT output into the NPZ files consumed by the
reconstruction (raw per-plane crossings only; generator truth is retained solely
for scoring).
