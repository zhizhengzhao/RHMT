#!/bin/bash
# Regenerate the benchmark scenes into ../data/ :
#   one centered U-shaped block per scene -- 3 materials x 2 backgrounds = 6 object
#   scenes -- plus 2 object-free background blanks (one per background, for calibration).
# The U is a single connected solid (g4/config/u.poly), centred at (0,0), embedded at
# half drum depth. One material per scene.
#
# The Geant4 executable is not bundled; this script documents the exact per-scene
# settings used to generate the benchmark. With a compatible Geant4 build present,
# run from sim/:  bash campaign.sh && bash convert_all.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
NBEAM=100000
HZ=117.25                      # block half-height in z (embedded, z-centred)
PF=$HERE/g4/config/u.poly      # the U cross-section (FILL contour, mm)
U="SAMPLE_SHAPE=polyfile SAMPLE_POLYFILE=$PF SAMPLE_HZ=$HZ SAMPLE_X=0 SAMPLE_Y=0 SAMPLE_Z=0"

# scene | background-material yaml | object env vars (empty = blank scan)
SCENES=(
  "blank_sio2|bench_bg_sio2.yaml|"
  "blank_concrete|bench_bg_concrete.yaml|"
  "u_water_sio2|bench_bg_sio2.yaml|$U SAMPLE_MAT=G4_WATER"
  "u_pb_sio2|bench_bg_sio2.yaml|$U SAMPLE_MAT=G4_Pb"
  "u_al_sio2|bench_bg_sio2.yaml|$U SAMPLE_MAT=G4_Al"
  "u_water_concrete|bench_bg_concrete.yaml|$U SAMPLE_MAT=G4_WATER"
  "u_pb_concrete|bench_bg_concrete.yaml|$U SAMPLE_MAT=G4_Pb"
  "u_al_concrete|bench_bg_concrete.yaml|$U SAMPLE_MAT=G4_Al"
)

mkdir -p "$HERE/work"
JOBS=$HERE/work/jobs.list; : > "$JOBS"
for entry in "${SCENES[@]}"; do
  IFS="|" read -r SCENE MAT ENVS <<< "$entry"
  case $SCENE in blank_*) NA=180; NB=700;; *) NA=80; NB=330;; esac   # blanks get more stats
  for i in $(seq 0 $((NA-1))); do echo "A|$SCENE|$MAT|$ENVS|$i" >> "$JOBS"; done
  for i in $(seq 0 $((NB-1))); do echo "B|$SCENE|$MAT|$ENVS|$i" >> "$JOBS"; done
done
echo "jobs: $(wc -l < "$JOBS")"

# run in parallel (override width with NPROC=...)
xargs -a "$JOBS" -d '\n' -P "${NPROC:-112}" -I{} bash "$HERE/run_one.sh" "{}" "$NBEAM"
echo "campaign done -> now run: bash convert_all.sh"
