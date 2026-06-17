#!/bin/bash
# One simulation job:  run_one.sh "TIER|SCENE|MAT_YAML|ENVS|I"  NBEAM
# The Geant4 detector executable is not bundled; this documents how one job is
# configured and invoked (./muPos).  TIER A = 4-plane (no field); B = 6-plane (3 T).
set -u
LINE=$1; NBEAM=$2
IFS="|" read -r TIER SCENE MAT ENVS I <<< "$LINE"
HERE=$(cd "$(dirname "$0")" && pwd)
G4=$HERE/g4
C=$G4/config
D=$HERE/work/$TIER/$SCENE
mkdir -p "$D/root_file"
OUT=$D/root_file/job_$I.root
[ -s "$OUT" ] && exit 0            # resumable

cd "$D"
[ -f Cry.mac ] || cp "$G4/Cry.mac" .
[ -e muPos ]   || ln -sf "$G4/build/muPos" .
MAC=mac_$I.mac
{ echo "/control/execute Cry.mac"
  echo "/run/initialize"
  echo "/rlt/SetFileName root_file/job_$I.root"
  echo "/run/printProgress 100000"
  echo "/run/beamOn $NBEAM"; } > "$MAC"

if [ "$TIER" = "A" ]; then
  VOL=$C/newrpc_readout.yaml:$C/newrpc.yaml:$C/newlayout4_L400_sio2.yaml; BF=""
else
  VOL=$C/newrpc_readout.yaml:$C/newrpc.yaml:$C/newlayout6_L400_sio2.yaml; BF="MUPOS_BFIELD=3"
fi

env $ENVS $BF \
  MUPOS_VOLUME_CONFIG="$VOL" \
  MUPOS_MATERIAL_CONFIG="$C/newrpc_material.yaml:$C/$MAT" \
  ./muPos "$MAC" > "root_file/job_$I.log" 2>&1
rc=$?; rm -f "$MAC"
[ $rc -eq 0 ] && [ "$I" != "0" ] && rm -f "root_file/job_$I.log"
exit $rc
