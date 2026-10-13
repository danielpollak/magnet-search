#!/bin/bash
# Production suite2p runs for the five photon-starved zebrafish sessions (2026-10-12,
# docs/resegmentation.md): tiffs x64, drift corrected in whole pixels beforehand, suite2p's
# registration off; then the full-resolution extraction. Local disk only.
#   bash docs/resegmentation/run_production.sh      # from the repo root
# Inputs: the original tiffs (hard-linked from the previous production runs, no NAS read) and the
# drift measured by docs/custom_registration/measure.py (copied into each run as drift_shifts.npz).
S2P=/c/Users/dan/anaconda3/envs/suite2p1.0/python.exe
ROOT=C:/Users/dan/suite2p_v1_production
DRIFT=C:/Users/dan/suite2p_v1_custom_reg
for s in 2022_09_14-fish2_cytoGCaMP 2022_09_15-fish1_cytoGCaMP 2022_10_01-fish1_cytoGCamp \
         2022_10_01-fish2_cytoGCaMP 2022_10_02-fish2_cytoGCaMP; do
  new=${s}_x64_drift
  mkdir -p "$ROOT/$new/tiffs"
  for f in "$ROOT/$s/tiffs/"*.tif; do [ -e "$ROOT/$new/tiffs/$(basename "$f")" ] || cp -l "$f" "$ROOT/$new/tiffs/"; done
  [ -e "$ROOT/$new/drift_shifts.npz" ] || cp "$DRIFT/$s/shifts.npz" "$ROOT/$new/drift_shifts.npz"
  echo "=== $(date +%T) suite2p $new"
  "$S2P" -u docs/suite2p_v1/run_suite2p_v1.py --session "$s" --local-root "$ROOT" --run-name "$new" \
      --scale 64 --shifts "$ROOT/$new/drift_shifts.npz" > "$ROOT/$new/run.log" 2>&1
  echo "    rc=$? $(tail -1 "$ROOT/$new/run.log")"
  echo "=== $(date +%T) extraction $new"
  "$S2P" -u pipeline/ophys_extraction.py --plane0 "$ROOT/$new/output/suite2p/plane0" --neuropil \
      > "$ROOT/$new/extraction.log" 2>&1
  echo "    rc=$? $(tail -1 "$ROOT/$new/extraction.log")"
done
echo "=== ALL DONE"
