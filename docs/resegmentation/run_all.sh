#!/bin/bash
# Re-segment the photon-starved zebrafish sessions (local disk only): drift measurement and tiff
# writing in parallel (CPU), then suite2p one run at a time (one GPU). 20220915_fish1 was done
# first, by hand (docs/custom_registration.md); it is skipped if its runs exist.
#   bash docs/resegmentation/run_all.sh        # from the repo root
PY=/c/Users/dan/anaconda3/envs/magneto2/python.exe
S2P=/c/Users/dan/anaconda3/envs/suite2p1.0/python.exe
CR=docs/custom_registration
OUT=docs/resegmentation
LOG=C:/Users/dan/suite2p_v1_custom_reg/logs
SESSIONS="2022_09_14-fish2_cytoGCaMP 2022_09_15-fish1_cytoGCaMP 2022_10_01-fish1_cytoGCamp 2022_10_01-fish2_cytoGCaMP 2022_10_02-fish2_cytoGCaMP"
prep() {   # drift, then both variants' tiffs, for one session
  s=$1; d=C:/Users/dan/suite2p_v1_custom_reg/$s
  [ -f $d/shifts.npz ] || $PY -u $CR/measure.py --session $s --csv $OUT/results_bins_$s.csv > $LOG/measure_$s.log 2>&1
  [ -f $OUT/results_bins_$s.csv ] || cp $CR/results_bins.csv $OUT/results_bins_$s.csv
  for v in corrected raw; do
    [ -d $d/s2p_${v}_x64/output/suite2p/plane0 ] || \
      $S2P -u $CR/segment.py --session $s --variant $v --tiffs-only > $LOG/tiffs_${v}_$s.log 2>&1
  done
}
echo "=== $(date +%T) prep start"
for s in $SESSIONS; do prep $s & done
wait
echo "=== $(date +%T) prep done"
for s in $SESSIONS; do for v in corrected raw; do
  d=C:/Users/dan/suite2p_v1_custom_reg/$s/s2p_${v}_x64/output/suite2p/plane0
  [ -f $d/stat.npy ] && continue
  $S2P -u $CR/segment.py --session $s --variant $v > $LOG/seg_${v}_$s.log 2>&1
  echo "=== $(date +%T) $s $v rc=$? $(tail -1 $LOG/seg_${v}_$s.log)"
done; done
echo "=== ALL DONE"
