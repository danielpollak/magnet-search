"""One-time NAS extraction of the oddball flash times, for oddball_stimulus.py.

The oddball (pigeon cerebellum, 20230221 and 20230228) is a full-screen
flash train: the screen alternates on/off every ~0.515 s, and on ~5% of
frames the state is held, giving a 1.5 s flash or gap (PsychoPy cell
"ON OFF stimuli" in MagnetSearch/code/experiments/2023 03 15 expts/
sound_visual_stim.py). The photodiode channel records every flash. The NWB
epochs only keep the padded analysis windows, so this script re-reads the
photodiode trace once from the NAS, runs the same Schmitt trigger as
_handle_oddball, and saves the flash onsets/offsets -- translated to the
Neuropixels clock, in seconds from the start of the recording, the same
clock as the recording's spike times; `rec_start_s` is where the recording
starts on the NWB Units table's concatenated timeline -- to untracked
data/oddball_ttl/{recname}.npz, which oddball_stimulus.py then loads.

    python pipeline/manuscript/oddball_extract_ttl.py

Requires NAS access to \\\\datanas\\family\\data_raw\\20230221 and 20230228.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import spikeinterface.extractors as se

from ephysio import openEphysIO
from magpyneto2 import schmitt

from pipeline.schema import load_experiment

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "data" / "oddball_ttl"
EXPERIMENTS = ("20230221", "20230228")


def extract(cfg, aux_cfg, cat_df):
    catrow = cat_df.loc[cat_df.recname == aux_cfg.recname].iloc[0]
    recroot_path = catrow.path.split(catrow.recname)[0] + catrow.recname
    recording = se.OpenEphysBinaryRecordingExtractor(recroot_path, stream_id=cfg.stream_id)
    ldr = openEphysIO.Loader(recroot_path.replace("\\", "/"), cntlbarcodes=cfg.cntlbarcodes)

    trace = np.array(recording.get_traces()[:, aux_cfg.channel], dtype="float64")
    iup, idown = schmitt(trace, thr_on=aux_cfg.thr_on, thr_off=aux_cfg.thr_off,
                         starttype=0, endtype=0)
    kw = dict(deststream=aux_cfg.deststream, sourcestream=aux_cfg.sourcestream,
              sourcebarcode=aux_cfg.sourcebarcode)
    ap_sr = ldr.samplingrate(ldr.spikestream())
    t_up = np.asarray(ldr.shifttime(iup, **kw)) / ap_sr
    t_down = np.asarray(ldr.shifttime(idown, **kw)) / ap_sr
    # Same concatenation offset as _build_contingency's beginning_time.
    row_i = cat_df.index.get_loc(catrow.name)
    rec_start_s = 0.0 if row_i == 0 else cat_df.iloc[row_i - 1]["cumulate"] / ap_sr
    return t_up, t_down, ap_sr, rec_start_s


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in EXPERIMENTS:
        cfg = load_experiment(str(REPO / "experiments" / f"{name}.yml"))
        cat_df = pd.read_csv(cfg.metadata_csv)
        cat_df["cumulate"] = np.cumsum(cat_df["nframes"].values)
        for aux_cfg in cfg.auxiliary_stimuli:
            if aux_cfg.kind != "oddball":
                continue
            t_up, t_down, ap_sr, rec_start_s = extract(cfg, aux_cfg, cat_df)
            out = OUT_DIR / f"{aux_cfg.recname}.npz"
            np.savez(out, t_up=t_up, t_down=t_down, ap_sr=ap_sr, rec_start_s=rec_start_s,
                     experiment=name)
            on = t_down - t_up if len(t_down) == len(t_up) else np.array([])
            print(f"{out}: {len(t_up)} onsets, {len(t_down)} offsets"
                  + (f", flash median {np.median(on):.3f} s" if on.size else ""))


if __name__ == "__main__":
    main()
