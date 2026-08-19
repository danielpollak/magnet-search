"""One-time NAS extraction for Fig 1's raw-voltage snippets.

Fig 1 (pipeline/manuscript/fig1.py) shows two short raw-voltage windows
(magnetic-stimulus "null" panel + white-noise "positive" panel) plus 100
randomly-sampled raw spike waveforms, all for a single exemplar unit. Those
are the only pieces of the whole figure that need NAS access -- everything
else (spike times, Fourier stats) comes from the pipeline's own
data/20230415.nwb. Rather than have fig1.py query the NAS live every time
it's regenerated, this script extracts those snippets once (with NAS access)
into untracked .npy files under data/fig1_raw/, which fig1.py then loads
directly.

Run manually, offline, whenever the exemplar unit/window constants below
change (they're kept in sync with fig1.py by importing them from it):

    python pipeline/manuscript/fig1_extract_raw_snippets.py

Requires NAS access to:
    \\\\datanas\\family\\data_raw\\20230415\\...
    \\\\datanas\\family\\data_aggregated\\20230415
"""
from pathlib import Path

import numpy as np

from ephysio import openEphysIO
from magpyneto2.utils import get_cluster_info

from pipeline import nwb_io
from pipeline.schema import load_experiment
from pipeline.manuscript.fig1 import (
    AGGREGATED_PATH,
    CLUSTER_ID,
    EXPERIMENT,
    MAG_CONTINGENCY,
    MAG_WINDOW,
    N_WAVEFORMS,
    RAW_DATA_ROOT,
    WAVEFORM_HALFWIDTH_MS,
    WAVEFORM_SEED,
    WN_CONTINGENCY,
    WN_WINDOW,
)
import format_parameters as FP

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "fig1_raw"


def open_loader(recname: str):
    contingency_path = (RAW_DATA_ROOT + f"\\{recname}").replace("\\", "/")
    return openEphysIO.Loader(contingency_path, cntlbarcodes=True)


def extract_snippet(ldr, window: tuple, ch: int) -> tuple:
    spike_sr = ldr.samplingrate(ldr.spikestream())
    t_on, t_off = window
    trace = ldr.data(ldr.spikestream())[int(t_on * spike_sr):int(t_off * spike_sr), ch]
    return trace, spike_sr


def extract_waveforms(ldr, ch: int, spk_times, half_width_samples: int, n: int, seed: int):
    spike_sr = ldr.samplingrate(ldr.spikestream())
    data = ldr.data(ldr.spikestream())
    n_total_samples = data.shape[0]

    spk_samples = (np.asarray(spk_times) * spike_sr).astype(int)
    # Drop spikes too close to either edge of the recording for a full window.
    valid = (spk_samples - half_width_samples >= 0) & (spk_samples + half_width_samples < n_total_samples)
    spk_samples = spk_samples[valid]

    rng = np.random.default_rng(seed)
    chosen = rng.choice(spk_samples, size=n, replace=False)

    waveforms = np.empty((n, 2 * half_width_samples + 1))
    for i, s in enumerate(chosen):
        waveforms[i] = data[s - half_width_samples:s + half_width_samples + 1, ch]
    return waveforms, spike_sr


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    udf = get_cluster_info(AGGREGATED_PATH)
    unitrow = udf.loc[udf.cluster_id == CLUSTER_ID].squeeze()
    ch = int(unitrow.ch)
    print(f"cluster {CLUSTER_ID} -> channel {ch}")

    for recname, window, outname in [
        (MAG_CONTINGENCY, MAG_WINDOW, "mag_trace.npy"),
        (WN_CONTINGENCY, WN_WINDOW, "wn_trace.npy"),
    ]:
        ldr = open_loader(recname)
        trace, spike_sr = extract_snippet(ldr, window, ch)
        out_path = OUT_DIR / outname
        np.save(out_path, trace)
        print(f"{recname}: window={window} spike_sr={spike_sr} -> {out_path} "
              f"({trace.shape[0]} samples)")

    # Waveforms: sampled from the WN recording (largest spike count).
    cfg = load_experiment(Path(__file__).resolve().parents[2] / "experiments" / f"{EXPERIMENT}.yml")
    io_r, nwbfile = nwb_io.read_nwbfile(str(Path(FP.DATA_DIR) / f"{EXPERIMENT}.nwb"))
    modulation_df = nwb_io.build_modulation_frame(nwbfile, good_only=cfg.good)
    io_r.close()
    spk_times = modulation_df.loc[
        (modulation_df.rec == WN_CONTINGENCY) & (modulation_df.id == CLUSTER_ID), "spk"
    ].values

    wn_ldr = open_loader(WN_CONTINGENCY)
    spike_sr = wn_ldr.samplingrate(wn_ldr.spikestream())
    half_width_samples = int(round(WAVEFORM_HALFWIDTH_MS / 1000 * spike_sr))
    waveforms, _ = extract_waveforms(wn_ldr, ch, spk_times, half_width_samples, N_WAVEFORMS, WAVEFORM_SEED)
    out_path = OUT_DIR / "wn_waveforms.npy"
    np.save(out_path, waveforms)
    print(f"waveforms: {waveforms.shape[0]} x {waveforms.shape[1]} samples "
          f"(+/-{WAVEFORM_HALFWIDTH_MS}ms) -> {out_path}")


if __name__ == "__main__":
    main()
