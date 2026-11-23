# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (magneto2)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # The mouse data: what each file in `data/precomputed/mouse/` is
#
# The mouse is the only species in the paper that the pipeline doesn't process. Its Fourier results enter `aggregate.py` as a frozen DataFrame, `mouse_analysis.pickle`. The other files in this directory are what that DataFrame was made from. They came from Kyu Hyun Lee (Meister lab, Caltech). This notebook opens each one in turn and says what it holds. It is the Python counterpart of `DJP_HKL_analysis.mlx`, the MATLAB live script that computed the first version of the statistic.
#
# Questions it answers:
#
# 1. **What was recorded?** One mouse, 2017-09-28, superior colliculus, with a 128-channel silicon probe. Two magnetic recordings, nominally 2 Hz and 4 Hz.
# 2. **Which probe, and is there a channel map?** The probe was a Masmanidis lab **128DN**: 4 shanks × 32 sites. No channel map is in these files; each unit carries only a channel number (`masks`).
# 3. **How do the files relate to each other and to `mouse_analysis.pickle`?** Sections 3–6.
#
# There is no raw voltage here. The `.rhd` names are Intan's file extension, but the `*.rhd` folders hold spike-sorted output and event times exported to MATLAB, not Intan data files.

# %%
from pathlib import Path
import zipfile, re

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import pymupdf
from IPython.display import Image, display

try:
    HERE = Path(__file__).resolve().parent
except NameError:  # running as a notebook
    HERE = Path.cwd()
assert (HERE / "mouse_analysis.pickle").exists(), HERE

RECS = {"magnetic_170928_025824": "nominal 2 Hz", "magnetic4hz_170928_031259": "nominal 4 Hz"}

# %% [markdown]
# ## 1. The directory

# %%
rows = []
for p in sorted(HERE.rglob("*")):
    if p.is_file() and p.name not in (".DS_Store",) and p.suffix not in (".py", ".ipynb"):
        rows.append({"file": str(p.relative_to(HERE)).replace("\\", "/"), "kB": round(p.stat().st_size / 1024, 1)})
pd.DataFrame(rows)

# %% [markdown]
# | File | What it is |
# |---|---|
# | `Recording 20170928.pdf` | Kyu Hyun Lee's notes for the session: animal, probe, depth, every stimulus block, and how the magnet's sync pulse relates to the field (section 2) |
# | `DJP_HKL_analysis.mlx` | MATLAB live script (a zip of XML) computing a Fourier statistic per unit (section 3) |
# | `<rec>.rhd/magstim.mat` | Times of the magnet's TTL sync pulse, one per field period (section 4) |
# | `<rec>.rhd/spiketrain.mat` | Spike times of the 51 sorted units, their Kilosort cluster ids and a channel number per unit (section 4) |
# | `<rec>.rhd/stim.mat` | Visual-stimulus trigger times; empty here, since no visual stimulus ran during the magnetic recordings |
# | `<rec>.rhd/wheel.mat` | Running-wheel velocity at 100 Hz (section 4) |
# | `<rec>.rhd.mat` | `chat`: the `.mlx`'s output for that recording (section 5) |
# | `mouse_analysis.pickle` | The Fourier results the pipeline uses (section 6) |
# | `.DS_Store` | macOS folder metadata; ignore |

# %% [markdown]
# ## 2. The recording notes: animal, probe and magnetic stimulus

# %%
pdf = pymupdf.open(HERE / "Recording 20170928.pdf")
text = "\n".join(page.get_text() for page in pdf)
print(text[:text.find("Baseline")])
print("...")
i = text.find("Magnetic stimulus")
print(text[i:])

# %% [markdown]
# The notes name the probe as **"Batch4 128DN 2"**: probe number 2 of fabrication batch 4 of the 128DN design. The 128DN is one of the open-source silicon probes from Sotiris Masmanidis's lab (UCLA; Du et al. 2011, Yang et al. 2020). Lee et al. (eLife 2020;9:e50678), the lab's paper on the superior colliculus, says they used probe types 128A, 128AN and 128DN, recorded with an Intan RHD2000 128-channel amplifier board and RHD2000 USB interface board, and spike-sorted with KiloSort, then curated in Phy. Methods sections that use the 128DN describe it as:
#
# - 4 shanks, 150 µm apart
# - 32 sites per shank, 25 µm apart vertically, spanning 0–775 µm above each tip
# - 100 µm² electrode area
#
# The probe tip was ~1800 µm below the brain surface, with DiD on the probe to find the track afterwards. The notes also say the magnet was 3–5 cm from the right side of the head, driven with a 2 Hz and then a 4 Hz sine for 10 min each, at ~300 µT peak to trough (a smartphone magnetometer reading). The TTL sync pulse steps up at the start of each field period and down at the half period; the notes' oscilloscope screenshot is below.

# %%
for page_i in (3, 4):
    for img_i, img in enumerate(pdf[page_i].get_images()):
        pix = pymupdf.Pixmap(pdf, img[0])
        if pix.n > 4:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        display(Image(data=pix.tobytes("png"), width=420))

# %% [markdown]
# ## 3. `DJP_HKL_analysis.mlx`: the MATLAB statistic
#
# An `.mlx` is a zip; its code is in `matlab/document.xml`.

# %%
with zipfile.ZipFile(HERE / "DJP_HKL_analysis.mlx") as z:
    xml = z.read("matlab/document.xml").decode("utf8")
code = "\n\n".join(re.findall(r"<!\[CDATA\[(.*?)\]\]>", xml, flags=re.S))
print(code)

# %% [markdown]
# For each unit with more than 100 spikes between the first and last magnet trigger, it computes the Fourier coefficient of the spike train, $c(f) = \frac{1}{T}\sum_k e^{2\pi i f t_k}$, at the stimulus frequency `fr` and at the 100 off-frequency bins in `fr ± 0.5 Hz` (0.01 Hz steps). `chat` is $|c(f_r)|$ divided by the mean of the standard deviations of the off-frequency coefficients' real and imaginary parts. That is the same idea as the pipeline's NFC.
#
# Two things to note. `fr = 4` is hard-coded, but the "4 Hz" magnet actually ran at 4.49 Hz (section 4). And the script loads the 4 Hz recording, with the 2 Hz recording's lines commented out.

# %% [markdown]
# ## 4. The `.rhd` folders

# %%
def load_rec(rec):
    d = HERE / f"{rec}.rhd"
    st = sio.loadmat(d / "spiketrain.mat")
    return dict(
        trig=sio.loadmat(d / "magstim.mat")["magstim_trigger"].ravel(),
        cellid=st["cellid"].ravel(),
        channel=st["masks"].ravel(),
        spikes=[x.ravel() for x in st["spiketrain"].ravel()],
        duration=st["recordingDuration"].item(),
        stim=sio.loadmat(d / "stim.mat")["stim_trigger"],
        wheel_t=sio.loadmat(d / "wheel.mat")["wheeltime"].ravel(),
        wheel_v=sio.loadmat(d / "wheel.mat")["velocity"].ravel(),
    )

data = {rec: load_rec(rec) for rec in RECS}
for rec, d in data.items():
    print(f"{rec} ({RECS[rec]}):")
    for name in ("magstim.mat", "spiketrain.mat", "stim.mat", "wheel.mat"):
        print(f"  {name}: {sio.whosmat(HERE / f'{rec}.rhd' / name)}")

# %% [markdown]
# ### `magstim.mat`: the magnet's sync pulse
#
# `magstim_trigger` holds times in seconds, one per field period: the rising edges of the sync pulse. Their spacing gives the frequency the magnet really ran at, 2.08 Hz and 4.49 Hz, not the nominal 2 and 4 Hz. `mouse_analysis.pickle` uses these measured values (`freq` 2.08 and 4.49).

# %%
fig, axes = plt.subplots(1, 2, figsize=(8, 2.5), sharey=True)
for ax, (rec, d) in zip(axes, data.items()):
    period = np.diff(d["trig"])
    ax.hist(period * 1e3, bins=60, color="steelblue")
    ax.set_title(f"{RECS[rec]}: {1 / np.median(period):.3f} Hz, {len(d['trig'])} periods\n"
                 f"{d['trig'][0]:.1f}–{d['trig'][-1]:.1f} s of {d['duration']:.0f} s", fontsize=9)
    ax.set_xlabel("Trigger interval (ms)")
axes[0].set_ylabel("Count")
plt.show()

# %% [markdown]
# ### `spiketrain.mat`: the sorted units
#
# - `cellid`: the unit's Kilosort/Phy cluster id.
# - `spiketrain`: one cell per unit, spike times in seconds from the start of the recording.
# - `masks`: one integer per unit, 1–127, most likely the unit's peak channel (MATLAB 1-based). The name is a guess at its origin, not a description.
# - `recordingDuration`: seconds.
#
# Both recordings have the same 51 units in the same order, so they were sorted together as one concatenated session, then split back into recordings.

# %%
a, b = (data[r] for r in RECS)
print("same cluster ids and channels in both recordings:",
      np.array_equal(a["cellid"], b["cellid"]) and np.array_equal(a["channel"], b["channel"]))
units = pd.DataFrame({
    "cellid": a["cellid"], "channel": a["channel"],
    **{f"rate {RECS[r]} (Hz)": [len(x) / data[r]["duration"] for x in data[r]["spikes"]] for r in RECS},
})
units.describe().round(2)

# %%
fig, axes = plt.subplots(2, 1, figsize=(8, 4.5), sharex=True)
for ax, (rec, d) in zip(axes, data.items()):
    order = np.argsort(d["channel"])
    for row, u in enumerate(order):
        s = d["spikes"][u]
        s = s[(s > d["trig"][0]) & (s < d["trig"][0] + 5)]
        ax.vlines(s - d["trig"][0], row, row + 0.8, color="k", linewidth=0.4)
    for t in d["trig"][(d["trig"] >= d["trig"][0]) & (d["trig"] < d["trig"][0] + 5)]:
        ax.axvline(t - d["trig"][0], color="steelblue", linewidth=0.5, alpha=0.5)
    ax.set_ylabel("Unit (by channel)")
    ax.set_title(f"{RECS[rec]}: first 5 s of stimulation, magnet triggers in blue", fontsize=9)
axes[-1].set_xlabel("Time from first trigger (s)")
plt.show()

# %% [markdown]
# ### `wheel.mat` and `stim.mat`
#
# `wheel.mat` is the running-wheel velocity (`velocity`, sampled every 10 ms at `wheeltime`). The units of velocity aren't stated. `stim.mat`'s `stim_trigger` is empty in both recordings: it holds visual-stimulus triggers, and none ran during the magnetic recordings.

# %%
fig, ax = plt.subplots(figsize=(8, 2.2))
for rec, d in data.items():
    ax.plot(d["wheel_t"], d["wheel_v"], linewidth=0.4, label=RECS[rec])
ax.set_xlabel("Time (s)")
ax.set_ylabel("Wheel velocity")
ax.legend(fontsize=8, frameon=False)
plt.show()
print({RECS[r]: d["stim"].shape for r, d in data.items()})

# %% [markdown]
# ## 5. `<rec>.rhd.mat`: the `.mlx` output
#
# Each holds `chat`, 48 values: one per unit with more than 100 spikes during stimulation. Re-running the `.mlx` algorithm in Python reproduces **both** files to rounding error at `fr = 4`, including the 2 Hz recording's. So both were made with the script as it stands, only the `load` lines swapped, and the 2 Hz recording's `chat` is its statistic at 4 Hz, not at its own frequency. These files are not used by the pipeline.

# %%
def mlx_chat(d, fr):
    out = []
    for st in d["spikes"]:
        st = st[(st > d["trig"][0]) & (st < d["trig"][-1])]
        if len(st) <= 100:
            continue
        T = st[-1] - st[0]
        fs = np.round(np.arange(3.5, 4.5 + 1e-9, 0.01), 2)  # as in the .mlx
        c = np.array([np.exp(2j * np.pi * f * st).sum() / T for f in fs])
        on = np.isclose(fs, fr)
        out.append(abs(c[on][0]) / ((np.std(c[~on].real, ddof=1) + np.std(c[~on].imag, ddof=1)) / 2))
    return np.array(out)

for rec, d in data.items():
    chat = sio.loadmat(HERE / f"{rec}.rhd.mat")["chat"].ravel()
    print(f"{rec} ({RECS[rec]}): {len(chat)} values, "
          f"max |Python - file| at fr = 4: {np.max(np.abs(mlx_chat(d, 4) - chat)):.1e}")

# %% [markdown]
# ## 6. `mouse_analysis.pickle`: what the pipeline uses
#
# One row per unit and recording, in the legacy pickle-era column names (`aggregate.py` renames them: `pp` → `p_value`, `nn` → `spk_count`, `rr` → `NFC`, `2f_rr`/`2f_pp` → `2f_NFC`/`2f_p_value`). It was not made by the `.mlx`: it is at the measured frequencies (2.08 and 4.49 Hz) with `Q` = 100 off-frequency bins, the legacy Python pipeline's `fit_fourier_sig` settings, and its `nn` spike counts come from a slightly different window than the `.mlx`'s. Its `id` is the `cellid` above. It keeps 50 and 49 of the 51 units: cluster 310 is missing from both recordings and cluster 542 from the 2 Hz one. Its `date` column says 2023-09-28, which is wrong: the recording was 2017-09-28. `aggregate.py` corrects it (`_PRECOMPUTED_FIXES`); the pickle itself stays frozen.

# %%
pkl = pd.read_pickle(HERE / "mouse_analysis.pickle")
display(pkl.head())
display(pkl.groupby("rec").agg(units=("id", "size"), freq=("freq", "first"), Q=("Q", "first"),
                                p_below_005=("pp", lambda p: int((p < 0.05).sum()))))
for rec, d in data.items():
    print(f"{RECS[rec]}: units not in the pickle: {sorted(set(d['cellid']) - set(pkl.loc[pkl.rec == rec, 'id']))}")

# %% [markdown]
# ## 7. The channel map: what we can and can't say
#
# **What the data give:** a channel number per unit (`masks`, 1–127) and nothing else. There are no site coordinates, no channel-map file (Kilosort's `chanMap.mat`) and no raw data. So these files cannot place units on the probe.
#
# **What the probe implies:** 128 channels = 4 shanks × 32 sites, 150 µm between shanks, 25 µm between sites. If the channel numbers were ordered shank by shank, `channel // 32` would give the shank. That ordering is not guaranteed: the mapping from Intan amplifier channel to probe site depends on the probe's connector wiring and is defined in the lab's channel-map file. The histogram below shows the units' channels with those hypothetical shank boundaries.
#
# **To get the real map:** ask Kyu Hyun Lee or the Meister lab for the Kilosort channel map used for this session (`chanMap.mat` for the 128DN), or the Masmanidis lab's 128DN wiring file.
#
# **What can be reported now:** "Recordings were made in the superior colliculus of a head-fixed C57BL/6 mouse with a 128-channel, 4-shank silicon probe (128DN, Masmanidis lab, UCLA; Du et al. 2011), with its tip ~1800 µm below the brain surface, through an Intan RHD2000 system, and spike-sorted with KiloSort and Phy (Lee et al. 2020)."

# %%
fig, ax = plt.subplots(figsize=(8, 2.2))
ax.hist(units["channel"], bins=np.arange(0.5, 128.5, 1), color="k")
for edge in (32.5, 64.5, 96.5):
    ax.axvline(edge, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlim(0.5, 128.5)
ax.set_xlabel("Unit's channel (masks, 1-based)")
ax.set_ylabel("Units")
ax.set_title("Dashed: shank boundaries IF channels were numbered shank by shank (unverified)", fontsize=9)
plt.show()
