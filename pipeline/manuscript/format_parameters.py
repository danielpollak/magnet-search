# Shared formatting parameters for all manuscript figures.
# Import as: import format_parameters as FP

# ── Font ───────────────────────────────────────────────────────────────────────
FONT_FAMILY = "arial"
FS_BODY     = 10      # default body text (figs 1, 3)
FS_BODY_LG  = 10      # large body text (fig 2)
FS_BODY_XL  = 10     # extra-large body text (fig 4)
FS_PANEL    = 11     # subfigure label (A, B, C…)
# Standardized 2026-08-28 across every manuscript figure: previously each
# figure/plotting helper hardcoded its own ad hoc legend fontsize (5-7pt)
# while titles were mostly left unset and rendered at matplotlib's much
# larger default 'axes.titlesize' ('large', ~1.2x FS_BODY) -- legends ended
# up far smaller than titles with no numeric relationship between the two.
# FS_LEGEND is now capped at 90% of FS_TITLE, and both are set explicitly
# (never left to fall back on rcParams defaults) everywhere they're used.
FS_TITLE    = 9       # axis title (was 10)
FS_LEGEND   = 8       # legend text, all figures (~89% of FS_TITLE; was 5-7pt in various places)

# ── Figure dimensions (figsize) ────────────────────────────────────────────────
FIGSIZE_FIG1 = (8.5, 6)       # composite NPIX + GCaMP + ECDF
FIGSIZE_FIG2 = (8.5, 6.5)   # excess counts + distributions
FIGSIZE_FIG3 = (8, 4)       # p/q-value uniformity
FIGSIZE_FIG4 = (11, 8)    # modulation sensitivity (A left / B right in a squished top band, C+D side-by-side below)

# ── File paths ────────────────────────────────────────────────────────────────────
# Anchored to this file's own location (repo_root/pipeline/manuscript/
# format_parameters.py -> repo_root), not the process's CWD -- these used to
# be plain relative strings ("../../data"), which only resolved correctly
# when the script was run with pipeline/manuscript/ as the working
# directory. Running a figure script the way its own docstring says to
# (`python pipeline/manuscript/fig4.py` from the repo root) resolved
# "../../data" to two directories *above* the repo instead, which doesn't
# exist -- confirmed via a real FileNotFoundError on data/*.nwb.
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent.parent

OUT_DIR          = str(_REPO_ROOT / "figs" / "paper")
DATA_DIR         = str(_REPO_ROOT / "data")
PARQUET_PATH     = str(_REPO_ROOT / "data" / "manuscript" / "all_fourier_df.parquet")

# ── Save settings ──────────────────────────────────────────────────────────────
DPI = 300
DPI_FIG2 = 300  # fig 2 uses default matplotlib DPI; save is bbox_inches only
DPI_FIG4 = 300  # fig 4 uses default matplotlib DPI; save is bbox_inches only

# ── Line widths ────────────────────────────────────────────────────────────────
LW_TRACE    = 1      # signal traces
LW_REFERENCE = 0.8   # reference lines (y=x, null line, etc)
LW_THIN     = 0.5    # thin borders
LW_CONTOUR  = 0.5    # cell contours

# ── Marker sizes ───────────────────────────────────────────────────────────────
MS_DATA  = 4     # standard scatter plot marker
MS_SMALL = 3     # dense scatter plots

# ── Alpha (transparency) ───────────────────────────────────────────────────────
ALPHA_TRACE      = 0.5      # semi-transparent traces (fig 3)
ALPHA_CONFIDENCE = 0.2      # confidence bands (fig 1 ECDF)
ALPHA_SCATTER    = 0.7      # scatter points (fig 4)
ALPHA_CELL_EXEMPLAR = 0.5   # exemplar cell mask (fig 1)
ALPHA_CELL_OTHER = 0.6      # other cell masks (fig 1)

# ── Colors ─────────────────────────────────────────────────────────────────────
COLOR_MAG     = "steelblue"  # magnetic stimulation
COLOR_VIS     = "coral"      # visual stimulation
COLOR_CELL_EX = "red"        # exemplar cell (filled)
COLOR_CELL_BG = "cyan"       # background cells (outline)
COLOR_NULL    = "gray"       # null hypothesis line

# ── GridSpec parameters ────────────────────────────────────────────────────────
WSPACE_DEFAULT = 0.3    # default column spacing
HSPACE_DEFAULT = 0.3    # default row spacing
WSPACE_TIGHT   = 0.25   # tighter spacing
HSPACE_TIGHT   = 0.25   # tighter spacing
