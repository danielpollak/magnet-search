# Shared formatting parameters for all manuscript figures.
# Import as: import format_parameters as FP

# ── Font ───────────────────────────────────────────────────────────────────────
FONT_FAMILY = "arial"
FS_BODY     = 6      # default body text (figs 1, 3)
FS_BODY_LG  = 8      # large body text (fig 2)
FS_BODY_XL  = 10     # extra-large body text (fig 4)
FS_PANEL    = 11     # subfigure label (A, B, C…)
FS_TITLE    = 8      # axis title
FS_LEGEND   = 5      # legend text (figs 2, 4)
FS_LEGEND_LG = 7     # larger legend (fig 1)

# ── Figure dimensions (figsize) ────────────────────────────────────────────────
FIGSIZE_FIG1 = (9, 6)       # composite NPIX + GCaMP + ECDF
FIGSIZE_FIG2 = (6.5, 6.5)   # excess counts + distributions
FIGSIZE_FIG3 = (6, 3)       # p/q-value uniformity
FIGSIZE_FIG4 = (6.5, 4)     # modulation sensitivity

# ── File paths ────────────────────────────────────────────────────────────────────
OUT_DIR          = "../../figs/paper"
DATA_DIR         = "../../data"
PARQUET_PATH     = "../../data/manuscript/all_fourier_df.parquet"

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
