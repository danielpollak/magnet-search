"""Fig 3 supplement — p-value and q-value uniformity across neuron populations,
2F harmonic (`2f_p_value`/`sens_2f`) instead of Fig3's 1F (`p_value`/`sens`).

Reuses fig3.py's `plot_fig3` unchanged (see its `pval_col`/`sens_col`/`out_name`/
`suptitle` parameters) -- same waves, same conditions, same layout, just reading
the 2F columns instead of the 1F ones. `all_fourier_df`'s `2f_p_value`/`sens_2f`
are NaN for any (rec, freq) group with no paired 2F harmonic (e.g. medaka's
independent mag/visual groups, or 2F skipped past Nyquist -- see CLAUDE.md);
those rows flow through exactly as NaN `p_value` rows already do in Fig3 itself
(not dropped, `storey_qvalues`/`np.percentile` treat them the same way there
too) -- this script deliberately does not special-case them, so the 1F and 2F
figures stay an apples-to-apples comparison.

Requires:
  data/manuscript/all_fourier_df.parquet  (run python pipeline/aggregate.py first)
  ecdfbounds library

Usage:
    python pipeline/manuscript/fig3_supp.py
    python pipeline/manuscript/fig3_supp.py --out-dir figs/paper
"""
import argparse
from pathlib import Path

# Detect if running in Jupyter notebook (must do this before matplotlib.use)
try:
    from IPython import get_ipython
    in_notebook = get_ipython() is not None
except ImportError:
    in_notebook = False

import matplotlib
if not in_notebook:
    matplotlib.use("Agg")
import pandas as pd

from fig3 import plot_fig3

import format_parameters as FP


def main():
    parser = argparse.ArgumentParser(description="Generate Fig 3 supplement (2F p/q-value uniformity)")
    parser.add_argument("--out-dir", default=FP.OUT_DIR, help="Output directory for PDFs")
    parser.add_argument("--parquet", default=FP.PARQUET_PATH,
                        help=f"Path to all_fourier_df.parquet (default: {FP.PARQUET_PATH})")
    args = parser.parse_args([] if in_notebook else None)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.parquet} ...")
    all_fourier_df = pd.read_parquet(args.parquet)
    plot_fig3(all_fourier_df, out_dir, pval_col="2f_p_value", sens_col="sens_2f",
              out_name="Fig3_supp.pdf", suptitle="2F harmonic")


if __name__ == "__main__":
    if in_notebook:
        get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")
    main()
