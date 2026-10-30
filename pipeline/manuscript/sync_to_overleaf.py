"""
Copy pipeline-generated manuscript figures into the Overleaf git clone.

Two separate things live in this repo:
  - figs/paper/*.pdf              -- pipeline figure output (gitignored, fully
                                      regenerable by pipeline/manuscript/fig*.py)
  - pipeline/manuscript/<overleaf_project_id>/figs/...
                                      -- a separate git clone of the live Overleaf
                                      project (git.overleaf.com), tracked in ITS
                                      OWN git history since Overleaf needs actual
                                      PDF bytes to compile

There is no automatic link between the two -- this script is the manual copy
step, made explicit instead of ad hoc. It only copies bytes; it never commits
or pushes. Review with `git status`/`git diff --stat` inside the Overleaf
clone and commit/push yourself when you're actually ready for a checkpoint
(see CLAUDE.md -- don't push on every regeneration, only at deliberate
milestones, to avoid growing Overleaf's git history with every intermediate
tweak).

Usage:
    python pipeline/manuscript/sync_to_overleaf.py [--dry-run]
"""
import argparse
import filecmp
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_FIGS = REPO_ROOT / "figs" / "paper"
OVERLEAF_CLONE = REPO_ROOT / "pipeline" / "manuscript" / "64c94b448ce6028c013b0143"
OVERLEAF_FIGS = OVERLEAF_CLONE / "figs"

# Pipeline output filename -> destination path relative to OVERLEAF_FIGS.
# Only figures actually wired into main.tex (or an explicitly pending
# supplement, like Fig3S1_second_harmonic) belong here -- add an entry once a new figure
# is actually referenced by \includegraphics, not before, so we don't seed
# unused files into the Overleaf project.
MAPPING = {
    "Fig1.pdf": "Fig1.pdf",
    # main.tex's \includegraphics for Figure 1 currently points at the wide
    # variant (the same figure with the cartoons cropped at the coil band,
    # which renders them wider) -- both are kept in sync so switching the
    # \includegraphics back is a one-word edit with no re-upload.
    "Fig1_wide.pdf": "Fig1_wide.pdf",
    "Fig2.pdf": "Fig2.pdf",
    "Fig3.pdf": "Fig3.pdf",
    "Fig4.pdf": "Fig4.pdf",
    # Keys are relative to figs/paper/, which keeps supplements and
    # diagnostics in subfolders.
    "supplement/Fig3S1_second_harmonic.pdf": "supplementary/Fig3S1_second_harmonic.pdf",
    # Only page 2 (every recording pooled) is in the manuscript; page 1
    # (split by occurrence) was cut 2026-10-30.
    "supplement/Fig3S2_all_recordings.pdf": "supplementary/Fig3S2_all_recordings.pdf",
    "diagnostic/Fig3S3_reproducibility.pdf": "supplementary/Fig3S3_reproducibility.pdf",
    "diagnostic/Fig3S4_bucketings.pdf": "supplementary/Fig3S4_bucketings.pdf",
    "appendix/Fig_1_3.pdf": "appendix/Fig_1_3.pdf",
}


def human(n_bytes):
    for unit in ("B", "KB", "MB"):
        if n_bytes < 1024:
            return f"{n_bytes:.0f}{unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f}GB"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report what would change, copy nothing")
    args = parser.parse_args()

    if not OVERLEAF_CLONE.is_dir():
        sys.exit(f"Overleaf clone not found at {OVERLEAF_CLONE}")

    changed, unchanged, missing = [], [], []
    for src_name, dest_rel in MAPPING.items():
        src = PIPELINE_FIGS / src_name
        dest = OVERLEAF_FIGS / dest_rel
        if not src.is_file():
            missing.append(src)
            continue
        if dest.is_file() and filecmp.cmp(src, dest, shallow=False):
            unchanged.append((src_name, dest_rel))
            continue
        changed.append((src, dest, src_name, dest_rel))

    if missing:
        print("MISSING pipeline output (run the corresponding fig*.py first):")
        for src in missing:
            print(f"  {src.relative_to(REPO_ROOT)}")
        print()

    if unchanged:
        print("Unchanged (already in sync):")
        for src_name, dest_rel in unchanged:
            print(f"  {src_name}")
        print()

    if not changed:
        print("Nothing to sync.")
        return

    verb = "Would copy" if args.dry_run else "Copying"
    print(f"{verb}:")
    for src, dest, src_name, dest_rel in changed:
        old_size = human(dest.stat().st_size) if dest.is_file() else "new"
        new_size = human(src.stat().st_size)
        print(f"  {src_name} -> figs/{dest_rel}  ({old_size} -> {new_size})")
        if not args.dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)

    print()
    if args.dry_run:
        print("Dry run -- nothing copied. Re-run without --dry-run to apply.")
    else:
        print(f"Copied. Review before committing:")
        print(f"  cd {OVERLEAF_CLONE.relative_to(REPO_ROOT)}")
        print(f"  git status --short")
        print(f"  git diff --stat")
        print(f"  git add figs && git commit -m '...' && git push origin main")


if __name__ == "__main__":
    main()
