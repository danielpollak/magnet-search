"""
Analysis stage entry point.

Usage:
    python pipeline/analysis.py --experiment 20230413_firstsite
    python pipeline/analysis.py --all [--workers 4]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXPERIMENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments")


def _analyze_one(args):
    """Worker function — accepts (yaml_path, log_path, quiet) tuple for pool.map."""
    yaml_path, log_path, quiet = args
    from pathlib import Path
    from pipeline.schema import load_experiment
    from pipeline.dispatch import run_analysis
    from pipeline.log_utils import run_with_logging

    cfg = load_experiment(yaml_path)

    NO_PROCESSING_PARADIGMS = {"engert", "manual"}
    if cfg.paradigm not in NO_PROCESSING_PARADIGMS and not os.path.exists(cfg.processing_path()):
        print(f"[analysis] {cfg.name}: processing output not found, skipping", flush=True)
        return

    def _run():
        print(f"[analysis] {cfg.name} ({cfg.paradigm})")
        run_analysis(cfg)
        print(f"[analysis] {cfg.name} done -> {cfg.analysis_path()}")

    run_with_logging(_run, cfg.name, Path(log_path), quiet=quiet)


def main():
    parser = argparse.ArgumentParser(description="Run analysis stage")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment", help="Experiment name (without .yml)")
    group.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (--all only)")
    parser.add_argument("--filter", dest="filter_", metavar="SUBSTR",
                        help="Only run experiments whose name contains SUBSTR (--all only)")
    args = parser.parse_args()

    from pathlib import Path
    from pipeline.log_utils import get_log_path
    log_path = get_log_path("analysis")
    print(f"Logging to {log_path}")

    if args.experiment:
        yaml_path = os.path.join(EXPERIMENTS_DIR, f"{args.experiment}.yml")
        if not os.path.exists(yaml_path):
            print(f"Error: {yaml_path} not found")
            sys.exit(1)
        _analyze_one((yaml_path, str(log_path), False))
    else:
        import glob
        yaml_paths = sorted(glob.glob(os.path.join(EXPERIMENTS_DIR, "*.yml")))
        if args.filter_:
            yaml_paths = [p for p in yaml_paths if args.filter_ in os.path.basename(p)]
        if not yaml_paths:
            print(f"No YAML files found in {EXPERIMENTS_DIR}" +
                  (f" matching '{args.filter_}'" if args.filter_ else ""))
            sys.exit(1)
        worker_args = [(p, str(log_path), True) for p in yaml_paths]
        if args.workers > 1:
            from multiprocessing import Pool
            with Pool(args.workers) as pool:
                pool.map(_analyze_one, worker_args)
        else:
            for a in worker_args:
                _analyze_one(a)


if __name__ == "__main__":
    main()
