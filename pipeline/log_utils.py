"""
Logging utilities for parallel pipeline runs.

Each worker captures its own stdout, writes the full output + any traceback
to a shared log file (append-mode, safe across processes), and prints a
one-line summary to the console.

Single-experiment runs tee output to both the console and the log.
"""
import io
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


def get_log_path(stage: str) -> Path:
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"{stage}_{ts}.log"


@contextmanager
def _capture_stdout():
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        yield buf
    finally:
        sys.stdout = old


def _append_to_log(log_path: Path, name: str, status: str,
                   captured: str, tb: str) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"[{status}] {name}\n")
        f.write(f"{'=' * 60}\n")
        if captured:
            f.write(captured)
            if not captured.endswith("\n"):
                f.write("\n")
        if tb:
            f.write("\n--- TRACEBACK ---\n")
            f.write(tb)


def run_with_logging(fn, name: str, log_path: Path, quiet: bool = True) -> bool:
    """Run fn(), routing output to log_path.

    Parameters
    ----------
    fn      : callable  zero-argument function to run
    name    : str       experiment name shown in log and console summary
    log_path: Path      log file to append to (created if missing)
    quiet   : bool      if True, suppress stdout on console (--all mode);
                        if False, tee full output to console too (--experiment mode)

    Returns True on success, False on failure.
    """
    if quiet:
        with _capture_stdout() as buf:
            try:
                fn()
                captured, tb, status = buf.getvalue(), "", "OK"
            except Exception:
                captured, tb, status = buf.getvalue(), traceback.format_exc(), "FAIL"
    else:
        try:
            fn()
            captured, tb, status = "", "", "OK"
        except Exception:
            captured, tb, status = "", traceback.format_exc(), "FAIL"
            print(tb, file=sys.stderr)

    _append_to_log(log_path, name, status, captured, tb)

    # Console summary
    tag = "[OK  ]" if status == "OK" else "[FAIL]"
    print(f"{tag} {name}  (see {log_path.name})", flush=True)
    if status == "FAIL":
        last = next(
            (l.strip() for l in reversed(tb.splitlines()) if l.strip()), "unknown")
        print(f"       {last}", flush=True)

    return status == "OK"
