"""
Main orchestrator — full pipeline from calibration to Isaac Sim values.

Run order:
  Step 1   — Calib 1  : side-view camera intrinsics
  Step 2   — Calib 2  : wrist camera intrinsics
  Step 3   — Calib 3  : wrist camera hand-eye calibration
  Step 4   — Phase 1  : side-view pose capture
  Step 5   — Phase 2  : transform chain
  Step 6   — Phase 3  : Isaac Sim axis conversion

To run a single step independently:
  Import and call its run() directly — each module loads its own inputs from disk.

To skip a step (e.g. calibration already done):
  Comment out that block below — phases read from calibration_files/ automatically.

Log file
--------
  Every run writes to results/session_XXXXX/session.log
  Console shows INFO and above. Log file contains everything (DEBUG level).
"""

import pathlib
import datetime

from logging_utils import setup_logging, get_logger
from config import RESULTS_BASE_DIR


def create_session_dir() -> pathlib.Path:
    """Create and return a timestamped session folder under results/."""
    timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = pathlib.Path(RESULTS_BASE_DIR) / f"session_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def main():
    # ── Session folder + logging ──────────────────────────────────────────────
    # Session dir created first so the log file goes inside it.
    session_dir = create_session_dir()
    log_path    = setup_logging(session_dir=session_dir)

    logger = get_logger(__name__)
    logger.info("Session folder : %s", session_dir)
    logger.info("Log file       : %s", log_path)

    # ── Lazy imports (after logging is configured) ────────────────────────────
    from calibration.calib1_intrinsics_sideview import run as calib1
    from calibration.calib2_intrinsics_wrist    import run as calib2
    from calibration.calib3_handeye_wrist       import run as calib3
    from extraction.extract_calibration         import run as calib1_extraction
    from extraction.convert_verify              import run as convert_verify
    from phases.phase1_capture                  import run as phase1
    from phases.phase2_transform                import run as phase2
    from phases.phase3_conversion               import run as phase3

    # ── Step 1: Side-view camera intrinsics ───────────────────────────────────
    # calib1()

    # ── Step 1.1 / 1.2: Sensor extraction ────────────────────────────────────
    # calib1_extraction()
    # convert_verify()

    # ── Step 2: Wrist camera intrinsics ──────────────────────────────────────
    # calib2()

    # ── Step 3: Wrist hand-eye calibration ───────────────────────────────────
    # calib3()

    # ── Steps 4–6: Pose pipeline ─────────────────────────────────────────────
    p1_data = phase1(session_dir)
    p2_data = phase2(session_dir, phase1_data=p1_data)
    p3_data = phase3(session_dir, phase2_data=p2_data)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()