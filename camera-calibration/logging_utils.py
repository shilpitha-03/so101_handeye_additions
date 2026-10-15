"""
calibration/logging_utils.py
============================
Central logging configuration for the calibration pipeline.

Usage
-----
From main.py (session run):
    from calibration.logging_utils import setup_logging
    setup_logging(session_dir=session_dir)   # logs to results/session_XXX/session.log

From any standalone script (e.g. python calibration/so101_fk.py):
    The module calls _ensure_logger() at import time automatically.
    Logs go to logs/standalone_YYYYMMDD_HHMMSS.log under the project root.

Getting a module logger:
    from calibration.logging_utils import get_logger
    logger = get_logger(__name__)   # e.g. "calibration_pipeline.calib3"

Rules
-----
- Console (StreamHandler): INFO and above only.
  Shows prompts, capture confirmations, warnings, errors.
- Log file (FileHandler): DEBUG and above.
  Shows everything: raw ticks, matrices, reprojection errors, all numbers.
- setup_logging() is idempotent — safe to call multiple times, only acts once.
- All modules use child loggers under "calibration_pipeline" so a single
  file handler on the root logger captures everything.
"""

import logging
import pathlib
import datetime

# ── Constants ─────────────────────────────────────────────────────────────────

ROOT_LOGGER_NAME = "calibration_pipeline"

# Format: timestamp [LEVEL  ] [module_name     ] message
# Fixed-width level (7 chars) and module name (16 chars) for readable log files.
_FILE_FORMAT    = "%(asctime)s.%(msecs)03d [%(levelname)-7s] [%(name)-28s] %(message)s"
_CONSOLE_FORMAT = "%(asctime)s [%(levelname)-7s] %(message)s"
_DATE_FORMAT    = "%Y-%m-%d %H:%M:%S"

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_LOGS_DIR     = _PROJECT_ROOT / "logs"

_configured = False   # guard against double-setup


# ── Public API ────────────────────────────────────────────────────────────────

def setup_logging(session_dir: pathlib.Path | None = None) -> pathlib.Path:
    """
    Configure the root pipeline logger.  Call once at process start.

    Parameters
    ----------
    session_dir : pathlib.Path or None
        If provided (called from main.py), the log file is written to
        session_dir / "session.log".
        If None (standalone run), a timestamped file is created under
        project_root/logs/.

    Returns
    -------
    pathlib.Path — path to the log file that was opened.
    """
    global _configured
    if _configured:
        return _get_existing_log_path()

    root = logging.getLogger(ROOT_LOGGER_NAME)
    root.setLevel(logging.DEBUG)   # capture everything; handlers filter by level

    # ── Determine log file path ───────────────────────────────────────────────
    if session_dir is not None:
        log_path = pathlib.Path(session_dir) / "session.log"
    else:
        _LOGS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = _LOGS_DIR / f"standalone_{ts}.log"

    # ── File handler — DEBUG and above ────────────────────────────────────────
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT))

    # ── Console handler — INFO and above ─────────────────────────────────────
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT))

    root.addHandler(fh)
    root.addHandler(ch)

    # Suppress noisy third-party loggers at WARNING level so they do not
    # pollute our log file with serial-port debug spam.
    for noisy in ("lerobot", "serial", "scservo_sdk", "scs_sdk"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True

    # Log the header immediately so every log file is self-describing.
    root.info("=" * 70)
    root.info("Calibration pipeline — logging started")
    root.info(f"Log file : {log_path}")
    root.info("=" * 70)

    return log_path


def get_logger(module_name: str) -> logging.Logger:
    """
    Return a child logger under the pipeline root.

    Parameters
    ----------
    module_name : str
        Pass __name__ from the calling module.
        If the name already starts with ROOT_LOGGER_NAME, it is used as-is.
        Otherwise it is prefixed: "calibration_pipeline.<module_name>".

    Examples
    --------
    In calib3_handeye_wrist.py:
        logger = get_logger(__name__)
        # → logging.getLogger("calibration_pipeline.calib3_handeye_wrist")

    In so101_fk.py:
        logger = get_logger(__name__)
        # → logging.getLogger("calibration_pipeline.so101_fk")
    """
    _ensure_logger()   # auto-setup if called standalone before setup_logging()

    if module_name.startswith(ROOT_LOGGER_NAME):
        name = module_name
    else:
        # Strip leading "calibration." prefix if present (avoids double nesting)
        short = module_name.removeprefix("calibration.")
        name  = f"{ROOT_LOGGER_NAME}.{short}"

    return logging.getLogger(name)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ensure_logger() -> None:
    """
    Called at import time by every module.
    If setup_logging() has not been called yet (standalone run), initialise
    with a timestamped standalone log file so nothing is lost.
    """
    if not _configured:
        setup_logging(session_dir=None)


def _get_existing_log_path() -> pathlib.Path:
    """Return the path of the FileHandler already attached to the root logger."""
    root = logging.getLogger(ROOT_LOGGER_NAME)
    for h in root.handlers:
        if isinstance(h, logging.FileHandler):
            return pathlib.Path(h.baseFilename)
    return _LOGS_DIR   # fallback, should never happen