"""Turn downloaded datasets into processed files, a manifest and split files.

See run.py for the command line; prepare_one and prepare_all are for notebooks.
"""

from woofdecoder.data.prepare.run import PrepareResult, dry_run, prepare_all, prepare_one

__all__ = ["PrepareResult", "dry_run", "prepare_all", "prepare_one"]
