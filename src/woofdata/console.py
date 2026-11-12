"""Console set-up shared by the download and prepare command lines."""

from __future__ import annotations

import io
import sys

CONSOLE_ENCODING: str = "utf-8"
# A character the console cannot show becomes "?" instead of stopping the program.
CONSOLE_ERRORS: str = "replace"


def use_utf8_output() -> None:
    """Switch stdout and stderr to UTF-8 with errors="replace".

    On a Windows console with a legacy code page (e.g. cp949 on Korean Windows) printing
    "—" or emoji raises UnicodeEncodeError, which crashed the downloader's summary table
    after the files had already downloaded.
    """
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding=CONSOLE_ENCODING, errors=CONSOLE_ERRORS)
