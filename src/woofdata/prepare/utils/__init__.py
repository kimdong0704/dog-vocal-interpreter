"""Shared helpers for the per-source converters.

Everything here is pure Python (no ffmpeg, no shell) so prepare gives identical results
on Windows and macOS. File lists are always sorted, and every path written to disk is
relative to the data folder with forward slashes.
"""
