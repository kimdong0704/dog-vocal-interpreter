"""Load one sample of a task from its manifest row: audio, images, targets and labels.

Every path comes from the manifest, relative to get_data_dir().
"""

from woofdata.loader.audio.segment import load_audio_segment
from woofdata.loader.sample import load_sample

__all__ = ["load_audio_segment", "load_sample"]
