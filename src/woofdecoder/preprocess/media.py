from __future__ import annotations

from pathlib import Path

import av
import soxr
import numpy as np

from woofdecoder.config import load_config

MEDIA_CONFIG = load_config()["media"]
TARGET_SAMPLE_RATE: int = MEDIA_CONFIG["sample_rate"] # 16k HZ
FRAMES_PER_SEGMENT: int = MEDIA_CONFIG["frames_per_segment"] # 4
RESAMPLE_QUALITY: str = "HQ" # soxr default quality value
TIME_TOLERANCE_S: float = 0.001 # 0.999 to 1 second


def describe_media(path: str | Path, verbose: bool = True) -> dict:
    with av.open(str(path)) as container:
        info: dict = {
            "duration_s": container.duration / av.time_base if container.duration else 0.0,
            "has_audio": len(container.streams.audio) > 0,
            "has_video": len(container.streams.video) > 0
        }
        if info["has_audio"]:
            audio_stream = container.streams.audio[0]
            info["audio_sample_rate"] = audio_stream.rate
            info["audio_channels"] = audio_stream.channels
        if info["has_video"]:
            video_stream = container.streams.video[0]
            info["video_fps"] = float(video_stream.average_rate or 0)
            info["video_size"] = (video_stream.width, video_stream.height)
    
    if verbose:
        print(f"File: {Path(path).name}")
        print(f"Length: {info['duration_s']:.2f} s")
        if info["has_audio"]:
            print(f"Audio")
            print(f"{info['audio_sample_rate']:,} Hz")
            print(f"{info['audio_channels']} channel(s)")
        if info["has_video"]:
            width, height = info["video_size"]
            print(f"Video")
            print(f"{info['video_fps']:,} fps")
            print(f"{width} , {height}")
    
    return info

def to_mono_16k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    # to shape (num_samples, )
    if samples.ndim == 2:
        mono = samples.mean(axis=1)
    else:
        mono = samples
        
    # change to 16k mono if audio sample is not 
    if sample_rate != TARGET_SAMPLE_RATE:
        mono = soxr.resample(mono, sample_rate, TARGET_SAMPLE_RATE, quality=RESAMPLE_QUALITY)
        
    return mono.astype(np.float32, copy=False) 

def read_audio_track(path: str | Path) -> tuple[np.ndarray, int] | None:
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        sample_rate = stream.rate
        
        converter = av.AudioResampler(format="fltp")
        chunks: list[np.ndarray] = []
        for frame in container.decode(stream):
            for converted in converter.resample(frame):
                chunks.append(converted.to_ndarray()) # shape (num_channels, chunk_samples)
        for converted in converter.resample(None): 
            chunks.append(converted.to_ndarray())
        
    if not chunks:
        return None
    samples = np.concatenate(chunks, axis=1).T # shape (num_samples, num_channels)
    return samples.astype(np.float32, copy=False), sample_rate

def load_audio_16k(path: str | Path, verbose: bool = True) -> np.ndarray | None:
    if verbose:
        print(f"Loading audio from {Path(path).name}")
    
    track = read_audio_track(path)
    if track is None:
        if verbose:
            print("No audio track")
        return None
    
    samples, sample_rate = track
    if verbose:
        print(f"Read {samples.shape[0]:,} samples, {samples.shape[1]} channels, {sample_rate:,} Hz")
    
    mono = to_mono_16k(samples, sample_rate)
    if verbose:
        seconds = len(mono) / TARGET_SAMPLE_RATE
        print(f"Converted to mono {TARGET_SAMPLE_RATE:,} Hz: {len(mono):,}, samples = {seconds:.2f} s")
    
    return mono

def frame_times_in_segment(
    start_s: float,
    end_s: float,
    num_frames: int = FRAMES_PER_SEGMENT
) -> list[float]:
    slice_s = (end_s - start_s) / num_frames
    return [start_s + (index + 0.5) * slice_s for index in range(num_frames)]

def read_frame_at(container, stream, time_s: float) -> np.ndarray | None:
    container.seek(int(time_s / stream.time_base), stream=stream, backward=True)
    
    chosen_frame = None
    
    for frame in container.decode(stream):
        chosen_frame = frame
        if frame.time is not None and frame.time >= time_s -TIME_TOLERANCE_S:
            break
        
    if chosen_frame is None:
        return None
    
    return chosen_frame.to_ndarray(format="rgb24")

def sample_frames(
    path: str | Path,
    start_s: float,
    end_s: float,
    num_frames: int = FRAMES_PER_SEGMENT,
    verbose: bool = True
) -> tuple[list[np.ndarray], list[float]]:
    wanted_times = frame_times_in_segment(start_s, end_s, num_frames)
    if verbose:
        print(f"Sample {num_frames} frames from {start_s:.3f}-{end_s:.3f} s")
        print(f"Frame times: {[round(t, 3) for t in wanted_times]}")
    
    frames: list[np.ndarray] = []
    frame_times_s: list[float] = []
    with av.open(str(path)) as container:
        if not container.streams.video:
            if verbose:
                print("No video track")
            return frames, frame_times_s
        
        stream = container.streams.video[0]
        for index, time_s in enumerate(wanted_times, start=1):
            image = read_frame_at(container, stream, time_s)
            if image is None:
                if verbose:
                    print(f"Frame {index}/{num_frames} at {time_s:.3f} s, failed")
                continue
            
            frames.append(image)
            frame_times_s.append(time_s)
            
            if verbose:
                print(f"Frame {index}/{num_frames} at {time_s:.3f} s: {image.shape}")
    
    return frames, frame_times_s