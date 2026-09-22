"""Readers for the MCD-rPPG file formats.

Every format here was confirmed by opening real files on 2026-09-23 - none is assumed
from the paper or the dataset card. The quirks the readers handle are real and were
found by inspection:

  ecg/*.json      Medmon/OData export. 500 Hz, 12 leads, 30 s (dataX spans 0-29.998 s).
                  Lead `I` has 14900 samples while all 11 others have 15000, so the
                  leads CANNOT be stacked naively and `dataX` cannot be zipped to lead I.
  meta/*.txt      Per-frame timestamps, `{frame_index}\\t{ISO timestamp}`, index from 1,
                  after one blank line. 180 s per recording. The frame rate is NOT
                  constant: gaps reach 0.645 s and some files skip frame indices.
  ppg/*.PW        Text, CRLF, `{value}   {ISO timestamp}`. 8-bit values (0-255),
                  ~18,050 samples over ~180.5 s = 100 Hz ON AVERAGE ONLY - samples
                  arrive in bursts (1-3 ms apart, then ~14-16 ms), so the timestamps
                  are not a uniform clock. Resample before any HRV/spectral work.
  video/*.avi     Opened with PyAV; container fps is compared against the fps implied
                  by the matching meta/ timestamps.

`.PW` and `meta/` share one wall clock, which is what makes PPG-to-frame alignment
possible at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CLIP_LOW, CLIP_HIGH = 0, 255  # the 8-bit rails of the pulse-wave device


def _read_stamped_text(path: Path, value_name: str) -> pd.DataFrame:
    """Parse the shared `{value}<whitespace>{ISO date} {ISO time}` layout."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=[value_name, "date", "time"],
        skip_blank_lines=True,
        encoding="utf-8",
    )
    # format="ISO8601" is required, not cosmetic: most timestamps carry microseconds
    # but some land exactly on the second ("2023-10-23 13:38:02"). Letting pandas
    # infer the format from the first row raises partway through those files.
    df["timestamp"] = pd.to_datetime(
        df.pop("date") + " " + df.pop("time"), format="ISO8601"
    )
    return df


# --------------------------------------------------------------------------- ECG
@dataclass(frozen=True)
class Ecg:
    frequency: int
    leads: dict[str, np.ndarray]
    time: np.ndarray
    segments: dict
    path: Path

    @property
    def lead_lengths(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.leads.items()}

    @property
    def common_length(self) -> int:
        return min(len(v) for v in self.leads.values())

    @property
    def ragged(self) -> bool:
        """True when leads differ in length - always the case here (lead I is short)."""
        return len(set(self.lead_lengths.values())) > 1

    def matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """(12, n) array + matching time base, truncated to the shortest lead.

        Truncation is the only safe way to stack these leads; doing it here keeps
        every caller from re-discovering the lead-I quirk.
        """
        n = self.common_length
        return np.vstack([self.leads[k][:n] for k in self.leads]), self.time[:n]

    @property
    def has_segments(self) -> bool:
        """Only some exports carry computed PQ/QT/ST intervals."""
        return bool(self.segments.get("qtAverageValue"))


def read_ecg(path: str | Path) -> Ecg:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    leads = {c["title"]: np.asarray(c["values"], dtype=np.float64) for c in raw["data"]}
    return Ecg(
        frequency=int(raw["frequency"]),
        leads=leads,
        time=np.asarray(raw["dataX"], dtype=np.float64),
        segments=raw.get("segmentsData") or {},
        path=path,
    )


# -------------------------------------------------------------------------- meta
@dataclass(frozen=True)
class FrameTimestamps:
    frame_index: np.ndarray
    timestamp: np.ndarray  # datetime64[ns]
    path: Path
    _dt: np.ndarray = field(repr=False)

    @property
    def n_frames(self) -> int:
        return len(self.frame_index)

    @property
    def duration_s(self) -> float:
        return float((self.timestamp[-1] - self.timestamp[0]) / np.timedelta64(1, "s"))

    @property
    def fps_mean(self) -> float:
        return self.n_frames / self.duration_s

    @property
    def fps_median(self) -> float:
        return 1.0 / float(np.median(self._dt))

    @property
    def index_contiguous(self) -> bool:
        """False when the capture dropped frames outright."""
        return bool(
            np.array_equal(np.diff(self.frame_index), np.ones(self.n_frames - 1))
        )

    def gaps(self, factor: float = 2.5) -> np.ndarray:
        """Indices where the gap to the next frame exceeds `factor` x the median."""
        return np.flatnonzero(self._dt > factor * np.median(self._dt))


def read_meta(path: str | Path) -> FrameTimestamps:
    path = Path(path)
    df = _read_stamped_text(path, "frame_index")
    ts = df["timestamp"].to_numpy()
    return FrameTimestamps(
        frame_index=df["frame_index"].to_numpy(),
        timestamp=ts,
        path=path,
        _dt=np.diff(ts) / np.timedelta64(1, "s"),
    )


# ---------------------------------------------------------------------------- PW
@dataclass(frozen=True)
class PulseWave:
    value: np.ndarray  # 8-bit ints as recorded
    timestamp: np.ndarray  # datetime64[ns]
    path: Path

    @property
    def n_samples(self) -> int:
        return len(self.value)

    @property
    def duration_s(self) -> float:
        return float((self.timestamp[-1] - self.timestamp[0]) / np.timedelta64(1, "s"))

    @property
    def mean_rate_hz(self) -> float:
        """The nominal 100 Hz holds only as an average - see the module docstring."""
        return self.n_samples / self.duration_s

    @property
    def clipped_fraction(self) -> float:
        """Samples resting on the 8-bit rails; measured at ~0.31% across 30 files."""
        return float(np.mean((self.value <= CLIP_LOW) | (self.value >= CLIP_HIGH)))

    def resample(self, fs: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate onto a uniform `fs` grid.

        Required before any spectral or HRV analysis: the raw samples arrive in
        bursts, so treating them as evenly spaced would distort the spectrum.
        """
        t = (self.timestamp - self.timestamp[0]) / np.timedelta64(1, "s")
        grid = np.arange(0.0, t[-1], 1.0 / fs)
        return grid, np.interp(grid, t, self.value.astype(np.float64))


def read_pw(path: str | Path) -> PulseWave:
    path = Path(path)
    df = _read_stamped_text(path, "value")
    return PulseWave(
        value=df["value"].to_numpy(),
        timestamp=df["timestamp"].to_numpy(),
        path=path,
    )


# ----------------------------------------------------------------------- ppg_sync
@dataclass(frozen=True)
class SyncedPpg:
    """PPG resampled onto the video timeline: one row per matched video frame.

    Column 2 is the |time difference| between that video frame and the PPG sample
    chosen for it - a per-row sync-quality measure, median ~3.5-4 ms.

    Row i corresponds to frame i of the ENCODED VIDEO: for all three sample videos
    the row count equals the decoded frame count exactly (5383, 5377, 4290).
    It does NOT match `meta/`, which lists more timestamps - frames captured but never
    written to the .avi. Measured over 250 recordings: `meta` has 0 to 71 more rows
    (median 20), and never fewer. Align this to the video, not to meta/.
    """

    value: np.ndarray
    sync_error_s: np.ndarray
    path: Path

    @property
    def n_rows(self) -> int:
        return len(self.value)

    @property
    def median_sync_error_s(self) -> float:
        return float(np.median(self.sync_error_s))

    def poor_sync(self, threshold_s: float = 0.02) -> np.ndarray:
        """Rows whose frame/PPG mismatch exceeds `threshold_s` (~0.01-0.02% of rows)."""
        return np.flatnonzero(self.sync_error_s > threshold_s)


# ------------------------------------------------------------------------- video
@dataclass(frozen=True)
class VideoInfo:
    codec: str
    width: int
    height: int
    container_fps: float
    n_frames: int
    duration_s: float
    path: Path

    @property
    def measured_fps(self) -> float:
        """Frames actually decoded over the container duration."""
        return self.n_frames / self.duration_s


def video_info(path: str | Path, count_frames: bool = True) -> VideoInfo:
    """Container metadata via PyAV, optionally decoding to count real frames.

    The container's declared fps is a nominal rational (e.g. 30/1); the `meta/`
    timestamps show the true capture rate drifts, so both are reported rather than
    trusting the header.
    """
    import av  # imported lazily: only the video path needs PyAV

    path = Path(path)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        ctx = stream.codec_context
        declared = float(stream.average_rate) if stream.average_rate else float("nan")
        n = stream.frames
        if count_frames and not n:
            n = sum(1 for _ in container.decode(video=0))
        # These AVIs carry no container duration, so fall back to the stream's own
        # duration, then to frames/fps. Never leave it NaN silently.
        if container.duration:
            duration = float(container.duration / 1_000_000)
        elif stream.duration and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        else:
            duration = n / declared if declared else float("nan")
        return VideoInfo(
            codec=ctx.name,
            width=ctx.width,
            height=ctx.height,
            container_fps=declared,
            n_frames=int(n),
            duration_s=duration,
            path=path,
        )


def read_ppg_sync(path: str | Path) -> SyncedPpg:
    path = Path(path)
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["value", "sync_error_s"],
        skip_blank_lines=True,
        encoding="utf-8",
    )
    return SyncedPpg(
        value=df["value"].to_numpy(),
        sync_error_s=df["sync_error_s"].to_numpy(dtype=np.float64),
        path=path,
    )
