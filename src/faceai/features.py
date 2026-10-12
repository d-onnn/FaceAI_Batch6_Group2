"""Phase 3 - per-session features from the contact PPG (.PW) waveform.

    python -m faceai.features        # or: make features

One row per SESSION (patient_id, step). The .PW file is shared by all three cameras of a
session, so features are computed once per session, never per recording. Only DEV
sessions are processed: the held-out test set is sealed until Phase 10, so its PPG files
are never opened here.

Every feature is computed from first principles on the uniformly resampled waveform, so
each one can be explained and checked:

  heart rate      mean of validated inter-beat intervals (IBIs), and the Welch spectral peak
  variability     SDNN, RMSSD, pNN50 over validated IBIs (pulse-rate variability, not ECG HRV)
  respiration     rate from IBI (RIFV), beat-amplitude (RIAV) and baseline (RIIV) modulation,
                  plus their median - a simple fusion, not Charlton's quality-weighted one
  morphology      rise time, half-height width, systolic fraction and the second-derivative
                  b/a ratio of the ensemble-averaged beat
  quality         template-match correlation, skewness, kurtosis, beat-amplitude CV

Deliberately NOT computed: the APG c/d/e waves and the aging index. Their detection on an
8-bit, 100 Hz signal has not been validated here, and an unvalidated feature is worse than
none. They belong to the Phase 5 feature bank. Nor is a perfusion index (AC/DC): the raw
output is inverted 8-bit counts with no physical DC reference.

Noise injection (robustness test): Gaussian noise band-limited to the pulse band
(0.5-8 Hz, the same band the extractor keeps) is added to the resampled waveform, scaled so
that the IN-BAND SNR - pulsatile power over added-noise power, both within 0.5-8 Hz - is
exactly the stated level. White noise was tried first and rejected: across 0-50 Hz only
~15% of its power lands in the pulse band, so even a nominal 0 dB moved heart rate by
<0.4 bpm and the test exercised nothing. Levels run from +20 dB (near-clean) to -10 dB,
where heart rate itself is lost. Every draw is seeded from (SEED, subject, state, SNR),
so results reproduce exactly.
"""

from __future__ import annotations

import datetime as dt
import json
import warnings
from itertools import pairwise

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import interpolate, ndimage, signal, stats

from faceai.data import audit
from faceai.data import formats as F
from faceai.data.hub import LOCAL_DIR, ROOT
from faceai.splits import SEED, load_splits

FS = 100.0  # uniform resampling rate (the .PW device's average rate)
HR_BAND_HZ = audit.HR_BAND_HZ  # 0.7-4.0 Hz, 42-240 bpm
IBI_RANGE_S = (60.0 / 180.0, 60.0 / 40.0)  # plausible beats: 40-180 bpm
IBI_LOCAL_TOL = 0.25  # reject an IBI more than 25% from its rolling median (artefact)
IBI_LOCAL_WINDOW = 11  # beats in that rolling median
# Respiration: search 12-30 breaths/min after a 0.2-0.6 Hz band-pass. The band is the
# physiological prior around the paper's 15-24 label range. It was adopted after comparing
# five bands on 150 dev sessions (reports/03_cv_robustness.md): without the band-pass, a
# 6-30 search collapses onto the 1/f drift of the beat series (median ~9 breaths/min).
RESP_BAND_HZ = (0.2, 0.5)
RESP_PREFILTER_HZ = (0.2, 0.6)
RESP_FS = 4.0  # beat series are interpolated onto this grid for spectral analysis
MIN_BEATS = 30  # fewer valid beats than this and the session's features are NaN
ONSET_SEARCH_S = 0.4  # the beat onset is the minimum within this window before a peak
SNR_LEVELS_DB = (20, 10, 5, 0, -5, -10)  # in-band, see the module docstring
NOISE_BAND_HZ = (0.5, 8.0)  # the elgendi cleaning band

OUT_DIR = ROOT / "artifacts" / "features"
STATE_CODE = {"before": 0, "after": 1}

# Model inputs. Bookkeeping columns (n_peaks, valid_ibi_frac, clipped_frac) are kept in
# the table for transparency but are not fed to models.
MODEL_FEATURES = [
    "hr_mean_bpm",
    "hr_spectral_bpm",
    "ibi_sdnn_ms",
    "ibi_rmssd_ms",
    "ibi_pnn50_pct",
    "resp_rate_rifv",
    "resp_rate_riav",
    "resp_rate_riiv",
    "resp_rate_fused",
    "rise_time_s",
    "width50_s",
    "systolic_frac",
    "apg_b_a",
    "amp_cv",
    "skewness",
    "kurtosis",
    "sqi_template_r",
]
BOOKKEEPING = ["n_peaks", "valid_ibi_frac", "clipped_frac"]
ALL_FEATURES = MODEL_FEATURES + BOOKKEEPING


def features_path(snr_db: float | None = None):
    name = (
        "ppg_clean.parquet" if snr_db is None else f"ppg_snr{int(snr_db):+d}db.parquet"
    )
    return OUT_DIR / name


# ------------------------------------------------------------------- signal helpers
def load_session_ppg(patient_id: int, step: str) -> np.ndarray:
    """The session's contact PPG on a uniform FS grid, INVERTED to physiological polarity.

    The raw .PW waveform is upside down: on 40 sampled dev sessions (verified
    2026-10-12) its median trough-to-peak time is 0.545 s and peak-to-trough 0.140 s,
    the reverse of a real pulse (fast systolic upstroke, slow diastolic decay) in
    every one. `CLIP_HIGH - raw` restores the physiological shape and keeps values
    in 0-255. Rate features are unaffected by polarity; morphology features are not.
    """
    _, v = F.read_pw(LOCAL_DIR / f"ppg/{patient_id}_{step}.PW").resample(FS)
    return F.CLIP_HIGH - v


def _pulsatile(x: np.ndarray, fs: float = FS) -> np.ndarray:
    import neurokit2 as nk

    return np.asarray(nk.ppg_clean(x, sampling_rate=fs, method="elgendi"))


def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """x plus pulse-band Gaussian noise at an exact in-band SNR (see the module docstring)."""
    p_signal = float(np.var(_pulsatile(x)))
    lo, hi = NOISE_BAND_HZ
    b, a = signal.butter(3, [lo / (FS / 2), hi / (FS / 2)], btype="band")
    noise = signal.filtfilt(b, a, rng.normal(0.0, 1.0, size=x.shape))
    noise *= np.sqrt(p_signal / 10.0 ** (snr_db / 10.0) / np.var(noise))
    return x + noise


def noise_rng(patient_id: int, step: str, snr_db: float) -> np.random.Generator:
    # SeedSequence entropy must be non-negative, hence the offset on the (signed) SNR.
    return np.random.default_rng(
        [SEED, int(patient_id), STATE_CODE[step], int(snr_db) + 100]
    )


def _spectral_peak_hz(v: np.ndarray, fs: float, band: tuple[float, float]) -> float:
    v = signal.detrend(v)
    freqs, power = signal.welch(v, fs=fs, nperseg=int(min(len(v), fs * 64)))
    keep = (freqs >= band[0]) & (freqs <= band[1])
    if not keep.any() or not np.isfinite(power[keep]).all():
        return np.nan
    return float(freqs[keep][np.argmax(power[keep])])


def _resp_rate(t: np.ndarray, v: np.ndarray) -> float:
    """Breaths/min from a beat-by-beat series v sampled at irregular beat times t."""
    if len(t) < MIN_BEATS:
        return np.nan
    grid = np.arange(t[0], t[-1], 1.0 / RESP_FS)
    if len(grid) < 4 * RESP_FS * 10:  # need at least ~10 s of series
        return np.nan
    series = signal.detrend(interpolate.interp1d(t, v, kind="linear")(grid))
    lo, hi = RESP_PREFILTER_HZ
    b, a = signal.butter(2, [lo / (RESP_FS / 2), hi / (RESP_FS / 2)], btype="band")
    hz = _spectral_peak_hz(signal.filtfilt(b, a, series), RESP_FS, RESP_BAND_HZ)
    return 60.0 * hz


def _valid_ibis(ibi: np.ndarray) -> np.ndarray:
    lo, hi = IBI_RANGE_S
    ok = (ibi >= lo) & (ibi <= hi)
    local = ndimage.median_filter(ibi, size=IBI_LOCAL_WINDOW, mode="nearest")
    return ok & (np.abs(ibi - local) <= IBI_LOCAL_TOL * local)


def _first_crossing(y: np.ndarray, level: float, start: int, step: int) -> int | None:
    """First index from `start`, walking by `step`, where y drops below `level`."""
    i = start
    while 0 <= i < len(y):
        if y[i] < level:
            return i
        i += step
    return None


# ------------------------------------------------------------------------ extractor
def extract(x: np.ndarray, fs: float = FS) -> dict[str, float]:
    """All features for one session's resampled contact PPG. NaN where undefined."""
    import neurokit2 as nk

    out = dict.fromkeys(ALL_FEATURES, np.nan)
    out["clipped_frac"] = float(
        np.mean((x <= F.CLIP_LOW + 0.5) | (x >= F.CLIP_HIGH - 0.5))
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clean = _pulsatile(x, fs)
        try:
            peaks = np.asarray(
                nk.ppg_findpeaks(clean, sampling_rate=fs, method="elgendi")[
                    "PPG_Peaks"
                ],
                dtype=int,
            )
        except Exception:  # noqa: BLE001 - a failed detection is a NaN row, not a crash
            peaks = np.array([], dtype=int)
    out["n_peaks"] = float(len(peaks))
    if len(peaks) < MIN_BEATS + 1:
        return out

    ibi = np.diff(peaks) / fs
    ok = _valid_ibis(ibi)
    out["valid_ibi_frac"] = float(ok.mean())
    if ok.sum() < MIN_BEATS:
        return out

    # -- rate and variability (validated IBIs only; successive pairs both valid)
    good = ibi[ok]
    out["hr_mean_bpm"] = 60.0 / float(good.mean())
    out["hr_spectral_bpm"] = 60.0 * _spectral_peak_hz(clean, fs, HR_BAND_HZ)
    ms = good * 1000.0
    out["ibi_sdnn_ms"] = float(np.std(ms, ddof=1))
    pair = ok[:-1] & ok[1:]
    if pair.sum() >= MIN_BEATS:
        d = np.diff(ibi * 1000.0)[pair]
        out["ibi_rmssd_ms"] = float(np.sqrt(np.mean(d**2)))
        out["ibi_pnn50_pct"] = float(100.0 * np.mean(np.abs(d) > 50.0))

    # -- beat onsets: the minimum in the window before each peak (never past the last peak)
    w = int(ONSET_SEARCH_S * fs)
    onsets = np.full(len(peaks), -1)
    for i, p in enumerate(peaks):
        lo = max(0, p - w, peaks[i - 1] + 1 if i > 0 else 0)
        if p - lo >= 2:
            onsets[i] = lo + int(np.argmin(clean[lo:p]))
    beat_ok = np.zeros(len(peaks), dtype=bool)
    beat_ok[1:] = ok  # beat i is valid if the IBI ending at it is valid
    beat_ok &= onsets >= 0
    bp, bo = peaks[beat_ok], onsets[beat_ok]
    if len(bp) < MIN_BEATS:
        return out
    amp = clean[bp] - clean[bo]
    t_beat = bp / fs

    # -- respiration from three modulations of the beat series
    rifv = _resp_rate(peaks[1:][ok] / fs, good)
    riav = _resp_rate(t_beat, amp)
    riiv = _resp_rate(t_beat, x[bo])
    out["resp_rate_rifv"], out["resp_rate_riav"], out["resp_rate_riiv"] = (
        rifv,
        riav,
        riiv,
    )
    rates = [r for r in (rifv, riav, riiv) if np.isfinite(r)]
    if rates:
        out["resp_rate_fused"] = float(np.median(rates))

    # -- amplitude features
    if np.all(amp > 0):
        out["amp_cv"] = float(np.std(amp, ddof=1) / np.mean(amp))
    out["skewness"] = float(stats.skew(clean))
    out["kurtosis"] = float(stats.kurtosis(clean))
    out["rise_time_s"] = float(np.median((bp - bo) / fs))

    # -- ensemble-averaged beat: onset-aligned, fixed length = median IBI, peak scaled to 1
    length = int(np.rint(np.median(good) * fs))
    segs = []
    for p, o in zip(bp, bo, strict=True):
        if o + length <= len(clean) and clean[p] > clean[o]:
            seg = clean[o : o + length]
            segs.append((seg - seg[0]) / (clean[p] - seg[0]))
    if len(segs) < MIN_BEATS:
        return out
    segs = np.vstack(segs)
    template = segs.mean(axis=0)
    r = [np.corrcoef(s, template)[0, 1] for s in segs]
    out["sqi_template_r"] = float(np.nanmean(r))

    # Upsample the template to 1 kHz for timing and derivative features.
    t = np.arange(length) / fs
    tt = np.arange(0.0, t[-1], 0.001)
    fine = interpolate.CubicSpline(t, template)(tt)
    ipk = int(np.argmax(fine))
    out["systolic_frac"] = float(ipk / len(fine))
    half = 0.5 * fine[ipk]
    up = _first_crossing(fine, half, ipk, -1)
    down = _first_crossing(fine, half, ipk, +1)
    if up is not None and down is not None:
        out["width50_s"] = float((down - up) / 1000.0)
    # Second derivative (APG) of the smoothed template: a = first maximum before the
    # systolic peak, b = the minimum that follows it, still before the peak.
    if ipk > 40:
        smooth = signal.savgol_filter(fine, window_length=31, polyorder=3)
        apg = np.gradient(np.gradient(smooth))
        ia = int(np.argmax(apg[:ipk]))
        if apg[ia] > 0 and ipk - ia > 2:
            ib = ia + 1 + int(np.argmin(apg[ia + 1 : ipk]))
            out["apg_b_a"] = float(apg[ib] / apg[ia])
    return out


# ------------------------------------------------------- respiration band comparison
RESP_VARIANTS = {  # name: (search band Hz, band-pass Hz or None)
    "6-30 /min, no band-pass": ((0.1, 0.5), None),
    "6-30 /min, 0.1-0.6 Hz band-pass": ((0.1, 0.5), (0.1, 0.6)),
    "9-30 /min, no band-pass": ((0.15, 0.5), None),
    "9-30 /min, 0.15-0.6 Hz band-pass": ((0.15, 0.5), (0.15, 0.6)),
    "12-30 /min, 0.2-0.6 Hz band-pass (adopted)": ((0.2, 0.5), (0.2, 0.6)),
}


def _variant_rate(t, v, band, prefilter) -> float:
    grid = np.arange(t[0], t[-1], 1.0 / RESP_FS)
    s = signal.detrend(interpolate.interp1d(t, v)(grid))
    if prefilter:
        b, a = signal.butter(2, [f / (RESP_FS / 2) for f in prefilter], btype="band")
        s = signal.filtfilt(b, a, s)
    return 60.0 * _spectral_peak_hz(s, RESP_FS, band)


def _resp_series(patient_id: int, step: str):
    import neurokit2 as nk

    x = load_session_ppg(patient_id, step)
    c = _pulsatile(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pk = np.asarray(
            nk.ppg_findpeaks(c, sampling_rate=FS, method="elgendi")["PPG_Peaks"]
        )
    ibi = np.diff(pk) / FS
    ok = _valid_ibis(ibi)
    w = int(ONSET_SEARCH_S * FS)
    ons = np.array(
        [
            max(0, p - w, q + 1) + np.argmin(c[max(0, p - w, q + 1) : p])
            for q, p in pairwise(pk)
        ]
    )
    return pk[1:][ok] / FS, ibi[ok], (c[pk[1:]] - c[ons])[ok], x[ons][ok]


def compare_resp_bands(
    labels: pd.DataFrame, n: int = 150, seed: int = 2
) -> pd.DataFrame:
    """The comparison behind RESP_BAND_HZ: fused rate vs the physician label, n dev sessions.

    `labels` is indexed by (patient_id, step) with a `respiratory` column. Reproduces the
    numbers quoted in reports/03_cv_robustness.md.
    """
    sess = dev_sessions()
    sess = sess[
        ~((sess["patient_id"] == 8706) & (sess["step"] == "after"))
    ]  # unreadable
    sess = sess.sample(n, random_state=seed)
    series = Parallel(n_jobs=12)(
        delayed(_resp_series)(r.patient_id, r.step) for r in sess.itertuples()
    )
    y = labels.loc[
        list(zip(sess["patient_id"], sess["step"], strict=True)), "respiratory"
    ].to_numpy()
    rows = []
    for name, (band, pre) in RESP_VARIANTS.items():
        fused = np.array(
            [
                np.median([_variant_rate(t, v, band, pre) for v in (ibi, amp, base)])
                for t, ibi, amp, base in series
            ]
        )
        rows.append(
            {
                "variant": name,
                "mae_vs_label": float(np.mean(np.abs(fused - y))),
                "pearson_r": float(np.corrcoef(fused, y)[0, 1]),
                "median_estimate": float(np.median(fused)),
            }
        )
    rows.append(
        {
            "variant": "constant (label median of these sessions)",
            "mae_vs_label": float(np.mean(np.abs(y - np.median(y)))),
            "pearson_r": np.nan,
            "median_estimate": float(np.median(y)),
        }
    )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- batch run
def dev_sessions() -> pd.DataFrame:
    """(patient_id, step, fold) for every DEV session. Test sessions are never listed."""
    splits = load_splits()
    dev = splits[splits["split"] == "dev"]
    sess = dev.drop_duplicates(subset=["patient_id", "step"])[
        ["patient_id", "step", "fold"]
    ]
    return sess.sort_values(["patient_id", "step"]).reset_index(drop=True)


def _one(patient_id: int, step: str, snr_db: float | None) -> dict:
    row = {"patient_id": int(patient_id), "step": step}
    try:
        x = load_session_ppg(patient_id, step)
    except ValueError:
        # An unparseable file (one known: 8706_after.PW, write-corrupted with a merged
        # timestamp, a NUL-byte run and a truncated last line). Reported, never repaired.
        return {**row, **dict.fromkeys(ALL_FEATURES, np.nan), "readable": False}
    if snr_db is not None:
        x = add_noise(x, snr_db, noise_rng(patient_id, step, snr_db))
    return {**row, **extract(x), "readable": True}


def build(snr_db: float | None = None, n_jobs: int = 12) -> pd.DataFrame:
    sess = dev_sessions()
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_one)(r.patient_id, r.step, snr_db) for r in sess.itertuples()
    )
    return pd.DataFrame(rows)


def write(df: pd.DataFrame, snr_db: float | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(features_path(snr_db), index=False)


def load_features(snr_db: float | None = None) -> pd.DataFrame:
    path = features_path(snr_db)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing - build it with `make features`")
    return pd.read_parquet(path)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--noise", action="store_true", help="also build every SNR level")
    args = ap.parse_args()
    log = {
        "built": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "levels": {},
    }
    for snr in [None, *(SNR_LEVELS_DB if args.noise else ())]:
        df = build(snr)
        write(df, snr)
        unreadable = df.loc[~df["readable"], ["patient_id", "step"]]
        failed = int((df["readable"] & df["hr_mean_bpm"].isna()).sum())
        key = "clean" if snr is None else f"{snr} dB"
        log["levels"][key] = {
            "sessions": len(df),
            "unreadable_files": [
                f"{r.patient_id}_{r.step}.PW" for r in unreadable.itertuples()
            ],
            "failed_extraction": failed,
        }
        print(
            f"{key:>6}: {len(df)} sessions, {len(unreadable)} unreadable, {failed} failed"
        )
    (OUT_DIR / "build_log.json").write_text(
        json.dumps(log, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
