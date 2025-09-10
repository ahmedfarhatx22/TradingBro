"""
Adaptive DTG Zero-Crossings & Event Spans
----------------------------------------
Purpose:
  - Extract ALL zero-crossing temperatures from a DTG curve (adaptive).
  - Pair crossings into weight-loss "events" (two temps per peak: start/end).
  - Optional smoothing for robustness; linear interpolation for "exact" T_cross.

What you get:
  * zero_crossings_table_for_file(dtg_path, ...) -> crossings DataFrame
      Columns: [Crossing, T_cross, Direction, Role, DTG_before, DTG_after]
        Direction: '+->-' (start of loss) or '- ->+' (end of loss)
        Role:       'start_of_loss' or 'end_of_loss'
  * event_spans_for_file(dtg_path, ...) -> events DataFrame
      Columns: [Event, T_start, T_end, Width, T_peak (optional), PeakAmp_DTG (optional)]

Batch helpers are provided for folders/lists of files.

Notes:
  - Set use_smoothed=False if you strictly want crossings on the RAW DTG trace.
  - 'Exact' T_cross is linear interpolation between the two samples that straddle zero
    (on smoothed or raw vector per your choice).

Requires: numpy, pandas
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Iterable, Dict

__all__ = [
    "load_dtg",
    "smooth_dtg_values",
    "compute_zero_crossings",
    "zero_crossings_table_for_dtg",
    "zero_crossings_table_for_file",
    "pair_crossings_to_events",
    "event_spans_for_file",
    "zero_crossings_table_batch",
    "event_spans_batch",
]

# -------------------------
# Loading (DTG only)
# -------------------------

def load_dtg(dtg_path: str, skiprows: int = 30) -> pd.DataFrame:
    """
    Robust DTG loader.
    Returns DataFrame with columns: ['T','DTG'] (sorted by T).
    Tries to detect columns even if headers differ.
    """
    df = pd.read_csv(dtg_path, encoding="ISO-8859-1", skiprows=skiprows, header=0)
    cols_lower = [c.lower() for c in df.columns]

    # Temperature column
    if "temperature" in cols_lower:
        tcol = df.columns[cols_lower.index("temperature")]
    else:
        # choose first numeric column that looks like temperature
        tcol = None
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().mean() > 0.7 and s.between(0, 1500).mean() > 0.6:
                tcol = c
                break
        if tcol is None:
            tcol = df.columns[0]

    # DTG column
    if "dtg" in cols_lower:
        dcol = df.columns[cols_lower.index("dtg")]
    else:
        # choose a column with sign changes and reasonable magnitude
        dcol = None
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if s.empty:
                continue
            if s.min() < 0 and s.max() > 0:
                dcol = c
                break
        if dcol is None:
            # fallback to last column
            dcol = df.columns[-1]

    out = pd.DataFrame({
        "T": pd.to_numeric(df[tcol], errors="coerce"),
        "DTG": pd.to_numeric(df[dcol], errors="coerce"),
    }).dropna().sort_values("T").reset_index(drop=True)
    return out

# -------------------------
# Smoothing (optional)
# -------------------------

def smooth_dtg_values(y: np.ndarray, window: int = 21) -> np.ndarray:
    """Simple moving average; ensures odd window ≥3."""
    if window < 3:
        window = 3
    if window % 2 == 0:
        window += 1
    if len(y) < window:
        return y.copy()
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")

# -------------------------
# Zero-crossings (linear interpolation)
# -------------------------

def compute_zero_crossings(T: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute all zero-crossing temperatures (linear interpolation).
    Includes exact zeros at sample points and sign-change between consecutive samples.
    Returns sorted 1D array of T_cross.
    """
    zT: List[float] = []
    for i in range(1, len(Y)):
        y0, y1 = Y[i-1], Y[i]
        t0, t1 = T[i-1], T[i]

        # exact zero at i-1
        if y0 == 0:
            zT.append(float(t0))

        # sign change between i-1 and i
        if (y0 < 0 and y1 > 0) or (y0 > 0 and y1 < 0):
            if y1 != y0:
                t_cross = t0 + (0 - y0) * (t1 - t0) / (y1 - y0)
                zT.append(float(t_cross))

    if not zT:
        return np.array([], dtype=float)
    # unique + sorted
    zT = sorted(set(zT))
    return np.array(zT, dtype=float)

# -------------------------
# Crossings table (adaptive)
# -------------------------

def zero_crossings_table_for_dtg(
    dtg_df: pd.DataFrame,
    smooth_window: int = 21,
    tmin: float = 30.0,
    tmax: float = 1000.0,
    use_smoothed: bool = True,
) -> pd.DataFrame:
    """
    Make a table of ALL DTG zero-crossings in [tmin, tmax].
    Columns:
      - Crossing (1..N)
      - T_cross (°C)
      - Direction: '+->-' (positive→negative) or '- ->+' (negative→positive)
      - Role: 'start_of_loss' or 'end_of_loss' (based on Direction)
      - DTG_before, DTG_after (values around crossing for sanity)
    """
    T = dtg_df["T"].values
    Yraw = dtg_df["DTG"].values
    Y = smooth_dtg_values(Yraw, smooth_window) if use_smoothed else Yraw

    # restrict to analysis window
    mask = (T >= tmin) & (T <= tmax)
    T = T[mask]; Y = Y[mask]

    rows: List[Dict[str, float | str | int]] = []
    cross_temps = compute_zero_crossings(T, Y)

    # determine direction by sampling a tiny epsilon on each side
    # for exact sample zeros we infer using adjacent points
    for idx, tc in enumerate(cross_temps, start=1):
        eps = 1e-6
        # pick close-by points using interpolation
        y_before = float(np.interp(tc - eps, T, Y))
        y_after  = float(np.interp(tc + eps, T, Y))
        if y_before > 0 and y_after < 0:
            direction = "+->-"
            role = "start_of_loss"
        elif y_before < 0 and y_after > 0:
            direction = "- ->+"
            role = "end_of_loss"
        else:
            # fallback: decide by looking slightly further away
            y_b2 = float(np.interp(tc - 1e-3, T, Y))
            y_a2 = float(np.interp(tc + 1e-3, T, Y))
            if y_b2 > 0 and y_a2 < 0:
                direction, role = "+->-", "start_of_loss"
            elif y_b2 < 0 and y_a2 > 0:
                direction, role = "- ->+", "end_of_loss"
            else:
                # ambiguous/noise
                direction, role = "ambiguous", "unknown"

        rows.append({
            "Crossing": idx,
            "T_cross": float(tc),
            "Direction": direction,
            "Role": role,
            "DTG_before": y_before,
            "DTG_after": y_after,
        })

    return pd.DataFrame(rows).sort_values("T_cross").reset_index(drop=True)

def zero_crossings_table_for_file(
    dtg_path: str,
    sample: str = "Sample",
    timepoint: str = "TP",
    skiprows: int = 30,
    smooth_window: int = 21,
    tmin: float = 30.0,
    tmax: float = 1000.0,
    use_smoothed: bool = True,
) -> pd.DataFrame:
    """
    Load a DTG CSV and return the crossings table with Sample/Timepoint columns.
    """
    dtg_df = load_dtg(dtg_path, skiprows=skiprows)
    zc = zero_crossings_table_for_dtg(
        dtg_df,
        smooth_window=smooth_window,
        tmin=tmin, tmax=tmax,
        use_smoothed=use_smoothed,
    )
    zc.insert(0, "Sample", sample)
    zc.insert(1, "Timepoint", timepoint)
    zc.insert(2, "File", os.path.basename(dtg_path))
    return zc

# -------------------------
# Pair crossings into weight-loss events (two temps per peak)
# -------------------------

def pair_crossings_to_events(
    crossings_df: pd.DataFrame,
    dtg_df: Optional[pd.DataFrame] = None,
    min_width: float = 8.0,
    enforce_negative_between: bool = True,
    use_smoothed: bool = True,
    smooth_window: int = 21,
) -> pd.DataFrame:
    """
    Pair start_of_loss and end_of_loss into events.
    Returns DataFrame with:
      [Event, T_start, T_end, Width]  (+ optional T_peak, PeakAmp_DTG if dtg_df is provided)

    Logic:
      - We take each 'start_of_loss' and pair it with the next 'end_of_loss'.
      - If 'enforce_negative_between' is True, we check DTG(midpoint) < 0 between them.
      - 'min_width' discards ultra-narrow events (°C).
    """
    starts = crossings_df[crossings_df["Role"] == "start_of_loss"].sort_values("T_cross")
    ends   = crossings_df[crossings_df["Role"] == "end_of_loss"].sort_values("T_cross")

    events: List[Tuple[float, float]] = []
    e_idx = 0
    for _, srow in starts.iterrows():
        sT = float(srow["T_cross"])
        # find the first end crossing after sT
        end_after = ends[ends["T_cross"] > sT]
        if end_after.empty:
            continue
        eT = float(end_after.iloc[0]["T_cross"])
        # drop this end from future pairing
        ends = ends[ends["T_cross"] > eT]
        # sanity filters
        if (eT - sT) < min_width:
            continue
        if enforce_negative_between and (dtg_df is not None):
            T = dtg_df["T"].values
            Yraw = dtg_df["DTG"].values
            Y = smooth_dtg_values(Yraw, smooth_window) if use_smoothed else Yraw
            mid = 0.5 * (sT + eT)
            y_mid = float(np.interp(mid, T, Y))
            if y_mid >= 0:
                # not a weight-loss segment
                continue
        events.append((sT, eT))

    # Build table
    rows: List[Dict[str, float | int]] = []
    if dtg_df is not None:
        T = dtg_df["T"].values
        Yraw = dtg_df["DTG"].values
        Y = smooth_dtg_values(Yraw, smooth_window) if use_smoothed else Yraw

    for i, (a, b) in enumerate(events, start=1):
        row = {
            "Event": i,
            "T_start": round(a, 3),
            "T_end": round(b, 3),
            "Width": round(b - a, 3),
        }
        if dtg_df is not None:
            mask = (T >= a) & (T <= b)
            if mask.any():
                idx_min = np.argmin(Y[mask])
                T_region = T[mask]
                Y_region = Y[mask]
                row["T_peak"] = round(float(T_region[idx_min]), 3)
                row["PeakAmp_DTG"] = round(float(Y_region[idx_min]), 6)
        rows.append(row)

    return pd.DataFrame(rows)

def event_spans_for_file(
    dtg_path: str,
    sample: str = "Sample",
    timepoint: str = "TP",
    skiprows: int = 30,
    smooth_window: int = 21,
    tmin: float = 30.0,
    tmax: float = 1000.0,
    use_smoothed: bool = True,
    min_width: float = 8.0,
    enforce_negative_between: bool = True,
) -> pd.DataFrame:
    """
    Convenience: from a DTG CSV, produce the TWO temperatures per event (start/end).
    Returns DataFrame with Sample, Timepoint, File, Event, T_start, T_end, Width, T_peak, PeakAmp_DTG.
    """
    dtg_df = load_dtg(dtg_path, skiprows=skiprows)
    # apply analysis window
    dtg_df = dtg_df[(dtg_df["T"] >= tmin) & (dtg_df["T"] <= tmax)].reset_index(drop=True)
    crossings = zero_crossings_table_for_dtg(
        dtg_df,
        smooth_window=smooth_window, tmin=tmin, tmax=tmax, use_smoothed=use_smoothed
    )
    events = pair_crossings_to_events(
        crossings, dtg_df=dtg_df, min_width=min_width,
        enforce_negative_between=enforce_negative_between,
        use_smoothed=use_smoothed, smooth_window=smooth_window
    )
    # add metadata
    events.insert(0, "Sample", sample)
    events.insert(1, "Timepoint", timepoint)
    events.insert(2, "File", os.path.basename(dtg_path))
    return events

# -------------------------
# Batch helpers
# -------------------------

def zero_crossings_table_batch(
    files: Iterable[Tuple[str, str, str]],
    skiprows: int = 30,
    smooth_window: int = 21,
    tmin: float = 30.0,
    tmax: float = 1000.0,
    use_smoothed: bool = True,
) -> pd.DataFrame:
    """
    Batch crossings-only.
    files: iterable of (dtg_path, sample, timepoint)
    """
    out = []
    for dtg_path, sample, timepoint in files:
        out.append(
            zero_crossings_table_for_file(
                dtg_path, sample=sample, timepoint=timepoint, skiprows=skiprows,
                smooth_window=smooth_window, tmin=tmin, tmax=tmax, use_smoothed=use_smoothed
            )
        )
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def event_spans_batch(
    files: Iterable[Tuple[str, str, str]],
    skiprows: int = 30,
    smooth_window: int = 21,
    tmin: float = 30.0,
    tmax: float = 1000.0,
    use_smoothed: bool = True,
    min_width: float = 8.0,
    enforce_negative_between: bool = True,
) -> pd.DataFrame:
    """
    Batch event spans (two temps per peak).
    files: iterable of (dtg_path, sample, timepoint)
    """
    out = []
    for dtg_path, sample, timepoint in files:
        out.append(
            event_spans_for_file(
                dtg_path, sample=sample, timepoint=timepoint, skiprows=skiprows,
                smooth_window=smooth_window, tmin=tmin, tmax=tmax, use_smoothed=use_smoothed,
                min_width=min_width, enforce_negative_between=enforce_negative_between
            )
        )
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ---------- REPLACEMENT: robust pairing + safe tables ----------

def pair_crossings_to_events_stack(
    crossings_df: pd.DataFrame,
    dtg_df: Optional[pd.DataFrame] = None,
    min_width: float = 8.0,
    enforce_negative_between: bool = True,
    use_smoothed: bool = True,
    smooth_window: int = 21,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Robustly pair crossings into weight-loss events using a stack:
      - push on '+->-' (start_of_loss)
      - pop on '- ->+' (end_of_loss)
    Produces a table with columns:
      ['Event','T_start','T_end','Width',('T_peak','PeakAmp_DTG' if dtg_df provided)]
    Always returns a DataFrame with these columns (may be empty).
    """
    cols_base = ["Event","T_start","T_end","Width"]
    cols_full = cols_base + (["T_peak","PeakAmp_DTG"] if dtg_df is not None else [])
    rows = []

    if crossings_df is None or crossings_df.empty:
        return pd.DataFrame(columns=cols_full)

    zc = crossings_df.sort_values("T_cross").reset_index(drop=True)
    if debug:
        print("Crossings found:", len(zc))
        print(zc[["T_cross","Direction","Role"]])

    starts_stack: List[float] = []

    # Prepare DTG arrays if we’ll compute T_peak/amp
    if dtg_df is not None:
        T = dtg_df["T"].values
        Yraw = dtg_df["DTG"].values
        Y = smooth_dtg_values(Yraw, smooth_window) if use_smoothed else Yraw

    event_idx = 0
    for _, r in zc.iterrows():
        role = str(r.get("Role", ""))
        t   = float(r["T_cross"])

        if role == "start_of_loss":
            starts_stack.append(t)
        elif role == "end_of_loss":
            # pair with the latest unmatched start (LIFO) to avoid crossing overlaps
            if starts_stack:
                sT = starts_stack.pop()     # LIFO pairing
                eT = t
                if (eT - sT) >= min_width:
                    event_idx += 1
                    row = {
                        "Event": event_idx,
                        "T_start": round(sT, 3),
                        "T_end": round(eT, 3),
                        "Width": round(eT - sT, 3),
                    }
                    if dtg_df is not None:
                        m = (T >= sT) & (T <= eT)
                        if m.any():
                            idx_min = np.argmin(Y[m])
                            T_region = T[m]
                            Y_region = Y[m]
                            row["T_peak"] = round(float(T_region[idx_min]), 3)
                            row["PeakAmp_DTG"] = round(float(Y_region[idx_min]), 6)
                        else:
                            row["T_peak"] = np.nan
                            row["PeakAmp_DTG"] = np.nan
                    rows.append(row)
            # if there’s an end with no start, we ignore it

    # Build DF; ensure columns exist even if rows == []
    if not rows:
        return pd.DataFrame(columns=cols_full)
    return pd.DataFrame(rows, columns=cols_full)


def event_spans_for_file(
    dtg_path: str,
    sample: str = "Sample",
    timepoint: str = "TP",
    skiprows: int = 30,
    smooth_window: int = 21,
    tmin: float = 30.0,
    tmax: float = 1000.0,
    use_smoothed: bool = True,
    min_width: float = 8.0,
    enforce_negative_between: bool = True,   # kept for API parity; check uses 'mid<0' below
    debug: bool = False,
) -> pd.DataFrame:
    """
    Produce the TWO temperatures per event (start/end) by pairing crossings.
    Always returns columns: Sample, Timepoint, File, Event, T_start, T_end, Width, T_peak, PeakAmp_DTG.
    May be empty if no complete (start,end) pairs exist.
    """
    dtg_df = load_dtg(dtg_path, skiprows=skiprows)
    dtg_df = dtg_df[(dtg_df["T"] >= tmin) & (dtg_df["T"] <= tmax)].reset_index(drop=True)

    crossings = zero_crossings_table_for_dtg(
        dtg_df, smooth_window=smooth_window, tmin=tmin, tmax=tmax, use_smoothed=use_smoothed
    )

    # Pair with stack method (more robust)
    events = pair_crossings_to_events_stack(
        crossings, dtg_df=dtg_df, min_width=min_width,
        enforce_negative_between=enforce_negative_between,
        use_smoothed=use_smoothed, smooth_window=smooth_window, debug=debug
    )

    # Add metadata & ensure columns
    for col in ["T_peak","PeakAmp_DTG"]:
        if col not in events.columns:
            events[col] = np.nan

    events.insert(0, "Sample", sample)
    events.insert(1, "Timepoint", timepoint)
    events.insert(2, "File", os.path.basename(dtg_path))

    # Ensure column order and presence (even if empty)
    cols = ["Sample","Timepoint","File","Event","T_start","T_end","Width","T_peak","PeakAmp_DTG"]
    return events.reindex(columns=cols)
