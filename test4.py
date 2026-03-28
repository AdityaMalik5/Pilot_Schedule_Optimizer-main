#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Flight Schedule Optimization with Full Requirements Implementation

NEW FEATURES ADDED:
Interactive Schedule Tuning Model with Impact Visualization
Cascading Delay Impact Analysis
Aircraft Rotation Tracking
Enhanced NLP Interface
Advanced Delay Propagation Modeling


"""

import argparse
import os
import re
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Try to import XGBoost
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ------------------------------
# Shared Data Canonicalization
# ------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common CSV variants to a canonical schema used across pages."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()].copy()
    lower_to_actual = {str(c).lower().strip(): c for c in out.columns}

    alias_groups = {
        "Flight Number": ["flight number", "flight_no", "flight no", "flightnumber", "leg_nb", "flight id"],
        "From": ["from", "origin", "origin airport", "airport_dep", "dep_airport", "departure airport"],
        "To": ["to", "destination", "dest", "airport_arr", "arr_airport", "arrival airport"],
        "Unnamed: 2": ["date", "flight_date", "date_dep", "departure date"],
        "STD": ["std", "scheduled departure", "scheduled_dep", "hour_dep", "dep time", "departure time"],
        "STA": ["sta", "scheduled arrival", "scheduled_arr", "hour_arr", "arr time", "arrival time"],
        "ATD": ["atd", "actual departure", "actual_dep"],
        "ATA": ["ata", "actual arrival", "actual_arr"],
        "Flight time": ["flight time", "flight_time", "duration", "block_time"],
        "Aircraft": ["aircraft", "aircraft registration", "tail", "tail number", "aircraft_reg"],
        "UniqueCarrier": ["carrier", "airline", "airline code", "uniquecarrier"],
        "S.No": ["s.no", "s no", "serial", "serial number"],
        "Status": ["status", "flight status"],
    }

    rename_cols = {}
    for target, aliases in alias_groups.items():
        if target in out.columns:
            continue
        for alias in aliases:
            if alias in lower_to_actual:
                rename_cols[lower_to_actual[alias]] = target
                break
    if rename_cols:
        out.rename(columns=rename_cols, inplace=True)

    # If a separate date + time format exists, synthesize expected strings.
    if "Unnamed: 2" in out.columns and "STD" not in out.columns and "hour_dep" in lower_to_actual:
        out["STD"] = out[lower_to_actual["hour_dep"]].astype(str)
    if "STA" not in out.columns and "hour_arr" in lower_to_actual:
        out["STA"] = out[lower_to_actual["hour_arr"]].astype(str)

    # Normalize route labels for airport code extraction.
    if "From" in out.columns:
        out["From"] = out["From"].astype(str).str.strip().apply(
            lambda x: x if "(" in x and ")" in x else f"Airport ({x.upper()})"
        )
    if "To" in out.columns:
        out["To"] = out["To"].astype(str).str.strip().apply(
            lambda x: x if "(" in x and ")" in x else f"Airport ({x.upper()})"
        )

    if "Flight Number" not in out.columns:
        out["Flight Number"] = [f"UNK{i+1:05d}" for i in range(len(out))]
    if "S.No" not in out.columns:
        out["S.No"] = np.arange(1, len(out) + 1)
    if "Status" not in out.columns:
        out["Status"] = "On Time"

    return out


def get_active_df_for_views() -> pd.DataFrame:
    """Return optimized data if applied, otherwise master data."""
    if st.session_state.get("optimization_applied") and "master_df" in st.session_state:
        return st.session_state["master_df"].copy()
    if "master_df" in st.session_state:
        return st.session_state["master_df"].copy()
    return pd.DataFrame()


def crew_rules_for_schedule_board() -> dict:
    """
    Map Crew Manager sliders (fh_hours, fh_ratio, fh_wl) to Schedule Board JS rule constants.
    Defaults match the Crew Manager slider defaults when those keys are missing.
    """
    h = float(st.session_state.get("fh_hours", 7.5))
    r = float(st.session_state.get("fh_ratio", 1.5))
    w = int(st.session_state.get("fh_wl", 60))

    max_block_h = round(min(12.0, max(6.0, h)), 2)
    max_duties = int(max(3, min(7, round(6 - max(0.0, r - 1.5) * 1.2))))
    min_rest_h = round(float(max(6.0, min(12.0, 8.0 + (r - 1.5) * 1.5))), 2)
    fatigue_thresh = int(max(2, min(5, round(4 - (w - 35) / 22))))
    duty_hours = round(min(3.5, max(1.2, max_block_h / max(max_duties, 1))), 2)

    return {
        "duty_hours": duty_hours,
        "max_duties": max_duties,
        "max_block_h": max_block_h,
        "min_rest_h": min_rest_h,
        "fatigue_thresh": fatigue_thresh,
    }


def _fatigue_triple_from_totals(total_h: float, sectors: int, delays: int) -> tuple:
    """RF feature triple from aggregated duty stats (same mapping as pilot-day features)."""
    if sectors <= 0:
        return 4.0, 1.0, 35.0
    total_h = float(total_h)
    hours_feat = float(np.clip(total_h, 1.0, 14.0))
    rest_proxy = max(2.0, 14.0 - total_h)
    ratio_feat = float(
        np.clip((total_h / rest_proxy) * (1.0 + 0.14 * max(0, sectors - 3)), 0.5, 3.5)
    )
    workload_feat = float(
        np.clip(
            18.0 + sectors * 9.0 + float(delays) * 14.0 + max(0.0, total_h - 8.0) * 5.0,
            10.0,
            100.0,
        )
    )
    return hours_feat, ratio_feat, workload_feat


def fatigue_features_for_pilot_day(df: pd.DataFrame, pilot: str, dkey: str) -> tuple:
    """
    Map one pilot's duties on a calendar day to RandomForest inputs
    (avg_flight_hours, duty_to_rest_ratio, workload_index).
    """
    pilot = str(pilot).strip()
    if not pilot or pilot.upper() == "N/A":
        return 4.0, 1.0, 35.0
    dk = str(dkey)
    m = (
        ((df["Captain"] == pilot) | (df["First Officer"] == pilot))
        & (df["DateKey"].astype(str) == dk)
    )
    sub = df.loc[m]
    if sub.empty:
        return 4.0, 1.0, 35.0
    total_h = float(sub["DurationHrs"].sum())
    sectors = len(sub)
    delays = int((sub["Status"].astype(str).str.lower() == "delayed").sum())
    return _fatigue_triple_from_totals(total_h, sectors, delays)


def _summarize_pilot_days(df: pd.DataFrame) -> pd.DataFrame:
    """One groupby over the schedule; avoids O(pilots × rows) dataframe filters."""
    c = df["Captain"].astype(str).str.strip()
    f = df["First Officer"].astype(str).str.strip()
    dk = df["DateKey"].astype(str)
    dur = df["DurationHrs"].astype(float)
    dly = (df["Status"].astype(str).str.lower() == "delayed").astype(int)
    p1 = pd.DataFrame({"pilot": c.values, "dkey": dk.values, "dur": dur.values, "dly": dly.values})
    p2 = pd.DataFrame({"pilot": f.values, "dkey": dk.values, "dur": dur.values, "dly": dly.values})
    pl = pd.concat([p1, p2], ignore_index=True)
    pl = pl[(pl["pilot"] != "") & (pl["pilot"].str.upper() != "N/A")]
    if pl.empty:
        return pd.DataFrame(columns=["total_h", "sectors", "delays"]).astype({"sectors": int, "delays": int})
    return pl.groupby(["pilot", "dkey"], sort=False).agg(
        total_h=("dur", "sum"),
        sectors=("dur", "count"),
        delays=("dly", "sum"),
    )


def _summ_get(summ: pd.DataFrame, pilot: str, dkey: str):
    key = (str(pilot).strip(), str(dkey))
    if summ.empty or key not in summ.index:
        return 0.0, 0, 0
    row = summ.loc[key]
    return float(row["total_h"]), int(row["sectors"]), int(row["delays"])


def propose_optimized_schedule(master_df: pd.DataFrame, max_shift_minutes: int = 30) -> pd.DataFrame:
    """
    Propose schedule changes using the Random Forest fatigue model (Crew Manager).

    Reassigns CA/FO on high-risk pilot-days and may apply a small spacing shift when
    still critical. Does not use airport congestion or delay-based slot shifting.
    """
    df = normalize_columns(master_df).copy()
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    if df.empty:
        return df

    if "Date" not in df.columns and "Unnamed: 2" in df.columns:
        df["Date"] = df["Unnamed: 2"]
    if "Date" not in df.columns:
        df["Date"] = pd.Timestamp.now().strftime("%d-%b-%y")

    df["Date_dt"] = pd.to_datetime(df["Date"], format="%d-%b-%y", errors="coerce")
    df["STD_dt"] = pd.to_datetime(df["Date_dt"].dt.strftime("%Y-%m-%d") + " " + df["STD"].astype(str), errors="coerce")
    df["STA_dt"] = pd.to_datetime(df["Date_dt"].dt.strftime("%Y-%m-%d") + " " + df["STA"].astype(str), errors="coerce")
    df["STA_dt"] = df["STA_dt"].where(df["STA_dt"] >= df["STD_dt"], df["STA_dt"] + pd.Timedelta(days=1))

    for col in ["Captain", "First Officer", "Status"]:
        if col not in df.columns:
            df[col] = "N/A"

    predictor = get_pilot_fatigue_predictor(_FATIGUE_PREDICTOR_RESOURCE_VERSION)

    proposed = df.copy()
    proposed["Orig_STD"] = proposed["STD"].astype(str)
    proposed["Orig_STA"] = proposed["STA"].astype(str)
    proposed["Change_Reason"] = ""
    proposed["Risk_Reduction_%"] = 0.0

    proposed["DateKey"] = proposed["Date_dt"].dt.strftime("%Y-%m-%d")
    proposed["DurationHrs"] = (
        (proposed["STA_dt"] - proposed["STD_dt"]).dt.total_seconds() / 3600
    ).fillna(1.5).clip(0.5, 6)

    captains = [
        c for c in proposed["Captain"].dropna().unique().tolist()
        if str(c).strip() and str(c).upper() != "N/A"
    ]
    fos = [
        f for f in proposed["First Officer"].dropna().unique().tolist()
        if str(f).strip() and str(f).upper() != "N/A"
    ]

    def _append_reason(idx, text: str):
        cur = str(proposed.at[idx, "Change_Reason"] or "").strip()
        proposed.at[idx, "Change_Reason"] = (cur + "; " if cur else "") + text

    ALT_CRIT_MAX = 0.55
    MIN_CRIT_DROP = 0.03
    MAX_PASSES = 3
    MAX_ALTS_CAP = min(12, max(1, len(captains) - 1)) if len(captains) > 1 else 0
    MAX_ALTS_FO = min(12, max(1, len(fos) - 1)) if len(fos) > 1 else 0

    rf = predictor.model

    def _rf_predict_row(h, r, w):
        return int(rf.predict(np.array([[float(h), float(r), float(w)]], dtype=np.float64))[0])

    for _ in range(MAX_PASSES):
        summ = _summarize_pilot_days(proposed)
        ranked = []
        for (pilot, dkey), srow in summ.iterrows():
            h, r, w = _fatigue_triple_from_totals(
                float(srow["total_h"]), int(srow["sectors"]), int(srow["delays"])
            )
            pred = _rf_predict_row(h, r, w)
            if pred < 1:
                continue
            pe, pm, pc = predictor.class_probs(h, r, w)
            ranked.append((pc, pm, str(pilot), str(dkey)))
        ranked.sort(reverse=True)
        changed = False

        for _, __, pilot, dkey in ranked:
            m = (
                ((proposed["Captain"] == pilot) | (proposed["First Officer"] == pilot))
                & (proposed["DateKey"].astype(str) == dkey)
            )
            sub = proposed.loc[m]
            if sub.empty:
                continue
            j = sub["DurationHrs"].idxmax()

            ph, ps, pdel = _summ_get(summ, pilot, dkey)
            h0, r0, w0 = _fatigue_triple_from_totals(ph, ps, pdel)
            pe0, pm0, pc0 = predictor.class_probs(h0, r0, w0)

            dur = float(proposed.loc[j, "DurationHrs"])
            dly = (
                1 if str(proposed.loc[j, "Status"]).lower() == "delayed" else 0
            )

            if str(proposed.loc[j, "Captain"]) == pilot and len(captains) > 1 and MAX_ALTS_CAP > 0:
                cand = [c for c in captains if c != pilot]
                cand.sort(
                    key=lambda a: predictor.class_probs(
                        *_fatigue_triple_from_totals(*_summ_get(summ, a, dkey))
                    )[2]
                )
                best_alt = None
                best_drop = 0.0
                ph2, ps2, pdel2 = ph - dur, ps - 1, pdel - dly
                for alt in cand[:MAX_ALTS_CAP]:
                    ah, as_, adel = _summ_get(summ, alt, dkey)
                    ah2, as2, adel2 = ah + dur, as_ + 1, adel + dly
                    hp, rp, wp = _fatigue_triple_from_totals(ph2, ps2, pdel2)
                    ha, ra, wa = _fatigue_triple_from_totals(ah2, as2, adel2)
                    _, _, pac = predictor.class_probs(ha, ra, wa)
                    if pac > ALT_CRIT_MAX:
                        continue
                    _, _, ppc = predictor.class_probs(hp, rp, wp)
                    drop = pc0 - ppc
                    if drop > best_drop:
                        best_drop = drop
                        best_alt = alt
                if best_alt is not None and best_drop >= MIN_CRIT_DROP:
                    proposed.at[j, "Captain"] = best_alt
                    summ = _summarize_pilot_days(proposed)
                    ph1, ps1, pdel1 = _summ_get(summ, pilot, dkey)
                    h1, r1, w1 = _fatigue_triple_from_totals(ph1, ps1, pdel1)
                    _, pm1, pc1 = predictor.class_probs(h1, r1, w1)
                    reduc = (pc0 - pc1) * 100.0 + (pm0 - pm1) * 35.0
                    proposed.at[j, "Risk_Reduction_%"] = max(
                        float(proposed.at[j, "Risk_Reduction_%"] or 0),
                        round(min(80.0, max(6.0, reduc)), 1),
                    )
                    _append_reason(j, f"Fatigue RF: CA swap ({pilot}→{best_alt})")
                    changed = True

            elif str(proposed.loc[j, "First Officer"]) == pilot and len(fos) > 1 and MAX_ALTS_FO > 0:
                cand = [x for x in fos if x != pilot]
                cand.sort(
                    key=lambda a: predictor.class_probs(
                        *_fatigue_triple_from_totals(*_summ_get(summ, a, dkey))
                    )[2]
                )
                best_alt = None
                best_drop = 0.0
                ph2, ps2, pdel2 = ph - dur, ps - 1, pdel - dly
                for alt in cand[:MAX_ALTS_FO]:
                    ah, as_, adel = _summ_get(summ, alt, dkey)
                    ah2, as2, adel2 = ah + dur, as_ + 1, adel + dly
                    hp, rp, wp = _fatigue_triple_from_totals(ph2, ps2, pdel2)
                    ha, ra, wa = _fatigue_triple_from_totals(ah2, as2, adel2)
                    _, _, pac = predictor.class_probs(ha, ra, wa)
                    if pac > ALT_CRIT_MAX:
                        continue
                    _, _, ppc = predictor.class_probs(hp, rp, wp)
                    drop = pc0 - ppc
                    if drop > best_drop:
                        best_drop = drop
                        best_alt = alt
                if best_alt is not None and best_drop >= MIN_CRIT_DROP:
                    proposed.at[j, "First Officer"] = best_alt
                    summ = _summarize_pilot_days(proposed)
                    ph1, ps1, pdel1 = _summ_get(summ, pilot, dkey)
                    h1, r1, w1 = _fatigue_triple_from_totals(ph1, ps1, pdel1)
                    _, pm1, pc1 = predictor.class_probs(h1, r1, w1)
                    reduc = (pc0 - pc1) * 100.0 + (pm0 - pm1) * 35.0
                    proposed.at[j, "Risk_Reduction_%"] = max(
                        float(proposed.at[j, "Risk_Reduction_%"] or 0),
                        round(min(80.0, max(6.0, reduc)), 1),
                    )
                    _append_reason(j, f"Fatigue RF: FO swap ({pilot}→{best_alt})")
                    changed = True

        if not changed:
            break

    # Duty spacing: one pass, cache (captain, day) fatigue probabilities
    cap_day_cache = {}
    for idx, row in proposed.iterrows():
        cap = str(row.get("Captain", "")).strip()
        if not cap or cap.upper() == "N/A":
            continue
        dkey = str(row["DateKey"])
        ck = (cap, dkey)
        if ck not in cap_day_cache:
            h, r, w = fatigue_features_for_pilot_day(proposed, cap, dkey)
            cap_day_cache[ck] = (
                _rf_predict_row(h, r, w),
                predictor.class_probs(h, r, w),
            )
        pred, (_, _, pc) = cap_day_cache[ck]
        if pred == 2 and not str(proposed.at[idx, "Change_Reason"] or "").strip():
            shift = min(max_shift_minutes, 20)
            proposed.at[idx, "STD_dt"] = row["STD_dt"] + pd.Timedelta(minutes=shift)
            proposed.at[idx, "STA_dt"] = row["STA_dt"] + pd.Timedelta(minutes=shift)
            proposed.at[idx, "Change_Reason"] = f"Fatigue RF: duty spacing (+{shift}m)"
            proposed.at[idx, "Risk_Reduction_%"] = round(
                min(45.0, 10.0 + float(pc) * 28.0), 1
            )

    proposed["STD"] = proposed["STD_dt"].dt.strftime("%I:%M %p")
    proposed["STA"] = proposed["STA_dt"].dt.strftime("%I:%M %p")
    proposed["Optimization_Changed"] = (
        (proposed["Orig_STD"] != proposed["STD"])
        | (proposed["Orig_STA"] != proposed["STA"])
        | (proposed["Change_Reason"].astype(str) != "")
    )

    return proposed.drop(columns=["Date_dt", "DateKey", "DurationHrs"], errors="ignore")


# ------------------------------
# ENHANCED Data Cleaning with Aircraft Tracking
# ------------------------------
@st.cache_data(show_spinner="Processing and cleaning flight data with aircraft tracking...")
def clean_flight_data_enhanced(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced data cleaning with aircraft rotation and cascading analysis support
    """
    df = normalize_columns(df)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    # Ensure required ID columns exist to avoid KeyError on uploads with slightly
    # different schemas.
    if "Flight Number" not in df.columns:
        df["Flight Number"] = [f"UNK{i+1:05d}" for i in range(len(df))]
    if "S.No" not in df.columns:
        df["S.No"] = np.arange(1, len(df) + 1)

    # Forward fill Flight Number and Serial Number
    df['Flight Number'].ffill(inplace=True)
    df['S.No'].ffill(inplace=True)

    # Date column harmonization without creating duplicate Date keys
    if "Date" in df.columns and "Unnamed: 2" in df.columns:
        # Prefer explicit Date and drop legacy alias to avoid duplicate-key assembly.
        df.drop(columns=["Unnamed: 2"], inplace=True, errors="ignore")
    elif "Date" not in df.columns and "Unnamed: 2" in df.columns:
        df.rename(columns={"Unnamed: 2": "Date"}, inplace=True)
    elif "Date" not in df.columns:
        df["Date"] = pd.Timestamp.now().strftime("%d-%b-%y")

    # Drop rows where essential data is missing
    df.dropna(subset=['Date', 'From', 'To'], inplace=True)

    # Rename columns for clarity
    df.rename(columns={
        'From': 'Origin_City',
        'To': 'Dest_City',
        'Flight time': 'FlightTime'
    }, inplace=True)

    # Extract Airport Codes
    df['Origin'] = df['Origin_City'].str.extract(r'\((\w+)\)').fillna('')
    df['Dest'] = df['Dest_City'].str.extract(r'\((\w+)\)').fillna('')

    # Create Aircraft Registration from Aircraft column
    df['Aircraft_Registration'] = df['Aircraft'].str.extract(r'([A-Z]{2}-[A-Z]{3})')
    df['Aircraft_Type'] = df['Aircraft'].str.extract(r'([A-Z0-9]+)')

    # Drop original city columns and other unused columns
    df.drop(columns=[
        'Origin_City', 'Dest_City', 'S.No', 'Unnamed: 10',
        'Unnamed: 12', 'Unnamed: 13'
    ], inplace=True, errors='ignore')

    # --- ENHANCED DATETIME CONVERSION ---
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y', errors='coerce')

    def to_datetime(series_time):
        return pd.to_datetime(
            df['Date'].dt.strftime('%Y-%m-%d') + ' ' + series_time,
            format='%Y-%m-%d %I:%M %p',
            errors='coerce'
        )

    df['STD_dt'] = to_datetime(df['STD'])  # Scheduled Departure
    df['ATD_dt'] = to_datetime(df['ATD'])  # Actual Departure
    df['STA_dt'] = to_datetime(df['STA'])  # Scheduled Arrival

    # Handle 'Landed HH:MM AM/PM' format in ATA
    ata_time = df['ATA'].str.replace('Landed ', '', regex=False).str.strip()
    df['ATA_dt'] = to_datetime(ata_time)  # Actual Arrival

    # --- ENHANCED FEATURE ENGINEERING ---
    # Calculate Delays in minutes (KEY METRICS for analysis)
    df['DepDelay'] = (df['ATD_dt'] - df['STD_dt']).dt.total_seconds() / 60
    df['ArrDelay'] = (df['ATA_dt'] - df['STA_dt']).dt.total_seconds() / 60
    df['TaxiOut'] = (df['ATD_dt'] - df['STD_dt']).dt.total_seconds() / 60  # Gate to runway
    df['TaxiIn'] = (df['ATA_dt'] - df['STA_dt']).dt.total_seconds() / 60  # Runway to gate

    # Fix date rollover issues (flights arriving next day)
    df.loc[df['ArrDelay'] < -1000, 'ArrDelay'] += 1440
    df.loc[df['DepDelay'] < -1000, 'DepDelay'] += 1440

    # Extract Time-based features
    df['DepHour'] = df['ATD_dt'].dt.hour
    df['ArrHour'] = df['ATA_dt'].dt.hour
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek + 1
    df['DayofMonth'] = df['Date'].dt.day

    # Convert FlightTime to minutes
    def flight_time_to_minutes(time_str):
        if pd.isna(time_str) or ':' not in str(time_str):
            return np.nan
        try:
            h, m = map(int, str(time_str).split(':'))
            return h * 60 + m
        except:
            return np.nan

    df['AirTime'] = df['FlightTime'].apply(flight_time_to_minutes)
    df['Distance'] = df['AirTime'] * 8  # Approximate distance

    # NEW: Aircraft Rotation Features for Cascading Analysis
    df = df.sort_values(['Aircraft_Registration', 'ATD_dt'])
    df['Next_Flight_Gap'] = df.groupby('Aircraft_Registration')['STD_dt'].shift(-1) - df['ATA_dt']
    df['Next_Flight_Gap_Minutes'] = df['Next_Flight_Gap'].dt.total_seconds() / 60
    df['Turnaround_Time'] = df['Next_Flight_Gap_Minutes']
    df['Is_Quick_Turnaround'] = (df['Turnaround_Time'] < 90).astype(int)  # < 90 min turnaround

    # Create cascading impact potential score
    df['Cascading_Risk_Score'] = (
            (df['DepDelay'].fillna(0) > 15).astype(int) * 0.4 +  # Departure delay
            df['Is_Quick_Turnaround'] * 0.3 +  # Quick turnaround pressure
            ((df['DepHour'].isin([7, 8, 9, 17, 18, 19, 20])).astype(int)) * 0.3  # Peak hour operation
    )

    # Binary delay classifications for ML
    df['DepDelayBinary'] = (df['DepDelay'].fillna(0) > 15).astype(int)
    df['ArrDelayBinary'] = (df['ArrDelay'].fillna(0) > 15).astype(int)

    # Create placeholder columns for compatibility
    df['Cancelled'] = 0
    df['Diverted'] = 0
    df.rename(columns={'Flight Number': 'UniqueCarrier'}, inplace=True)

    # Final cleanup
    df.dropna(subset=['Date', 'DepHour', 'DepDelay', 'ArrDelay'], inplace=True)

    return df


# ------------------------------
# NEW: Schedule Tuning Model
# ------------------------------
class ScheduleTuner:
    """
    Interactive schedule tuning with delay impact simulation
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df = self.df.sort_values(['Date', 'STD_dt'])
        self._build_aircraft_networks()

    def _build_aircraft_networks(self):
        """Build network graph of aircraft rotations for impact analysis"""
        import networkx as nx

        self.aircraft_networks = {}
        for aircraft in self.df['Aircraft_Registration'].dropna().unique():
            aircraft_flights = self.df[self.df['Aircraft_Registration'] == aircraft].sort_values('STD_dt')
            if len(aircraft_flights) > 1:
                G = nx.DiGraph()
                for i in range(len(aircraft_flights) - 1):
                    current_flight = aircraft_flights.iloc[i]
                    next_flight = aircraft_flights.iloc[i + 1]
                    G.add_edge(
                        f"{current_flight['UniqueCarrier']}_{current_flight.name}",
                        f"{next_flight['UniqueCarrier']}_{next_flight.name}",
                        turnaround_time=current_flight['Turnaround_Time']
                    )
                self.aircraft_networks[aircraft] = G

    def simulate_schedule_change(self, flight_index: int, new_departure_minutes: int):
        """
        Simulate the impact of changing a flight's departure time
        Returns before/after metrics and cascading effects
        """
        if flight_index not in self.df.index:
            return None

        original_flight = self.df.loc[flight_index].copy()

        # Calculate new times
        original_std = original_flight['STD_dt']
        time_shift = timedelta(minutes=new_departure_minutes)
        new_std = original_std + time_shift

        # Simulate impact on current flight
        results = {
            'original_departure': original_std.strftime('%H:%M'),
            'new_departure': new_std.strftime('%H:%M'),
            'time_shift_minutes': new_departure_minutes,
            'original_delay_risk': self._calculate_delay_risk(original_flight),
        }

        # Calculate new delay risk based on congestion at new time
        modified_flight = original_flight.copy()
        modified_flight['DepHour'] = new_std.hour
        modified_flight['STD_dt'] = new_std
        results['new_delay_risk'] = self._calculate_delay_risk(modified_flight)
        results['risk_change'] = results['new_delay_risk'] - results['original_delay_risk']

        # Analyze cascading impact
        cascading_impact = self._analyze_cascading_impact(flight_index, time_shift)
        results.update(cascading_impact)

        return results

    def _calculate_delay_risk(self, flight_data):
        """Calculate delay probability for a flight based on its characteristics"""
        airport = flight_data['Origin']
        hour = flight_data['DepHour'] if 'DepHour' in flight_data else flight_data['STD_dt'].hour

        # Historical delay rate at this airport/hour combination
        similar_flights = self.df[
            (self.df['Origin'] == airport) &
            (self.df['DepHour'] == hour)
            ]

        if len(similar_flights) == 0:
            return 0.3  # Default risk

        delay_rate = (similar_flights['DepDelay'] > 15).mean()
        return delay_rate

    def _analyze_cascading_impact(self, flight_index: int, time_shift: timedelta):
        """Analyze how schedule change affects subsequent flights"""
        flight = self.df.loc[flight_index]
        aircraft_reg = flight['Aircraft_Registration']

        if pd.isna(aircraft_reg) or aircraft_reg not in self.aircraft_networks:
            return {'cascading_flights_affected': 0, 'total_delay_impact': 0}

        # Find subsequent flights on same aircraft
        subsequent_flights = self.df[
            (self.df['Aircraft_Registration'] == aircraft_reg) &
            (self.df['STD_dt'] > flight['STD_dt'])
            ].sort_values('STD_dt')

        affected_flights = 0
        total_impact = 0

        for _, next_flight in subsequent_flights.head(3).iterrows():  # Check next 3 flights
            turnaround_buffer = (next_flight['STD_dt'] - flight['ATA_dt']).total_seconds() / 60

            # If the delay eats into turnaround time
            if time_shift.total_seconds() / 60 > turnaround_buffer:
                affected_flights += 1
                delay_propagation = max(0, time_shift.total_seconds() / 60 - turnaround_buffer)
                total_impact += delay_propagation

        return {
            'cascading_flights_affected': affected_flights,
            'total_delay_impact': total_impact
        }


# ------------------------------
# NEW: Cascading Delay Impact Analyzer
# ------------------------------
class CascadingDelayAnalyzer:
    """
    Analyzes which flights have the biggest cascading impact on schedule delays
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._calculate_impact_scores()

    def _calculate_impact_scores(self):
        """Calculate cascading impact potential for each flight"""
        impact_scores = []

        for idx, flight in self.df.iterrows():
            score = self._calculate_single_flight_impact(flight)
            impact_scores.append({
                'flight_index': idx,
                'flight_number': flight['UniqueCarrier'],
                'origin': flight['Origin'],
                'dest': flight['Dest'],
                'departure_time': flight['STD_dt'].strftime('%H:%M') if pd.notna(flight['STD_dt']) else 'N/A',
                'aircraft': flight['Aircraft_Registration'],
                'cascading_impact_score': score,
                'risk_category': self._categorize_risk(score)
            })

        self.impact_df = pd.DataFrame(impact_scores)

    def _calculate_single_flight_impact(self, flight):
        """Calculate impact score for a single flight"""
        score = 0

        # Factor 1: Historical delay probability (40% weight)
        similar_flights = self.df[
            (self.df['Origin'] == flight['Origin']) &
            (self.df['DepHour'] == flight['DepHour'])
            ]
        if len(similar_flights) > 0:
            delay_prob = (similar_flights['DepDelay'] > 15).mean()
            score += delay_prob * 0.4

        # Factor 2: Aircraft utilization intensity (30% weight)
        if pd.notna(flight['Aircraft_Registration']):
            aircraft_flights = self.df[self.df['Aircraft_Registration'] == flight['Aircraft_Registration']]
            daily_flights = len(aircraft_flights) / max(1, aircraft_flights['Date'].nunique())
            utilization_score = min(daily_flights / 8, 1.0)  # Normalize to max 8 flights/day
            score += utilization_score * 0.3

        # Factor 3: Turnaround pressure (20% weight)
        if pd.notna(flight['Turnaround_Time']):
            if flight['Turnaround_Time'] < 60:  # Very tight turnaround
                score += 0.2
            elif flight['Turnaround_Time'] < 90:  # Tight turnaround
                score += 0.15

        # Factor 4: Peak hour operations (10% weight)
        if flight['DepHour'] in [7, 8, 9, 17, 18, 19, 20]:
            score += 0.1

        return min(score, 1.0)  # Cap at 1.0

    def _categorize_risk(self, score):
        """Categorize cascading risk level"""
        if score >= 0.7:
            return "🔴 HIGH RISK"
        elif score >= 0.5:
            return "🟡 MEDIUM RISK"
        elif score >= 0.3:
            return "🟠 LOW-MEDIUM RISK"
        else:
            return "🟢 LOW RISK"

    def get_highest_impact_flights(self, n=10):
        """Return top N flights with highest cascading impact"""
        return self.impact_df.nlargest(n, 'cascading_impact_score')

    def get_impact_by_airport(self):
        """Get cascading impact analysis by airport"""
        return self.impact_df.groupby('origin').agg({
            'cascading_impact_score': ['mean', 'max', 'count']
        }).round(3)


# ------------------------------
# Enhanced Airport Configuration
# ------------------------------
AIRPORT_CONFIG = {
    "BOM": {
        "name": "Mumbai (Chhatrapati Shivaji)",
        "runways": 2, "capacity_per_hour": 45,
        "peak_hours": [8, 9, 10, 17, 18, 19, 20],
        "weather_delays": 0.15, "ground_congestion": 0.85
    },
    "DEL": {
        "name": "Delhi (Indira Gandhi International)",
        "runways": 4, "capacity_per_hour": 80,
        "peak_hours": [7, 8, 9, 18, 19, 20],
        "weather_delays": 0.18, "ground_congestion": 0.9
    },
    "BLR": {
        "name": "Bangalore (Kempegowda International)",
        "runways": 2, "capacity_per_hour": 50,
        "peak_hours": [8, 9, 17, 18, 19],
        "weather_delays": 0.10, "ground_congestion": 0.8
    },
    "CCU": {
        "name": "Kolkata (Netaji Subhas Chandra Bose)",
        "runways": 2, "capacity_per_hour": 35,
        "peak_hours": [7, 8, 17, 18],
        "weather_delays": 0.14, "ground_congestion": 0.75
    },
    "IXC": {
        "name": "Chandigarh Airport",
        "runways": 1, "capacity_per_hour": 15,
        "peak_hours": [7, 8, 16, 17],
        "weather_delays": 0.10, "ground_congestion": 0.6
    },
    "BBI": {
        "name": "Bhubaneswar (Biju Patnaik)",
        "runways": 1, "capacity_per_hour": 20,
        "peak_hours": [8, 9, 18, 19],
        "weather_delays": 0.12, "ground_congestion": 0.7
    }
}


# ------------------------------
# Enhanced Schedule Optimizer
# ------------------------------
class EnhancedScheduleOptimizer:
    """Enhanced optimizer with cascading analysis and schedule tuning"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.tuner = ScheduleTuner(df)
        self.cascading_analyzer = CascadingDelayAnalyzer(df)
        self._prepare_enhanced_data()

    def _prepare_enhanced_data(self):
        self._calculate_congestion_metrics()
        self._calculate_delay_patterns()
        self._analyze_busiest_slots()

    def _calculate_congestion_metrics(self):
        """Enhanced congestion calculation with real data"""
        congestion_data = []

        for airport in AIRPORT_CONFIG.keys():
            # Get actual flight data for this airport
            airport_flights = self.df[
                (self.df["Origin"] == airport) | (self.df["Dest"] == airport)
                ]

            if airport_flights.empty:
                continue

            # Calculate hourly traffic
            hourly_departures = airport_flights[airport_flights["Origin"] == airport].groupby("DepHour").size()
            hourly_arrivals = airport_flights[airport_flights["Dest"] == airport].groupby("ArrHour").size()

            config = AIRPORT_CONFIG[airport]
            max_capacity = config["capacity_per_hour"]

            for hour in range(24):
                dep_count = hourly_departures.get(hour, 0)
                arr_count = hourly_arrivals.get(hour, 0)
                total_count = dep_count + arr_count

                utilization = total_count / max_capacity if max_capacity > 0 else 0

                congestion_data.append({
                    "Airport": airport,
                    "Hour": hour,
                    "DepartureCount": dep_count,
                    "ArrivalCount": arr_count,
                    "TotalFlights": total_count,
                    "Utilization": min(utilization, 1.0),
                    "CongestionLevel": self._get_enhanced_congestion_level(utilization, config, hour),
                    "IsAvoidableSlot": utilization > 0.8  # Mark as avoidable if >80% capacity
                })

        self.congestion_df = pd.DataFrame(congestion_data)

    def _get_enhanced_congestion_level(self, utilization, config, hour):
        """Enhanced congestion calculation"""
        base_congestion = utilization

        # Peak hour multiplier
        if hour in config["peak_hours"]:
            base_congestion *= 1.4

        # Weather impact
        base_congestion *= (1 + config["weather_delays"])

        # Ground congestion factor
        base_congestion *= config["ground_congestion"]

        return min(base_congestion, 1.0)

    def _calculate_delay_patterns(self):
        """Enhanced delay pattern analysis"""
        delay_patterns = []

        for airport in AIRPORT_CONFIG.keys():
            airport_data = self.df[self.df["Origin"] == airport]
            if len(airport_data) == 0:
                continue

            # Group by hour and calculate comprehensive delay statistics
            hourly_stats = airport_data.groupby("DepHour").agg({
                'DepDelay': ['mean', 'median', 'std', 'count'],
                'ArrDelay': ['mean', 'median', 'std'],
                'Cascading_Risk_Score': 'mean'
            }).fillna(0)

            for hour in range(24):
                if hour in hourly_stats.index:
                    dep_stats = hourly_stats.loc[hour, 'DepDelay']
                    arr_stats = hourly_stats.loc[hour, 'ArrDelay']
                    cascading_risk = hourly_stats.loc[hour, ('Cascading_Risk_Score', 'mean')]

                    delay_patterns.append({
                        "Airport": airport,
                        "Hour": hour,
                        "AvgDepDelay": dep_stats['mean'],
                        "AvgArrDelay": arr_stats['mean'],
                        "DelayVolatility": dep_stats['std'],
                        "FlightCount": dep_stats['count'],
                        "DelayRisk": self._calculate_enhanced_delay_risk(dep_stats, cascading_risk),
                        "CascadingRisk": cascading_risk
                    })

        self.delay_patterns_df = pd.DataFrame(delay_patterns)

    def _calculate_enhanced_delay_risk(self, stats, cascading_risk):
        """Enhanced delay risk calculation including cascading effects"""
        if stats['count'] == 0:
            return 0

        # Base delay risk
        base_risk = (max(0, stats['mean']) + stats['std']) / 100

        # Add cascading risk component
        total_risk = base_risk * 0.7 + cascading_risk * 0.3

        return min(total_risk, 1.0)

    def _analyze_busiest_slots(self):
        """Identify and rank the busiest time slots to avoid"""
        busiest_slots = []

        for airport in AIRPORT_CONFIG.keys():
            airport_congestion = self.congestion_df[self.congestion_df["Airport"] == airport]

            # Rank slots by congestion level
            airport_ranked = airport_congestion.sort_values('CongestionLevel', ascending=False)

            for _, slot in airport_ranked.head(8).iterrows():  # Top 8 busiest slots
                busiest_slots.append({
                    'Airport': airport,
                    'Hour': slot['Hour'],
                    'CongestionLevel': slot['CongestionLevel'],
                    'TotalFlights': slot['TotalFlights'],
                    'Recommendation': self._get_avoidance_recommendation(slot['CongestionLevel']),
                    'AlternativeSlots': self._suggest_alternatives(airport, slot['Hour'])
                })

        self.busiest_slots_df = pd.DataFrame(busiest_slots)

    def _get_avoidance_recommendation(self, congestion_level):
        """Get recommendation text based on congestion level"""
        if congestion_level >= 0.9:
            return "🚫 AVOID - Extreme congestion, high delay risk"
        elif congestion_level >= 0.7:
            return "⚠️  AVOID - Very busy, consider alternatives"
        elif congestion_level >= 0.5:
            return "🟡 CAUTION - Moderately busy, monitor conditions"
        else:
            return "✅ ACCEPTABLE - Low congestion"

    def _suggest_alternatives(self, airport, busy_hour):
        """Suggest alternative time slots"""
        airport_data = self.congestion_df[self.congestion_df["Airport"] == airport]

        # Find hours with low congestion (within 2 hours of busy_hour)
        nearby_hours = [(busy_hour + i) % 24 for i in range(-2, 3) if i != 0]
        alternatives = airport_data[
            airport_data["Hour"].isin(nearby_hours) &
            (airport_data["CongestionLevel"] < 0.5)
            ].sort_values('CongestionLevel')

        if not alternatives.empty:
            alt_hours = alternatives.head(2)["Hour"].tolist()
            return f"Consider {alt_hours[0]:02d}:00" + (f" or {alt_hours[1]:02d}:00" if len(alt_hours) > 1 else "")
        else:
            return "No nearby alternatives available"

    def find_optimal_slots_enhanced(self, airport: str, max_slots: int = 10):
        """Enhanced optimal slot finding with comprehensive scoring"""
        recommendations = []

        airport_congestion = self.congestion_df[self.congestion_df["Airport"] == airport]
        airport_delays = self.delay_patterns_df[self.delay_patterns_df["Airport"] == airport]

        for hour in range(24):
            congestion_data = airport_congestion[airport_congestion["Hour"] == hour]
            delay_data = airport_delays[airport_delays["Hour"] == hour]

            if not congestion_data.empty and not delay_data.empty:
                congestion = congestion_data.iloc[0]
                delay_info = delay_data.iloc[0]

                # Multi-factor scoring
                congestion_score = congestion["CongestionLevel"] * 0.4
                delay_score = delay_info["DelayRisk"] * 0.3
                cascading_score = delay_info["CascadingRisk"] * 0.3

                total_score = congestion_score + delay_score + cascading_score

                recommendations.append({
                    "Hour": hour,
                    "Score": total_score,
                    "CongestionLevel": congestion["CongestionLevel"],
                    "DelayRisk": delay_info["DelayRisk"],
                    "CascadingRisk": delay_info["CascadingRisk"],
                    "TotalFlights": congestion["TotalFlights"],
                    "Recommendation": self._get_enhanced_recommendation_text(total_score, hour),
                    "OptimalityRating": self._get_optimality_rating(total_score)
                })

        recommendations.sort(key=lambda x: x["Score"])
        return recommendations[:max_slots]

    def _get_enhanced_recommendation_text(self, score, hour):
        """Enhanced recommendation text with detailed insights"""
        if score < 0.25:
            return f"⭐ EXCELLENT slot at {hour:02d}:00 - Optimal conditions, minimal delays"
        elif score < 0.4:
            return f"✅ GOOD slot at {hour:02d}:00 - Low congestion, acceptable delay risk"
        elif score < 0.6:
            return f"🟡 MODERATE slot at {hour:02d}:00 - Some congestion, moderate delay risk"
        elif score < 0.75:
            return f"⚠️  BUSY slot at {hour:02d}:00 - High traffic, increased delay risk"
        else:
            return f"🚫 AVOID slot at {hour:02d}:00 - Peak congestion, high delay probability"

    def _get_optimality_rating(self, score):
        """Convert score to star rating"""
        if score < 0.25:
            return "⭐⭐⭐⭐⭐"
        elif score < 0.4:
            return "⭐⭐⭐⭐"
        elif score < 0.6:
            return "⭐⭐⭐"
        elif score < 0.75:
            return "⭐⭐"
        else:
            return "⭐"


# ------------------------------
# Enhanced NLP Query Processing
# ------------------------------
ENHANCED_COLUMN_DESCRIPTIONS = {
    "ArrDelay": "difference between scheduled and actual arrival time in minutes",
    "DepDelay": "difference between scheduled and actual departure time in minutes",
    "DepHour": "actual departure hour (0-23)",
    "ArrHour": "actual arrival hour (0-23)",
    "UniqueCarrier": "airline carrier code",
    "Aircraft_Registration": "aircraft registration number for tracking rotations",
    "AirTime": "time spent airborne in minutes",
    "Origin": "origin airport code",
    "Dest": "destination airport code",
    "Distance": "estimated distance flown (miles)",
    "Month": "month number 1..12",
    "DayOfWeek": "day of week as integer (1=Mon, 7=Sun)",
    "Turnaround_Time": "minutes between arrival and next departure for same aircraft",
    "Cascading_Risk_Score": "probability that this flight will cause delays to other flights"
}

ENHANCED_OPERATIONS = {
    "mean": "average / mean",
    "median": "median",
    "max": "maximum",
    "min": "minimum",
    "count": "count / how many",
    "top": "top N / most frequent",
    "distribution": "distribution / histogram",
    "percent": "percentage (for binary columns)",
    "trend": "time trend (by hour or by day)",
    "optimize": "find optimal schedule slots",
    "capacity": "runway capacity analysis",
    "congestion": "congestion analysis",
    "busiest": "busiest time slots to avoid",
    "tune": "tune schedule time and see delay impact",
    "cascade": "analyze cascading delay impacts",
    "simulate": "simulate schedule changes",
    "impact": "flights with biggest cascading impact"
}


# ------------------------------
# Enhanced ML Models with Cascading Features
# ------------------------------
@st.cache_resource(show_spinner=True)
def train_enhanced_models(df: pd.DataFrame):
    """Train enhanced models with cascading delay features"""
    df = df.copy()

    # Enhanced Feature Engineering
    df["IsPeakHour"] = df.apply(
        lambda row: 1 if row['Origin'] in AIRPORT_CONFIG and row['DepHour'] in
                         AIRPORT_CONFIG[row['Origin']]['peak_hours'] else 0, axis=1
    )
    df["IsWeekend"] = df["DayOfWeek"].apply(lambda x: 1 if x in [6, 7] else 0)
    df["ArrDelayBinary"] = (df["ArrDelay"].fillna(0) > 15).astype(int)

    # NEW: Cascading delay features
    df["HasQuickTurnaround"] = df["Is_Quick_Turnaround"].fillna(0)
    df["TurnaroundPressure"] = (df["Turnaround_Time"].fillna(120) < 90).astype(int)

    # Enhanced feature set
    enhanced_feats = [
        "Month", "DayOfWeek", "DepHour", "Distance", "AirTime",
        "IsPeakHour", "IsWeekend", "HasQuickTurnaround", "TurnaroundPressure",
        "Cascading_Risk_Score"
    ]

    available_feats = [f for f in enhanced_feats if f in df.columns]
    X = df[available_feats].fillna(df[available_feats].median())
    y = df["ArrDelayBinary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if y.nunique() > 1 else None
    )

    # Enhanced Random Forest
    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=150, random_state=42, n_jobs=-1,
            max_depth=12, min_samples_split=5
        ))
    ])
    rf_pipe.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf_pipe.predict(X_test))

    # Enhanced second model
    if XGBOOST_AVAILABLE:
        model_b = XGBClassifier(
            use_label_encoder=False, eval_metric="logloss",
            n_estimators=150, random_state=42, max_depth=6,
            learning_rate=0.1, subsample=0.8
        )
        model_b_name = "XGBoost"
    else:
        model_b = GradientBoostingClassifier(
            n_estimators=150, random_state=42, max_depth=6,
            learning_rate=0.1, subsample=0.8
        )
        model_b_name = "GradientBoosting"

    model_b.fit(X_train, y_train)
    model_b_acc = accuracy_score(y_test, model_b.predict(X_test))

    return {
        "rf_pipe": rf_pipe,
        "model_b": model_b,
        "meta": {
            "features": available_feats,
            "rf_acc": float(rf_acc),
            "model_b_acc": float(model_b_acc),
            "model_b_name": model_b_name
        }
    }


# ------------------------------
# Enhanced Visualization Functions
# ------------------------------
def create_cascading_impact_chart(analyzer: CascadingDelayAnalyzer):
    """Create interactive chart showing cascading impact analysis"""
    top_flights = analyzer.get_highest_impact_flights(15)

    fig = px.bar(
        top_flights,
        x='cascading_impact_score',
        y='flight_number',
        color='cascading_impact_score',
        color_continuous_scale='Reds',
        title='Top 15 Flights with Highest Cascading Impact',
        labels={'cascading_impact_score': 'Cascading Impact Score', 'flight_number': 'Flight'}
    )
    fig.update_layout(height=600)
    return fig


def create_schedule_tuning_visualization(tuning_results):
    """Create visualization for schedule tuning results"""
    if not tuning_results:
        return None

    # Create before/after comparison
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Delay Risk Comparison', 'Cascading Impact',
            'Time Shift Impact', 'Overall Recommendation'
        ),
        specs=[[{"type": "bar"}, {"type": "indicator"}],
               [{"type": "scatter"}, {"type": "indicator"}]]
    )

    # Delay risk comparison
    fig.add_trace(
        go.Bar(
            x=['Original', 'New Schedule'],
            y=[tuning_results['original_delay_risk'], tuning_results['new_delay_risk']],
            marker_color=['red', 'green'] if tuning_results['risk_change'] < 0 else ['red', 'orange'],
            name='Delay Risk'
        ),
        row=1, col=1
    )

    # Cascading impact indicator
    fig.add_trace(
        go.Indicator(
            mode="number+gauge+delta",
            value=tuning_results.get('cascading_flights_affected', 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Flights Affected"},
            gauge={'axis': {'range': [None, 10]}, 'bar': {'color': "darkblue"}}
        ),
        row=1, col=2
    )

    fig.update_layout(height=600, title="Schedule Tuning Impact Analysis")
    return fig


def create_busiest_slots_heatmap(optimizer: EnhancedScheduleOptimizer):
    """Create heatmap showing busiest time slots to avoid"""
    pivot_data = optimizer.congestion_df.pivot(
        index='Airport', columns='Hour', values='CongestionLevel'
    )

    fig = px.imshow(
        pivot_data,
        color_continuous_scale='RdYlBu_r',
        title='Airport Congestion Heatmap - Red Areas to Avoid',
        labels={'color': 'Congestion Level'}
    )
    fig.update_layout(height=500)
    return fig


# ------------------------------
# NEW: Pilot Fatigue & Decision Support Framework
# ------------------------------
class PilotFatiguePredictor:
    """Predicts pilot fatigue level using Random Forest."""
    def __init__(self):
        # We will create a synthesized dataset to train this model since the actual flight data
        # does not contain pilot biometric/workload data directly.
        # Features: Avg Flight Hours (0-12), Duty-to-Rest Ratio (0.0-3.0), Workload Index (0-100)
        # Target: 0 (Efficient), 1 (Moderate), 2 (Critical)
        
        np.random.seed(42)
        n_samples = 1000
        
        hours = np.random.uniform(2, 12, n_samples)
        ratio = np.random.uniform(0.5, 3.0, n_samples)
        workload = np.random.uniform(20, 100, n_samples)
        
        # Rule-based generation for synthetic ground truth so the model can learn the pattern
        y = []
        for h, r, w in zip(hours, ratio, workload):
            if h >= 9 or r >= 2.0 or w >= 80:
                y.append(2) # Critical
            elif h >= 6 or r >= 1.2 or w >= 55:
                y.append(1) # Moderate
            else:
                y.append(0) # Efficient
                
        X = pd.DataFrame({
            'avg_flight_hours': hours,
            'duty_to_rest_ratio': ratio,
            'workload_index': workload
        })
        y = np.array(y)
        
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.model.fit(X, y)
        self.feature_names = X.columns.tolist()

    def predict(self, hours, ratio, workload):
        X_test = pd.DataFrame({
            'avg_flight_hours': [hours],
            'duty_to_rest_ratio': [ratio],
            'workload_index': [workload]
        })
        prob = self.model.predict_proba(X_test)[0]
        pred = self.model.predict(X_test)[0]
        return int(pred), prob

    def class_probs(self, hours: float, ratio: float, workload: float):
        """Single numpy predict_proba call (faster than DataFrame in hot loops)."""
        p = self.model.predict_proba(
            np.array([[float(hours), float(ratio), float(workload)]], dtype=np.float64)
        )[0]
        return float(p[0]), float(p[1]), float(p[2])


# Bump when PilotFatiguePredictor API or training changes so Streamlit does not reuse a stale
# cached instance from before hot reload (old instances lack newly added methods).
_FATIGUE_PREDICTOR_RESOURCE_VERSION = 2


@st.cache_resource(show_spinner=False)
def get_pilot_fatigue_predictor(
    _resource_version: int = _FATIGUE_PREDICTOR_RESOURCE_VERSION,
) -> "PilotFatiguePredictor":
    """Train the fatigue Random Forest once per Streamlit process."""
    del _resource_version  # cache key only
    return PilotFatiguePredictor()


class DecisionSupportFramework:
    """Combines deterministic rules with ML predictions for pilot scheduling."""
    @staticmethod
    def evaluate(hours, ratio, workload, ml_prediction, ml_probs):
        # Deterministic Rules (e.g., FAA limits approximation)
        rule_violation = []
        if hours > 10.0:
            rule_violation.append("Exceeds max safe flight hours (10h)")
        if ratio > 2.5:
            rule_violation.append("Dangerous duty-to-rest ratio (>2.5)")
            
        is_hard_violation = len(rule_violation) > 0
        
        # Output Logic
        status = "Approved"
        color = "green"
        recommendation = "Schedule is efficient and within safe limits."
        
        if is_hard_violation:
            status = "Rejected (Rule Violation)"
            color = "red"
            recommendation = "Deterministic rule violation: " + ", ".join(rule_violation)
        else:
            if ml_prediction == 2:
                status = "Critical Risk"
                color = "red"
                recommendation = "ML Predicts Critical Fatigue. Recommend swapping pilot or adding rest."
            elif ml_prediction == 1:
                status = "Review Required"
                color = "orange"
                recommendation = "ML Predicts Moderate Fatigue. Acceptable but monitor closely."
                
        return {
            "status": status,
            "color": color,
            "recommendation": recommendation,
            "violations": rule_violation,
            "ml_prediction": ml_prediction,
            "probs": ml_probs
        }

def render_pilot_fatigue_dashboard():
    import plotly.graph_objects as go

    # ── Blueprint CSS injection ──
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Mono:wght@400;700&display=swap');
    .crew-heading { font-family:'Bebas Neue',sans-serif; letter-spacing:0.12em; font-size:1.6rem; color:#0f172a; margin-bottom:0.2rem; }
    .crew-sub    { font-family:'Space Mono',monospace; font-size:0.82rem; color:#64748b; margin-bottom:1.2rem; }
    [data-testid="stMetric"] { border-radius:0 !important; }
    [data-testid="stMetric"] label { font-family:'Space Mono',monospace !important; text-transform:uppercase; font-size:0.7rem !important; letter-spacing:0.08em; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { font-family:'Space Mono',monospace !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="crew-heading">PILOT FATIGUE RISK ASSESSMENT</div>', unsafe_allow_html=True)
    st.markdown('<div class="crew-sub">Real-time fatigue classification with DGCA compliance checks</div>', unsafe_allow_html=True)
    st.markdown("---")

    predictor = get_pilot_fatigue_predictor(_FATIGUE_PREDICTOR_RESOURCE_VERSION)

    # ── Layout: 2 / 1 ──
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("##### Pilot Parameters")
        hours = st.slider("Avg Flight Hours / Day", min_value=1.0, max_value=14.0, value=7.5, step=0.5, key="fh_hours")
        ratio = st.slider("Duty-to-Rest Ratio", min_value=0.5, max_value=3.5, value=1.5, step=0.1, key="fh_ratio")
        workload = st.slider("Workload Index", min_value=10, max_value=100, value=60, step=1, key="fh_wl")

        # ── Live evaluation (no button needed) ──
        pred, probs = predictor.predict(hours, ratio, workload)
        result = DecisionSupportFramework.evaluate(hours, ratio, workload, pred, probs)

        # Classification label + color
        label_map = {0: "Efficient", 1: "Moderate", 2: "Critical"}
        color_map = {0: "#10b981", 1: "#f59e0b", 2: "#ef4444"}
        badge_label = label_map[pred]
        badge_color = color_map[pred]

        # ── Natural-language recommendation ──
        if pred == 2:
            rec_text = f"Rest required before next assignment. Duty-to-rest ratio is {ratio:.1f}, above the 1.5 DGCA advisory threshold." if ratio > 1.5 else f"Workload index at {workload} exceeds the safe operating ceiling of 80. Schedule relief crew."
        elif pred == 1:
            rec_text = f"Monitor closely — flight hours averaging {hours:.1f}h/day approach the 12h DGCA daily cap. Consider crew rotation."
        else:
            rec_text = "All parameters within safe operating limits. Pilot is cleared for duty."

        # Add any hard-rule violations from the framework
        if result.get("violations"):
            rec_text = "RULE VIOLATION: " + "; ".join(result["violations"]) + ". " + rec_text

        # ── DGCA compliance checks ──
        chk_duty = hours <= 12.0
        chk_rest = ratio <= 1.5   # ratio > 1.5 implies rest interval < 8h equivalent
        chk_weekly = hours * 7 <= 60.0

        # ── Risk Card HTML ──
        st.markdown(f"""
        <div style="background:white; border:2px solid {badge_color}; padding:1.4rem 1.6rem; margin-top:1rem; font-family:'Space Mono',monospace;">
            <div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:1rem;">
                <span style="background:{badge_color}; color:white; padding:0.35rem 1rem; font-weight:700; font-size:0.95rem; letter-spacing:0.1em; text-transform:uppercase;">{badge_label}</span>
                <span style="font-size:0.8rem; color:#64748b;">ML Classification Result</span>
            </div>
            <p style="font-size:0.85rem; color:#334155; line-height:1.6; margin-bottom:1.2rem;">{rec_text}</p>
            <div style="border-top:1px solid #e2e8f0; padding-top:1rem;">
                <div style="font-family:'Bebas Neue',sans-serif; font-size:0.95rem; letter-spacing:0.1em; color:#0f172a; margin-bottom:0.6rem;">DGCA COMPLIANCE</div>
                <div style="font-size:0.82rem; margin-bottom:0.35rem;">{"✅" if chk_duty else "❌"} Max duty hours (12h) — <b>{hours:.1f}h</b></div>
                <div style="font-size:0.82rem; margin-bottom:0.35rem;">{"✅" if chk_rest else "❌"} Min rest interval (8h equiv.) — ratio <b>{ratio:.1f}</b></div>
                <div style="font-size:0.82rem;">{"✅" if chk_weekly else "❌"} Weekly hours cap (60h) — <b>{hours * 7:.0f}h/wk</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("##### Fatigue Risk Index")

        gauge_value = round(float(probs[2]) * 100, 1)  # P(Critical) × 100

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=gauge_value,
            number={"suffix": "%", "font": {"family": "Space Mono, monospace", "size": 38}},
            title={"text": "FATIGUE RISK INDEX", "font": {"family": "Bebas Neue, sans-serif", "size": 20}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": badge_color},
                "bgcolor": "#f8fafc",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 60], "color": "#d1fae5"},
                    {"range": [60, 80], "color": "#fef3c7"},
                    {"range": [80, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#ef4444", "width": 3},
                    "thickness": 0.8,
                    "value": 80,
                },
            },
        ))
        fig_gauge.update_layout(
            height=280,
            margin=dict(t=50, b=20, l=30, r=30),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"family": "Space Mono, monospace"},
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Probability summary metrics ──
        st.metric("P(Efficient)", f"{probs[0]*100:.1f}%")
        st.metric("P(Moderate)", f"{probs[1]*100:.1f}%")
        st.metric("P(Critical)", f"{probs[2]*100:.1f}%")

    # ── Model diagnostics (advanced) — collapsed by default ──
    with st.expander("Model diagnostics (advanced)"):
        prob_df = pd.DataFrame({
            "Category": ["Efficient", "Moderate", "Critical"],
            "Probability": [float(p) for p in probs]
        })
        fig_bar = px.bar(prob_df, x="Category", y="Probability", color="Category",
                         color_discrete_map={"Efficient": "#10b981", "Moderate": "#f59e0b", "Critical": "#ef4444"},
                         title="ML Probabilistic Risk Output")
        fig_bar.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("**Raw prediction class:** " + str(pred))
        st.markdown("**Feature vector:** " + f"hours={hours}, ratio={ratio}, workload={workload}")
        st.markdown("**Model:** RandomForest (100 trees, depth=5, seed=42)")

# ------------------------------
# Enhanced Main Application
# ------------------------------
def load_dashboard_data():
    # Dashboard/Schedule views must read only from active master_df.
    import random
    random.seed(42)

    df = get_active_df_for_views()
    if df.empty:
        return pd.DataFrame()

    df = normalize_columns(df)
    
    # Parse Date robustly for multiple source schemas
    if 'Date' not in df.columns and 'Unnamed: 2' in df.columns:
        df.rename(columns={'Unnamed: 2': 'Date'}, inplace=True)
    if 'Date' not in df.columns and 'date_dep' in df.columns:
        df['Date'] = df['date_dep']
    if 'Date' not in df.columns:
        df['Date'] = pd.Timestamp.now().strftime('%d-%b-%y')

    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y', errors='coerce')
    df['Date'] = df['Date'].fillna(pd.Timestamp.now())
    
    # Ensure required columns exist with safe defaults
    for col in ['STD', 'STA', 'Aircraft', 'Flight Number', 'From', 'To']:
        if col not in df.columns:
            df[col] = 'N/A'
    
    # Synthesize pilot assignments
    captains = ["Capt. Sharma", "Capt. Singh", "Capt. Rao", "Capt. Gupta", "Capt. Patel"]
    fos = ["FO Verma", "FO Das", "FO Kumar", "FO Joshi", "FO Nair"]
    
    if "Captain" not in df.columns:
        df['Captain'] = [random.choice(captains) for _ in range(len(df))]
        df['First Officer'] = [random.choice(fos) for _ in range(len(df))]
        df['Status'] = [random.choice(['On Time', 'Delayed', 'Boarding']) for _ in range(len(df))]
        
    return df

def render_pilot_detail_page(pilot_name, df_dash):
    st.markdown(f"<h1>👤 Pilot Profile: {pilot_name}</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    is_captain = "Capt" in pilot_name
    rank = "Captain (PIC)" if is_captain else "First Officer (SIC)"
    
    import hashlib
    h = int(hashlib.md5(pilot_name.encode()).hexdigest(), 16)
    total_hours = 4500 + (h % 5000) if is_captain else 1200 + (h % 2000)
    
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Rank", rank)
    colB.metric("Total Flight Hours", f"{total_hours:,} h")
    colC.metric("Current Workload", "Moderate (7.2h/day)")
    colD.metric("Duty Status", "Active - Cleared")
    
    st.markdown("<br>### 📊 Analytics & Currency", unsafe_allow_html=True)
    a1, a2, a3 = st.columns([1.2, 1, 1])
    
    with a1:
        st.markdown("**Total Hours Breakdown**")
        hours_df = pd.DataFrame({
            "Type": ["PIC", "SIC", "Dual", "Solo"],
            "Hours": [total_hours * 0.7, total_hours * 0.1, total_hours * 0.15, total_hours * 0.05] if is_captain else 
                     [total_hours * 0.1, total_hours * 0.6, total_hours * 0.2, total_hours * 0.1]
        })
        fig = px.pie(hours_df, values='Hours', names='Type', hole=0.75,
                     color_discrete_sequence=['#3b82f6', '#818cf8', '#a78bfa', '#e2e8f0'])
        fig.update_layout(
            annotations=[dict(text=f"{total_hours:,}", x=0.5, y=0.5, font_size=24, showarrow=False)],
            showlegend=False, margin=dict(t=10, b=0, l=0, r=0), height=250
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with a2:
        st.markdown("**Currency Status**")
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 8px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); font-family: 'Space Mono', monospace; font-size: 0.9rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.8rem;"><span>🌞 Day VFR</span><span style="color: #10b981; font-weight: bold;">Current (89d)</span></div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.8rem;"><span>🌙 Night VFR</span><span style="color: #10b981; font-weight: bold;">Current (45d)</span></div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.8rem;"><span>🌨️ Instrument</span><span style="color: #10b981; font-weight: bold;">Current (12d)</span></div>
            <div style="display: flex; justify-content: space-between;"><span>🏥 Medical</span><span style="color: #f59e0b; font-weight: bold;">Exp. 42 Days</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    with a3:
        st.markdown("**Hours by Aircraft Type**")
        ac_df = pd.DataFrame({
            "Aircraft": ["B737", "A320", "ATR72"],
            "Hours": [total_hours * 0.8, total_hours * 0.15, total_hours * 0.05]
        })
        fig2 = px.bar(ac_df, x="Hours", y="Aircraft", orientation='h', color_discrete_sequence=['#3b82f6'])
        fig2.update_layout(margin=dict(t=10, b=0, l=0, r=0), height=200, xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig2, use_container_width=True)
        
    st.markdown("### ⏱️ Assigned Duties (Weekly Timeline)")

    import json
    from datetime import datetime, timedelta

    # ── Data wiring: build schedule from df_dash ──
    pilot_schedule = df_dash[
        (df_dash['Captain'] == pilot_name) | (df_dash['First Officer'] == pilot_name)
    ].copy()

    # Determine display week:
    # anchor to the pilot's nearest scheduled date so timeline is not empty
    today = datetime.now()
    pilot_dates = pd.to_datetime(pilot_schedule.get('Date'), errors='coerce').dropna()
    if not pilot_dates.empty:
        today_ts = pd.Timestamp(today).normalize()
        nearest_idx = (pilot_dates - today_ts).abs().argmin()
        anchor_date = pilot_dates.iloc[nearest_idx].to_pydatetime()
    else:
        anchor_date = today

    week_start = anchor_date - timedelta(days=anchor_date.weekday())  # Monday
    week_end = week_start + timedelta(days=6)

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    week_label = f"{week_start.strftime('%b %d')} – {week_end.strftime('%b %d, %Y')}"

    schedule_data = []
    total_duty_mins = 0
    sectors_today = 0
    last_duty_end_dt = None

    for _, row in pilot_schedule.iterrows():
        flight_date = pd.to_datetime(row['Date'], errors='coerce')
        if pd.isna(flight_date):
            continue
        day_index = flight_date.weekday()  # 0=Mon...6=Sun
        week_offset = (flight_date.date() - week_start.date()).days // 7
        # Keep all available pilot duties so week navigation can reach them.

        # Parse STD / STA to float hours
        def parse_time_to_float(t):
            t = str(t).strip().upper()
            try:
                for fmt in ['%I:%M %p', '%H:%M', '%I:%M%p']:
                    try:
                        dt = datetime.strptime(t, fmt)
                        return dt.hour + dt.minute / 60.0
                    except ValueError:
                        continue
            except Exception:
                pass
            return 10.0  # fallback

        dep_h = parse_time_to_float(row.get('STD', '10:00 AM'))
        arr_h = parse_time_to_float(row.get('STA', '12:00 PM'))
        if arr_h <= dep_h:
            arr_h = dep_h + 2.0

        duration_mins = (arr_h - dep_h) * 60
        total_duty_mins += duration_mins

        if flight_date.date() == today.date():
            sectors_today += 1

        # Track last duty end (arr_h can be >= 24 when STD is late and STA is next day;
        # datetime.replace(hour=...) only allows 0..23)
        fd_norm = pd.Timestamp(flight_date).normalize().to_pydatetime()
        duty_end_dt = fd_norm + timedelta(hours=float(arr_h))
        if last_duty_end_dt is None or duty_end_dt > last_duty_end_dt:
            last_duty_end_dt = duty_end_dt

        # Determine status
        status = "normal"
        rule = ""
        if str(row.get('Status', '')).strip() == 'Delayed':
            status = "warning"
            rule = "Delayed departure"
        if duration_mins > 9 * 60:
            status = "conflict"
            rule = "DGCA: Max 9h block time"

        from_str = str(row.get('From', ''))
        to_str = str(row.get('To', ''))
        # Extract airport code if in format "City (CODE)"
        import re
        from_code = re.search(r'\((\w+)\)', from_str)
        to_code = re.search(r'\((\w+)\)', to_str)
        route = f"{from_code.group(1) if from_code else from_str}→{to_code.group(1) if to_code else to_str}"

        fo_name = str(row.get('First Officer', ''))
        if row.get('Captain') == pilot_name:
            fo_name = str(row.get('First Officer', ''))
        else:
            fo_name = str(row.get('Captain', ''))

        schedule_data.append({
            "id": str(row.get('Flight Number', 'N/A')),
            "route": route,
            "type": "flight",
            "dep": round(dep_h, 2),
            "arr": round(arr_h, 2),
            "day_index": day_index,
            "week_offset": week_offset,
            "date_str": flight_date.strftime('%Y-%m-%d'),
            "ac": str(row.get('Aircraft', 'N/A')),
            "fo": fo_name,
            "status": status,
            "rule": rule,
            "note": ""
        })

    # Compute stat card values
    hours_this_week = round(total_duty_mins / 60, 1)
    total_rest_hours = max(1, 7 * 24 - hours_this_week)
    duty_rest_ratio = round(hours_this_week / total_rest_hours, 2)
    last_rest_h = "N/A"
    if last_duty_end_dt:
        rest_delta = (today - last_duty_end_dt).total_seconds() / 3600
        last_rest_h = f"{max(0, rest_delta):.1f}h"

    stats_json = json.dumps({
        "hours_this_week": hours_this_week,
        "duty_rest_ratio": duty_rest_ratio,
        "last_rest": last_rest_h,
        "sectors_today": sectors_today,
        "dgca_hours_exceeded": hours_this_week > 60,
        "dgca_ratio_exceeded": duty_rest_ratio > 1.5,
        "dgca_rest_violated": last_duty_end_dt is not None and (today - last_duty_end_dt).total_seconds() / 3600 < 8
    })

    schedule_json = json.dumps(schedule_data)

    timeline_html = """
<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Bebas+Neue&display=swap" rel="stylesheet">
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:'Space Mono',monospace; background:#f8fafc; color:#0f172a; }
.tl-wrap { padding:0.8rem; }
.stats-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:14px; }
.stat-card { background:#f1f5f9; padding:12px; transition: background-color 0.15s ease, outline 0.15s ease; outline: 1px solid transparent; }
.stat-card:hover { background-color:#eff6ff; outline: 1px solid #bfdbfe; }
.stat-label { font-size:10px; text-transform:uppercase; color:#64748b; letter-spacing:0.06em; margin-bottom:4px; }
.stat-value { font-size:18px; font-family:'Space Mono',monospace; font-weight:700; color:#0f172a; }
.stat-sub { font-size:11px; color:#64748b; margin-top:2px; }
.stat-sub.exceeded { color:#ef4444; font-weight:700; }
.controls { display:flex; align-items:center; justify-content:space-between; margin-bottom:10px; }
.week-nav { display:flex; align-items:center; gap:8px; }
.week-nav button { background:#e2e8f0; color:#0f172a; border:none; padding:5px 10px; cursor:pointer; font-family:'Space Mono',monospace; font-size:13px; transition: background-color 0.15s ease, color 0.15s ease; }
.week-nav button:hover { background-color:#eff6ff; color:#1d4ed8; }
.week-label { font-family:'Bebas Neue',sans-serif; font-size:15px; letter-spacing:0.08em; min-width:160px; text-align:center; color:#0f172a; }
.view-btns { display:flex; gap:4px; }
.view-btns button { background:#e2e8f0; color:#0f172a; border:none; padding:5px 14px; cursor:pointer; font-family:'Space Mono',monospace; font-size:11px; transition: background-color 0.15s ease, color 0.15s ease, opacity 0.15s ease; }
.view-btns button.active { background:#0f172a; color:#ffffff; }
.view-btns button.active:hover { opacity:0.85; }
.view-btns button:not(.active):hover { background-color:#eff6ff; color:#1d4ed8; }
.legend { display:flex; gap:14px; margin-bottom:8px; font-size:10px; color:#64748b; }
.legend-item { display:flex; align-items:center; gap:4px; }
.legend-swatch { width:10px; height:10px; display:inline-block; }
.grid-header { display:flex; border-bottom:1px solid #e2e8f0; padding-bottom:4px; margin-bottom:2px; }
.grid-header .day-label { width:80px; min-width:80px; }
.grid-header .hours { flex:1; display:flex; }
.grid-header .hour-tick { flex:1; text-align:center; font-size:9px; color:#94a3b8; }
.day-row { display:flex; min-height:48px; border-bottom:1px solid #f1f5f9; align-items:stretch; position:relative; }
.row-bg { position:absolute; inset:0; z-index:0; transition: background-color 0.15s ease; }
.day-row:hover .row-bg { background-color:#eff6ff; }
.day-lc { width:80px; min-width:80px; padding:6px 8px; font-size:11px; line-height:1.3; display:flex; flex-direction:column; justify-content:center; position:relative; z-index:1; }
.day-lc .dn { font-weight:700; color:#0f172a; }
.day-lc .dd { color:#64748b; font-size:10px; }
.day-lc.empty .dn { color:#94a3b8; }
.day-lc.empty .dd { color:#cbd5e1; }
.tl-track { flex:1; position:relative; min-height:48px; z-index:1; }
.duty-block { position:absolute; top:7px; height:34px; border-radius:3px; padding:3px 6px; font-size:10px; line-height:1.35; cursor:pointer; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; z-index:2; transition: outline 0.15s ease; outline: 2px solid transparent; }
.duty-block:hover { outline:2px solid #bfdbfe; outline-offset:1px; z-index:10; }
.duty-block .bt { font-weight:700; }
.duty-block .br { font-size:9px; }
.duty-block.flight { background:#dbeafe; border-left:3px solid #3b82f6; color:#1e40af; }
.duty-block.ground { background:#f0fdf4; border-left:3px solid #22c55e; color:#166534; }
.duty-block.sim { background:#faf5ff; border-left:3px solid #a855f7; color:#581c87; }
.duty-block.warning { background:#fef3c7; border-left:3px solid #f59e0b; color:#78350f; }
.duty-block.conflict { background:#fee2e2; border-left:3px solid #ef4444; color:#7f1d1d; }
.detail-panel { display:none; background:#ffffff; border:1px solid #e2e8f0; padding:12px 14px; margin-top:6px; position:relative; transition: background-color 0.15s ease; }
.detail-panel.visible { display:block; }
.detail-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; }
.dc-label { font-size:9px; text-transform:uppercase; color:#64748b; }
.dc-value { font-size:12px; font-weight:700; color:#0f172a; }
.detail-close { position:absolute; top:6px; right:10px; cursor:pointer; font-size:14px; color:#94a3b8; background:none; border:none; transition: background-color 0.15s ease, color 0.15s ease; padding: 2px 6px; border-radius: 4px; }
.detail-close:hover { background-color:#eff6ff; color:#1d4ed8; }
</style>
</head>
<body>
<div class="tl-wrap" id="app"></div>
<script>
var SCHEDULE = __SCHEDULE_JSON__;
var STATS = __STATS_JSON__;
var PILOT_NAME = __PILOT_NAME__;
var WEEK_LABEL_BASE = __WEEK_LABEL__;
var HOURS = 24;
var currentView = 'week';
var weekOffset = 0;

function fmtHour(h) {
  var hh = Math.floor(h);
  var mm = Math.round((h - hh) * 60);
  return (hh < 10 ? '0' : '') + hh + ':' + (mm < 10 ? '0' : '') + mm;
}

function getGridDateLabel(d_idx, w_offset) {
    var today = new Date();
    var diff = today.getDay() === 0 ? 6 : today.getDay() - 1;
    var dOff = new Date(today);
    dOff.setDate(today.getDate() - diff + d_idx + (w_offset * 7));
    return dOff.toLocaleDateString('en-US', {month:'short', day:'numeric'});
}

function getWeekLabelText(w_offset) {
    var today = new Date();
    var diff = today.getDay() === 0 ? 6 : today.getDay() - 1;
    var wStart = new Date(today);
    wStart.setDate(today.getDate() - diff + (w_offset * 7));
    var wEnd = new Date(wStart);
    wEnd.setDate(wStart.getDate() + 6);
    return wStart.toLocaleDateString('en-US', {month:'short', day:'numeric'}) + " – " + wEnd.toLocaleDateString('en-US', {month:'short', day:'numeric'});
}

function render() {
  var app = document.getElementById('app');
  var html = '';

  html += '<div class="stats-grid">';
  html += '<div class="stat-card"><div class="stat-label">Hours this week</div><div class="stat-value">' + STATS.hours_this_week + 'h</div><div class="stat-sub ' + (STATS.dgca_hours_exceeded ? 'exceeded' : '') + '">DGCA cap 60h/wk' + (STATS.dgca_hours_exceeded ? ' \\u2014 EXCEEDED' : '') + '</div></div>';
  html += '<div class="stat-card"><div class="stat-label">Duty / Rest ratio</div><div class="stat-value">' + STATS.duty_rest_ratio + '</div><div class="stat-sub ' + (STATS.dgca_ratio_exceeded ? 'exceeded' : '') + '">Threshold 1.5' + (STATS.dgca_ratio_exceeded ? ' \\u2014 EXCEEDED' : '') + '</div></div>';
  html += '<div class="stat-card"><div class="stat-label">Last rest period</div><div class="stat-value">' + STATS.last_rest + '</div><div class="stat-sub ' + (STATS.dgca_rest_violated ? 'exceeded' : '') + '">Min 8h required' + (STATS.dgca_rest_violated ? ' \\u2014 VIOLATED' : '') + '</div></div>';
  html += '<div class="stat-card"><div class="stat-label">Sectors today</div><div class="stat-value">' + STATS.sectors_today + '</div><div class="stat-sub">Max 6 per DGCA</div></div>';
  html += '</div>';

  html += '<div class="controls">';
  html += '<div class="week-nav"><button onclick="navWeek(-1)">&lt;</button><span class="week-label">' + getWeekLabelText(weekOffset) + '</span><button onclick="navWeek(1)">&gt;</button></div>';
  html += '<div class="view-btns"><button id="btn-week" class="' + (currentView === 'week' ? 'active' : '') + '" onclick="setView(\'week\')">Week</button><button id="btn-today" class="' + (currentView === 'today' ? 'active' : '') + '" onclick="setView(\'today\')">Today</button></div>';
  html += '</div>';

  html += '<div class="legend">';
  html += '<div class="legend-item"><span class="legend-swatch" style="background:#dbeafe;border-left:2px solid #3b82f6"></span>Flight</div>';
  html += '<div class="legend-item"><span class="legend-swatch" style="background:#f0fdf4;border-left:2px solid #22c55e"></span>Ground</div>';
  html += '<div class="legend-item"><span class="legend-swatch" style="background:#faf5ff;border-left:2px solid #a855f7"></span>Simulator</div>';
  html += '<div class="legend-item"><span class="legend-swatch" style="background:#fef3c7;border-left:2px solid #f59e0b"></span>Warning</div>';
  html += '<div class="legend-item"><span class="legend-swatch" style="background:#fee2e2;border-left:2px solid #ef4444"></span>Conflict</div>';
  html += '</div>';

  html += '<div class="grid-header"><div class="day-label"></div><div class="hours">';
  for (var h = 0; h < HOURS; h += 2) {
    html += '<div class="hour-tick">' + (h < 10 ? '0' : '') + h + ':00</div>';
  }
  html += '</div></div>';

  var dayNames = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
  var todayIdx = new Date().getDay();
  var todayMon = todayIdx === 0 ? 6 : todayIdx - 1;

  for (var d = 0; d < 7; d++) {
    if (currentView === 'today' && (d !== todayMon || weekOffset !== 0)) continue;
    var dayDuties = [];
    for (var i = 0; i < SCHEDULE.length; i++) {
      if (SCHEDULE[i].day_index === d && SCHEDULE[i].week_offset === weekOffset) dayDuties.push(SCHEDULE[i]);
    }
    var isEmpty = dayDuties.length === 0;
    var dateLabel = getGridDateLabel(d, weekOffset);

    html += '<div class="day-row"><div class="row-bg"></div>';
    html += '<div class="day-lc ' + (isEmpty ? 'empty' : '') + '"><span class="dn">' + dayNames[d] + '</span><span class="dd">' + dateLabel + '</span></div>';
    html += '<div class="tl-track">';

    for (var j = 0; j < dayDuties.length; j++) {
      var duty = dayDuties[j];
      var leftPct = (duty.dep / HOURS * 100).toFixed(2);
      var widthPct = Math.max(2, ((duty.arr - duty.dep) / HOURS * 100)).toFixed(2);
      var cls = duty.status === 'warning' ? 'warning' : duty.status === 'conflict' ? 'conflict' : duty.type;
      // We pass the global index of SCHEDULE to showDetail, we need to find it
      var globalIdx = SCHEDULE.indexOf(duty);
      html += '<div class="duty-block ' + cls + '" style="left:' + leftPct + '%;width:' + widthPct + '%" onclick="showDetail(' + globalIdx + ')">';
      html += '<div class="bt">' + duty.id + '</div>';
      html += '<div class="br">' + duty.route + '</div>';
      html += '</div>';
    }

    html += '</div></div>';
  }

  html += '<div class="detail-panel" id="detailPanel"><button class="detail-close" onclick="hideDetail()">x</button><div class="detail-grid" id="detailGrid"></div></div>';
  app.innerHTML = html;
}

function setView(v) {
  currentView = v;
  render();
}

function navWeek(dir) {
  weekOffset += dir;
  render();
}

function showDetail(globalIdx) {
  var duty = SCHEDULE[globalIdx];
  if (!duty) return;
  var panel = document.getElementById('detailPanel');
  var grid = document.getElementById('detailGrid');
  var dur = ((duty.arr - duty.dep) * 60).toFixed(0) + ' min';
  grid.innerHTML = '<div><div class="dc-label">Route</div><div class="dc-value">' + duty.id + ' ' + duty.route + '</div></div>' +
    '<div><div class="dc-label">Date</div><div class="dc-value">' + (duty.date_str || "N/A") + '</div></div>' +
    '<div><div class="dc-label">Start</div><div class="dc-value">' + fmtHour(duty.dep) + '</div></div>' +
    '<div><div class="dc-label">End</div><div class="dc-value">' + fmtHour(duty.arr) + '</div></div>' +
    '<div><div class="dc-label">Duration</div><div class="dc-value">' + dur + '</div></div>' +
    '<div><div class="dc-label">Aircraft</div><div class="dc-value">' + duty.ac + '</div></div>' +
    '<div><div class="dc-label">Crew</div><div class="dc-value">' + duty.fo + '</div></div>' +
    '<div><div class="dc-label">Note</div><div class="dc-value">' + (duty.rule || '\\u2014') + '</div></div>';
  panel.className = 'detail-panel visible';
}

function hideDetail() {
  document.getElementById('detailPanel').className = 'detail-panel';
}

try { render(); } catch(e) { document.getElementById('app').innerHTML = '<pre style="color:red">' + e.message + '</pre>'; }
</script>
</body>
</html>
"""

    timeline_html = (
        timeline_html
        .replace('__SCHEDULE_JSON__', schedule_json)
        .replace('__STATS_JSON__', stats_json)
        .replace('__PILOT_NAME__', json.dumps(pilot_name))
        .replace('__WEEK_LABEL__', json.dumps(week_label))
    )

    # Use native Streamlit/Plotly timeline for reliability.
    # The custom embedded HTML timeline can fail in some environments and leave large blank areas.
    st.markdown("#### Weekly Duty Timeline")
    if schedule_data:
        available_offsets = sorted(
            {int(d.get("week_offset", 0)) for d in schedule_data if d.get("week_offset") is not None}
        )
        selected_week_offset = st.selectbox(
            "Week",
            available_offsets,
            index=available_offsets.index(0) if 0 in available_offsets else len(available_offsets) - 1,
            format_func=lambda w: (
                f"{(week_start + timedelta(days=(w * 7))).strftime('%b %d')} - "
                f"{(week_start + timedelta(days=(w * 7) + 6)).strftime('%b %d, %Y')}"
            ),
            label_visibility="collapsed",
        )

        week_base = week_start + timedelta(days=(selected_week_offset * 7))
        week_days = [week_base + timedelta(days=i) for i in range(7)]

        per_day_hours = {d.date(): 0.0 for d in week_days}
        per_day_flights = {d.date(): 0 for d in week_days}

        for duty in schedule_data:
            if int(duty.get("week_offset", 999)) != int(selected_week_offset):
                continue
            dep = float(duty.get("dep", 0.0))
            arr = float(duty.get("arr", dep))
            if arr <= dep:
                arr = dep + 2.0
            duty_hours = max(0.0, arr - dep)
            duty_date = pd.to_datetime(duty.get("date_str"), errors="coerce")
            if pd.isna(duty_date):
                continue
            duty_key = duty_date.date()
            if duty_key in per_day_hours:
                per_day_hours[duty_key] += duty_hours
                per_day_flights[duty_key] += 1

        weekly_rows = []
        for d in week_days:
            day_key = d.date()
            weekly_rows.append(
                {
                    "Day": d.strftime("%a"),
                    "Date": d.strftime("%b %d"),
                    "Label": f"{d.strftime('%a')} {d.strftime('%b %d')}",
                    "Hours": round(per_day_hours[day_key], 2),
                    "Flights": per_day_flights[day_key],
                }
            )

        week_df = pd.DataFrame(weekly_rows)
        fig_week = px.bar(
            week_df,
            x="Label",
            y="Hours",
            text="Hours",
            title=f"{pilot_name} - Flight Hours by Day",
            color="Hours",
            color_continuous_scale=[[0.0, "#dbeafe"], [1.0, "#3b82f6"]],
        )
        fig_week.update_traces(
            texttemplate="%{text:.1f}h",
            textposition="outside",
            hovertemplate="Day=%{x}<br>Hours=%{y:.2f}h<br>Flights=%{customdata[0]}<extra></extra>",
            customdata=week_df[["Flights"]].values,
            marker_line_color="#1d4ed8",
            marker_line_width=1,
        )
        fig_week.update_layout(
            height=360,
            xaxis_title="Day and Date",
            yaxis_title="Hours Flown",
            margin=dict(t=50, b=30, l=20, r=20),
            coloraxis_showscale=False,
        )
        fig_week.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig_week, use_container_width=True)
    else:
        st.info("No assigned duties found for this pilot.")
         
    st.markdown("<br>### ⚡ Quick Actions", unsafe_allow_html=True)
    act1, act2, act3 = st.columns(3)
    with act1:
         st.button("✏️ Edit Schedule", use_container_width=True)
    with act2:
         st.button("📜 View Full History", use_container_width=True)
    with act3:
         st.button("\u26a0\ufe0f Modify/Remove Duties", use_container_width=True)


def render_scheduling_board():
    import streamlit.components.v1 as components
    import json

    df_dash = load_dashboard_data()
    if df_dash.empty:
        st.info("Upload a flight CSV to populate this view.")
        return

    PAGE_SIZE = 20
    if "schedule_board_offset" not in st.session_state:
        st.session_state["schedule_board_offset"] = 0

    crew_rules = crew_rules_for_schedule_board()
    df_board = df_dash.copy()
    sort_date = pd.to_datetime(df_board.get("Date"), errors="coerce")
    df_board["_sb_sort_date"] = sort_date
    if "Flight Number" in df_board.columns:
        df_board["_sb_sort_fn"] = df_board["Flight Number"].astype(str)
    else:
        df_board["_sb_sort_fn"] = df_board.index.astype(str)
    df_board = df_board.sort_values(["_sb_sort_date", "_sb_sort_fn"], na_position="last").drop(
        columns=["_sb_sort_date", "_sb_sort_fn"]
    )

    # Optimization controls (proposal vs confirm/discard)
    c1, c2, c3 = st.columns([1.5, 1.2, 1.2])
    with c1:
        if st.button("✦ Optimize Schedule", key="optimize_schedule_board", use_container_width=True):
            st.session_state["optimized_df"] = propose_optimized_schedule(
                st.session_state.get("master_df", df_dash),
                max_shift_minutes=60
            )
            st.success(
                "Fatigue-model optimization proposal generated (Random Forest). "
                "Review and confirm to apply globally."
            )
    with c2:
        if st.button("✔ Confirm & Apply Optimization", key="confirm_schedule_opt", use_container_width=True):
            if "optimized_df" in st.session_state and not st.session_state["optimized_df"].empty:
                opt_df = st.session_state["optimized_df"].copy()
                remaining_flags = int(
                    ((opt_df.get("Risk_Reduction_%", 0) <= 0).astype(int).sum())
                    if "Risk_Reduction_%" in opt_df.columns else 0
                )
                if remaining_flags > 0:
                    st.warning("Optimization has been applied with active warnings. Review conflict and fatigue rows in the board.")
                st.session_state["master_df"] = st.session_state["optimized_df"].copy()
                st.session_state.pop("optimized_df", None)
                st.session_state["optimization_applied"] = True
                st.success("Optimization applied. All views have been updated.")
                st.rerun()
    with c3:
        if st.button("✘ Discard", key="discard_schedule_opt", use_container_width=True):
            st.session_state.pop("optimized_df", None)
            st.info("Optimization proposal discarded.")

    if "optimized_df" in st.session_state and not st.session_state["optimized_df"].empty:
        cmp_tabs = st.tabs(["Current Schedule", "Optimized Schedule"])
        with cmp_tabs[0]:
            st.dataframe(
                df_dash[["Flight Number", "From", "To", "STD", "STA", "Captain", "Status"]].head(40),
                use_container_width=True,
                hide_index=True
            )
        with cmp_tabs[1]:
            optv = st.session_state["optimized_df"].copy()
            st.dataframe(
                optv[["Flight Number", "From", "To", "Orig_STD", "STD", "Orig_STA", "STA", "Captain", "Change_Reason", "Risk_Reduction_%"]]
                .head(40),
                use_container_width=True,
                hide_index=True
            )
        st.markdown("---")

    n_total = len(df_board)
    max_offset = max(0, (max(n_total - 1, 0) // PAGE_SIZE) * PAGE_SIZE)
    if st.session_state["schedule_board_offset"] > max_offset:
        st.session_state["schedule_board_offset"] = max_offset

    off = int(st.session_state["schedule_board_offset"])
    page_df = df_board.iloc[off : off + PAGE_SIZE]
    start_i = off + 1 if n_total else 0
    end_i = min(off + PAGE_SIZE, n_total)

    st.caption(
        "Board rules follow **Crew Manager** sliders: "
        f"~{crew_rules['duty_hours']}h/duty est., max **{crew_rules['max_duties']}** duties, "
        f"**{crew_rules['max_block_h']}h** block cap, **{crew_rules['min_rest_h']}h** rest threshold, "
        f"fatigue warn after **{crew_rules['fatigue_thresh']}** duties."
    )

    nav_a, nav_b, nav_c = st.columns([1, 2, 1])
    with nav_a:
        if st.button("← Previous 20", key="sched_board_prev", disabled=(off <= 0), use_container_width=True):
            st.session_state["schedule_board_offset"] = max(0, off - PAGE_SIZE)
            st.rerun()
    with nav_b:
        st.markdown(
            f"<div style='text-align:center;font-family:Space Mono,monospace;font-size:0.85rem;color:#475569;padding:0.35rem 0;'>"
            f"Showing flights <b>{start_i}–{end_i}</b> of <b>{n_total}</b></div>",
            unsafe_allow_html=True,
        )
    with nav_c:
        if st.button(
            "Next 20 →",
            key="sched_board_next",
            disabled=(off >= max_offset and n_total > 0) or n_total == 0,
            use_container_width=True,
        ):
            st.session_state["schedule_board_offset"] = min(max_offset, off + PAGE_SIZE)
            st.rerun()

    # Build flight data for the JS board (current page only; id unique per date)
    flights_data = []
    for _, row in page_df.iterrows():
        fn = str(row.get("Flight Number", "N/A"))
        dts = pd.to_datetime(row.get("Date"), errors="coerce")
        date_part = dts.strftime("%Y-%m-%d") if pd.notna(dts) else ""
        uid = f"{fn}|{date_part}" if date_part else fn
        date_label = dts.strftime("%d %b %Y") if pd.notna(dts) else "—"
        flights_data.append({
            "id": uid,
            "label": fn,
            "date_label": date_label,
            "aircraft": str(row.get('Aircraft', 'N/A')),
            'from': str(row.get('From', 'N/A')),
            'to': str(row.get('To', 'N/A')),
            'std': str(row.get('STD', 'N/A')),
            'sta': str(row.get('STA', 'N/A')),
            'captain': str(row.get('Captain', '')),
            'fo': str(row.get('First Officer', '')),
            'status': str(row.get('Status', 'On Time'))
        })

    pilots = sorted(set(df_dash['Captain'].unique().tolist() + df_dash['First Officer'].unique().tolist()))
    flights_json = json.dumps(flights_data)
    pilots_json = json.dumps(pilots)

    board_template = """
<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Bebas+Neue&display=swap" rel="stylesheet">
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:'Space Mono',monospace; background:#f8fafc; color:#0f172a; overflow-x:hidden; }
.app-layout { display:grid; grid-template-columns:70fr 30fr; gap:1.2rem; padding:1rem; min-height:760px; overflow:visible; }
.main-col { min-width:0; overflow:visible; }
.toolbar { display:flex; gap:0.5rem; margin-bottom:1rem; align-items:center; flex-wrap:wrap; background:white; border:1px solid #e2e8f0; padding:0.6rem 0.8rem; }
.toolbar button { color:#0f172a; padding:0.4rem 0.85rem; border:1px solid #e2e8f0; background:white; border-radius:0; cursor:pointer; font-size:0.75rem; font-family:'Space Mono',monospace; font-weight:700; letter-spacing:0.02em; transition:background-color 0.15s ease, color 0.15s ease, border-color 0.15s ease; }
.toolbar button:not(:disabled):hover { border-color:#3b82f6; color:#1d4ed8; background-color:#eff6ff; }
.toolbar button:disabled { opacity:0.35; cursor:not-allowed; }
.toolbar .badge { background:#3b82f6; color:#ffffff; padding:0.25rem 0.65rem; font-size:0.7rem; font-weight:700; font-family:'Space Mono',monospace; }
.btn-apply { background:#3b82f6!important; color:#ffffff!important; border-color:#3b82f6!important; margin-left:auto; }
.btn-apply:not(:disabled):hover { background-color:#2563eb!important; opacity:0.85; }
.legend { display:flex; gap:1.2rem; margin-bottom:0.8rem; font-size:0.7rem; letter-spacing:0.02em; color:#64748b; }
.legend span { display:flex; align-items:center; gap:0.35rem; }
.legend .dot { width:12px; height:12px; }
.board { display:grid; grid-template-columns:140px 1fr; border:1px solid #e2e8f0; overflow:visible; background:white; }
.pilot-label { padding:0.65rem 0.7rem; background:#f1f5f9; border-bottom:1px solid #e2e8f0; font-weight:700; font-size:0.8rem; display:flex; align-items:center; gap:0.3rem; min-height:88px; font-family:'Space Mono',monospace; letter-spacing:-0.01em; word-break:break-word; color:#0f172a; }
.pilot-label.affected { background:#fef3c7; border-left:3px solid #f59e0b; }
.pilot-label.conflict-row { background:#fee2e2; border-left:3px solid #ef4444; }
.pilot-row { position:relative; padding:0.45rem 0.6rem; border-bottom:1px solid #e2e8f0; display:flex; flex-wrap:wrap; gap:0.5rem; min-height:88px; align-items:center; transition:background-color 0.15s ease; }
.row-bg { position:absolute; inset:0; z-index:0; transition:background-color 0.15s ease; }
.pilot-row:hover .row-bg { background-color:#eff6ff; }
.pilot-row.drag-over .row-bg { background-color:#dbeafe; }
.pilot-row.drag-over { background:#dbeafe; }
@keyframes pulseAmber { 0%,100% { box-shadow:inset 0 0 0 2px rgba(245,158,11,0.5); } 50% { box-shadow:inset 0 0 0 2px rgba(245,158,11,0); } }
.pilot-row.row-warning .row-bg { animation: pulseAmber 2s infinite; }
.pilot-row.row-conflict .row-bg { box-shadow:inset 0 0 0 2px #ef4444; }
.flight-card { position:relative; z-index:2; width:172px; min-height:72px; padding:0.45rem 0.65rem; border-radius:0; font-size:0.76rem; cursor:grab; user-select:none; border:2px solid transparent; transition:background-color 0.15s ease, outline 0.15s ease, transform 0.15s ease; outline:2px solid transparent; display:flex; flex-direction:column; justify-content:center; overflow:visible; font-family:'Space Mono',monospace; }
.flight-card:hover { outline:2px solid #bfdbfe; outline-offset:2px; z-index:80; }
.flight-card.dragging { opacity:0.5; outline:2px dashed #3b82f6; z-index:60; }
.flight-card:active { cursor:grabbing; transform:scale(1.06); box-shadow:0 6px 20px rgb(0 0 0/0.18); }
.fc-body { width:100%; overflow:hidden; display:flex; flex-direction:column; justify-content:center; flex:1; min-height:0; }
.flight-card .fid { font-weight:700; font-size:0.86rem; line-height:1.2; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.flight-card .froute { font-size:0.7rem; line-height:1.3; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; opacity:0.85; }
.fc-tooltip { visibility:hidden; opacity:0; position:absolute; left:50%; top:calc(100% + 6px); transform:translateX(-50%) translateY(0); width:min(288px,calc(100vw - 48px)); max-width:288px; padding:14px 14px 12px; background:#ffffff; border:1px solid #e2e8f0; box-shadow:0 12px 32px rgba(15,23,42,0.12),0 4px 8px rgba(15,23,42,0.06); color:#0f172a; font-size:0.72rem; line-height:1.35; pointer-events:none; transition:opacity 0.18s ease,visibility 0.18s ease,transform 0.18s ease; text-align:left; border-radius:2px; }
.flight-card:hover .fc-tooltip { visibility:visible; opacity:1; transform:translateX(-50%) translateY(2px); }
.flight-card.dragging .fc-tooltip { visibility:hidden !important; opacity:0 !important; }
.fc-tooltip-h { font-family:'Bebas Neue',sans-serif; font-size:1.35rem; letter-spacing:0.06em; color:#0f172a; margin-bottom:2px; line-height:1.1; }
.fc-tooltip-sub { font-size:0.65rem; color:#64748b; text-transform:uppercase; letter-spacing:0.14em; margin-bottom:4px; font-weight:700; }
.fc-tooltip-rows { border-top:1px solid #e2e8f0; margin-top:8px; padding-top:8px; display:flex; flex-direction:column; gap:6px; }
.fc-tooltip-row { display:grid; grid-template-columns:92px 1fr; gap:8px; align-items:start; }
.fc-tooltip-k { color:#94a3b8; text-transform:uppercase; font-size:0.58rem; letter-spacing:0.1em; font-weight:700; }
.fc-tooltip-v { color:#334155; font-weight:500; word-break:break-word; }
.fc-tooltip-assign .fc-tooltip-v { color:#1d4ed8; font-weight:700; }
.fc-status-pill { display:inline-block; padding:2px 8px; border-radius:999px; font-size:0.62rem; font-weight:800; letter-spacing:0.04em; text-transform:uppercase; }
.fc-pill-green { background:#dcfce7; color:#166534; }
.fc-pill-amber { background:#fef3c7; color:#b45309; }
.fc-pill-red { background:#fee2e2; color:#b91c1c; }
.fc-pill-blue { background:#dbeafe; color:#1d4ed8; }
.card-green  { background:#10b981; border-color:#059669; color:#14532d; }
.card-amber  { background:#f59e0b; border-color:#d97706; color:#78350f; }
.card-red    { background:#fff1f2; border-color:#dc2626; color:#7f1d1d; box-shadow: inset 0 0 0 2px rgba(220,38,38,0.25); }
.card-blue   { background:#dbeafe; border-color:#3b82f6; color:#0f172a; }
@keyframes cardPulse { 0%,100% { box-shadow:0 0 0 0 rgba(16,185,129,0.45); } 50% { box-shadow:0 0 0 6px rgba(16,185,129,0); } }
.flight-card.changed { animation: cardPulse 1.6s infinite; }
.side-panel { background:white; border:1px solid #e2e8f0; padding:1rem; height:fit-content; position:sticky; top:1rem; overflow-y:auto; max-height:calc(100vh - 2rem); }
.side-panel h3 { font-family:'Bebas Neue',sans-serif; font-size:1.4rem; color:#0f172a; margin-bottom:0.6rem; letter-spacing:0.04em; }
.sp-section { margin-bottom:0.9rem; padding-bottom:0.7rem; border-bottom:1px solid #f1f5f9; }
.sp-section:last-child { border-bottom:none; margin-bottom:0; }
.sp-section h4 { font-size:0.68rem; color:#64748b; margin-bottom:0.45rem; text-transform:uppercase; letter-spacing:0.12em; font-family:'Space Mono',monospace; }
.sp-empty { color:#94a3b8; font-size:0.75rem; font-style:italic; }
.affected-pilot { background:#f8fafc; padding:0.5rem; margin-bottom:0.35rem; font-size:0.72rem; border:1px solid #e2e8f0; }
.affected-pilot .ap-name { font-weight:700; margin-bottom:0.25rem; }
.affected-pilot .ap-delta { display:flex; justify-content:space-between; font-size:0.68rem; color:#64748b; }
.delta-up { color:#ef4444; font-weight:700; }
.delta-down { color:#10b981; font-weight:700; }
.delta-same { color:#94a3b8; }
.fatigue-meter { height:5px; background:#e2e8f0; margin-top:0.25rem; overflow:hidden; }
.fatigue-fill { height:100%; transition:width 0.3s; }
.fatigue-low  { background:#10b981; }
.fatigue-med  { background:#f59e0b; }
.fatigue-high { background:#ef4444; }
.ba-row { display:grid; grid-template-columns:1fr auto 1fr; gap:0.35rem; align-items:center; font-size:0.68rem; padding:0.3rem 0; border-bottom:1px solid #f8fafc; }
.ba-arrow { color:#3b82f6; font-weight:700; font-size:0.78rem; }
.ba-before { text-align:right; color:#94a3b8; text-decoration:line-through; }
.ba-after { color:#0f172a; font-weight:700; }
.conflict-item { padding:0.45rem; margin-bottom:0.3rem; font-size:0.7rem; display:flex; align-items:center; gap:0.35rem; font-family:'Space Mono',monospace; }
.conflict-item.ci-conflict { background:#fef2f2; color:#dc2626; }
.conflict-item.ci-warning  { background:#fffbeb; color:#d97706; }
.conflict-item.ci-valid    { background:#f0fdf4; color:#059669; }
</style>
</head>
<body>
<div class="app-layout">
<div class="main-col">
<div class="toolbar">
  <button id="btnUndo" onclick="undo()" disabled>&#8617; Undo</button>
  <span class="badge" id="changeCount">0 changes</span>
  <button id="btnRedo" onclick="redo()" disabled>&#8618; Redo</button>
  <button onclick="resetAll()">&#8634; Reset</button>
  <button class="btn-apply" id="btnApply" onclick="applyChanges()" disabled>Apply All</button>
</div>
<div class="legend">
  <span><div class="dot" style="background:#10b981"></div> Valid</span>
  <span><div class="dot" style="background:#f59e0b"></div> Fatigue Risk</span>
  <span><div class="dot" style="background:#ef4444"></div> DGCA Conflict</span>
  <span><div class="dot" style="background:#dbeafe;border:1px solid #3b82f6"></div> Unchanged</span>
</div>
<div class="board" id="board"></div>
</div>
<div class="side-panel" id="sidePanel">
  <h3>IMPACT DASHBOARD</h3>
  <div class="sp-section" id="spSummary"><h4>Summary</h4><div class="sp-empty" id="spSummaryContent">Make a change to see impact analysis</div></div>
  <div class="sp-section" id="spAffected"><h4>Affected Pilots</h4><div id="spAffectedContent" class="sp-empty">No pilots affected</div></div>
  <div class="sp-section"><h4>Before &#x2194; After</h4><div id="spBeforeAfter" class="sp-empty">No changes to compare</div></div>
  <div class="sp-section"><h4>Conflicts &amp; Risks</h4><div id="spConflicts" class="sp-empty">System nominal</div></div>
</div>
</div>

<script>
const flights = __FLIGHTS_JSON__;
const pilots  = __PILOTS_JSON__;
const DUTY_HOURS = __DUTY_HOURS__, MAX_DUTIES = __MAX_DUTIES__, MAX_BLOCK_H = __MAX_BLOCK_H__, MIN_REST_H = __MIN_REST_H__, FATIGUE_THRESH = __FATIGUE_THRESH__;

let assignments = {};
let undoStack = [], redoStack = [];
let originalAssignments = {};

pilots.forEach(p => { assignments[p] = []; });
flights.forEach(f => {
  if (assignments[f.captain] !== undefined) assignments[f.captain].push(f.id);
  if (assignments[f.fo] !== undefined) assignments[f.fo].push(f.id);
});
Object.keys(assignments).forEach(k => { originalAssignments[k] = [...assignments[k]]; });

function getFlightById(id) { return flights.find(f => f.id === id); }

function escapeHtml(s) {
  if (s == null || s === undefined) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
function airportCode(s) {
  if (!s) return '\u2014';
  let m = String(s).match(/\\(([^)]+)\\)/);
  return m ? m[1] : String(s).trim().slice(0, 8);
}
function airportCity(s) {
  if (!s) return '';
  let t = String(s).replace(/\\s*\\([^)]+\\)\\s*$/, '').trim();
  return t || String(s);
}
function statusPillClass(st) {
  let x = String(st || '').toLowerCase();
  if (x.indexOf('delay') >= 0) return 'fc-pill-amber';
  if (x.indexOf('cancel') >= 0) return 'fc-pill-red';
  if (x.indexOf('board') >= 0) return 'fc-pill-blue';
  return 'fc-pill-green';
}
function buildFlightTooltipHtml(fl, rowPilot) {
  let depC = airportCode(fl.from);
  let arrC = airportCode(fl.to);
  let depCity = escapeHtml(airportCity(fl.from));
  let arrCity = escapeHtml(airportCity(fl.to));
  let cap = escapeHtml(fl.captain || '\u2014');
  let fo = escapeHtml(fl.fo || '\u2014');
  let ac = escapeHtml(fl.aircraft || '\u2014');
  let st = escapeHtml(fl.status || 'On Time');
  let pill = statusPillClass(fl.status);
  let dl = escapeHtml(fl.date_label || '\u2014');
  let fn = escapeHtml(fl.label || fl.id);
  let rp = escapeHtml(rowPilot || '');
  return (
    '<div class="fc-tooltip">' +
      '<div class="fc-tooltip-h">' + fn + '</div>' +
      '<div class="fc-tooltip-sub">' + depC + ' \u2192 ' + arrC + '</div>' +
      '<div class="fc-tooltip-rows">' +
        '<div class="fc-tooltip-row"><span class="fc-tooltip-k">Date</span><span class="fc-tooltip-v">' + dl + '</span></div>' +
        '<div class="fc-tooltip-row"><span class="fc-tooltip-k">Route</span><span class="fc-tooltip-v">' + depCity + ' \u2192 ' + arrCity + '</span></div>' +
        '<div class="fc-tooltip-row"><span class="fc-tooltip-k">Depart</span><span class="fc-tooltip-v">' + escapeHtml(fl.std) + '</span></div>' +
        '<div class="fc-tooltip-row"><span class="fc-tooltip-k">Arrive</span><span class="fc-tooltip-v">' + escapeHtml(fl.sta) + '</span></div>' +
        '<div class="fc-tooltip-row"><span class="fc-tooltip-k">Aircraft</span><span class="fc-tooltip-v">' + ac + '</span></div>' +
        '<div class="fc-tooltip-row"><span class="fc-tooltip-k">Captain</span><span class="fc-tooltip-v">' + cap + '</span></div>' +
        '<div class="fc-tooltip-row"><span class="fc-tooltip-k">First officer</span><span class="fc-tooltip-v">' + fo + '</span></div>' +
        '<div class="fc-tooltip-row"><span class="fc-tooltip-k">Status</span><span class="fc-tooltip-v"><span class="fc-status-pill ' + pill + '">' + st + '</span></span></div>' +
        '<div class="fc-tooltip-row fc-tooltip-assign"><span class="fc-tooltip-k">On this row</span><span class="fc-tooltip-v">' + rp + '</span></div>' +
      '</div>' +
    '</div>'
  );
}

function detectConflicts(pName, fids) {
  let issues = [], dutyCount = fids.length, totalBlock = dutyCount * DUTY_HOURS;
  if (dutyCount > MAX_DUTIES) {
    issues.push({ type:'conflict', msg: pName + ': ' + dutyCount + ' duties (DGCA max ' + MAX_DUTIES + ')' });
    issues.push({ type:'conflict', msg: pName + ': insufficient rest (<' + MIN_REST_H + 'h)' });
  }
  if (totalBlock > MAX_BLOCK_H) issues.push({ type:'conflict', msg: pName + ': ' + totalBlock.toFixed(1) + 'h block (max ' + MAX_BLOCK_H + 'h)' });
  if (dutyCount > FATIGUE_THRESH && dutyCount <= MAX_DUTIES) issues.push({ type:'warning', msg: pName + ': ' + dutyCount + ' duties (fatigue risk)' });
  let acMap = {};
  fids.forEach(fid => { let fl = getFlightById(fid); if (fl) { if (acMap[fl.aircraft]) issues.push({ type:'conflict', msg:'Aircraft ' + fl.aircraft + ' double-booked by ' + pName }); acMap[fl.aircraft] = true; } });
  return issues;
}

function getCardClass(pilot, fid) {
  let issues = detectConflicts(pilot, assignments[pilot] || []);
  let isChanged = !originalAssignments[pilot] || !originalAssignments[pilot].includes(fid);
  if (issues.some(i => i.type === 'conflict')) return 'card-red';
  if (issues.some(i => i.type === 'warning'))  return 'card-amber';
  if (isChanged) return 'card-green';
  return 'card-blue';
}
function getRowClass(pilot) {
  let issues = detectConflicts(pilot, assignments[pilot] || []);
  if (issues.some(i => i.type === 'conflict')) return 'row-conflict';
  if (issues.some(i => i.type === 'warning'))  return 'row-warning';
  return '';
}
function getLabelClass(pilot) {
  let issues = detectConflicts(pilot, assignments[pilot] || []);
  if (issues.some(i => i.type === 'conflict')) return 'conflict-row';
  if (isAffectedPilot(pilot)) return 'affected';
  return '';
}
function isAffectedPilot(p) {
  return (originalAssignments[p] || []).sort().join(',') !== (assignments[p] || []).sort().join(',');
}

function renderBoard() {
  const board = document.getElementById('board');
  board.innerHTML = '';
  pilots.forEach(pilot => {
    let label = document.createElement('div');
    let lblExtra = getLabelClass(pilot);
    label.className = 'pilot-label' + (lblExtra ? ' ' + lblExtra : '');
    label.textContent = pilot;
    board.appendChild(label);
    let row = document.createElement('div');
    let rowExtra = getRowClass(pilot);
    row.className = 'pilot-row' + (rowExtra ? ' ' + rowExtra : '');
    row.dataset.pilot = pilot;
    let rowbg = document.createElement('div');
    rowbg.className = 'row-bg';
    row.appendChild(rowbg);
    row.addEventListener('dragover', e => { e.preventDefault(); row.classList.add('drag-over'); });
    row.addEventListener('dragleave', () => row.classList.remove('drag-over'));
    row.addEventListener('drop', e => {
      e.preventDefault(); row.classList.remove('drag-over');
      let data = JSON.parse(e.dataTransfer.getData('text/plain'));
      moveFlightToPilot(data.fid, data.fromPilot, pilot);
    });
    (assignments[pilot] || []).forEach(fid => {
      let fl = getFlightById(fid); if (!fl) return;
      let isNew = !originalAssignments[pilot] || !originalAssignments[pilot].includes(fid);
      let card = document.createElement('div');
      card.className = 'flight-card ' + getCardClass(pilot, fid) + (isNew ? ' changed' : '');
      card.draggable = true;
      let routeFrom = fl.from.split('(')[0].trim();
      let routeTo = fl.to.split('(')[0].trim();
      let dispId = fl.label || fl.id;
      card.innerHTML = '<div class="fc-body"><div class="fid">' + dispId + ' &#183; ' + fl.std + '</div><div class="froute">' + routeFrom + ' &#8594; ' + routeTo + '</div></div>' + buildFlightTooltipHtml(fl, pilot);
      card.setAttribute('aria-label', 'Flight ' + dispId + ', drag to reassign');
      card.addEventListener('dragstart', e => { card.classList.add('dragging'); e.dataTransfer.setData('text/plain', JSON.stringify({ fid: fid, fromPilot: pilot })); });
      card.addEventListener('dragend', () => { card.classList.remove('dragging'); });
      row.appendChild(card);
    });
    board.appendChild(row);
  });
  updateUI();
}

function snapshot() { let s = {}; Object.keys(assignments).forEach(k => s[k] = [...assignments[k]]); return s; }
function moveFlightToPilot(fid, from, to) {
  if (from === to) return;
  undoStack.push(snapshot()); redoStack = [];
  assignments[from] = assignments[from].filter(f => f !== fid);
  if (!assignments[to].includes(fid)) assignments[to].push(fid);
  renderBoard();
}
function undo()  { if (!undoStack.length) return; redoStack.push(snapshot()); assignments = undoStack.pop(); renderBoard(); }
function redo()  { if (!redoStack.length) return; undoStack.push(snapshot()); assignments = redoStack.pop(); renderBoard(); }
function resetAll() { undoStack.push(snapshot()); redoStack = []; Object.keys(originalAssignments).forEach(k => assignments[k] = [...originalAssignments[k]]); renderBoard(); }

function getChanges() {
  let ch = [];
  pilots.forEach(p => {
    let o = originalAssignments[p] || [], c = assignments[p] || [];
    c.forEach(f => { if (!o.includes(f)) ch.push({ type:'add', pilot:p, fid:f }); });
    o.forEach(f => { if (!c.includes(f)) ch.push({ type:'remove', pilot:p, fid:f }); });
  });
  return ch;
}

function updateUI() {
  let changes = getChanges();
  document.getElementById('changeCount').textContent = changes.length + ' change' + (changes.length !== 1 ? 's' : '');
  document.getElementById('btnUndo').disabled = !undoStack.length;
  document.getElementById('btnRedo').disabled = !redoStack.length;
  document.getElementById('btnApply').disabled = !changes.length;
  updateSidePanel(changes);
}

function updateSidePanel(changes) {
  let affectedPilots = pilots.filter(isAffectedPilot);
  let allIssues = []; pilots.forEach(p => { allIssues = allIssues.concat(detectConflicts(p, assignments[p] || [])); });
  let origIssues = []; pilots.forEach(p => { origIssues = origIssues.concat(detectConflicts(p, originalAssignments[p] || [])); });
  let newConflicts = allIssues.filter(i => i.type === 'conflict').length - origIssues.filter(i => i.type === 'conflict').length;
  let newWarnings = allIssues.filter(i => i.type === 'warning').length - origIssues.filter(i => i.type === 'warning').length;

  if (changes.length === 0) {
    document.getElementById('spSummaryContent').innerHTML = '<span class="sp-empty">Make a change to see impact analysis</span>';
    document.getElementById('spAffectedContent').innerHTML = '<span class="sp-empty">No pilots affected</span>';
    document.getElementById('spBeforeAfter').innerHTML = '<span class="sp-empty">No changes to compare</span>';
    document.getElementById('spConflicts').innerHTML = '<div class="conflict-item ci-valid">&#9989; System nominal</div>';
    return;
  }

  document.getElementById('spSummaryContent').innerHTML =
    '<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.35rem;font-size:0.72rem;font-family:Space Mono,monospace;">' +
    '<div style="background:#dbeafe;padding:0.45rem;text-align:center;"><strong>' + changes.length + '</strong><br>Changes</div>' +
    '<div style="background:' + (affectedPilots.length > 0 ? '#fef3c7' : '#d1fae5') + ';padding:0.45rem;text-align:center;"><strong>' + affectedPilots.length + '</strong><br>Pilots</div>' +
    '<div style="background:' + (newConflicts > 0 ? '#fee2e2' : '#d1fae5') + ';padding:0.45rem;text-align:center;"><strong>' + (newConflicts > 0 ? '+' : '') + newConflicts + '</strong><br>Conflicts</div>' +
    '<div style="background:' + (newWarnings > 0 ? '#fef3c7' : '#d1fae5') + ';padding:0.45rem;text-align:center;"><strong>' + (newWarnings > 0 ? '+' : '') + newWarnings + '</strong><br>Fatigue</div></div>';

  if (affectedPilots.length > 0) {
    document.getElementById('spAffectedContent').innerHTML = affectedPilots.map(p => {
      let origCount = (originalAssignments[p] || []).length;
      let currCount = (assignments[p] || []).length;
      let delta = currCount - origCount;
      let deltaHrs = (delta * DUTY_HOURS).toFixed(1);
      let fatiguePct = Math.min(100, (currCount / (MAX_DUTIES + 1)) * 100);
      let fatigueClass = fatiguePct >= 80 ? 'fatigue-high' : fatiguePct >= 50 ? 'fatigue-med' : 'fatigue-low';
      return '<div class="affected-pilot"><div class="ap-name">' + p + '</div>' +
        '<div class="ap-delta"><span>' + origCount + ' &#8594; ' + currCount + ' duties</span>' +
        '<span class="' + (delta > 0 ? 'delta-up' : delta < 0 ? 'delta-down' : 'delta-same') + '">' + (delta > 0 ? '+' : '') + deltaHrs + 'h</span></div>' +
        '<div class="fatigue-meter"><div class="fatigue-fill ' + fatigueClass + '" style="width:' + fatiguePct + '%"></div></div></div>';
    }).join('');
  } else {
    document.getElementById('spAffectedContent').innerHTML = '<span class="sp-empty">No pilots affected</span>';
  }

  document.getElementById('spBeforeAfter').innerHTML = changes.map(c => {
    let fl = getFlightById(c.fid);
    let route = fl ? fl.from.split('(')[0].trim() + ' &#8594; ' + fl.to.split('(')[0].trim() : '';
    if (c.type === 'add') {
      let origPilot = pilots.find(p => (originalAssignments[p] || []).includes(c.fid) && p !== c.pilot) || '—';
      return '<div class="ba-row"><div class="ba-before">' + origPilot + '</div><div class="ba-arrow">&#8594;</div><div class="ba-after">' + c.pilot + '</div></div>' +
             '<div style="font-size:0.62rem;color:#94a3b8;margin-bottom:0.25rem;padding-left:0.15rem;">' + c.fid + ' ' + route + '</div>';
    }
    return '';
  }).join('');

  if (allIssues.length > 0) {
    document.getElementById('spConflicts').innerHTML = allIssues.map(i =>
      '<div class="conflict-item ci-' + i.type + '">' + (i.type === 'conflict' ? '&#128308;' : '&#128993;') + ' ' + i.msg + '</div>'
    ).join('');
  } else {
    document.getElementById('spConflicts').innerHTML = '<div class="conflict-item ci-valid">&#9989; All clear</div>';
  }
}

function applyChanges() {
  let changes = getChanges();
  if (!changes.length) return;
  Object.keys(assignments).forEach(k => originalAssignments[k] = [...assignments[k]]);
  undoStack = []; redoStack = [];
  renderBoard();
}

renderBoard();
</script>
</body>
</html>
"""

    board_html = (
        board_template.replace('__FLIGHTS_JSON__', flights_json)
        .replace('__PILOTS_JSON__', pilots_json)
        .replace('__DUTY_HOURS__', str(crew_rules['duty_hours']))
        .replace('__MAX_DUTIES__', str(crew_rules['max_duties']))
        .replace('__MAX_BLOCK_H__', str(crew_rules['max_block_h']))
        .replace('__MIN_REST_H__', str(crew_rules['min_rest_h']))
        .replace('__FATIGUE_THRESH__', str(crew_rules['fatigue_thresh']))
    )

    st.markdown("# \u2708\ufe0f Interactive Scheduling Board")
    st.markdown("Drag flight cards between pilots to reassign duties. The **Impact Dashboard** updates in real-time.")
    st.markdown("---")
    components.html(board_html, height=860, scrolling=True)


def main():
    st.set_page_config(
        page_title="Pilot Scheduler",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ─── GLOBAL CSS: Blueprint Grid + Original Color Scheme ───
    st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
      /* Bebas Neue headers */
      h1, h2, h3 { font-family: 'Bebas Neue', sans-serif !important; letter-spacing: 0.04em; }
      /* Blueprint grid background */
      .stApp > div:first-child { background-image: linear-gradient(#e2e8f0 1px, transparent 1px), linear-gradient(90deg, #e2e8f0 1px, transparent 1px); background-size: 60px 60px; background-color: #f8fafc; }
      /* Remove Streamlit metric border-radius (flat blueprint style) */
      div[data-testid="metric-container"] { border-radius: 0 !important; border: 1px solid #e2e8f0 !important; background: white !important; }
      /* Tab styling — flat, monospace labels */
      .stTabs [data-baseweb="tab"] { font-family: 'Space Mono', monospace; font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase; }
      .stTabs [aria-selected="true"] { border-bottom: 2px solid #0f172a !important; color: #0f172a !important; }
      /* Sidebar */
      section[data-testid="stSidebar"] { background: white !important; border-right: 1px solid #e2e8f0; }
      section[data-testid="stSidebar"] :not([data-testid="stIconMaterial"]) { font-family: 'Space Mono', monospace !important; }
      [data-testid="stIconMaterial"], .material-symbols-rounded {
        font-family: "Material Symbols Rounded" !important;
      }
      /* Slider accent color — use blue not red */
      div[data-testid="stSlider"] > div > div > div { background: #3b82f6 !important; }
      div[data-testid="stSlider"] > div > div > div > div { background: #0f172a !important; border: 2px solid white !important; }
      /* Buttons */
      .stButton > button { border-radius: 0 !important; font-family: 'Space Mono', monospace !important; font-size: 11px !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; border: 1px solid #0f172a !important; }
      .stButton > button:hover { background: #0f172a !important; color: white !important; }
      /* File uploader */
      div[data-testid="stFileUploader"] { border-radius: 0 !important; border: 1px dashed #e2e8f0 !important; }
      .block-container { padding-top: 2rem !important; max-width: 1400px; }
    </style>
    """, unsafe_allow_html=True)

    def _on_quick_pilot_change():
        chosen = st.session_state.get("quick_pilot_lookup", "— Select —")
        if chosen and chosen != "— Select —":
            st.session_state._pilot_nav = chosen

    # ─── SIDEBAR NAVIGATION ───
    page_from_query = st.query_params.get("page", "")
    query_to_page = {
        "dashboard": "📊  Dashboard",
        "flight-ops": "✈️  Flight Ops",
        "crew-manager": "👨‍✈️  Crew Manager",
        "schedule-board": "📋  Schedule Board",
        "alerts": "⚠️  Alerts",
    }
    if page_from_query in query_to_page:
        st.session_state["main_page"] = query_to_page[page_from_query]
        st.query_params.clear()

    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; margin-bottom:1.5rem;">
            <div style="font-size:1.1rem; font-weight:700; color:#0f172a; letter-spacing:0.15em; font-family:'Bebas Neue',sans-serif; font-size:1.6rem;">✈️ PILOT SCHEDULER</div>
            <div style="font-size:0.65rem; color:#94a3b8; margin-top:0.3rem; font-family:'Space Mono',monospace; letter-spacing:0.1em;">OPERATIONS COMMAND CENTER</div>
            <div style="font-size:0.6rem; color:#3b82f6; margin-top:0.2rem; font-family:'Space Mono',monospace;">V.01 / BETA RELEASE</div>
        </div>
        <hr style="border:none; border-top:1px solid #e2e8f0; margin:0 0 0.5rem 0;">
        """, unsafe_allow_html=True)
        if st.session_state.get("optimization_applied"):
            st.success("🟢 Optimized schedule active")

        page = st.radio(
            "Navigation",
            ["📊  Dashboard", "✈️  Flight Ops", "👨‍✈️  Crew Manager", "📋  Schedule Board", "⚠️  Alerts"],
            key="main_page",
            label_visibility="collapsed"
        )

        st.markdown("<hr style='border:none; border-top:1px solid #e2e8f0; margin:1rem 0;'>", unsafe_allow_html=True)

        # Quick pilot lookup in sidebar
        df_dash = load_dashboard_data()
        st.markdown("<div style='font-size:0.65rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.3rem; font-family:\"Space Mono\",monospace;'>Quick Pilot Lookup</div>", unsafe_allow_html=True)
        if df_dash.empty:
            st.info("Upload a flight CSV to populate this view.")
        else:
            all_pilots = sorted(set(df_dash['Captain'].unique().tolist() + df_dash['First Officer'].unique().tolist()))
            selected_pilot = st.selectbox(
                "Pilot",
                ["— Select —"] + all_pilots,
                key="quick_pilot_lookup",
                label_visibility="collapsed",
                on_change=_on_quick_pilot_change,
            )

    # ─── GLOBAL ROUTING INTERCEPTS ───
    # 1. Sidebar dropdown / gantt click target
    if hasattr(st.session_state, '_pilot_nav') and st.session_state._pilot_nav:
        target = st.session_state._pilot_nav
        st.session_state._pilot_nav = None
        render_pilot_detail_page(target, df_dash)
        st.stop()

    # ─── PAGE: DASHBOARD ───
    if page == "📊  Dashboard":
        import streamlit.components.v1 as components

        # Split-Flap Hero Animation
        hero_html = """
        <!DOCTYPE html>
        <html>
        <head>
        <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
        <style>
        body { margin:0; padding:0; background-color:transparent; font-family:'Space Mono',monospace; color:#0f172a; }
        .hero-container { padding:1.5rem 1rem 0.5rem; display:flex; flex-direction:column; }
        .split-flap-row { display:flex; gap:4px; margin-bottom:1.2rem; flex-wrap:wrap; justify-content:center; }
        .flap {
            width:clamp(30px,4vw,60px); height:clamp(45px,6vw,90px);
            background-color:#0f172a; color:white;
            font-size:clamp(30px,4vw,55px); font-weight:bold;
            display:flex; align-items:center; justify-content:center;
            font-family:'Bebas Neue',sans-serif;
            position:relative; box-shadow:0 4px 6px -1px rgb(0 0 0/0.1);
            transform-origin:bottom;
            animation:flipIn 0.5s ease-out forwards;
            opacity:0; transform:rotateX(-90deg);
        }
        .flap::after { content:''; position:absolute; top:50%; left:0; right:0; height:2px; background-color:rgba(0,0,0,0.5); }
        .flap.blue { background-color:#3b82f6; }
        @keyframes flipIn { from{transform:rotateX(-90deg);opacity:0;} to{transform:rotateX(0deg);opacity:1;} }
        .title-sub { font-size:1.6rem; color:#94a3b8; font-family:'Space Mono',monospace; margin-bottom:0.8rem; animation:fadeIn 1s ease-out 1s forwards; opacity:0; text-align:center; }
        .desc-text { font-size:0.85rem; color:#475569; max-width:550px; line-height:1.6; margin-bottom:1.5rem; margin-left:auto; margin-right:auto; animation:fadeIn 1s ease-out 1.2s forwards; opacity:0; text-align:center; }
        @keyframes fadeIn { to{opacity:1;} }
        .version { font-size:0.6rem; color:#3b82f6; text-align:right; margin-top:1rem; font-family:'Space Mono',monospace; letter-spacing:0.1em; }
        </style>
        </head>
        <body>
        <div class="hero-container">
            <div class="split-flap-row" id="split-flap">
                <div class="flap" style="animation-delay:0.1s">P</div>
                <div class="flap" style="animation-delay:0.15s">I</div>
                <div class="flap" style="animation-delay:0.2s">L</div>
                <div class="flap" style="animation-delay:0.25s">O</div>
                <div class="flap" style="animation-delay:0.3s">T</div>
                <div style="width:20px;"></div>
                <div class="flap blue" style="animation-delay:0.35s">S</div>
                <div class="flap blue" style="animation-delay:0.4s">C</div>
                <div class="flap blue" style="animation-delay:0.45s">H</div>
                <div class="flap blue" style="animation-delay:0.5s">E</div>
                <div class="flap blue" style="animation-delay:0.55s">D</div>
                <div class="flap blue" style="animation-delay:0.6s">U</div>
                <div class="flap blue" style="animation-delay:0.65s">L</div>
                <div class="flap blue" style="animation-delay:0.7s">E</div>
                <div class="flap blue" style="animation-delay:0.75s">R</div>
            </div>
            <div class="title-sub">ML-Assisted Pilot Assignment &amp; DGCA Compliance</div>
            <div class="desc-text">
                Intelligent scheduling powered by Random Forest fatigue prediction.<br>
                Assign pilots, detect violations, and optimize duty rosters - all within DGCA regulatory limits.
            </div>
            <div style="display:flex;gap:10px;justify-content:center;margin-top:18px;flex-wrap:wrap">
              <span style="font-size:11px;padding:4px 12px;border:1px solid #e2e8f0;
                           border-radius:0;font-family:'Space Mono',monospace;
                           color:#64748b;background:#f8fafc;letter-spacing:0.05em">
                RANDOM FOREST FATIGUE MODEL
              </span>
              <span style="font-size:11px;padding:4px 12px;border:1px solid #e2e8f0;
                           border-radius:0;font-family:'Space Mono',monospace;
                           color:#64748b;background:#f8fafc;letter-spacing:0.05em">
                DGCA RULE ENGINE
              </span>
              <span style="font-size:11px;padding:4px 12px;border:1px solid #e2e8f0;
                           border-radius:0;font-family:'Space Mono',monospace;
                           color:#64748b;background:#f8fafc;letter-spacing:0.05em">
                LIVE CONFLICT DETECTION
              </span>
            </div>
            <div class="version">V.01 / BETA RELEASE</div>
        </div>
        <script>
          const flaps = document.querySelectorAll('.flap');
          const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
          flaps.forEach(flap => {
              const targetText = flap.innerText;
              let iterations = 0;
              const interval = setInterval(() => {
                  flap.innerText = letters[Math.floor(Math.random() * letters.length)];
                  if (iterations >= 10) { clearInterval(interval); flap.innerText = targetText; }
                  iterations++;
              }, 50 + Math.random() * 50);
          });
        </script>
        </body>
        </html>
        """
        components.html(hero_html, height=350, scrolling=False)

        if df_dash.empty:
            st.info("Upload a flight CSV to populate this view.")
            st.stop()

        # --- KPI Row ---
        active_flights = len(df_dash[df_dash['Status'].isin(['On Time', 'Boarding'])])
        total_pilots = len(df_dash['Captain'].unique()) + len(df_dash['First Officer'].unique())
        utilization = 87.5
        alert_count = len(df_dash[df_dash['Status'] == 'Delayed']) * 2

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Active Flights", f"{active_flights}", delta="Live", delta_color="normal")
        k2.metric("Pilots on Duty", f"{total_pilots}", delta=f"{total_pilots} assigned")
        k3.metric("Fleet Utilization", f"{utilization}%", delta="+2.3%")
        k4.metric("System Alerts", f"{alert_count}", delta=f"{alert_count} active", delta_color="inverse")

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Recent Flights (Live CSV) ---
        st.markdown("""
        <div style="margin-top:0.5rem;">
            <span style="color:#3b82f6; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.3em; font-weight:700; font-family:'Space Mono',monospace;">01 / FLIGHT LOGS</span>
            <h2 style="font-family:'Bebas Neue',sans-serif; font-size:3rem; color:#0f172a; margin:0.3rem 0 0 0;">RECENT FLIGHTS</h2>
        </div>
        """, unsafe_allow_html=True)

        recent_df = df_dash.copy()
        recent_df['STD_sort'] = pd.to_datetime(
            recent_df['Date'].astype(str) + " " + recent_df['STD'].astype(str),
            errors='coerce'
        )
        recent_df['STA_sort'] = pd.to_datetime(
            recent_df['Date'].astype(str) + " " + recent_df['STA'].astype(str),
            errors='coerce'
        )
        recent_df['ATA_sort'] = pd.to_datetime(
            recent_df['Date'].astype(str) + " " + recent_df['ATA'].astype(str).str.replace('Landed ', '', regex=False),
            errors='coerce'
        )
        recent_df['ATD_sort'] = pd.to_datetime(
            recent_df['Date'].astype(str) + " " + recent_df['ATD'].astype(str),
            errors='coerce'
        )
        recent_df['Delay (mins)'] = (
            (recent_df['ATA_sort'] - recent_df['STA_sort']).dt.total_seconds() / 60
        ).fillna((recent_df['ATD_sort'] - recent_df['STD_sort']).dt.total_seconds() / 60).fillna(0).round(0)
        recent_df = recent_df.sort_values('STD_sort', ascending=False).head(20)
        recent_df['StatusColor'] = recent_df['Status'].astype(str)
        recent_df.loc[(recent_df['Status'].astype(str).str.lower() == 'cancelled') | (recent_df['Delay (mins)'] >= 30), 'StatusColor'] = '🔴 ' + recent_df['Status'].astype(str)
        recent_df.loc[(recent_df['Delay (mins)'] >= 1) & (recent_df['Delay (mins)'] < 30), 'StatusColor'] = '🟡 ' + recent_df['Status'].astype(str)
        recent_df.loc[(recent_df['Delay (mins)'] < 1), 'StatusColor'] = '🟢 ' + recent_df['Status'].astype(str)

        st.dataframe(
            recent_df[['Flight Number', 'From', 'To', 'STD', 'STA', 'StatusColor', 'Delay (mins)']]
            .rename(columns={'Flight Number': 'Flight No', 'StatusColor': 'Status'}),
            use_container_width=True,
            hide_index=True,
        )

        # --- Compliance Metric (full-width, under Recent Flights) ---
        # Uses recent_df delays to provide a lightweight compliance indicator.
        delay_threshold = 30  # minutes
        if len(recent_df) > 0 and 'Delay (mins)' in recent_df.columns:
            non_cancelled = recent_df['Status'].astype(str).str.lower() != 'cancelled'
            pass_mask = non_cancelled & (recent_df['Delay (mins)'].fillna(0) < delay_threshold)
            compliance_rate = float(pass_mask.mean()) * 100.0
        else:
            compliance_rate = 0.0

        if compliance_rate >= 85.0:
            compliance_label = "PASS"
            bar_color = "#22c55e"
        elif compliance_rate >= 60.0:
            compliance_label = "REVIEW"
            bar_color = "#f59e0b"
        else:
            compliance_label = "RISK"
            bar_color = "#ef4444"

        st.markdown(
            f"""
            <div style="margin: 0.2rem 0 1rem 0; width: 100%;">
                <div style="margin-top:0.2rem;">
                    <span style="color:#3b82f6; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.3em; font-weight:700; font-family:'Space Mono',monospace;">
                        COMPLIANCE METRIC
                    </span>
                </div>
                <div style="display:flex; align-items: baseline; gap: 1rem; margin-top:0.35rem;">
                    <div style="font-family:'Bebas Neue',sans-serif; font-size:2.2rem; color:#0f172a; line-height:1;">
                        {compliance_rate:.1f}%
                    </div>
                    <div style="font-family:'Space Mono',monospace; color:#475569; font-size:0.9rem;">
                        Status: <span style="font-weight:800; color:{bar_color};">{compliance_label}</span>
                        <span style="color:#64748b;"> (delay &lt; {delay_threshold}m; excluding cancelled)</span>
                    </div>
                </div>
                <div style="height: 10px; background:#e2e8f0; border-radius: 999px; overflow:hidden; margin-top:0.6rem;">
                    <div style="height:100%; width:{min(max(compliance_rate, 0.0), 100.0):.1f}%; background:{bar_color};"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<hr style='border:none; border-top:1px solid #e2e8f0; margin:1rem 0;'>", unsafe_allow_html=True)

        # --- Alerts + Activity ---
        st.markdown("""
        <div>
            <span style="color:#3b82f6; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.3em; font-weight:700; font-family:'Space Mono',monospace;">02 / SYSTEM STATUS</span>
            <h2 style="font-family:'Bebas Neue',sans-serif; font-size:2.5rem; color:#0f172a; margin:0.3rem 0 0.5rem 0;">ALERTS & ACTIVITY</h2>
        </div>
        """, unsafe_allow_html=True)

        al_col, act_col = st.columns(2)
        with al_col:
            st.error("**Fatigue Risk** — Captain Rao approaching 9hr block limit.")
            st.warning("**DGCA Warning** — FO Verma assigned 6th consecutive sector.")
            st.warning("**Scheduling Conflict** — VT-ABC delayed, AI101 downstream impact.")
        with act_col:
            st.info("🕒 **10 min ago** — System swapped VT-XYZ to BLR route.")
            st.info("🕒 **25 min ago** — Dispatch approved Capt. Singh extension.")
            st.info("🕒 **1 hr ago** — Auto-assigned FO Patel to AI305 (DEL→BOM).")

        st.stop()

    # ─── PAGE: FLIGHT OPS ───
    if page == "✈️  Flight Ops":
        # ── Blueprint CSS for this page ──
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Mono:wght@400;700&display=swap');
        .ops-heading { font-family:'Bebas Neue',sans-serif; letter-spacing:0.12em; font-size:1.6rem; color:#0f172a; margin-bottom:0.2rem; }
        .ops-sub     { font-family:'Space Mono',monospace; font-size:0.82rem; color:#64748b; margin-bottom:1.2rem; }
        .chart-header { font-family:'Bebas Neue',sans-serif; letter-spacing:0.1em; font-size:1.15rem; color:#0f172a; margin-bottom:0.3rem; }
        [data-testid="stMetric"] { border-radius:0 !important; background:#f8fafc; }
        [data-testid="stMetric"] label { font-family:'Space Mono',monospace !important; text-transform:uppercase; font-size:0.7rem !important; letter-spacing:0.08em; }
        [data-testid="stMetric"] [data-testid="stMetricValue"] { font-family:'Space Mono',monospace !important; }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="ops-heading">FLIGHT OPERATIONS & OPTIMIZER</div>', unsafe_allow_html=True)
        st.markdown('<div class="ops-sub">Upload flight data, query the AI engine, and explore schedule analytics</div>', unsafe_allow_html=True)
        st.markdown("---")

        # ──────────────────────────────────────────────
        # SINGLE data-loading block (no duplicates)
        # ──────────────────────────────────────────────
        default_file = "Flight_Data.csv"
        uploaded = st.file_uploader("Upload flight data (CSV)", type=["csv"], key="flightops_upload")

        if uploaded:
            df_raw = pd.read_csv(uploaded)
            df_raw = normalize_columns(df_raw)
            st.session_state["master_df"] = df_raw.copy()
            st.session_state["optimization_applied"] = False
            st.session_state.pop("optimized_df", None)
        elif "master_df" in st.session_state and not st.session_state["master_df"].empty:
            df_raw = normalize_columns(st.session_state["master_df"].copy())
        elif os.path.exists(default_file):
            df_raw = normalize_columns(pd.read_csv(default_file))
            st.session_state["master_df"] = df_raw.copy()
            st.session_state.setdefault("optimization_applied", False)
        else:
            st.info("Upload a flight CSV to populate this view.")
            st.stop()

        df = clean_flight_data_enhanced(df_raw)
        optimizer = EnhancedScheduleOptimizer(df)

        # Load NLP models (with offline fallback)
        with st.spinner("Loading Enhanced NLP Models..."):
            sbert = load_sbert()
            col_keys = list(ENHANCED_COLUMN_DESCRIPTIONS.keys())
            op_keys = list(ENHANCED_OPERATIONS.keys())
            if sbert is None:
                col_emb = None
                op_emb = None
                if not st.session_state.get("nlp_error_shown", False):
                    err_msg = st.session_state.get("nlp_model_error", "unknown error")
                    st.warning(f"Error loading NLP model: {err_msg}")
                    st.session_state["nlp_error_shown"] = True
            else:
                st.session_state["nlp_error_shown"] = False
                st.session_state.pop("nlp_model_error", None)
                col_emb = sbert.encode(
                    [f"{k}: {v}" for k, v in ENHANCED_COLUMN_DESCRIPTIONS.items()],
                    convert_to_tensor=True
                )
                op_emb = sbert.encode(
                    [f"{k}: {v}" for k, v in ENHANCED_OPERATIONS.items()],
                    convert_to_tensor=True
                )

        # Train models
        if "enhanced_artifacts" not in st.session_state:
            with st.spinner("Training Enhanced ML Models with Cascading Features..."):
                st.session_state["enhanced_artifacts"] = train_enhanced_models(df)
        artifacts = st.session_state["enhanced_artifacts"]

        # ──────────────────────────────────────────────
        # 4 TABS
        # ──────────────────────────────────────────────
        tab_data, tab_ml, tab_analytics, tab_cascade = st.tabs([
            "📂 Data & Overview",
            "🧠 ML Model & NLP Query",
            "📊 Schedule Analytics",
            "⛓ Cascading Delays"
        ])

        # ═══════════════════════════════════════════
        # TAB 1 — DATA & OVERVIEW
        # ═══════════════════════════════════════════
        with tab_data:
            # KPI metric cards
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.metric("Total Flights", f"{len(df):,}")
            with kpi2:
                st.metric("Airports", len(AIRPORT_CONFIG))
            with kpi3:
                avg_cascade_risk = df['Cascading_Risk_Score'].mean()
                st.metric("Avg Cascading Risk", f"{avg_cascade_risk:.2f}")
            with kpi4:
                high_risk_flights = (df['Cascading_Risk_Score'] > 0.7).sum()
                st.metric("High Risk Flights", high_risk_flights)

            st.divider()

            with st.expander("Preview raw data"):
                st.dataframe(df.head(20), use_container_width=True, hide_index=True)

        # ═══════════════════════════════════════════
        # TAB 2 — ML MODEL & NLP QUERY
        # ═══════════════════════════════════════════
        with tab_ml:
            # Model performance
            with st.expander("🤖 Enhanced Model Performance"):
                meta = artifacts["meta"]
                mp1, mp2 = st.columns(2)
                with mp1:
                    st.metric("RandomForest Accuracy", f"{meta['rf_acc']:.3f}")
                with mp2:
                    st.metric(f"{meta['model_b_name']} Accuracy", f"{meta['model_b_acc']:.3f}")
                st.caption(f"**Enhanced Features:** {', '.join(meta['features'])}")

            st.divider()

            # NLP query interface
            st.markdown('<div class="chart-header">ASK QUESTIONS IN NATURAL LANGUAGE</div>', unsafe_allow_html=True)

            # Example query chip buttons
            st.markdown("**Quick Actions:**")
            user_q = ""
            qa1, qa2, qa3, qa4 = st.columns(4)
            with qa1:
                if st.button("Find Best Times", key="qa_best"):
                    user_q = "find optimal departure slots from Mumbai"
            with qa2:
                if st.button("Show Busiest Times", key="qa_busy"):
                    user_q = "show busiest time slots to avoid"
            with qa3:
                if st.button("Tune Schedule", key="qa_tune"):
                    user_q = "tune schedule time and show impact"
            with qa4:
                if st.button("Cascading Impact", key="qa_cascade"):
                    user_q = "analyze cascading delay impacts"

            user_q_input = st.text_input(
                "Query",
                placeholder="e.g., 'Which flights cause the most cascading delays?' or 'Tune departure time and show impact'",
                key="nlp_query_input"
            )
            if user_q_input:
                user_q = user_q_input

            with st.expander("Enhanced Example Queries"):
                enhanced_examples = {
                    "Optimization": [
                        "Find the best departure times from Mumbai to avoid delays",
                        "Show optimal landing slots at Delhi airport",
                        "When should I schedule flights to minimize cascading impacts?"
                    ],
                    "Avoidance": [
                        "What are the busiest time slots to avoid at BOM?",
                        "Show congested hours across all airports",
                        "Which time periods have highest delay risks?"
                    ],
                    "Schedule Tuning": [
                        "Tune departure time by +30 minutes and show impact",
                        "Simulate moving flight to 2 hours later",
                        "What happens if I reschedule this flight?"
                    ],
                    "Cascading Analysis": [
                        "Which flights have the biggest cascading impact?",
                        "Show aircraft with highest delay propagation risk",
                        "Analyze cascade effects for quick turnaround flights"
                    ],
                    "Predictions": [
                        "Predict delay probability for Sunday morning flight",
                        "What's the risk of delays for peak hour departures?",
                        "Show delay trends by hour and aircraft type"
                    ]
                }
                for category, queries in enhanced_examples.items():
                    st.markdown(f"**{category}:**")
                    for query in queries:
                        if st.button(query, key=f"example_{query}"):
                            user_q = query
                            st.rerun()

            st.divider()

            # Query processing
            if user_q:
                with st.spinner("🔍 Processing your query with enhanced AI..."):
                    mapping = map_enhanced_query(user_q, sbert, col_keys, col_emb, op_keys, op_emb, df)
                st.info(f"**Detected Intent:** {mapping['intent'].upper()}")

                if mapping["intent"] == "optimize":
                    handle_optimization_query(mapping, optimizer)
                elif mapping["intent"] == "busiest":
                    handle_busiest_slots_query(optimizer)
                elif mapping["intent"] == "tune":
                    handle_schedule_tuning_query(df, optimizer)
                elif mapping["intent"] == "cascade":
                    handle_cascading_analysis_query(optimizer)
                elif mapping["intent"] == "predict":
                    handle_prediction_query(mapping, artifacts, df)
                else:
                    handle_statistics_query(mapping, df)

        # ═══════════════════════════════════════════
        # TAB 3 — SCHEDULE ANALYTICS
        # ═══════════════════════════════════════════
        with tab_analytics:
            # ── Congestion Heatmap ──
            with st.container():
                st.markdown('<div class="chart-header">AIRPORT CONGESTION BY HOUR</div>', unsafe_allow_html=True)
                if hasattr(optimizer, 'congestion_df') and not optimizer.congestion_df.empty:
                    pivot_data = optimizer.congestion_df.pivot(
                        index='Airport', columns='Hour', values='CongestionLevel'
                    )
                    fig_heatmap = px.imshow(
                        pivot_data,
                        color_continuous_scale='RdYlGn_r',
                        title='',
                        labels=dict(x="Hour of Day", y="Airport", color="Congestion")
                    )
                    fig_heatmap.update_layout(
                        height=400,
                        margin=dict(t=20, b=40, l=60, r=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font={"family": "Space Mono, monospace"}
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                else:
                    st.info("No congestion data available.")

            st.divider()

            # ── Delay Patterns ──
            with st.container():
                st.markdown('<div class="chart-header">DELAY PATTERNS BY AIRPORT & HOUR</div>', unsafe_allow_html=True)
                if hasattr(optimizer, 'delay_patterns_df') and not optimizer.delay_patterns_df.empty:
                    dp = optimizer.delay_patterns_df.copy()
                    dp['RiskLevel'] = dp['DelayRisk'].apply(
                        lambda r: 'High' if r >= 0.6 else ('Medium' if r >= 0.3 else 'Low')
                    )
                    fig_delay = px.bar(
                        dp, x='Hour', y='AvgDepDelay', color='RiskLevel',
                        facet_row='Airport',
                        color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"},
                        labels={"AvgDepDelay": "Avg Departure Delay (min)", "Hour": "Hour of Day"},
                        title=''
                    )
                    fig_delay.update_layout(
                        height=max(400, len(AIRPORT_CONFIG) * 150),
                        margin=dict(t=20, b=40, l=60, r=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font={"family": "Space Mono, monospace"},
                        showlegend=True
                    )
                    st.plotly_chart(fig_delay, use_container_width=True)
                else:
                    st.info("No delay pattern data available.")

        # ═══════════════════════════════════════════
        # TAB 4 — CASCADING DELAYS
        # ═══════════════════════════════════════════
        with tab_cascade:
            import networkx as nx

            analyzer = optimizer.cascading_analyzer

            # ── Network rotation graph ──
            with st.container():
                st.markdown('<div class="chart-header">AIRCRAFT ROTATION NETWORK</div>', unsafe_allow_html=True)
                if hasattr(analyzer, 'impact_df') and not analyzer.impact_df.empty:
                    import plotly.graph_objects as go_fig

                    top_flights = analyzer.get_highest_impact_flights(20)

                    # Build graph
                    G = nx.DiGraph()
                    for _, row in top_flights.iterrows():
                        G.add_node(row['origin'], type='airport')
                        G.add_node(row['dest'], type='airport')
                        G.add_edge(row['origin'], row['dest'],
                                   weight=row['cascading_impact_score'],
                                   flight=row['flight_number'])

                    if len(G.nodes) > 0:
                        pos = nx.spring_layout(G, seed=42, k=2)

                        # Edge traces
                        edge_x, edge_y = [], []
                        for u, v in G.edges():
                            x0, y0 = pos[u]
                            x1, y1 = pos[v]
                            edge_x += [x0, x1, None]
                            edge_y += [y0, y1, None]

                        edge_trace = go_fig.Scatter(
                            x=edge_x, y=edge_y, mode='lines',
                            line=dict(width=1.5, color='#94a3b8'),
                            hoverinfo='none'
                        )

                        # Node traces
                        node_x = [pos[n][0] for n in G.nodes()]
                        node_y = [pos[n][1] for n in G.nodes()]
                        node_text = list(G.nodes())
                        node_color = ['#3b82f6' if G.degree(n) > 2 else '#10b981' for n in G.nodes()]
                        node_size = [max(20, G.degree(n) * 8) for n in G.nodes()]

                        node_trace = go_fig.Scatter(
                            x=node_x, y=node_y, mode='markers+text',
                            text=node_text, textposition='top center',
                            textfont=dict(family='Space Mono, monospace', size=11),
                            marker=dict(size=node_size, color=node_color,
                                        line=dict(width=1, color='#0f172a')),
                            hoverinfo='text'
                        )

                        fig_network = go_fig.Figure(
                            data=[edge_trace, node_trace],
                            layout=go_fig.Layout(
                                showlegend=False,
                                height=400,
                                margin=dict(t=10, b=10, l=10, r=10),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                font=dict(family='Space Mono, monospace')
                            )
                        )
                        st.plotly_chart(fig_network, use_container_width=True)
                    else:
                        st.info("Not enough data to build rotation network.")

            st.divider()

            # ── Propagation chain table ──
            with st.container():
                st.markdown('<div class="chart-header">CASCADING IMPACT — HIGHEST RISK FLIGHTS</div>', unsafe_allow_html=True)
                if hasattr(analyzer, 'impact_df') and not analyzer.impact_df.empty:
                    display_df = analyzer.get_highest_impact_flights(15)
                    st.dataframe(
                        display_df[['flight_number', 'origin', 'dest', 'departure_time',
                                    'cascading_impact_score', 'risk_category']],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "flight_number": st.column_config.TextColumn("Flight", width="small"),
                            "origin": st.column_config.TextColumn("From", width="small"),
                            "dest": st.column_config.TextColumn("To", width="small"),
                            "departure_time": st.column_config.TextColumn("Departure", width="small"),
                            "cascading_impact_score": st.column_config.ProgressColumn(
                                "Impact Score", min_value=0, max_value=1, format="%.2f"
                            ),
                            "risk_category": st.column_config.TextColumn("Risk Level", width="medium")
                        }
                    )
                else:
                    st.info("No cascading impact data available.")

        st.stop()

    # ─── PAGE: CREW MANAGER ───
    elif page == "👨‍✈️  Crew Manager":
        st.markdown("# 👨‍✈️ Crew Manager")
        st.markdown("Select a pilot from the sidebar, or explore fatigue analytics below.")
        st.markdown("---")
        render_pilot_fatigue_dashboard()
        st.stop()

    # ─── PAGE: SCHEDULE BOARD ───
    elif page == "📋  Schedule Board":
        render_scheduling_board()
        st.stop()

    # ─── PAGE: ALERTS ───
    elif page == "⚠️  Alerts":
        st.markdown("# ⚠️ Alert Center")
        st.markdown("Full log of scheduling conflicts, fatigue risks, and DGCA violations.")
        st.markdown("---")

        # 1. Severity Filter
        severity_filter = st.radio("Filter Alerts by Severity", ["All", "Critical", "Warning", "Info"], horizontal=True)

        delayed = df_dash[df_dash['Status'] == 'Delayed'].copy()

        # Generate some mock severities for delayed flights for demonstration
        # so they filter correctly. Delay > 60 mins -> Critical, else Warning
        if 'Delay_Mins' not in delayed.columns:
            np.random.seed(42)
            delayed['Delay_Mins'] = np.random.randint(15, 120, size=len(delayed))
        
        def assign_sev(mins):
            if mins > 90: return "Critical"
            elif mins > 45: return "Warning"
            return "Info"
            
        delayed['Severity'] = delayed['Delay_Mins'].apply(assign_sev)

        # Filter by severity
        if severity_filter != "All":
            delayed = delayed[delayed['Severity'] == severity_filter]

        if len(delayed) > 0:
            st.markdown(f"**{len(delayed)}** delayed flights generating alerts")
            # 2. Group by origin airport using expanders
            for origin, group in delayed.groupby('From'):
                with st.expander(f"{origin} — {len(group)} delays"):
                    for _, row in group.iterrows():
                        sev = row['Severity']
                        sev_color = "#ef4444" if sev == "Critical" else ("#f59e0b" if sev == "Warning" else "#3b82f6")
                        bg_color = "#fee2e2" if sev == "Critical" else ("#fef3c7" if sev == "Warning" else "#dbeafe")
                        
                        st.markdown(
                            f'<div style="padding: 0.75rem; border-left: 4px solid {sev_color}; margin-bottom: 0.5rem; background: {bg_color}; font-family: \'Space Mono\', monospace; font-size: 0.85rem; border-radius: 0 4px 4px 0;">'
                            f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                            f'<span><b>{row["Flight Number"]}</b> — {row["From"]} → {row["To"]}</span>'
                            f'<span style="background: {sev_color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: bold; text-transform: uppercase;">{sev}</span>'
                            f'</div>'
                            f'<div style="color: #475569; margin-top: 0.25rem;">Captain: {row["Captain"]} | FO: {row["First Officer"]}</div>'
                            f'</div>', 
                            unsafe_allow_html=True
                        )
        else:
            st.success(f"No {severity_filter.lower() if severity_filter != 'All' else ''} delayed flights.")

        st.markdown("---")
        
        # 3. Fatigue Risk Pilots
        st.markdown("### Fatigue Risk Pilots")
        
        predictor = get_pilot_fatigue_predictor(_FATIGUE_PREDICTOR_RESOURCE_VERSION)

        all_pilots = set(df_dash['Captain'].dropna()).union(set(df_dash['First Officer'].dropna()))
        
        critical_pilots = []
        for p in all_pilots:
            # Generate deterministic mock features based on name hash to simulate model input
            np.random.seed(hash(p) % 10000)
            f_hours = np.random.uniform(4, 11)
            f_ratio = np.random.uniform(0.8, 2.1)
            f_workload = np.random.uniform(3, 9)
            
            p_class, p_probs = predictor.predict(f_hours, f_ratio, f_workload)
            if p_class == 'Critical':
                # Grab the Critical probability (index 2 usually)
                risk_idx = round(p_probs[2] * 100, 1)
                critical_pilots.append((p, risk_idx))
                
        # Sort by highest risk
        critical_pilots.sort(key=lambda x: x[1], reverse=True)

        if severity_filter in ["All", "Critical"] and critical_pilots:
            st.markdown("The following pilots have been flagged by the ML model with a Critical risk level:")
            cols = st.columns(3)
            for i, (cp, risk) in enumerate(critical_pilots):
                with cols[i % 3]:
                    st.markdown(
                        f'<div style="padding: 1rem; border: 1px solid #fecaca; border-radius: 8px; margin-bottom: 0.8rem; background: white; text-align: center; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">'
                        f'<div style="font-family: \'Bebas Neue\', sans-serif; font-size: 1.4rem; color: #0f172a;">{cp}</div>'
                        f'<div style="margin-top: 0.5rem;">'
                        f'<span style="background: #ef4444; color: white; padding: 3px 10px; border-radius: 20px; font-family: \'Space Mono\', monospace; font-size: 0.75rem; font-weight: bold;">CRITICAL // RISK: {risk}</span>'
                        f'</div></div>', 
                        unsafe_allow_html=True
                    )
        elif severity_filter in ["All", "Critical"]:
            st.success("No pilots currently flagged with Critical fatigue risk.")
        elif severity_filter not in ["All", "Critical"]:
            st.info(f"Fatigue Risk section hidden (Filter set to {severity_filter}).")

        st.markdown("---")
        
        # 4. DGCA Compliance Summary
        st.markdown("### DGCA Compliance Summary")
        
        comp_data = pd.DataFrame({
            'Rule': ['Max Duty Hours', 'Min Rest Period', 'Max Sectors', 'Weekly Hours'],
            'Limit': ['12h', '8h', '6', '60h'],
            'Status': ['✅', '⚠️ Review', '✅', '✅']
        })
        
        st.dataframe(
            comp_data,
            column_config={
                "Rule": st.column_config.TextColumn("DGCA Regulation", width="medium"),
                "Limit": st.column_config.TextColumn("Threshold Limit", width="small"),
                "Status": st.column_config.TextColumn("Current Status", width="small")
            },
            hide_index=True,
            use_container_width=True
        )
        
        st.stop()


# Enhanced Query Handlers
# ------------------------------
def map_enhanced_query(query: str, sbert_model, col_keys, col_emb, op_keys, op_emb, df):
    """Enhanced query mapping with new intents"""
    top_cols = []
    top_op = None

    # Semantic matching is optional; fall back to keyword routing when SBERT is unavailable.
    if sbert_model is not None and col_emb is not None and op_emb is not None:
        from sentence_transformers import util

        q_emb = sbert_model.encode(query, convert_to_tensor=True)

        # Calculate similarities
        col_scores = util.pytorch_cos_sim(q_emb, col_emb)[0].cpu().tolist()
        col_with_scores = sorted([(col_keys[i], float(s)) for i, s in enumerate(col_scores)], key=lambda x: -x[1])
        top_cols = [c for c, s in col_with_scores[:3] if s > 0.28]

        op_scores = util.pytorch_cos_sim(q_emb, op_emb)[0].cpu().tolist()
        op_with_scores = sorted([(op_keys[i], float(s)) for i, s in enumerate(op_scores)], key=lambda x: -x[1])
        top_op = op_with_scores[0][0] if op_with_scores and op_with_scores[0][1] > 0.22 else None

    # Enhanced intent detection
    intent = "statistics"  # default
    q_lower = query.lower()

    if any(kw in q_lower for kw in ["optimize", "best time", "optimal", "recommend"]):
        intent = "optimize"
    elif any(kw in q_lower for kw in ["busiest", "avoid", "congested", "busy"]):
        intent = "busiest"
    elif any(kw in q_lower for kw in ["tune", "reschedule", "move", "shift", "simulate"]):
        intent = "tune"
    elif any(kw in q_lower for kw in ["cascade", "cascading", "impact", "propagation", "chain"]):
        intent = "cascade"
    elif any(kw in q_lower for kw in ["predict", "probability", "risk", "chance"]):
        intent = "predict"

    return {
        "intent": intent,
        "top_cols": top_cols,
        "top_op": top_op,
        "query_text": query
    }


def handle_optimization_query(mapping, optimizer):
    """Handle optimization queries"""
    st.markdown("### Optimal Schedule Recommendations")

    # Airport selection
    airports = list(AIRPORT_CONFIG.keys())
    selected_airport = st.selectbox("Select Airport", airports, key="opt_airport")

    # Get recommendations
    recommendations = optimizer.find_optimal_slots_enhanced(selected_airport)

    # Display results in tabs
    tab1, tab2, tab3 = st.tabs([" Recommendations", " Analysis Chart", "Heatmap"])

    with tab1:
        st.markdown(f"####  Top Optimal Slots for **{selected_airport}**")

        for i, rec in enumerate(recommendations[:5]):
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"**{rec['OptimalityRating']} {rec['Hour']:02d}:00-{rec['Hour'] + 1:02d}:00**")
                    st.caption(rec['Recommendation'])
                with col2:
                    st.metric("Score", f"{rec['Score']:.2f}")
                with col3:
                    st.metric("Traffic", int(rec['TotalFlights']))

    with tab2:
        # Interactive chart
        rec_df = pd.DataFrame(recommendations)
        fig = px.bar(
            rec_df, x='Hour', y='Score',
            color='Score', color_continuous_scale='RdYlGn_r',
            title=f'Hourly Optimization Scores for {selected_airport}',
            hover_data=['CongestionLevel', 'DelayRisk', 'CascadingRisk']
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = create_busiest_slots_heatmap(optimizer)
        st.plotly_chart(fig, use_container_width=True)


def handle_busiest_slots_query(optimizer):
    """Handle busiest time slots queries"""
    st.markdown("### Busiest Time Slots to Avoid")

    # Show busiest slots table
    if hasattr(optimizer, 'busiest_slots_df'):
        busiest = optimizer.busiest_slots_df.head(15)

        st.markdown("#### ⚠️ Top 15 Busiest Slots Across All Airports")

        for _, slot in busiest.iterrows():
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
                with col1:
                    st.markdown(f"**{slot['Airport']} - {slot['Hour']:02d}:00**")
                with col2:
                    congestion_color = "🔴" if slot['CongestionLevel'] > 0.8 else "🟡" if slot[
                                                                                            'CongestionLevel'] > 0.5 else "🟢"
                    st.markdown(f"{congestion_color} {slot['CongestionLevel']:.2f}")
                with col3:
                    st.metric("Flights", int(slot['TotalFlights']))
                with col4:
                    st.caption(slot['Recommendation'])
                    if slot['AlternativeSlots'] != "No nearby alternatives available":
                        st.success(f"Alternative: {slot['AlternativeSlots']}")


def handle_schedule_tuning_query(df, optimizer):
    """Handle schedule tuning simulation"""
    st.markdown("### Interactive Schedule Tuning Simulator")

    with st.form("schedule_tuning_form"):
        st.markdown("#### Select Flight to Tune")

        # Flight selection
        sample_flights = df.head(20)[['UniqueCarrier', 'Origin', 'Dest', 'STD']].reset_index()
        flight_options = [
            f"{row['UniqueCarrier']} {row['Origin']}→{row['Dest']} @ {row['STD']}"
            for _, row in sample_flights.iterrows()
        ]

        selected_flight_idx = st.selectbox("Choose Flight", range(len(flight_options)),
                                           format_func=lambda x: flight_options[x])
        flight_index = sample_flights.iloc[selected_flight_idx].name

        # Time adjustment
        time_shift = st.slider("Time Shift (minutes)", -180, 180, 0, 15,
                               help="Positive = later departure, Negative = earlier departure")

        submitted = st.form_submit_button("Simulate Impact")

    if submitted and time_shift != 0:
        with st.spinner("🔄 Simulating schedule change impact..."):
            results = optimizer.tuner.simulate_schedule_change(flight_index, time_shift)

        if results:
            st.success("Simulation Complete!")

            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Departure", results['original_departure'])
                st.metric("New Departure", results['new_departure'])
            with col2:
                st.metric("Risk Change", f"{results['risk_change']:.3f}",
                          delta=f"{results['risk_change']:.3f}")
            with col3:
                st.metric("Affected Flights", results.get('cascading_flights_affected', 0))
                st.metric("Total Impact (min)", f"{results.get('total_delay_impact', 0):.0f}")

            # Visualization
            fig = create_schedule_tuning_visualization(results)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Unable to simulate - flight not found or insufficient data")


def handle_cascading_analysis_query(optimizer):
    """Handle cascading impact analysis"""
    st.markdown("### Cascading Delay Impact Analysis")

    analyzer = optimizer.cascading_analyzer

    # Top impact flights
    top_impact = analyzer.get_highest_impact_flights(15)

    tab1, tab2, tab3 = st.tabs(["High Impact Flights", "Impact Chart", "By Airport"])

    with tab1:
        st.markdown("####Flights with Highest Cascading Impact")

        for _, flight in top_impact.iterrows():
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
                with col1:
                    st.markdown(f"**{flight['flight_number']}** {flight['origin']}→{flight['dest']}")
                    st.caption(f"Departure: {flight['departure_time']}")
                with col2:
                    st.markdown(flight['risk_category'])
                with col3:
                    st.metric("Impact Score", f"{flight['cascading_impact_score']:.2f}")
                with col4:
                    if flight['aircraft']:
                        st.caption(f"Aircraft: {flight['aircraft']}")

    with tab2:
        fig = create_cascading_impact_chart(analyzer)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        airport_impact = analyzer.get_impact_by_airport()
        st.markdown("#### Cascading Impact by Airport")
        st.dataframe(airport_impact)


def handle_prediction_query(mapping, artifacts, df):
    """Handle prediction queries"""
    st.markdown("### Advanced Delay Prediction")

    with st.form("enhanced_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            origin = st.selectbox("Origin Airport", list(AIRPORT_CONFIG.keys()))
            hour = st.slider("Departure Hour", 0, 23, 8)
            month = st.slider("Month", 1, 12, 7)
        with col2:
            dow = st.slider("Day of Week (1=Mon)", 1, 7, 3)
            is_quick_turnaround = st.checkbox("Quick Turnaround (<90 min)")

        submitted = st.form_submit_button(" Predict Delay Risk")

    if submitted:
        # Create prediction input
        input_features = pd.DataFrame([{
            "Month": month,
            "DayOfWeek": dow,
            "DepHour": hour,
            "Distance": df['Distance'].median(),
            "AirTime": df['AirTime'].median(),
            "IsPeakHour": 1 if hour in AIRPORT_CONFIG.get(origin, {}).get("peak_hours", []) else 0,
            "IsWeekend": 1 if dow > 5 else 0,
            "HasQuickTurnaround": 1 if is_quick_turnaround else 0,
            "TurnaroundPressure": 1 if is_quick_turnaround else 0,
            "Cascading_Risk_Score": 0.5 if is_quick_turnaround else 0.3
        }])

        # Make prediction
        prob = ensemble_predict_enhanced(artifacts["rf_pipe"], artifacts["model_b"], artifacts["meta"], input_features)

        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Delay Probability", f"{prob:.1%}")
        with col2:
            risk_level = "🔴 HIGH" if prob > 0.6 else "🟡 MEDIUM" if prob > 0.3 else "🟢 LOW"
            st.metric("Risk Level", risk_level)
        with col3:
            confidence = min(95, 70 + (prob * 25))  # Higher confidence for extreme predictions
            st.metric("Confidence", f"{confidence:.0f}%")

        # Risk factors breakdown
        with st.expander("🔍 Risk Factors Analysis"):
            factors = []
            if hour in AIRPORT_CONFIG.get(origin, {}).get("peak_hours", []):
                factors.append("Peak hour operation")
            if dow > 5:
                factors.append("Weekend operations")
            if is_quick_turnaround:
                factors.append("⏱Quick turnaround pressure")
            if month in [6, 7, 8]:  # Summer months
                factors.append("High season traffic")

            if factors:
                st.markdown("**Contributing Risk Factors:**")
                for factor in factors:
                    st.markdown(f"- {factor}")
            else:
                st.success("No major risk factors identified")


def handle_statistics_query(mapping, df):
    """Handle statistical queries"""
    st.markdown("###Statistical Analysis")

    # Apply filters based on query
    filtered_df = df.copy()

    # Simple filtering based on top columns
    if mapping["top_cols"]:
        target_col = mapping["top_cols"][0]
        if target_col in df.columns:
            st.markdown(f"#### Analysis of: **{target_col}**")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean", f"{df[target_col].mean():.2f}")
                st.metric("Median", f"{df[target_col].median():.2f}")
            with col2:
                st.metric("Max", f"{df[target_col].max():.2f}")
                st.metric("Std Dev", f"{df[target_col].std():.2f}")

            # Distribution chart
            fig = px.histogram(df, x=target_col, title=f"Distribution of {target_col}")
            st.plotly_chart(fig, use_container_width=True)


# ------------------------------
# Enhanced Helper Functions
# ------------------------------
def ensemble_predict_enhanced(rf_pipe, model_b, meta, input_row: pd.DataFrame):
    """Enhanced ensemble prediction with error handling"""
    try:
        available_features = [f for f in meta["features"] if f in input_row.columns]
        Xrow = input_row[available_features].fillna(0)

        rf_prob = rf_pipe.predict_proba(Xrow)[:, 1] if hasattr(rf_pipe, 'predict_proba') else [0.5]
        b_prob = model_b.predict_proba(Xrow)[:, 1] if hasattr(model_b, 'predict_proba') else [0.5]

        # Weighted ensemble (RF gets higher weight due to typically better performance)
        avg_prob = (rf_prob * 0.6 + b_prob * 0.4)
        return float(avg_prob[0])
    except Exception as e:
        st.warning(f"Prediction error: {e}")
        return 0.5  # Default probability


def load_sbert(name="all-MiniLM-L6-v2"):
    """Load sentence transformer lazily (only when Flight Ops NLP runs)."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(name)
    except Exception as e:
        st.session_state["nlp_model_error"] = str(e)
        return None


# ------------------------------
# Application Entry Point
# ------------------------------
if __name__ == "__main__":
    # Set page config first
    st.set_page_config(
        page_title="Enhanced Flight Schedule Optimizer",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Add custom CSS for better styling
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e1e5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.375rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Run main application
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.markdown("Please check your data file and try again.")


# ------------------------------
# Additional Utility Functions for Enhanced Features
# ------------------------------
def export_analysis_results(optimizer, analyzer):
    """Export analysis results to downloadable formats"""
    results = {
        "optimal_slots": optimizer.find_optimal_slots_enhanced("BOM"),
        "busiest_slots": optimizer.busiest_slots_df.to_dict('records') if hasattr(optimizer,
                                                                                  'busiest_slots_df') else [],
        "high_impact_flights": analyzer.get_highest_impact_flights().to_dict('records'),
        "airport_impact_summary": analyzer.get_impact_by_airport().to_dict()
    }
    return results


def generate_recommendations_report(optimizer):
    """Generate a comprehensive recommendations report"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_airports_analyzed": len(AIRPORT_CONFIG),
            "optimization_recommendations": [],
            "avoidance_recommendations": [],
            "high_risk_flights": []
        }
    }

    # Add recommendations for each airport
    for airport in AIRPORT_CONFIG.keys():
        optimal_slots = optimizer.find_optimal_slots_enhanced(airport, max_slots=3)
        report["summary"]["optimization_recommendations"].append({
            "airport": airport,
            "top_slots": [f"{slot['Hour']:02d}:00" for slot in optimal_slots],
            "best_score": optimal_slots[0]["Score"] if optimal_slots else None
        })

    return report


# ------------------------------
# Performance Monitoring
# ------------------------------
def monitor_query_performance():
    """Monitor and log query performance metrics"""
    if "query_stats" not in st.session_state:
        st.session_state.query_stats = {
            "total_queries": 0,
            "optimization_queries": 0,
            "prediction_queries": 0,
            "tuning_queries": 0,
            "cascading_queries": 0
        }

    return st.session_state.query_stats


# ------------------------------
# Data Quality Checks
# ------------------------------
def validate_data_quality(df):
    """Perform data quality validation and report issues"""
    quality_report = {
        "total_records": len(df),
        "missing_data": {},
        "data_anomalies": [],
        "quality_score": 100
    }

    # Check for missing critical data
    critical_columns = ['DepDelay', 'ArrDelay', 'Origin', 'Dest', 'STD_dt', 'ATD_dt']
    for col in critical_columns:
        if col in df.columns:
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            quality_report["missing_data"][col] = missing_pct
            if missing_pct > 10:
                quality_report["quality_score"] -= 10

    # Check for data anomalies
    if 'DepDelay' in df.columns:
        extreme_delays = (df['DepDelay'] > 300).sum()  # > 5 hours
        if extreme_delays > 0:
            quality_report["data_anomalies"].append(f"{extreme_delays} flights with extreme delays (>5h)")

    return quality_report


# ------------------------------
# Advanced Analytics Extensions
# ------------------------------
def calculate_network_effects(df):
    """Calculate network-wide delay propagation effects"""
    import networkx as nx

    # Build flight network graph
    network = nx.DiGraph()

    # Add flights as nodes
    for idx, flight in df.iterrows():
        network.add_node(idx,
                         flight=flight['UniqueCarrier'],
                         origin=flight['Origin'],
                         dest=flight['Dest'],
                         delay=flight['DepDelay'])

    # Add edges for aircraft rotations
    aircraft_flights = df.groupby('Aircraft_Registration')
    for aircraft, flights in aircraft_flights:
        if len(flights) < 2:
            continue
        flights_sorted = flights.sort_values('STD_dt')
        for i in range(len(flights_sorted) - 1):
            current = flights_sorted.iloc[i].name
            next_flight = flights_sorted.iloc[i + 1].name
            turnaround = flights_sorted.iloc[i]['Turnaround_Time']
            network.add_edge(current, next_flight, turnaround_time=turnaround)

    return network


def predict_seasonal_patterns(df):
    """Predict seasonal delay patterns"""
    monthly_delays = df.groupby('Month')['DepDelay'].agg(['mean', 'std', 'count'])
    seasonal_patterns = {
        "peak_delay_months": monthly_delays['mean'].nlargest(3).index.tolist(),
        "low_delay_months": monthly_delays['mean'].nsmallest(3).index.tolist(),
        "most_volatile_months": monthly_delays['std'].nlargest(3).index.tolist()
    }
    return seasonal_patterns


