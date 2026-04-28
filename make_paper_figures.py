"""Generate three research-paper figures focused on fatigue optimization.

Figures:
  1. fig_fatigue_tiers.png   — grouped bar chart: pilot-day counts per fatigue tier
                                (Efficient / Moderate / Critical), baseline vs optimized.
  2. fig_dgca_violations.png — bar chart: total DGCA violations, baseline vs optimized.
  3. fig_convergence.png     — line chart: optimizer fatigue objective vs iteration.

Data: monthly_schedule_30days_cleaned.csv (real schedule, not synthetic training data).
Model + optimizer logic mirrors test4.py (PilotFatiguePredictor + propose_optimized_schedule),
reimplemented here to avoid Streamlit import side-effects and to expose per-iteration
convergence history.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(ROOT, "monthly_schedule_30days_cleaned.csv")


# ---------- Fatigue feature mapping (copied verbatim from test4.py) ----------

def _fatigue_triple_from_totals(total_h: float, sectors: int, delays: int) -> tuple:
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
            10.0, 100.0,
        )
    )
    return hours_feat, ratio_feat, workload_feat


def _summarize_pilot_days(df: pd.DataFrame) -> pd.DataFrame:
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
        return pd.DataFrame(columns=["total_h", "sectors", "delays"])
    return pl.groupby(["pilot", "dkey"], sort=False).agg(
        total_h=("dur", "sum"), sectors=("dur", "count"), delays=("dly", "sum"),
    )


def _summ_get(summ: pd.DataFrame, pilot: str, dkey: str):
    key = (str(pilot).strip(), str(dkey))
    if summ.empty or key not in summ.index:
        return 0.0, 0, 0
    row = summ.loc[key]
    return float(row["total_h"]), int(row["sectors"]), int(row["delays"])


# ---------- Random Forest fatigue predictor (same training recipe as test4.py) ----------

class PilotFatiguePredictor:
    def __init__(self):
        np.random.seed(42)
        n = 1000
        hours = np.random.uniform(2, 12, n)
        ratio = np.random.uniform(0.5, 3.0, n)
        workload = np.random.uniform(20, 100, n)
        y = []
        for h, r, w in zip(hours, ratio, workload):
            if h >= 9 or r >= 2.0 or w >= 80:
                y.append(2)
            elif h >= 6 or r >= 1.2 or w >= 55:
                y.append(1)
            else:
                y.append(0)
        X = pd.DataFrame({
            "avg_flight_hours": hours,
            "duty_to_rest_ratio": ratio,
            "workload_index": workload,
        })
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.model.fit(X, np.array(y))

    def class_probs(self, h, r, w):
        p = self.model.predict_proba(np.array([[float(h), float(r), float(w)]]))[0]
        return float(p[0]), float(p[1]), float(p[2])

    def predict_tier(self, h, r, w):
        return int(self.model.predict(np.array([[float(h), float(r), float(w)]]))[0])


# ---------- CSV loading + prep ----------

def load_monthly_schedule() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df.columns = [str(c).strip() for c in df.columns]

    # Date parsing — CSV has "01-Jan-00" style in "Unnamed: 2"; fall back to schedule_day.
    date_col = "Unnamed: 2" if "Unnamed: 2" in df.columns else None
    if date_col:
        parsed = pd.to_datetime(df[date_col], format="%d-%b-%y", errors="coerce")
    else:
        parsed = pd.to_datetime("2024-01-01") + pd.to_timedelta(df["schedule_day"] - 1, unit="D")
    # If parsing fails or the "year 2000" placeholder is being used, synthesize from schedule_day.
    if parsed.isna().all() or parsed.dt.year.min() < 2001:
        parsed = pd.to_datetime("2024-01-01") + pd.to_timedelta(df["schedule_day"].astype(int) - 1, unit="D")
    df["DateKey"] = parsed.dt.strftime("%Y-%m-%d")

    # Duration from "Flight time" "HH:MM" strings.
    def _to_hours(s):
        try:
            h, m = str(s).split(":")
            return int(h) + int(m) / 60.0
        except Exception:
            return 1.5
    df["DurationHrs"] = df["Flight time"].apply(_to_hours).clip(0.5, 6).fillna(1.5)

    for c in ("Captain", "First Officer", "Status"):
        if c not in df.columns:
            df[c] = "N/A"
        df[c] = df[c].astype(str).str.strip()

    return df


# ---------- Baseline metrics ----------

def tier_counts(df: pd.DataFrame, predictor: PilotFatiguePredictor) -> dict:
    summ = _summarize_pilot_days(df)
    counts = {0: 0, 1: 0, 2: 0}
    for (_, _), row in summ.iterrows():
        h, r, w = _fatigue_triple_from_totals(row["total_h"], row["sectors"], row["delays"])
        tier = predictor.predict_tier(h, r, w)
        counts[tier] += 1
    return counts


def dgca_violations(df: pd.DataFrame) -> int:
    """DGCA-style rule violations, summed across all categories.

    Per pilot-day:
      - daily duty hours > 12
      - duty-to-rest proxy ratio > 1.5
    Per pilot (rolling 7-day window):
      - weekly duty hours > 60
    Each violation condition contributes +1. This matches the user-selected
    'Single total-violation count' interpretation.
    """
    summ = _summarize_pilot_days(df).reset_index()
    if summ.empty:
        return 0
    total = 0
    # Daily checks
    for _, row in summ.iterrows():
        h, r, _ = _fatigue_triple_from_totals(row["total_h"], int(row["sectors"]), int(row["delays"]))
        if row["total_h"] > 12.0:
            total += 1
        if r > 1.5:
            total += 1
    # Weekly rolling check per pilot
    summ["dkey_dt"] = pd.to_datetime(summ["dkey"])
    for pilot, grp in summ.groupby("pilot"):
        grp = grp.sort_values("dkey_dt").set_index("dkey_dt")
        rolling = grp["total_h"].rolling("7D").sum()
        total += int((rolling > 60.0).sum())
    return total


def fatigue_objective(df: pd.DataFrame, predictor: PilotFatiguePredictor) -> float:
    """Sum of P(critical) across all pilot-days. Lower is better."""
    summ = _summarize_pilot_days(df)
    s = 0.0
    for (_, _), row in summ.iterrows():
        h, r, w = _fatigue_triple_from_totals(row["total_h"], row["sectors"], row["delays"])
        _, _, pc = predictor.class_probs(h, r, w)
        s += pc
    return s


# ---------- Optimizer with per-iteration convergence logging ----------

def optimize_with_convergence(master_df: pd.DataFrame, predictor: PilotFatiguePredictor,
                              max_passes: int = 8):
    """Reassign CA/FO on high-risk pilot-days, recording the fatigue objective
    after every pass. Mirrors the core loop of propose_optimized_schedule in
    test4.py, but without the Streamlit caching and with convergence history.
    """
    df = master_df.copy()
    captains = [c for c in df["Captain"].unique() if str(c).strip() and str(c).upper() != "N/A"]
    fos = [f for f in df["First Officer"].unique() if str(f).strip() and str(f).upper() != "N/A"]

    ALT_CRIT_MAX = 0.55
    MIN_CRIT_DROP = 0.03
    MAX_ALTS = min(12, max(1, max(len(captains), len(fos)) - 1))

    history = [fatigue_objective(df, predictor)]

    for _ in range(max_passes):
        summ = _summarize_pilot_days(df)
        ranked = []
        for (pilot, dkey), srow in summ.iterrows():
            h, r, w = _fatigue_triple_from_totals(srow["total_h"], int(srow["sectors"]), int(srow["delays"]))
            tier = predictor.predict_tier(h, r, w)
            if tier < 1:
                continue
            _, pm, pc = predictor.class_probs(h, r, w)
            ranked.append((pc, pm, str(pilot), str(dkey)))
        ranked.sort(reverse=True)

        changed = False
        for _, __, pilot, dkey in ranked:
            m = ((df["Captain"] == pilot) | (df["First Officer"] == pilot)) & (df["DateKey"] == dkey)
            sub = df.loc[m]
            if sub.empty:
                continue
            j = sub["DurationHrs"].idxmax()

            ph, ps, pdel = _summ_get(summ, pilot, dkey)
            h0, r0, w0 = _fatigue_triple_from_totals(ph, ps, pdel)
            _, _, pc0 = predictor.class_probs(h0, r0, w0)

            dur = float(df.loc[j, "DurationHrs"])
            dly = 1 if str(df.loc[j, "Status"]).lower() == "delayed" else 0

            role = "Captain" if str(df.loc[j, "Captain"]) == pilot else "First Officer"
            pool = captains if role == "Captain" else fos
            if len(pool) <= 1:
                continue
            cand = [c for c in pool if c != pilot]
            cand.sort(key=lambda a: predictor.class_probs(
                *_fatigue_triple_from_totals(*_summ_get(summ, a, dkey))
            )[2])

            best_alt, best_drop = None, 0.0
            ph2, ps2, pdel2 = ph - dur, max(ps - 1, 0), max(pdel - dly, 0)
            for alt in cand[:MAX_ALTS]:
                ah, as_, adel = _summ_get(summ, alt, dkey)
                ha, ra, wa = _fatigue_triple_from_totals(ah + dur, as_ + 1, adel + dly)
                _, _, pac = predictor.class_probs(ha, ra, wa)
                if pac > ALT_CRIT_MAX:
                    continue
                hp, rp, wp = _fatigue_triple_from_totals(ph2, ps2, pdel2)
                _, _, ppc = predictor.class_probs(hp, rp, wp)
                drop = pc0 - ppc
                if drop > best_drop:
                    best_drop, best_alt = drop, alt

            if best_alt is not None and best_drop >= MIN_CRIT_DROP:
                df.at[j, role] = best_alt
                summ = _summarize_pilot_days(df)
                changed = True

        history.append(fatigue_objective(df, predictor))
        if not changed:
            break

    return df, history


# ---------- Plots ----------

def plot_tier_counts(baseline: dict, optimized: dict, out_path: str):
    labels = ["Efficient", "Moderate", "Critical"]
    b = [baseline[0], baseline[1], baseline[2]]
    o = [optimized[0], optimized[1], optimized[2]]
    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars1 = ax.bar(x - w / 2, b, w, label="Baseline", color="#d9534f", edgecolor="black", linewidth=0.6)
    bars2 = ax.bar(x + w / 2, o, w, label="Optimized", color="#5cb85c", edgecolor="black", linewidth=0.6)
    for bars in (bars1, bars2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{int(h)}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Pilot-Days")
    ax.set_title("Fatigue Tier Distribution — Baseline vs Optimized Schedule")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_dgca_violations(baseline: int, optimized: int, out_path: str):
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    bars = ax.bar(["Baseline", "Optimized"], [baseline, optimized],
                  color=["#d9534f", "#5cb85c"], edgecolor="black", linewidth=0.6, width=0.55)
    for rect in bars:
        h = rect.get_height()
        ax.annotate(f"{int(h)}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=11)
    if baseline > 0:
        reduction = 100.0 * (baseline - optimized) / baseline
        ax.set_title(f"DGCA Compliance Violations\n(↓ {reduction:.1f}% after fatigue optimization)")
    else:
        ax.set_title("DGCA Compliance Violations")
    ax.set_ylabel("Total Violations")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_convergence(history: list, out_path: str):
    iters = np.arange(len(history))
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.plot(iters, history, marker="o", color="#2a6fb5", linewidth=2.2, markersize=7)
    ax.fill_between(iters, history, min(history), color="#2a6fb5", alpha=0.08)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fatigue Objective — Σ P(Critical) over pilot-days")
    ax.set_title("Optimizer Convergence on Fatigue Objective")
    ax.set_xticks(iters)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, v in enumerate(history):
        ax.annotate(f"{v:.2f}", xy=(i, v), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color="#2a6fb5")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


# ---------- Main ----------

def main():
    print("[1/6] Loading monthly schedule...")
    df = load_monthly_schedule()
    print(f"      rows={len(df)}, days={df['DateKey'].nunique()}, "
          f"captains={df['Captain'].nunique()}, fos={df['First Officer'].nunique()}")

    print("[2/6] Training fatigue Random Forest...")
    predictor = PilotFatiguePredictor()

    print("[3/6] Computing baseline metrics...")
    base_tiers = tier_counts(df, predictor)
    base_viol = dgca_violations(df)
    print(f"      tiers={base_tiers}, violations={base_viol}")

    print("[4/6] Running optimizer with convergence logging...")
    optimized_df, history = optimize_with_convergence(df, predictor, max_passes=8)
    print(f"      convergence history: {[round(h, 3) for h in history]}")

    print("[5/6] Computing optimized metrics...")
    opt_tiers = tier_counts(optimized_df, predictor)
    opt_viol = dgca_violations(optimized_df)
    print(f"      tiers={opt_tiers}, violations={opt_viol}")

    print("[6/6] Rendering figures...")
    plot_tier_counts(base_tiers, opt_tiers, os.path.join(ROOT, "fig_fatigue_tiers.png"))
    plot_dgca_violations(base_viol, opt_viol, os.path.join(ROOT, "fig_dgca_violations.png"))
    plot_convergence(history, os.path.join(ROOT, "fig_convergence.png"))

    print("\nSaved:")
    print("  fig_fatigue_tiers.png")
    print("  fig_dgca_violations.png")
    print("  fig_convergence.png")


if __name__ == "__main__":
    main()
