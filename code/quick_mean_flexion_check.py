import numpy as np
import pandas as pd
import os
import glob
import requests
from scipy import stats

KP = {
    "nose": 0, "l_shoulder": 5, "r_shoulder": 6,
    "l_elbow": 7, "r_elbow": 8, "l_wrist": 9, "r_wrist": 10,
    "l_hip": 11, "r_hip": 12, "l_knee": 13, "r_knee": 14,
    "l_ankle": 15, "r_ankle": 16,
}
WINDOW_START = 50
RELEASE_BAND = list(range(174, 183))
CONF_THRESH = 0.3

PLAYER_WINDOW_MAP = {
    645261: {"name": "Sandy Alcantara", "season": 2023,
             "base": r"C:\Users\noahl\OneDrive\fatigue_cv\multi_pitcher_pull\window1"},
    663556: {"name": "Shane McClanahan", "season": 2023,
             "base": r"C:\Users\noahl\OneDrive\fatigue_cv\multi_pitcher_pull\window3"},
    543135: {"name": "Nathan Eovaldi", "season": 2023,
             "base": r"C:\Users\noahl\OneDrive\fatigue_cv\multi_pitcher_pull\window2"},
    666142: {"name": "Cole Ragans", "season": 2025,
             "base": r"C:\Users\noahl\OneDrive\fatigue_cv\multi_pitcher_pull\window3"},
    668390: {"name": "Cole Winn", "season": 2026,
             "base": r"C:\Users\noahl\OneDrive\fatigue_cv\multi_pitcher_pull\window2"},
}
SKENES_ID = 694973
SKENES_BASE = r"C:\Users\noahl\OneDrive\fatigue_cv\skenes_2026_season"


def angle_between(a, b, c):
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    cos_a = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def metrics_for_frame(kp):
    def pt(name):
        return kp[KP[name], :2]

    def vis(name):
        return kp[KP[name], 2] > CONF_THRESH

    m = {}
    for side in ["l", "r"]:
        hip, knee, ankle = f"{side}_hip", f"{side}_knee", f"{side}_ankle"
        m[f"{side}_knee_flexion"] = (
            angle_between(pt(hip), pt(knee), pt(ankle))
            if (vis(hip) and vis(knee) and vis(ankle)) else np.nan
        )
    return m


def load_pitcher(npz_dir, progress, prefix=""):
    rows = []
    for _, row in progress.iterrows():
        npz_path = os.path.join(npz_dir, f"{prefix}{row['play_id']}.npz")
        if not os.path.exists(npz_path):
            continue
        data = np.load(npz_path)
        keypoints = data["keypoints"]
        band_metrics = []
        for fi in RELEASE_BAND:
            local_idx = fi - WINDOW_START
            if local_idx < 0 or local_idx >= keypoints.shape[0]:
                continue
            kp = keypoints[local_idx]
            if np.isnan(kp).all():
                continue
            band_metrics.append(metrics_for_frame(kp))
        if not band_metrics:
            continue
        band_df = pd.DataFrame(band_metrics)
        rows.append({
            "play_id": row["play_id"],
            "game_pk": row["game_pk"],
            "n_release_frames_used": len(band_metrics),
            "r_knee_flexion_mean": band_df["r_knee_flexion"].mean(),
        })
    return pd.DataFrame(rows)


def iqr_filter(series, k=2.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return series.between(q1 - k * iqr, q3 + k * iqr)


print("loading all 6 pitchers...")
all_slopes = []
all_negative_count = 0
all_start_count = 0

for pid, info in PLAYER_WINDOW_MAP.items():
    name, season, base = info["name"], info["season"], info["base"]
    safe_name = name.replace(" ", "_")
    progress = pd.read_csv(os.path.join(base, f"progress_{safe_name}_{season}.csv"))
    progress = progress.reset_index().rename(columns={"index": "true_chron_order"})
    keypoints_dir = os.path.join(base, "keypoints")
    prefix = f"{pid}_{season}_"

    df = load_pitcher(keypoints_dir, progress, prefix=prefix)
    df = df.merge(progress[["play_id", "true_chron_order"]], on="play_id")
    mask = iqr_filter(df["r_knee_flexion_mean"]) & (df["n_release_frames_used"] >= 7)
    df = df[mask].sort_values("true_chron_order").reset_index(drop=True)
    df["pitch_in_start"] = df.groupby("game_pk").cumcount() + 1
    start_order = {gp: i + 1 for i, gp in enumerate(df.drop_duplicates("game_pk")["game_pk"])}
    df["start_number"] = df["game_pk"].map(start_order)

    for sn, g in df.groupby("start_number"):
        if len(g) < 15:
            continue
        slope, intercept, r, p, se = stats.linregress(g["pitch_in_start"], g["r_knee_flexion_mean"])
        all_slopes.append(slope)
        all_start_count += 1
        if slope < 0:
            all_negative_count += 1
    print(f"{name}: loaded and processed")

progress = pd.read_csv(os.path.join(SKENES_BASE, "skenes_2026_progress.csv"))
progress = progress.reset_index().rename(columns={"index": "true_chron_order"})
keypoints_dir = os.path.join(SKENES_BASE, "keypoints")
df = load_pitcher(keypoints_dir, progress, prefix="")
df = df.merge(progress[["play_id", "true_chron_order"]], on="play_id")
mask = iqr_filter(df["r_knee_flexion_mean"]) & (df["n_release_frames_used"] >= 7)
df = df[mask].sort_values("true_chron_order").reset_index(drop=True)
df["pitch_in_start"] = df.groupby("game_pk").cumcount() + 1
start_order = {gp: i + 1 for i, gp in enumerate(df.drop_duplicates("game_pk")["game_pk"])}
df["start_number"] = df["game_pk"].map(start_order)

for sn, g in df.groupby("start_number"):
    if len(g) < 15:
        continue
    slope, intercept, r, p, se = stats.linregress(g["pitch_in_start"], g["r_knee_flexion_mean"])
    all_slopes.append(slope)
    all_start_count += 1
    if slope < 0:
        all_negative_count += 1
print("Paul Skenes: loaded and processed")

all_slopes = np.array(all_slopes)
t, p = stats.ttest_1samp(all_slopes, 0)

pd.DataFrame({"slope": all_slopes}).to_csv(
    r"C:\Users\noahl\OneDrive\fatigue_cv\multi_pitcher_pull\pooled_within_start_slopes.csv",
    index=False
)
print(f"\nraw per-start slopes saved to: pooled_within_start_slopes.csv")

print("\n" + "=" * 70)
print("POOLED WITHIN-START MEAN KNEE FLEXION SLOPE (6 pitchers, matches 100-start study metric)")
print("=" * 70)
print(f"total starts analyzed: {all_start_count}")
print(f"negative slopes: {all_negative_count} / {all_start_count} ({all_negative_count/all_start_count*100:.1f}%)")
print(f"mean slope: {all_slopes.mean():.5f} deg/pitch")
print(f"SD of slope distribution: {all_slopes.std():.5f} deg/pitch")
print(f"t = {t:.3f}, p = {p:.6f}")
print(f"\nfor comparison, 100-start study found: 76/100 negative, "
      f"pooled beta=-0.034 deg/pitch, p<0.0001, slope SD=0.022")
