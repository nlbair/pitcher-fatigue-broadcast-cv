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

INJURY_EVENTS = [
    {"player_id": 645261, "date": "2023-04-22", "desc": "Biceps tendinitis, missed 5 games"},
    {"player_id": 645261, "date": "2023-09-04", "desc": "Forearm flexor strain, 15-day IL"},
    {"player_id": 645261, "date": "2023-10-02", "desc": "Season-ending, missed final 24 games"},
    {"player_id": 663556, "date": "2023-08-03", "desc": "Left forearm tightness, pulled mid-start"},
    {"player_id": 663556, "date": "2023-08-21", "desc": "60-day IL / Tommy John surgery, season-ending"},
    {"player_id": 543135, "date": "2023-07-27", "desc": "Right forearm strain, 15-day IL"},
    {"player_id": 666142, "date": "2025-06-08", "desc": "Rotator cuff strain, 15-day IL"},
    {"player_id": 666142, "date": "2025-07-08", "desc": "Transferred to 60-day IL"},
    {"player_id": 666142, "date": "2025-09-17", "desc": "Season-ending, missed 87 games"},
    {"player_id": 668390, "date": "2026-04-24", "desc": "Right arm fatigue, 15-day IL"},
    {"player_id": 668390, "date": "2026-08-28", "desc": "Rotator cuff strain, 15-day IL"},
    {"player_id": 668390, "date": "2026-09-18", "desc": "60-day IL, season-ending"},
]


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
    for side in ["l", "r"]:
        shoulder, elbow, wrist = f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"
        m[f"{side}_elbow_flexion"] = (
            angle_between(pt(shoulder), pt(elbow), pt(wrist))
            if (vis(shoulder) and vis(elbow) and vis(wrist)) else np.nan
        )
    return m


def get_game_dates(pitcher_id, season):
    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats?stats=gameLog&season={season}&group=pitching"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    date_map = {}
    for stat_block in data.get("stats", []):
        for split in stat_block.get("splits", []):
            game = split.get("game", {})
            if "gamePk" in game:
                date_map[game["gamePk"]] = split.get("date")
    return date_map


def load_new_pitcher(player_id):
    info = PLAYER_WINDOW_MAP[player_id]
    name, season, base = info["name"], info["season"], info["base"]
    safe_name = name.replace(" ", "_")
    progress_path = os.path.join(base, f"progress_{safe_name}_{season}.csv")
    keypoints_dir = os.path.join(base, "keypoints")

    progress = pd.read_csv(progress_path)
    progress = progress.reset_index().rename(columns={"index": "true_chron_order"})

    date_map = get_game_dates(player_id, season)
    progress["game_date"] = progress["game_pk"].map(date_map)

    rows = []
    for _, row in progress.iterrows():
        npz_path = os.path.join(keypoints_dir, f"{player_id}_{season}_{row['play_id']}.npz")
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
            "true_chron_order": row["true_chron_order"],
            "play_id": row["play_id"],
            "game_pk": row["game_pk"],
            "game_date": row["game_date"],
            "n_release_frames_used": len(band_metrics),
            "r_knee_flexion_mean": band_df["r_knee_flexion"].mean(),
            "r_knee_flexion_sd": band_df["r_knee_flexion"].std(),
            "r_elbow_flexion_mean": band_df["r_elbow_flexion"].mean(),
            "r_elbow_flexion_sd": band_df["r_elbow_flexion"].std(),
        })

    df = pd.DataFrame(rows).sort_values("true_chron_order").reset_index(drop=True)
    df["player_id"] = player_id
    df["player_name"] = name
    df["season"] = season
    return df


def load_skenes():
    progress = pd.read_csv(os.path.join(SKENES_BASE, "skenes_2026_progress.csv"))
    progress = progress.reset_index().rename(columns={"index": "true_chron_order"})
    date_map = get_game_dates(SKENES_ID, 2026)
    progress["game_date"] = progress["game_pk"].map(date_map)
    keypoints_dir = os.path.join(SKENES_BASE, "keypoints")

    rows = []
    for _, row in progress.iterrows():
        npz_path = os.path.join(keypoints_dir, f"{row['play_id']}.npz")
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
            "true_chron_order": row["true_chron_order"],
            "play_id": row["play_id"],
            "game_pk": row["game_pk"],
            "game_date": row["game_date"],
            "n_release_frames_used": len(band_metrics),
            "r_knee_flexion_mean": band_df["r_knee_flexion"].mean(),
            "r_knee_flexion_sd": band_df["r_knee_flexion"].std(),
            "r_elbow_flexion_mean": band_df["r_elbow_flexion"].mean(),
            "r_elbow_flexion_sd": band_df["r_elbow_flexion"].std(),
        })

    df = pd.DataFrame(rows).sort_values("true_chron_order").reset_index(drop=True)
    df["player_id"] = SKENES_ID
    df["player_name"] = "Paul Skenes"
    df["season"] = 2026
    return df


def iqr_filter(series, k=2.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return series.between(q1 - k * iqr, q3 + k * iqr)


def clean_pitcher_df(df):
    mask = (
        iqr_filter(df["r_knee_flexion_mean"])
        & iqr_filter(df["r_knee_flexion_sd"])
        & (df["n_release_frames_used"] >= 7)
    )
    df = df[mask].copy()
    df["pitch_in_start"] = df.groupby("game_pk").cumcount() + 1
    start_order = {gp: i + 1 for i, gp in enumerate(df.drop_duplicates("game_pk")["game_pk"])}
    df["start_number"] = df["game_pk"].map(start_order)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


print("=" * 90)
print("LOADING ALL PITCHERS")
print("=" * 90)

all_players = {}
for pid in PLAYER_WINDOW_MAP:
    raw = load_new_pitcher(pid)
    clean = clean_pitcher_df(raw)
    all_players[pid] = {"raw": raw, "clean": clean}
    print(f"{PLAYER_WINDOW_MAP[pid]['name']}: {len(raw)} pitches loaded, "
          f"{len(clean)} survive cleaning ({len(clean)/max(len(raw),1)*100:.1f}%)")

skenes_raw = load_skenes()
skenes_clean = clean_pitcher_df(skenes_raw)
all_players[SKENES_ID] = {"raw": skenes_raw, "clean": skenes_clean}
print(f"Paul Skenes (control, no injuries): {len(skenes_raw)} pitches loaded, "
      f"{len(skenes_clean)} survive cleaning ({len(skenes_clean)/max(len(skenes_raw),1)*100:.1f}%)")

check_num = 0


def check(label):
    global check_num
    check_num += 1
    print(f"\n--- CHECK #{check_num}: {label} ---")


print("\n" + "=" * 90)
print("SECTION A: DATA INTEGRITY CHECKS")
print("=" * 90)

for pid, d in all_players.items():
    name = d["raw"]["player_name"].iloc[0] if len(d["raw"]) else PLAYER_WINDOW_MAP.get(pid, {}).get("name", pid)

    check(f"{name} -- duplicate play_id check")
    dupes = d["raw"]["play_id"].duplicated().sum()
    print(f"duplicates found: {dupes}")

    check(f"{name} -- n_release_frames_used distribution")
    print(d["raw"]["n_release_frames_used"].value_counts().sort_index().to_string())

    check(f"{name} -- pitches excluded by IQR/frame-count filter")
    excluded = len(d["raw"]) - len(d["clean"])
    print(f"excluded: {excluded} / {len(d['raw'])} ({excluded/max(len(d['raw']),1)*100:.1f}%)")

    check(f"{name} -- r_knee_flexion_mean sanity range (post-clean)")
    print(d["clean"]["r_knee_flexion_mean"].describe().to_string())

    check(f"{name} -- r_knee_flexion_sd sanity range (post-clean)")
    print(d["clean"]["r_knee_flexion_sd"].describe().to_string())

    check(f"{name} -- missing game_date entries")
    missing_dates = d["clean"]["game_date"].isna().sum()
    print(f"missing dates: {missing_dates} / {len(d['clean'])}")

print("\n" + "=" * 90)
print("SECTION B: WITHIN-START FATIGUE TREND (per pitcher)")
print("=" * 90)

within_start_results = {}
for pid, d in all_players.items():
    df = d["clean"]
    name = df["player_name"].iloc[0] if len(df) else pid

    check(f"{name} -- within-start knee_flexion_sd slope, averaged across starts")
    slopes = []
    for sn, g in df.groupby("start_number"):
        if len(g) < 15:
            continue
        slope, intercept, r, p, se = stats.linregress(g["pitch_in_start"], g["r_knee_flexion_sd"])
        slopes.append(slope)
    if slopes:
        t, p = stats.ttest_1samp(slopes, 0)
        print(f"mean within-start slope: {np.mean(slopes):.5f} deg/pitch | t={t:.3f} p={p:.4f} | n_starts={len(slopes)}")
        within_start_results[pid] = {"mean_slope": np.mean(slopes), "p": p}
    else:
        print("insufficient data")

    check(f"{name} -- within-start elbow_flexion_sd slope, averaged across starts")
    slopes_e = []
    for sn, g in df.groupby("start_number"):
        if len(g) < 15:
            continue
        slope, intercept, r, p, se = stats.linregress(g["pitch_in_start"], g["r_elbow_flexion_sd"])
        slopes_e.append(slope)
    if slopes_e:
        t, p = stats.ttest_1samp(slopes_e, 0)
        print(f"mean within-start elbow slope: {np.mean(slopes_e):.5f} deg/pitch | t={t:.3f} p={p:.4f}")

print("\n" + "=" * 90)
print("SECTION C: SEASON-LONG TREND (start-level, per pitcher)")
print("=" * 90)

for pid, d in all_players.items():
    df = d["clean"]
    name = df["player_name"].iloc[0] if len(df) else pid

    per_start = df.groupby("start_number").agg(
        knee_mean=("r_knee_flexion_mean", "mean"),
        knee_sd=("r_knee_flexion_sd", "mean"),
        elbow_sd=("r_elbow_flexion_sd", "mean"),
        n_pitches=("play_id", "count"),
    ).reset_index()

    check(f"{name} -- season-long knee_flexion_mean trend (start-level)")
    if len(per_start) >= 3:
        slope, intercept, r, p, se = stats.linregress(per_start["start_number"], per_start["knee_mean"])
        print(f"slope={slope:.4f} r2={r**2:.4f} p={p:.4f}")

    check(f"{name} -- season-long knee_flexion_sd trend (start-level)")
    if len(per_start) >= 3:
        slope, intercept, r, p, se = stats.linregress(per_start["start_number"], per_start["knee_sd"])
        print(f"slope={slope:.4f} r2={r**2:.4f} p={p:.4f}")

    check(f"{name} -- season-long elbow_flexion_sd trend (start-level)")
    if len(per_start) >= 3:
        slope, intercept, r, p, se = stats.linregress(per_start["start_number"], per_start["elbow_sd"])
        print(f"slope={slope:.4f} r2={r**2:.4f} p={p:.4f}")

    check(f"{name} -- fraction of variance in knee_flexion_mean explained by which start it is")
    grand_mean = df["r_knee_flexion_mean"].mean()
    if len(per_start) >= 2 and df["r_knee_flexion_mean"].var() > 0:
        between_var = ((per_start["knee_mean"] - grand_mean) ** 2 * per_start["n_pitches"]).sum() / len(df)
        total_var = df["r_knee_flexion_mean"].var()
        print(f"fraction: {between_var/total_var:.3f}")

print("\n" + "=" * 90)
print("SECTION D: BETWEEN-START CARRYOVER (per pitcher)")
print("=" * 90)

N = 10
for pid, d in all_players.items():
    df = d["clean"]
    name = df["player_name"].iloc[0] if len(df) else pid

    records = []
    for sn, g in df.groupby("start_number"):
        g = g.sort_values("pitch_in_start")
        n = len(g)
        if n < 6:
            continue
        k = min(N, max(n // 3, 1))
        first_n, last_n = g.iloc[:k], g.iloc[-k:]
        records.append({
            "start_number": sn,
            "first_knee_sd": first_n["r_knee_flexion_sd"].mean(),
            "last_knee_sd": last_n["r_knee_flexion_sd"].mean(),
        })
    start_summary = pd.DataFrame(records).sort_values("start_number").reset_index(drop=True)
    start_summary["next_first_knee_sd"] = start_summary["first_knee_sd"].shift(-1)
    pairs = start_summary.dropna(subset=["next_first_knee_sd"])

    check(f"{name} -- end-of-start instability predicting next start's opening instability")
    if len(pairs) >= 4:
        slope, intercept, r, p, se = stats.linregress(pairs["last_knee_sd"], pairs["next_first_knee_sd"])
        print(f"slope={slope:.4f} r={r:.4f} p={p:.4f} n_pairs={len(pairs)}")
    else:
        print("insufficient starts for carryover test")

print("\n" + "=" * 90)
print("SECTION E: INJURY-PROXIMITY CHECKS (the key question)")
print("=" * 90)

injury_summary_rows = []

for event in INJURY_EVENTS:
    pid = event["player_id"]
    if pid not in all_players:
        continue
    df = all_players[pid]["clean"]
    name = df["player_name"].iloc[0] if len(df) else pid
    event_date = pd.to_datetime(event["date"])

    prior = df[df["game_date"] < event_date].sort_values("game_date")
    if prior.empty:
        continue

    check(f"{name} -- last start before '{event['desc']}' ({event['date']})")
    last_start_num = prior["start_number"].max()
    last_start_data = prior[prior["start_number"] == last_start_num]
    days_before = (event_date - prior["game_date"].max()).days
    print(f"last start was {days_before} days before this event (start #{int(last_start_num)})")
    print(f"that start's mean knee_flexion_sd: {last_start_data['r_knee_flexion_sd'].mean():.2f}, "
          f"season baseline mean: {df['r_knee_flexion_sd'].mean():.2f}")

    check(f"{name} -- z-score of pre-event start's instability vs own season baseline")
    season_mean = df["r_knee_flexion_sd"].mean()
    season_sd = df["r_knee_flexion_sd"].std()
    if season_sd and season_sd > 0:
        z = (last_start_data["r_knee_flexion_sd"].mean() - season_mean) / season_sd
        print(f"z-score: {z:.3f} (>1.5 or so would suggest unusually elevated instability right before injury)")
    else:
        z = np.nan
        print("season_sd is zero or missing, cannot compute z-score")

    check(f"{name} -- trailing 3 starts before event vs rest of season (t-test)")
    trailing_starts = sorted(prior["start_number"].unique())[-3:]
    trailing_data = df[df["start_number"].isin(trailing_starts)]
    rest_data = df[~df["start_number"].isin(trailing_starts)]
    if len(trailing_data) > 3 and len(rest_data) > 3:
        t, p = stats.ttest_ind(trailing_data["r_knee_flexion_sd"].dropna(),
                                rest_data["r_knee_flexion_sd"].dropna(), equal_var=False)
        print(f"trailing-3-starts mean SD: {trailing_data['r_knee_flexion_sd'].mean():.2f} | "
              f"rest-of-season mean SD: {rest_data['r_knee_flexion_sd'].mean():.2f} | t={t:.3f} p={p:.4f}")
    else:
        p = np.nan
        print("insufficient data for comparison")

    check(f"{name} -- same trailing-3 test on elbow_flexion_sd")
    if len(trailing_data) > 3 and len(rest_data) > 3:
        t_e, p_e = stats.ttest_ind(trailing_data["r_elbow_flexion_sd"].dropna(),
                                    rest_data["r_elbow_flexion_sd"].dropna(), equal_var=False)
        print(f"trailing-3-starts mean elbow SD: {trailing_data['r_elbow_flexion_sd'].mean():.2f} | "
              f"rest-of-season mean elbow SD: {rest_data['r_elbow_flexion_sd'].mean():.2f} | t={t_e:.3f} p={p_e:.4f}")

    injury_summary_rows.append({
        "player": name, "event_date": event["date"], "description": event["desc"],
        "days_before_last_start": days_before, "last_start_number": int(last_start_num),
        "z_score_vs_season": z, "trailing3_vs_rest_p": p,
    })

check("FALSE-POSITIVE RATE CHECK -- how often does a random 3-start window look 'elevated' with no injury nearby?")
print("(sanity check: if trailing-3-before-injury windows look elevated at roughly the same rate as random")
print(" 3-start windows elsewhere in the same season, the injury-proximity signal is not distinguishable from noise)")
for pid, d in all_players.items():
    df = d["clean"]
    name = df["player_name"].iloc[0] if len(df) else pid
    starts = sorted(df["start_number"].dropna().unique())
    if len(starts) < 6:
        continue
    random_flags = 0
    n_trials = 0
    season_mean = df["r_knee_flexion_sd"].mean()
    season_sd = df["r_knee_flexion_sd"].std()
    if not season_sd or season_sd == 0:
        continue
    for i in range(len(starts) - 3):
        window_starts = starts[i:i+3]
        window_data = df[df["start_number"].isin(window_starts)]
        z = (window_data["r_knee_flexion_sd"].mean() - season_mean) / season_sd
        n_trials += 1
        if abs(z) > 1.5:
            random_flags += 1
    if n_trials > 0:
        print(f"{name}: {random_flags}/{n_trials} random 3-start windows exceed |z|>1.5 "
              f"({random_flags/n_trials*100:.1f}% base rate)")

print("\n" + "=" * 90)
print("SECTION F: CROSS-PITCHER COMPARISON (injured vs. Skenes control)")
print("=" * 90)

check("season-average instability (knee_flexion_sd): all 5 injured pitchers vs. Skenes control")
for pid, d in all_players.items():
    df = d["clean"]
    name = df["player_name"].iloc[0] if len(df) else pid
    print(f"{name:<20}: mean knee_flexion_sd = {df['r_knee_flexion_sd'].mean():.2f}  "
          f"(n={len(df)} clean pitches)")

check("season-average instability (elbow_flexion_sd): all 5 injured pitchers vs. Skenes control")
for pid, d in all_players.items():
    df = d["clean"]
    name = df["player_name"].iloc[0] if len(df) else pid
    print(f"{name:<20}: mean elbow_flexion_sd = {df['r_elbow_flexion_sd'].mean():.2f}  "
          f"(n={len(df)} clean pitches)")

check("within-start fatigue slope comparison across all 6 pitchers (does any show real within-game rise?)")
for pid, res in within_start_results.items():
    name = all_players[pid]["clean"]["player_name"].iloc[0] if len(all_players[pid]["clean"]) else pid
    flag = "SIGNIFICANT" if res["p"] < 0.05 else "not significant"
    print(f"{name:<20}: mean slope={res['mean_slope']:.5f}, p={res['p']:.4f} -- {flag}")

print("\n" + "=" * 90)
print(f"TOTAL CHECKS RUN: {check_num}")
print("=" * 90)

out_path = r"C:\Users\noahl\OneDrive\fatigue_cv\multi_pitcher_pull\injury_proximity_summary.csv"
pd.DataFrame(injury_summary_rows).to_csv(out_path, index=False)
print(f"\ninjury proximity summary saved to: {out_path}")
