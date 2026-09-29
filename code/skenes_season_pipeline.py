import requests
from bs4 import BeautifulSoup
import cv2
import numpy as np
import pandas as pd
import os
import time
from ultralytics import YOLO

PITCHER_ID = 694973
SEASON = 2026

OUTPUT_DIR = r"C:\Users\noahl\OneDrive\fatigue_cv\skenes_2026_season"
CLIPS_DIR = os.path.join(OUTPUT_DIR, "clips")
KEYPOINTS_DIR = os.path.join(OUTPUT_DIR, "keypoints")
PROGRESS_CSV = os.path.join(OUTPUT_DIR, "skenes_2026_progress.csv")

os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(KEYPOINTS_DIR, exist_ok=True)

DELETE_CLIPS_AFTER_PROCESSING = True

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

KP = {
    "nose": 0, "l_shoulder": 5, "r_shoulder": 6,
    "l_elbow": 7, "r_elbow": 8, "l_wrist": 9, "r_wrist": 10,
    "l_hip": 11, "r_hip": 12, "l_knee": 13, "r_knee": 14,
    "l_ankle": 15, "r_ankle": 16,
}

MOUND_X_FRAC = 0.35
MOUND_Y_FRAC = 0.65

WINDOW_START = 50
WINDOW_END = 240
RELEASE_FRAME = 178
RELEASE_BAND = list(range(174, 183))

POSE_CONF = 0.3
POSE_IMGSZ = 480
LOW_CONFIDENCE_THRESHOLD = 7

SAVE_EVERY = 10

pose_model = YOLO("yolov8s-pose.pt")


def get_game_pks_for_season(pitcher_id, season):
    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats?stats=gameLog&season={season}&group=pitching"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    game_pks = []
    for stat_block in data.get("stats", []):
        for split in stat_block.get("splits", []):
            game = split.get("game", {})
            if "gamePk" in game:
                game_pks.append((split.get("date"), game["gamePk"]))
    game_pks.sort(key=lambda x: x[0])
    return [g[1] for g in game_pks]


def get_pitcher_pitches(game_pk, pitcher_id):
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    all_plays = data["liveData"]["plays"]["allPlays"]
    rows = []
    for play in all_plays:
        matchup = play.get("matchup", {})
        pitcher = matchup.get("pitcher", {})
        if pitcher.get("id") != pitcher_id:
            continue
        about = play.get("about", {})
        for event in play.get("playEvents", []):
            if not event.get("isPitch", False):
                continue
            details = event.get("details", {})
            rows.append({
                "game_pk": game_pk,
                "inning": about.get("inning"),
                "pitch_number": event.get("pitchNumber"),
                "play_id": event.get("playId"),
                "pitch_type_desc": details.get("type", {}).get("description"),
            })
    return rows


def get_clip_url(play_id, max_retries=3, sleep_between=1.0):
    page_url = f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"
    for attempt in range(max_retries):
        try:
            resp = requests.get(page_url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            video_tag = soup.find("video")
            if video_tag:
                source_tag = video_tag.find("source")
                if source_tag and source_tag.get("src"):
                    return source_tag["src"]
                if video_tag.get("src"):
                    return video_tag["src"]
            return None
        except requests.RequestException:
            time.sleep(sleep_between)
    return None


def download_clip(url, out_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30, stream=True)
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except requests.RequestException:
            time.sleep(1)
    return False


def read_frame_range(video_path, start, end):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    for _ in range(end - start + 1):
        ret, frame = cap.read()
        if not ret:
            break
        out.append(frame)
    cap.release()
    return out


def pick_pitcher(people, frame_w, frame_h, prev_box=None):
    if not people:
        return None, None
    target_x = frame_w * MOUND_X_FRAC
    target_y = frame_h * MOUND_Y_FRAC

    def box_center(b):
        return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

    def iou(a, b):
        xA, yA = max(a[0], b[0]), max(a[1], b[1])
        xB, yB = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (a[2] - a[0]) * (a[3] - a[1])
        areaB = (b[2] - b[0]) * (b[3] - b[1])
        return inter / max(areaA + areaB - inter, 1e-8)

    if prev_box is not None:
        best, best_score = None, 0.0
        for box, kp in people:
            score = iou(prev_box, box)
            if score > best_score:
                best, best_score = (box, kp), score
        if best is not None and best_score > 0.15:
            return best

    best, best_dist = None, float("inf")
    for box, kp in people:
        cx, cy = box_center(box)
        dist = (cx - target_x) ** 2 + (cy - target_y) ** 2
        if dist < best_dist:
            best, best_dist = (box, kp), dist
    return best


def track_pitcher_batched(frames, conf=POSE_CONF, imgsz=POSE_IMGSZ):
    batch_results = pose_model(frames, verbose=False, conf=conf, imgsz=imgsz)
    tracked = []
    prev_box = None
    h, w = frames[0].shape[:2]
    for res in batch_results:
        people = []
        if res.keypoints is not None and res.boxes is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            kps = res.keypoints.data.cpu().numpy()
            for box, kp in zip(boxes, kps):
                people.append((box, kp))
        box, kp = pick_pitcher(people, w, h, prev_box=prev_box) if people else (None, None)
        tracked.append((box, kp))
        if box is not None:
            prev_box = box
    return tracked


def tracked_to_arrays(tracked):
    n = len(tracked)
    boxes_arr = np.full((n, 4), np.nan, dtype=np.float32)
    kps_arr = np.full((n, 17, 3), np.nan, dtype=np.float32)
    for i, (box, kp) in enumerate(tracked):
        if box is not None:
            boxes_arr[i] = box
        if kp is not None:
            kps_arr[i] = kp
    return boxes_arr, kps_arr


def load_progress():
    if os.path.exists(PROGRESS_CSV):
        return pd.read_csv(PROGRESS_CSV)
    return pd.DataFrame(columns=[
        "play_id", "game_pk", "inning", "pitch_number", "n_frames_pose",
        "n_detected", "detection_rate", "release_band_detected",
        "low_confidence_flag", "npz_path", "download_sec", "pose_sec", "total_sec",
    ])


def append_progress(rows):
    df_new = pd.DataFrame(rows)
    write_header = not os.path.exists(PROGRESS_CSV)
    df_new.to_csv(PROGRESS_CSV, mode="a", header=write_header, index=False)


print("gathering game log...")
game_pks = get_game_pks_for_season(PITCHER_ID, SEASON)
print(f"found {len(game_pks)} appearances in {SEASON}")

all_pitches = []
for gp in game_pks:
    all_pitches.extend(get_pitcher_pitches(gp, PITCHER_ID))

total_pitches = len(all_pitches)
print(f"found {total_pitches} total pitches across the season")

progress_df = load_progress()
already_done = set(progress_df["play_id"].astype(str)) if not progress_df.empty else set()
remaining_pitches = [p for p in all_pitches if str(p["play_id"]) not in already_done]

print(f"{len(already_done)} pitches already processed, {len(remaining_pitches)} remaining")

buffer_rows = []
run_start = time.perf_counter()
processed_this_run = 0

for idx, pitch in enumerate(remaining_pitches):
    play_id = pitch["play_id"]
    game_pk = pitch["game_pk"]
    clip_path = os.path.join(CLIPS_DIR, f"{play_id}.mp4")

    pitch_start = time.perf_counter()

    dl_start = time.perf_counter()
    clip_url = get_clip_url(play_id)
    got_clip = False
    if clip_url:
        got_clip = download_clip(clip_url, clip_path)
    dl_time = time.perf_counter() - dl_start

    n_frames_pose = 0
    n_detected = 0
    release_band_detected = 0
    pose_time = 0.0
    npz_path = ""

    if got_clip and os.path.exists(clip_path):
        pose_start = time.perf_counter()
        window_frames = read_frame_range(clip_path, WINDOW_START, WINDOW_END)
        n_frames_pose = len(window_frames)

        if window_frames:
            tracked = track_pitcher_batched(window_frames)
            n_detected = sum(1 for b, k in tracked if k is not None)

            for fi in RELEASE_BAND:
                local_idx = fi - WINDOW_START
                if 0 <= local_idx < len(tracked) and tracked[local_idx][1] is not None:
                    release_band_detected += 1

            boxes_arr, kps_arr = tracked_to_arrays(tracked)
            npz_path = os.path.join(KEYPOINTS_DIR, f"{play_id}.npz")
            np.savez_compressed(
                npz_path,
                boxes=boxes_arr,
                keypoints=kps_arr,
                window_start=WINDOW_START,
                window_end=WINDOW_END,
                release_frame=RELEASE_FRAME,
                game_pk=game_pk,
            )

        pose_time = time.perf_counter() - pose_start

    if DELETE_CLIPS_AFTER_PROCESSING and os.path.exists(clip_path):
        os.remove(clip_path)

    pitch_total = time.perf_counter() - pitch_start
    detection_rate = n_detected / n_frames_pose if n_frames_pose else 0.0
    low_confidence_flag = release_band_detected < LOW_CONFIDENCE_THRESHOLD

    if low_confidence_flag:
        print(f"  [!] low confidence: play_id {play_id} (game {game_pk}) — "
              f"only {release_band_detected}/9 release-band frames detected")

    buffer_rows.append({
        "play_id": play_id,
        "game_pk": game_pk,
        "inning": pitch["inning"],
        "pitch_number": pitch["pitch_number"],
        "n_frames_pose": n_frames_pose,
        "n_detected": n_detected,
        "detection_rate": round(detection_rate, 3),
        "release_band_detected": release_band_detected,
        "low_confidence_flag": low_confidence_flag,
        "npz_path": npz_path,
        "download_sec": round(dl_time, 2),
        "pose_sec": round(pose_time, 2),
        "total_sec": round(pitch_total, 2),
    })

    processed_this_run += 1

    if processed_this_run % SAVE_EVERY == 0 or idx == len(remaining_pitches) - 1:
        append_progress(buffer_rows)
        buffer_rows = []

        elapsed = time.perf_counter() - run_start
        avg_per_pitch = elapsed / processed_this_run
        pitches_left_after_this = len(remaining_pitches) - (idx + 1)
        eta_sec = pitches_left_after_this * avg_per_pitch

        print(f"[{processed_this_run}/{len(remaining_pitches)} this run | "
              f"{len(already_done) + processed_this_run}/{total_pitches} total] "
              f"avg {avg_per_pitch:.2f}s/pitch | "
              f"elapsed {elapsed/60:.1f} min | "
              f"ETA remaining: {eta_sec/60:.1f} min ({eta_sec/3600:.2f} hr)")

print("\ndone.")
print(f"progress log: {PROGRESS_CSV}")
print(f"keypoint arrays: {KEYPOINTS_DIR}")
