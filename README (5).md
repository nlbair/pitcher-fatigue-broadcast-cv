# Pitcher Fatigue Detection via Broadcast Video Pose Estimation

This repository contains the code, derived data, and figures supporting research on detecting mechanical pitching fatigue from broadcast video using computer vision, submitted to the MIT Sloan Sports Analytics Conference (SSAC27) Research Paper Competition.

## Overview

We use markerless pose estimation (YOLOv8-pose) on publicly available MLB broadcast video (Baseball Savant pitch clips) to extract pitching mechanics — primarily stride knee flexion — on a pitch-by-pitch basis, with no additional hardware or in-venue tracking required.

The research tests two things:

1. **Replication** — whether a within-outing mechanical fatigue signal (declining stride knee flexion as pitch count rises within a start), originally found in a 100-start random sample and a single-game case study (Jacob Misiorowski, 5/1/2026), holds up on an independent sample of pitchers.
2. **Injury proximity** — whether this fatigue signal shows any detectable elevation in the starts immediately preceding a real, documented pitcher injury, using five pitchers with media-reported 2023–2026 injuries and a healthy 2026 control (Paul Skenes).

## Data Sources

- **MLB Stats API** (`statsapi.mlb.com`) — game logs, pitch-by-pitch play data, play IDs
- **Baseball Savant** (`baseballsavant.mlb.com`) — pitch clip video resolution and download
- **Publicly reported injury/IL transaction data** — team and league injury announcements (dates and descriptions compiled from public reporting)

Raw broadcast video is not included in this repository due to copyright (owned by MLB/broadcast rightsholders). All video is publicly accessible via the cited sources above using the play IDs embedded in the derived data files, allowing full reproduction of the pipeline from scratch.

## Repository Structure

```
code/
  fullsaberseminarscript.ipynb          Original 100-start study + single-game
                                         (Misiorowski) case study pipeline
  skenes_season_pipeline.py             Production pipeline: full 2026 season,
                                         Paul Skenes (healthy control)
  multi_pitcher_pipeline.py             Production pipeline: full-season pulls for
                                         5 pitchers with documented injuries
                                         (run in parallel across 3 processes,
                                         split by pitcher workload)
  mega_sanity_check_injury_analysis.py  Data integrity checks, within-start /
                                         season-long / carryover fatigue tests,
                                         and the 12-event injury-proximity
                                         analysis (incl. false-positive check)
  quick_mean_flexion_check.py           Pooled within-start mean knee flexion
                                         slope test across all 6 pitchers,
                                         directly comparable to the original
                                         100-start study's headline metric

data/
  skenes_2026_progress.csv              Per-pitch processing log, Skenes 2026
                                         (detection quality, timing, frame counts)
  skenes_2026_metrics_summary.csv       Per-pitch release-band biomechanical
                                         metrics, Skenes 2026
  pooled_within_start_slopes.csv        Individual within-start knee flexion
                                         slopes, all 6 pitchers (141 starts)
  injury_proximity_summary.csv          Per-event summary: last start before each
                                         of 12 documented injuries, z-scores,
                                         trailing-window significance tests

figures/
  saberseminarvisual1.png               Case study: velocity vs. stride knee
                                         flexion across a single outing
  saberseminarvisual2.png               Within-outing mechanical breakdown:
                                         average angle and release-point
                                         instability vs. pitch number
```

## Methodology Summary

**Pose estimation pipeline:** YOLOv8-pose (Ultralytics), batched inference, run on a fixed frame window per pitch clip anchored to a validated ball-release frame. Pitcher identity across frames is resolved via IoU-based tracking against the prior frame's bounding box, with a mound-relative position heuristic as fallback. Stride knee flexion (hip-knee-ankle angle) and elbow flexion (shoulder-elbow-wrist angle) are computed at each frame in a release-band window and aggregated per pitch.

**Cleaning:** IQR-based outlier filtering (k=2.5) on knee flexion mean and standard deviation, plus a minimum release-band detection threshold, consistent across all pitchers studied.

**Statistical tests:**
- *Within-outing trend:* per-start linear regression of knee flexion (mean and SD) against pitch number within that start; slopes pooled and tested against zero.
- *Season-long trend:* start-level regression across a full season.
- *Carryover:* whether end-of-start instability predicts the next start's opening instability.
- *Injury proximity:* z-scored comparison of the 2–3 starts preceding each documented injury against that pitcher's own season baseline, with a random-window false-positive rate check to guard against large-sample-size significance artifacts.

## Known Limitations

The release-frame/window calibration was established via one-time visual confirmation on a single reference broadcast and reused across all broadcasts analyzed. Camera geometry, timing offset, and video quality vary across different games and stadiums, and this confound is present at meaningful scale (28–61% of variance in knee flexion mean explained simply by which start/broadcast it was). Within-outing comparisons are largely immune to this confound since camera setup is constant within a single broadcast; between-start and season-long comparisons are more vulnerable to it.

## Citation / Contact

Noah Bair, Syracuse University, Falk College of Sport
