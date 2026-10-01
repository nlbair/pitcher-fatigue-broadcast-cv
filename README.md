# Pitcher Mechanical Fatigue Detection from Broadcast Video — Case Study

This repository contains the code supporting the Jacob Misiorowski single-game case study portion of ongoing research into detecting in-game pitcher fatigue from broadcast video using markerless pose estimation.

## Scope

This notebook (`fullsaberseminarscript.ipynb`) covers **one real start in full detail**: Jacob Misiorowski (MLBAM ID 694819), May 1, 2026, vs. the Washington Nationals (game_pk 822744). It does not include the separate 100-start population-level study (62 unique pitchers, mixed-effects model, camera-shake negative control) referenced in the accompanying abstract; that analysis was conducted separately.

## Data Sources

- **MLB Stats API** (`statsapi.mlb.com`) — live game feed, pitch-by-pitch play data, play IDs
- **Baseball Savant** (`baseballsavant.mlb.com`) — pitch clip video resolution and download via the `sporty-videos` page wrapper

Raw broadcast video is not included due to copyright (owned by MLB/broadcast rightsholders). It is fully re-derivable from the play IDs in this notebook via the public sources above.

## What's in the Notebook

**1. Data acquisition**
Pulls every pitch Misiorowski threw in the target game via the MLB Stats API live feed, resolves each pitch's Baseball Savant clip URL, and downloads the clip.

**2. Pose estimation pipeline**
- YOLOv8-pose (Ultralytics) tracking, with a pitcher/hitter/catcher (PHC) gated coverage check to confirm the camera angle shows all three roles before trusting a clip
- Release-frame detection via a wrist-height heuristic, confirmed by one-time manual visual validation against a rendered frame strip
- A fixed, validated release window (frames 175–183) and tracking window (165–195) used once the release point was confirmed for this specific broadcast

**3. Biomechanical metrics (computed per frame)**
- Knee flexion (both legs), elbow flexion (both arms)
- Shoulder abduction proxy (upper arm angle from vertical)
- Trunk forward tilt and lateral lean
- Hip-shoulder separation (X-factor proxy), torso/pelvis rotation proxies
- Wrist height and position at release
- Semantic stride-leg / throwing-arm aliases applied based on pitcher handedness

**4. Statistical analysis**
- IQR-based outlier filtering (k=2.5) per pitch type
- Inning-level aggregation and linear regression (slope, r², p) of each metric against pitch count/inning
- Bootstrap confidence intervals on regression slopes
- ANCOVA-style comparison isolating a post-settling period from the full outing
- Linear vs. quadratic model comparison
- **Robustness checks:** a leverage test (does dropping the noisiest inning change the headline result?) and a multiple-comparisons correction (Benjamini-Hochberg FDR and Bonferroni) applied across all 45 regressions run (24 mean-metrics + 21 std-metrics)

**5. Visualization development**
Multiple iterative presentation formats built for the Saberseminar talk, including skeleton overlays on broadcast video, dual-panel velocity-vs-mechanics comparisons, HUD-style telemetry displays, motion-trail ("ghost trail") renders, a transition effect between visual styles ("mitosis"), slow-motion skeleton-only renders, and side-by-side pitch comparisons.

## Citation / Contact

Noah Bair, Syracuse University, Falk College of Sport
