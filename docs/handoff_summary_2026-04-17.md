# Player Model Handoff Summary (Updated for Persistent RunPod Data)

## Situation
- Goal: improve player detection using Roboflow sources + Zenodo comparator.
- RunPod disk persists across pod switches for your setup.
- So you should **reuse existing processed data** and only add missing datasets.

## Already Completed (Do NOT redo)

### Raw Roboflow already present
- `data/raw/roboflow_players/rf01`
- `data/raw/roboflow_players/rf02`
- `data/raw/roboflow_players/rf03`
- `data/raw/roboflow_players/rf04`

### Prepared player-only datasets already present
- `data/prepared/roboflow_players/rf01_player_only/rf01_player_only.yaml`
- `data/prepared/roboflow_players/rf02_player_only_det/rf02_player_only_det.yaml`
- `data/prepared/roboflow_players/rf03_player_only/rf03_player_only.yaml`
- `data/prepared/roboflow_players/rf04_player_only/rf04_player_only.yaml`

### Merged datasets already present
- RF-only: `data/merged_players_rf_v1/merged_players_rf_v1.yaml`
- Zenodo+RF: `data/merged_players_zenodo_rf_v1_fast/merged_players_zenodo_rf_v1_fast.yaml`

## What Is Still Missing
- Roboflow links 5, 6, 7 were not downloaded previously due expired signed URLs.
- Only these need fresh links + processing.

## Recommended next state after adding 5-7
1. Download/extract only `rf05`, `rf06`, `rf07`.
2. Create `source_with_path.yaml` for `rf05-07`.
3. Filter each to player-only detect format.
4. Validate new prepared sets.
5. Rebuild merged RF dataset (`merged_players_rf_v2`) with rf01..rf07.
6. Rebuild merged Zenodo+RF dataset (`merged_players_zenodo_rf_v2_fast`).
7. Train comparison runs (Zenodo vs RF-only vs Zenodo+RF).

## Notes
- `rf02` required `--truncate-values 5` (pose-style labels converted to detect format).
- `rf01` originally had a non-standard class name and was normalized to `player`.
- Training was stopped on request and should be restarted explicitly when ready.

