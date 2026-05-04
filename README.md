# IRL on COVID Masking Twitter Data

Per-user Deep MaxEnt IRL on cross-partisan engagement during 2020 COVID masking debate.
Each user is an MDP agent. We recover reward functions explaining their reply/quote behavior.

## MDP

- **Actions:** A0 = engage in-group, A1 = engage out-group
- **Features:** S1 (in-group stance agreement), S2 (out-group stance agreement)
- **States:** 4 (per-group median splits on S1 and S2)
- **Timestep:** each reply or quote tweet

## Data

- Tweets: `/data/dchu/covid_masking/masking_2020-*.parquet`
- Trajectory pkl (25,281 users, ≥10 actions): `/data/dchu/covid_masking_irl_action_v3/trajectories.pkl`

Trajectory format: list of dicts. Each dict has `user_id`, `political_gen` (0=lib, 1=con), `n_steps`, `trajectory`. Each step is `[date_str, S1, S2, action]`.

## Pipeline

### 1. Build trajectories from raw tweets
```bash
python build_trajectories_action_v3.py \
    --data_dir /data/dchu/covid_masking \
    --bot_dir /data/dchu/covid_mask_misc \
    --out_dir /data/dchu/covid_masking_irl_action_v3 \
    --n_workers 32 --min_actions 10
```

### 2. Run IRL
- Full year, all users: `run_deep_irl_v3.py`
- Pre/post shock split, separate models per user: `pre_vs_post_shock_irl.py`
- Rolling window around shock: `rolling_irl_shock_thr7.py`

## Rolling Window Around Shock (Trump COVID, Oct 2 2020)

Output dir: `/nas/home/dchu/IDP4CSS/Ashwin/Trajectories/pre_post_shock/rolling_window_around_shock/`

**Step 1.** Build cohort (users with ≥7 actions in every window):
```bash
python build_cohort_thr7_7win.py
# → trajectories_cohort_thr7_7win.pkl
```

**Step 2.** Run rolling IRL (7 windows, 30-day length, ends at T-6, T-3, T+0, T+3, T+6, T+9, T+12):
```bash
python rolling_irl_shock_thr7.py
# → rolling_irl_thr7_window_{w}_T{off}.parquet  (one per window)
```

State edges recomputed per window (per-group medians within that window).
Policy P(out-group) is comparable across windows; raw rewards are not.

## Output Schema (per-window parquet)

| Column | Meaning |
|---|---|
| `uid`, `pol` | user id, 0=lib / 1=con |
| `window_id`, `end_offset` | which window (offset from shock) |
| `n_actions`, `out_rate` | actions in window, fraction out-group |
| `acc`, `c0_acc`, `c1_acc` | policy accuracy on own actions |
| `reward_s{0..3}` | learned reward per state |
| `policy_out_s{0..3}` | P(A1=out-group) per state |

## State Encoding (4 states)

| State | S1 (in-group agreement) | S2 (out-group agreement) |
|---|---|---|
| S0 | low | low |
| S1 | low | high |
| S2 | high | low |
| S3 | high | high |

## Hyperparameters

3-layer sigmoid net (3-3-1), 3000 epochs, lr=0.01, gamma=0.95, l2=0.5, 32 parallel workers.

## Known Issues

- S1/S2 alone fail at predicting individual behavior (near-chance balanced accuracy on full-year)
- Adding S3 (personal cross-partisan momentum) is what makes IRL work, but S3 is excluded from the rolling-window analysis
- Liberal class imbalance (~82/18 in/out) limits liberal model performance
- Per-window IRL noise is high for users with few actions per window — cross-check policy shifts against observed behavior shifts before interpreting
