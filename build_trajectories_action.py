#!/usr/bin/env python3
"""
Build PER-ACTION IRL trajectories for COVID masking cross-partisan engagement.

Each reply or quote tweet is one timestep. No grouping by day or week.

MDP:
  Features: S1 (in-group stance agreement), S2 (out-group stance agreement),
            S3 (my cross-partisan engagement ratio over recent actions)
  Actions:  A0 = engage in-group, A1 = engage out-group
  
Key differences from weekly version:
  - Each engagement (reply/quote) is its own timestep
  - Action is directly observed per engagement (no majority voting)
  - S1/S2 computed from daily stance distributions, lagged 1 day
  - S3 = rolling cross-partisan ratio over last N actions (default 7)
  - Much longer trajectories per user (mean ~20 vs ~14 weekly)

Usage:
  python build_trajectories_action.py \
    --data_dir /data/dchu/covid_masking \
    --bot_dir /data/dchu/covid_mask_misc \
    --out_dir /data/dchu/covid_masking_irl_action \
    --n_workers 32 --min_actions 10 --max_gap_days 60 --s3_window 7
"""

import argparse
import glob
import multiprocessing as mp
import os
import pickle
import time
import gc

import numpy as np
import pandas as pd

# ====================================================================
# GLOBALS -- set in parent, inherited COW by forked workers
# ====================================================================
_G_USER_ATTRS = {}       # uid(str) -> (modal_stance:int, political_gen:int)
_G_STANCE_DIST = {}      # (date_str, pol_group) -> frac_pro_mask
_G_USER_ACTIONS = {}     # uid(str) -> list of (date_str, action:int) sorted by date
_G_USER_BASELINE = {}    # uid(str) -> overall cross-partisan ratio [0,1]
_G_SORTED_DATES = {}     # date_str -> index
_G_SORTED_DATES_LIST = []  # index -> date_str
_G_MIN_ACTIONS = 10
_G_MAX_GAP_DAYS = 60
_G_S3_WINDOW = 7


# ====================================================================
# CLI
# ====================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="/data/dchu/covid_masking")
    p.add_argument("--bot_dir", type=str, default="/data/dchu/covid_mask_misc")
    p.add_argument("--out_dir", type=str, default="/data/dchu/covid_masking_irl_action")
    p.add_argument("--n_workers", type=int, default=32)
    p.add_argument("--min_actions", type=int, default=10,
                   help="Min engagements to keep user")
    p.add_argument("--max_gap_days", type=int, default=60,
                   help="Max gap (days) between consecutive actions. Users with larger gaps dropped.")
    p.add_argument("--s3_window", type=int, default=7,
                   help="Rolling window (number of past actions) for S3")
    p.add_argument("--supp_labels", type=str,
                   default="/data/dchu/covid_mask_misc/supplementary_pol_labels.parquet")
    return p.parse_args()


# ====================================================================
# 1. LOAD & FILTER
# ====================================================================
def load_data(data_dir: str, bot_dir: str) -> pd.DataFrame:
    t0 = time.time()

    clean_ids = None
    for fname in ["clean_userids.parquet", "clean_user_ids.pkl",
                   "clean_user_ids.npy", "clean_user_ids.csv"]:
        fpath = os.path.join(bot_dir, fname)
        if not os.path.exists(fpath):
            continue
        if fname.endswith(".parquet"):
            cdf = pd.read_parquet(fpath)
            clean_ids = set(cdf.iloc[:, 0].astype(str))
            del cdf
        elif fname.endswith(".pkl"):
            with open(fpath, "rb") as f:
                clean_ids = set(str(x) for x in pickle.load(f))
        elif fname.endswith(".npy"):
            clean_ids = set(str(x) for x in np.load(fpath, allow_pickle=True))
        else:
            clean_ids = set(pd.read_csv(fpath, header=None)[0].astype(str))
        print(f"  Loaded {len(clean_ids):,} clean user IDs from {fname}")
        break

    if clean_ids is None:
        print(f"  WARNING: no bot filter found. Proceeding without.")

    files = sorted(glob.glob(os.path.join(data_dir, "masking_2020-*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files in {data_dir}")
    print(f"  Found {len(files)} parquet files")

    cols = [
        "userid", "date", "tweet_type", "stance_bin",
        "political_gen", "reply_userid", "qtd_userid",
    ]

    chunks = []
    for fp in files:
        df = pd.read_parquet(fp, columns=cols)
        chunks.append(df)
        print(f"    {os.path.basename(fp)}: {len(df):,}")
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    print(f"  Total: {len(df):,}")

    df["userid"] = df["userid"].astype(str)
    df["reply_userid"] = df["reply_userid"].astype(str)
    df.loc[df["reply_userid"].isin(["nan", "None", ""]), "reply_userid"] = np.nan

    # qtd_userid is float64, convert to string safely
    df["qtd_userid"] = df["qtd_userid"].astype("object")
    mask = df["qtd_userid"].notna()
    if mask.any():
        df.loc[mask, "qtd_userid"] = df.loc[mask, "qtd_userid"].astype(float).astype(np.int64).astype(str)
    df["qtd_userid"] = df["qtd_userid"].astype(str)
    df.loc[df["qtd_userid"].isin(["nan", "None", "", "NaN"]), "qtd_userid"] = np.nan

    if clean_ids is not None:
        n0 = len(df)
        df = df[df["userid"].isin(clean_ids)].copy()
        print(f"  Bot filter: {len(df):,} kept (removed {n0-len(df):,})")

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)

    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    print(f"  {df['date_str'].nunique()} unique dates, {time.time()-t0:.1f}s")
    return df


# ====================================================================
# 2. USER ATTRIBUTES
# ====================================================================
def compute_user_attrs(df: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()

    stance_counts = df.groupby(["userid", "stance_bin"]).size().reset_index(name="cnt")
    idx = stance_counts.groupby("userid")["cnt"].idxmax()
    stance_mode = stance_counts.loc[idx].set_index("userid")["stance_bin"].rename("modal_stance")

    pol_counts = df.groupby(["userid", "political_gen"]).size().reset_index(name="cnt")
    pol_idx = pol_counts.groupby("userid")["cnt"].idxmax()
    pol_mode = pol_counts.loc[pol_idx].set_index("userid")["political_gen"].rename("political_gen")

    attrs = pd.concat([stance_mode, pol_mode], axis=1)
    n0 = len(attrs)
    attrs = attrs.dropna(subset=["political_gen", "modal_stance"])
    attrs["political_gen"] = attrs["political_gen"].astype(int)
    attrs["modal_stance"] = attrs["modal_stance"].astype(int)

    print(f"  {n0:,} total users -> {len(attrs):,} with valid labels -- {time.time()-t0:.1f}s")
    return attrs


# ====================================================================
# 3. DAILY STANCE DISTRIBUTIONS (S1, S2)
# ====================================================================
def build_daily_stance_dist(df: pd.DataFrame, all_attrs: pd.DataFrame) -> dict:
    t0 = time.time()
    uw = df[["userid", "date_str"]].drop_duplicates()
    uw = uw.merge(
        all_attrs[["modal_stance", "political_gen"]],
        left_on="userid", right_index=True, how="inner",
    )
    result = uw.groupby(["date_str", "political_gen"])["modal_stance"].mean().to_dict()
    print(f"  {len(result)} entries -- {time.time()-t0:.1f}s")
    return result


# ====================================================================
# 4. BUILD PER-USER ACTION SEQUENCES
# ====================================================================
def build_user_action_sequences(df: pd.DataFrame, pol_map: pd.Series) -> dict:
    """
    For each user, build a chronologically sorted list of engagements.
    Each entry: (date_str, action) where action=0 (in-group) or 1 (out-group).
    """
    t0 = time.time()

    # Replies
    m_reply = (df["tweet_type"] == "reply") & df["reply_userid"].notna()
    reply_acts = df.loc[m_reply, ["userid", "reply_userid", "date_str", "date"]].copy()
    reply_acts.columns = ["author", "target", "date_str", "date"]
    print(f"  {len(reply_acts):,} reply actions")

    # Quotes
    m_quote = (df["tweet_type"] == "quoted_tweet") & df["qtd_userid"].notna()
    quote_acts = df.loc[m_quote, ["userid", "qtd_userid", "date_str", "date"]].copy()
    quote_acts.columns = ["author", "target", "date_str", "date"]
    print(f"  {len(quote_acts):,} quote actions")

    acts = pd.concat([reply_acts, quote_acts], ignore_index=True)
    del reply_acts, quote_acts
    print(f"  {len(acts):,} total engagements")

    acts["a_pol"] = acts["author"].map(pol_map)
    acts["t_pol"] = acts["target"].map(pol_map)
    acts.dropna(subset=["a_pol", "t_pol"], inplace=True)
    print(f"  {len(acts):,} with valid labels on both sides")

    acts["action"] = (acts["a_pol"] != acts["t_pol"]).astype(int)

    # Sort by date within each user
    acts = acts.sort_values(["author", "date"])

    # Build per-user action sequences
    result = {}
    for uid, grp in acts.groupby("author"):
        seq = list(zip(grp["date_str"].values, grp["action"].values))
        result[uid] = seq

    print(f"  {len(result):,} users with action sequences -- {time.time()-t0:.1f}s")
    return result


# ====================================================================
# 5. COMPUTE PER-USER BASELINE CROSS-PARTISAN RATIO
# ====================================================================
def compute_user_baselines(user_actions: dict) -> dict:
    t0 = time.time()
    result = {}
    for uid, seq in user_actions.items():
        n_out = sum(a for _, a in seq)
        n_total = len(seq)
        result[uid] = n_out / n_total if n_total > 0 else 0.5
    print(f"  {len(result):,} user baselines -- {time.time()-t0:.1f}s")
    return result


# ====================================================================
# 6. TRAJECTORY BUILDER (reads globals via COW)
# ====================================================================
def _build_traj_chunk(user_ids):
    """
    Worker: for each user, each engagement is one timestep.
    
    For action i:
      - Action: directly observed (in-group=0, out-group=1)
      - S1: in-group stance agreement from the day before this action
      - S2: out-group stance agreement from the day before this action
      - S3: cross-partisan ratio over last N actions [i-N..i-1]
    """
    out = []

    for uid in user_ids:
        entry = _G_USER_ATTRS.get(uid)
        if entry is None:
            continue
        my_stance, my_pol = entry
        other_pol = 1 - my_pol
        baseline = _G_USER_BASELINE.get(uid, 0.5)

        seq = _G_USER_ACTIONS.get(uid)
        if seq is None or len(seq) < _G_MIN_ACTIONS:
            continue

        # Check max gap
        skip = False
        for i in range(1, len(seq)):
            d1 = pd.Timestamp(seq[i - 1][0])
            d2 = pd.Timestamp(seq[i][0])
            gap = (d2 - d1).days
            if gap > _G_MAX_GAP_DAYS:
                skip = True
                break
        if skip:
            continue

        traj = []
        for i, (date_str, action) in enumerate(seq):
            # Need at least 1 prior day for S1/S2 lag
            dt_idx = _G_SORTED_DATES.get(date_str)
            if dt_idx is None or dt_idx < 1:
                continue

            # S1, S2: from the previous calendar day
            prev_date = _G_SORTED_DATES_LIST[dt_idx - 1]
            ig_frac = _G_STANCE_DIST.get((prev_date, my_pol), np.nan)
            s1 = ig_frac if my_stance == 1 else (1.0 - ig_frac)

            og_frac = _G_STANCE_DIST.get((prev_date, other_pol), np.nan)
            s2 = og_frac if my_stance == 1 else (1.0 - og_frac)

            if np.isnan(s1) or np.isnan(s2):
                continue

            # S3: cross-partisan ratio over last N actions
            lookback_start = max(0, i - _G_S3_WINDOW)
            if i > 0:
                recent = seq[lookback_start:i]
                n_out = sum(a for _, a in recent)
                n_total = len(recent)
                s3 = n_out / n_total
            else:
                s3 = baseline

            traj.append((date_str, s1, s2, s3, action))

        if len(traj) >= _G_MIN_ACTIONS:
            out.append({
                "user_id": uid,
                "political_gen": int(my_pol),
                "modal_stance": int(my_stance),
                "n_steps": len(traj),
                "trajectory": traj,
            })
    return out


def parallel_build(user_ids, n_workers):
    t0 = time.time()
    valid = [u for u in user_ids if u in _G_USER_ATTRS and u in _G_USER_ACTIONS]
    print(f"  {len(valid):,} valid users -> {n_workers} workers")

    chunks = np.array_split(valid, n_workers)
    chunks = [c.tolist() for c in chunks if len(c) > 0]

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_workers) as pool:
        batches = pool.map(_build_traj_chunk, chunks)

    trajs = [t for batch in batches for t in batch]
    print(f"  {len(trajs):,} trajectories -- {time.time()-t0:.1f}s")
    return trajs


# ====================================================================
# 7. SUMMARY & SAVE
# ====================================================================
def summarize_and_save(trajs, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "trajectories.pkl"), "wb") as f:
        pickle.dump(trajs, f)

    rows = []
    for t in trajs:
        uid, pol, st = t["user_id"], t["political_gen"], t["modal_stance"]
        for (dt, s1, s2, s3, a) in t["trajectory"]:
            rows.append((uid, pol, st, dt, s1, s2, s3, a))
    flat = pd.DataFrame(rows, columns=[
        "user_id", "political_gen", "modal_stance", "date",
        "S1_ingroup_stance_agree", "S2_outgroup_stance_agree",
        "S3_cross_partisan_ratio", "action",
    ])
    flat.to_parquet(os.path.join(out_dir, "trajectories_flat.parquet"), index=False)

    n = len(trajs)
    if n == 0:
        print("  WARNING: 0 trajectories produced.")
        return

    n_lib = sum(1 for t in trajs if t["political_gen"] == 0)
    steps = np.array([t["n_steps"] for t in trajs])
    out_rates = np.array([np.mean([s[4] for s in t["trajectory"]]) for t in trajs])
    s_all = np.array([[s[1], s[2], s[3]] for t in trajs for s in t["trajectory"]])

    r = []
    r.append("=" * 60)
    r.append("TRAJECTORY SUMMARY (PER-ACTION TIMESTEP)")
    r.append("=" * 60)
    r.append(f"Users:          {n:,}  (lib={n_lib:,}, con={n-n_lib:,})")
    r.append(f"Steps/user:     mean={steps.mean():.1f}  med={np.median(steps):.0f}  "
             f"range=[{steps.min()}, {steps.max()}]")
    r.append(f"User-steps:     {len(flat):,}")
    r.append(f"Out-grp rate:   mean={out_rates.mean():.3f}  med={np.median(out_rates):.3f}")
    r.append("")

    for i, nm in enumerate(["S1 (in-grp stance agree)",
                             "S2 (out-grp stance agree)",
                             "S3 (cross-partisan ratio)"]):
        c = s_all[:, i]
        c = c[~np.isnan(c)]
        if len(c) > 0:
            r.append(f"  {nm}: mean={c.mean():.4f}  std={c.std():.4f}  "
                     f"med={np.median(c):.4f}  [{c.min():.4f}, {c.max():.4f}]")

    n_baseline = sum(1 for t in trajs for j, s in enumerate(t["trajectory"])
                     if j == 0 and s[3] == _G_USER_BASELINE.get(t["user_id"], -1))
    r.append(f"  S3 baseline fallback (first action only): {n_baseline:,} / {n:,}")

    r.append("")
    r.append("--- By political group ---")
    for pol, lab in [(0, "Liberal"), (1, "Conservative")]:
        sub = flat[flat["political_gen"] == pol]
        if len(sub) > 0:
            r.append(f"  {lab}: {sub['user_id'].nunique():,} users, "
                     f"{len(sub):,} steps, out-rate={sub['action'].mean():.3f}, "
                     f"S3 mean={sub['S3_cross_partisan_ratio'].mean():.3f}")

    # Action distribution
    r.append("")
    r.append("--- Action distribution ---")
    for pol, lab in [(0, "Liberal"), (1, "Conservative")]:
        sub = flat[flat["political_gen"] == pol]
        if len(sub) > 0:
            a0 = (sub["action"] == 0).mean()
            a1 = (sub["action"] == 1).mean()
            r.append(f"  {lab}: A0(in-grp)={a0:.3f}  A1(out-grp)={a1:.3f}")

    # Pre/post shock
    r.append("")
    r.append("--- Pre/post Trump mask (2020-07-11) ---")
    shock_date = "2020-07-11"
    for pol, lab in [(None, "All"), (0, "Liberal"), (1, "Conservative")]:
        if pol is not None:
            sub_trajs = [t for t in trajs if t["political_gen"] == pol]
        else:
            sub_trajs = trajs

        n_both = 0
        pre_counts = []
        post_counts = []
        for t in sub_trajs:
            pre = [s for s in t["trajectory"] if s[0] < shock_date]
            post = [s for s in t["trajectory"] if s[0] >= shock_date]
            if pre and post:
                n_both += 1
                pre_counts.append(len(pre))
                post_counts.append(len(post))

        if n_both > 0:
            r.append(f"  {lab}: {n_both:,} users with both pre & post")
            r.append(f"    Pre steps:  mean={np.mean(pre_counts):.1f}  med={np.median(pre_counts):.0f}")
            r.append(f"    Post steps: mean={np.mean(post_counts):.1f}  med={np.median(post_counts):.0f}")
            for min_s in [3, 5, 7, 10]:
                both_n = sum(1 for p, q in zip(pre_counts, post_counts) if p >= min_s and q >= min_s)
                r.append(f"    Min {min_s} steps both sides: {both_n:,}")

    r.append("=" * 60)
    report = "\n".join(r)
    print(report)

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(report)

    print(f"\nOutputs -> {out_dir}/")
    print(f"  trajectories.pkl")
    print(f"  trajectories_flat.parquet")
    print(f"  summary.txt")


# ====================================================================
# MAIN
# ====================================================================
def main():
    global _G_USER_ATTRS, _G_STANCE_DIST, _G_USER_ACTIONS, _G_USER_BASELINE
    global _G_SORTED_DATES, _G_SORTED_DATES_LIST, _G_MIN_ACTIONS, _G_MAX_GAP_DAYS, _G_S3_WINDOW

    args = parse_args()
    print("=" * 60)
    print("IRL TRAJECTORY BUILDER -- COVID Masking (PER-ACTION)")
    print(f"  workers={args.n_workers}  min_actions={args.min_actions}  "
          f"max_gap_days={args.max_gap_days}  s3_window={args.s3_window}")
    print("=" * 60)

    # 1. Load
    print("\n[1/6] Loading...")
    df = load_data(args.data_dir, args.bot_dir)

    # 2. User attrs
    print("\n[2/6] User attributes...")
    all_attrs = compute_user_attrs(df)

    pol_map = all_attrs["political_gen"].copy()
    if os.path.exists(args.supp_labels):
        supp = pd.read_parquet(args.supp_labels)
        supp["userid"] = supp["userid"].astype(str)
        supp = supp.set_index("userid")["political_gen"]
        new_labels = supp[~supp.index.isin(pol_map.index)]
        pol_map = pd.concat([pol_map, new_labels])
        print(f"  Supplementary labels: {len(new_labels):,} new users added")
        print(f"  Total pol_map: {len(pol_map):,}")
        del supp, new_labels
    else:
        print(f"  No supplementary labels at {args.supp_labels}")

    # 3. Daily stance distributions
    print("\n[3/6] Daily stance dists (S1, S2)...")
    _G_STANCE_DIST = build_daily_stance_dist(df, all_attrs)

    # 4. Per-user action sequences
    print("\n[4/6] Building per-user action sequences...")
    _G_USER_ACTIONS = build_user_action_sequences(df, pol_map)

    # 5. User baselines
    print("\n[5/6] User baseline cross-partisan ratios...")
    _G_USER_BASELINE = compute_user_baselines(_G_USER_ACTIONS)

    # 6. Build trajectories
    print("\n[6/6] Building trajectories...")
    _G_USER_ATTRS = {
        uid: (int(row["modal_stance"]), int(row["political_gen"]))
        for uid, row in all_attrs.iterrows()
    }

    # Build sorted date index for S1/S2 lag lookup
    all_dates = sorted(df["date_str"].unique())
    _G_SORTED_DATES = {dt: i for i, dt in enumerate(all_dates)}
    _G_SORTED_DATES_LIST = all_dates

    _G_MIN_ACTIONS = args.min_actions
    _G_MAX_GAP_DAYS = args.max_gap_days
    _G_S3_WINDOW = args.s3_window

    print(f"  {len(all_dates)} unique dates: {all_dates[0]} .. {all_dates[-1]}")
    print(f"  S3 window: last {_G_S3_WINDOW} actions")
    print(f"  Max gap: {_G_MAX_GAP_DAYS} days")

    del df
    gc.collect()

    trajs = parallel_build(list(_G_USER_ACTIONS.keys()), args.n_workers)

    print("\n[SAVE]")
    summarize_and_save(trajs, args.out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()