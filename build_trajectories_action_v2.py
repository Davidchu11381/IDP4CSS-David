#!/usr/bin/env python3
"""
Build PER-ACTION IRL trajectories for COVID masking cross-partisan engagement.
V2: Dynamic tweet-level S1/S2.

Key change from v1:
  S1/S2 are now computed from ALL tweets (reply, quote, RT, original) from
  the previous day. For each action (reply/quote), we compare THAT action's
  stance_bin against the stance distribution of yesterday's tweets.

  S1 = fraction of in-group tweets yesterday that share this action's stance
  S2 = fraction of out-group tweets yesterday that share this action's stance

  This is dynamic (changes daily based on actual tweet content) and action-specific
  (different replies on the same day can have different S1/S2 if they have different stances).

MDP:
  Features: S1 (in-group stance agreement), S2 (out-group stance agreement),
            S3 (cross-partisan engagement ratio over recent actions)
  Actions:  A0 = engage in-group, A1 = engage out-group
  Timestep: each reply or quote tweet

Usage:
  python build_trajectories_action_v2.py \
    --data_dir /data/dchu/covid_masking \
    --bot_dir /data/dchu/covid_mask_misc \
    --out_dir /data/dchu/covid_masking_irl_action_v2 \
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
# GLOBALS
# ====================================================================
_G_USER_POL = {}          # uid(str) -> political_gen (int)
_G_DAILY_STANCE = {}      # (date_str, pol_group, stance_bin) -> count
_G_DAILY_TOTAL = {}       # (date_str, pol_group) -> total count
_G_USER_ACTIONS = {}      # uid(str) -> list of (date_str, stance_bin, action) sorted by date
_G_USER_BASELINE = {}     # uid(str) -> overall cross-partisan ratio
_G_SORTED_DATES = {}      # date_str -> index
_G_SORTED_DATES_LIST = [] # index -> date_str
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
    p.add_argument("--out_dir", type=str, default="/data/dchu/covid_masking_irl_action_v2")
    p.add_argument("--n_workers", type=int, default=32)
    p.add_argument("--min_actions", type=int, default=10)
    p.add_argument("--max_gap_days", type=int, default=60)
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
        print(f"  Loaded {len(clean_ids):,} clean user IDs from {fname}")
        break

    if clean_ids is None:
        print(f"  WARNING: no bot filter found.")

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

    # qtd_userid is float64
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
# 2. BUILD DAILY STANCE DISTRIBUTIONS (from ALL tweets)
# ====================================================================
def build_daily_stance_counts(df: pd.DataFrame) -> tuple:
    """
    For each (date, political_group, stance_bin), count number of tweets.
    Uses ALL tweet types: original, reply, quote, RT.
    """
    t0 = time.time()

    # Need political_gen and stance_bin
    sub = df.dropna(subset=["political_gen", "stance_bin"]).copy()
    sub["political_gen"] = sub["political_gen"].astype(int)
    sub["stance_bin"] = sub["stance_bin"].astype(int)

    # Count tweets per (date, pol, stance)
    counts = sub.groupby(["date_str", "political_gen", "stance_bin"]).size()
    stance_counts = counts.to_dict()

    # Total per (date, pol)
    totals = sub.groupby(["date_str", "political_gen"]).size().to_dict()

    print(f"  {len(stance_counts):,} (date, pol, stance) entries")
    print(f"  {len(totals):,} (date, pol) entries")
    print(f"  Using ALL tweet types for stance distribution -- {time.time()-t0:.1f}s")

    return stance_counts, totals


# ====================================================================
# 3. BUILD PER-USER ACTION SEQUENCES
# ====================================================================
def build_user_action_sequences(df: pd.DataFrame, pol_map: dict) -> dict:
    """
    For each user, build sorted list of (date_str, stance_bin, action).
    action = 0 (in-group target) or 1 (out-group target).
    stance_bin = the stance of THIS specific reply/quote.
    """
    t0 = time.time()

    # Replies
    m_reply = (df["tweet_type"] == "reply") & df["reply_userid"].notna()
    reply_acts = df.loc[m_reply, ["userid", "reply_userid", "date_str", "date", "stance_bin"]].copy()
    reply_acts.columns = ["author", "target", "date_str", "date", "stance_bin"]
    print(f"  {len(reply_acts):,} reply actions")

    # Quotes
    m_quote = (df["tweet_type"] == "quoted_tweet") & df["qtd_userid"].notna()
    quote_acts = df.loc[m_quote, ["userid", "qtd_userid", "date_str", "date", "stance_bin"]].copy()
    quote_acts.columns = ["author", "target", "date_str", "date", "stance_bin"]
    print(f"  {len(quote_acts):,} quote actions")

    acts = pd.concat([reply_acts, quote_acts], ignore_index=True)
    del reply_acts, quote_acts
    print(f"  {len(acts):,} total engagements")

    acts["a_pol"] = acts["author"].map(pol_map)
    acts["t_pol"] = acts["target"].map(pol_map)
    acts.dropna(subset=["a_pol", "t_pol", "stance_bin"], inplace=True)
    acts["stance_bin"] = acts["stance_bin"].astype(int)
    print(f"  {len(acts):,} with valid labels on both sides + stance")

    acts["action"] = (acts["a_pol"] != acts["t_pol"]).astype(int)
    acts = acts.sort_values(["author", "date"])

    result = {}
    for uid, grp in acts.groupby("author"):
        seq = list(zip(grp["date_str"].values, grp["stance_bin"].values, grp["action"].values))
        result[uid] = seq

    print(f"  {len(result):,} users with action sequences -- {time.time()-t0:.1f}s")
    return result


# ====================================================================
# 4. USER BASELINES
# ====================================================================
def compute_user_baselines(user_actions: dict) -> dict:
    t0 = time.time()
    result = {}
    for uid, seq in user_actions.items():
        n_out = sum(a for _, _, a in seq)
        n_total = len(seq)
        result[uid] = n_out / n_total if n_total > 0 else 0.5
    print(f"  {len(result):,} user baselines -- {time.time()-t0:.1f}s")
    return result


# ====================================================================
# 5. TRAJECTORY BUILDER
# ====================================================================
def _build_traj_chunk(user_ids):
    out = []

    for uid in user_ids:
        pol = _G_USER_POL.get(uid)
        if pol is None:
            continue
        other_pol = 1 - pol
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
        for i, (date_str, stance_bin, action) in enumerate(seq):
            dt_idx = _G_SORTED_DATES.get(date_str)
            if dt_idx is None or dt_idx < 1:
                continue

            # S1/S2 from PREVIOUS day's tweet-level stance distribution
            prev_date = _G_SORTED_DATES_LIST[dt_idx - 1]

            # S1: fraction of in-group tweets yesterday with same stance
            ig_same = _G_DAILY_STANCE.get((prev_date, pol, stance_bin), 0)
            ig_total = _G_DAILY_TOTAL.get((prev_date, pol), 0)
            if ig_total > 0:
                s1 = ig_same / ig_total
            else:
                s1 = np.nan

            # S2: fraction of out-group tweets yesterday with same stance
            og_same = _G_DAILY_STANCE.get((prev_date, other_pol, stance_bin), 0)
            og_total = _G_DAILY_TOTAL.get((prev_date, other_pol), 0)
            if og_total > 0:
                s2 = og_same / og_total
            else:
                s2 = np.nan

            if np.isnan(s1) or np.isnan(s2):
                continue

            # S3: cross-partisan ratio over last N actions
            lookback_start = max(0, i - _G_S3_WINDOW)
            if i > 0:
                recent = seq[lookback_start:i]
                n_out = sum(a for _, _, a in recent)
                s3 = n_out / len(recent)
            else:
                s3 = baseline

            traj.append((date_str, s1, s2, s3, action))

        if len(traj) >= _G_MIN_ACTIONS:
            out.append({
                "user_id": uid,
                "political_gen": int(pol),
                "n_steps": len(traj),
                "trajectory": traj,
            })
    return out


def parallel_build(user_ids, n_workers):
    t0 = time.time()
    valid = [u for u in user_ids if u in _G_USER_POL and u in _G_USER_ACTIONS]
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
# 6. SUMMARY & SAVE
# ====================================================================
def summarize_and_save(trajs, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "trajectories.pkl"), "wb") as f:
        pickle.dump(trajs, f)

    rows = []
    for t in trajs:
        uid, pol = t["user_id"], t["political_gen"]
        for (dt, s1, s2, s3, a) in t["trajectory"]:
            rows.append((uid, pol, dt, s1, s2, s3, a))
    flat = pd.DataFrame(rows, columns=[
        "user_id", "political_gen", "date",
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
    r.append("TRAJECTORY SUMMARY (PER-ACTION, DYNAMIC S1/S2)")
    r.append("=" * 60)
    r.append(f"Users:          {n:,}  (lib={n_lib:,}, con={n-n_lib:,})")
    r.append(f"Steps/user:     mean={steps.mean():.1f}  med={np.median(steps):.0f}  "
             f"range=[{steps.min()}, {steps.max()}]")
    r.append(f"User-steps:     {len(flat):,}")
    r.append(f"Out-grp rate:   mean={out_rates.mean():.3f}  med={np.median(out_rates):.3f}")
    r.append("")
    r.append("S1/S2 computed from ALL tweets (reply, quote, RT, original) from previous day.")
    r.append("S1 = fraction of in-group tweets yesterday with same stance as this action.")
    r.append("S2 = fraction of out-group tweets yesterday with same stance as this action.")
    r.append("")

    for i, nm in enumerate(["S1 (in-grp stance agree)",
                             "S2 (out-grp stance agree)",
                             "S3 (cross-partisan ratio)"]):
        c = s_all[:, i]
        c = c[~np.isnan(c)]
        if len(c) > 0:
            r.append(f"  {nm}: mean={c.mean():.4f}  std={c.std():.4f}  "
                     f"med={np.median(c):.4f}  [{c.min():.4f}, {c.max():.4f}]")

    r.append("")
    r.append("--- By political group ---")
    for pol, lab in [(0, "Liberal"), (1, "Conservative")]:
        sub = flat[flat["political_gen"] == pol]
        if len(sub) > 0:
            r.append(f"  {lab}: {sub['user_id'].nunique():,} users, "
                     f"{len(sub):,} steps, out-rate={sub['action'].mean():.3f}, "
                     f"S3 mean={sub['S3_cross_partisan_ratio'].mean():.3f}")

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
    for pol_label, pol_val in [("All", None), ("Liberal", 0), ("Conservative", 1)]:
        if pol_val is not None:
            sub_trajs = [t for t in trajs if t["political_gen"] == pol_val]
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
            r.append(f"  {pol_label}: {n_both:,} users with both pre & post")
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


# ====================================================================
# MAIN
# ====================================================================
def main():
    global _G_USER_POL, _G_DAILY_STANCE, _G_DAILY_TOTAL
    global _G_USER_ACTIONS, _G_USER_BASELINE
    global _G_SORTED_DATES, _G_SORTED_DATES_LIST
    global _G_MIN_ACTIONS, _G_MAX_GAP_DAYS, _G_S3_WINDOW

    args = parse_args()
    print("=" * 60)
    print("IRL TRAJECTORY BUILDER -- COVID Masking (PER-ACTION, DYNAMIC S1/S2)")
    print(f"  workers={args.n_workers}  min_actions={args.min_actions}  "
          f"max_gap_days={args.max_gap_days}  s3_window={args.s3_window}")
    print("=" * 60)

    # 1. Load
    print("\n[1/6] Loading...")
    df = load_data(args.data_dir, args.bot_dir)

    # 2. Build pol_map from all users
    print("\n[2/6] Building political label map...")
    # From masking dataset
    pol_from_data = df.dropna(subset=["political_gen"]).groupby("userid")["political_gen"].agg(
        lambda x: x.mode()[0]).astype(int).to_dict()
    print(f"  From masking data: {len(pol_from_data):,}")

    # From supplementary
    pol_map = dict(pol_from_data)
    if os.path.exists(args.supp_labels):
        supp = pd.read_parquet(args.supp_labels)
        supp["userid"] = supp["userid"].astype(str)
        n_new = 0
        for _, row in supp.iterrows():
            uid = row["userid"]
            if uid not in pol_map:
                pol_map[uid] = int(row["political_gen"])
                n_new += 1
        print(f"  Supplementary: {n_new:,} new users")
    print(f"  Total pol_map: {len(pol_map):,}")

    _G_USER_POL = pol_map

    # 3. Daily stance distributions from ALL tweets
    print("\n[3/6] Daily stance distributions (ALL tweet types)...")
    _G_DAILY_STANCE, _G_DAILY_TOTAL = build_daily_stance_counts(df)

    # 4. Per-user action sequences
    print("\n[4/6] Building per-user action sequences...")
    _G_USER_ACTIONS = build_user_action_sequences(df, pol_map)

    # 5. User baselines
    print("\n[5/6] User baseline cross-partisan ratios...")
    _G_USER_BASELINE = compute_user_baselines(_G_USER_ACTIONS)

    # 6. Build trajectories
    print("\n[6/6] Building trajectories...")
    all_dates = sorted(df["date_str"].unique())
    _G_SORTED_DATES = {dt: i for i, dt in enumerate(all_dates)}
    _G_SORTED_DATES_LIST = all_dates
    _G_MIN_ACTIONS = args.min_actions
    _G_MAX_GAP_DAYS = args.max_gap_days
    _G_S3_WINDOW = args.s3_window

    print(f"  {len(all_dates)} dates: {all_dates[0]} .. {all_dates[-1]}")
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