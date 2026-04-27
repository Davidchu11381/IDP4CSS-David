#!/usr/bin/env python3
"""
Build weekly IRL trajectories for COVID masking cross-partisan engagement.

MDP:
  Features: S1 (in-group stance agreement), S2 (out-group stance agreement),
            S3 (my cross-partisan engagement ratio over recent weeks)
  Actions:  A0 = engage in-group, A1 = engage out-group
  Reward:   R = w1*S1 + w2*S2 + w3*S3  (recovered by IRL)

Temporal structure:
  Action at week t.
  S1/S2 lagged from week t-1 (opinion environment).
  S3 = out-group replies / total replies over [t-W..t-1] (behavioral momentum).

All features are fractions in [0, 1]:
  S1: what fraction of my in-group shares my masking stance
  S2: what fraction of the out-group shares my masking stance
  S3: what fraction of my recent replies went to the out-group

S3 is the key feature for IRL viability: the user's action directly moves
S3 next period. When no replies exist in the lookback window (~15% of cases),
S3 defaults to the user's overall cross-partisan ratio.

Key design decisions:
  - S1/S2 stance distributions computed from ALL labeled users (full population).
  - Action index uses REPLIES + QUOTES (RTs are 91% in-group, drown out signal).
  - No stance-flipper filtering. Modal stance used as fixed label.
  - User IDs forced to strings throughout to avoid type mismatch on joins.
  - qtd_userid is float64 in raw data, converted to int->str carefully.

Usage:
  python build_trajectories.py --data_dir /data/dchu/covid_masking \
                               --bot_dir /data/dchu/covid_mask_misc \
                               --out_dir /data/dchu/covid_masking_irl \
                               --n_workers 32 --min_weeks 10 --s3_window 3

Memory strategy:
  Globals are set in the parent process before Pool fork.
  Workers inherit them copy-on-write -- no pickling of large dicts.
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
_G_USER_ATTRS = {}      # uid(str) -> (modal_stance:int, political_gen:int)
_G_STANCE_DIST = {}     # (year_week, pol_group) -> frac_pro_mask
_G_ACTIONS = {}         # (uid(str), year_week) -> (n_in, n_out, n_total, action)
_G_USER_BASELINE = {}   # uid(str) -> overall cross-partisan ratio [0,1]
_G_SORTED_WEEKS = []
_G_WEEK_TO_IDX = {}     # year_week -> index in _G_SORTED_WEEKS
_G_MIN_WEEKS = 10
_G_MIN_ACT = 1
_G_S3_WINDOW = 3


# ====================================================================
# CLI
# ====================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="/data/dchu/covid_masking")
    p.add_argument("--bot_dir", type=str, default="/data/dchu/covid_mask_misc")
    p.add_argument("--out_dir", type=str, default="/data/dchu/covid_masking_irl")
    p.add_argument("--n_workers", type=int, default=32)
    p.add_argument("--min_weeks", type=int, default=10,
                   help="Min active weeks with reply/RT activity to keep user")
    p.add_argument("--min_actions_per_week", type=int, default=1)
    p.add_argument("--s3_window", type=int, default=3,
                   help="Rolling window (weeks) for S3 cross-partisan ratio")
    p.add_argument("--supp_labels", type=str,
                   default="/data/dchu/covid_mask_misc/supplementary_pol_labels.parquet",
                   help="Supplementary political labels for reply targets outside masking dataset")
    return p.parse_args()


# ====================================================================
# 1. LOAD & FILTER
# ====================================================================
def load_data(data_dir: str, bot_dir: str) -> pd.DataFrame:
    t0 = time.time()

    # --- Load clean (non-bot) user IDs ---
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
        avail = os.listdir(bot_dir) if os.path.isdir(bot_dir) else []
        print(f"  WARNING: no clean_user_ids.* in {bot_dir}")
        print(f"  Files found: {avail[:20]}")
        print(f"  Proceeding WITHOUT bot filter.")

    # --- Load parquets ---
    files = sorted(glob.glob(os.path.join(data_dir, "masking_2020-*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files in {data_dir}")
    print(f"  Found {len(files)} parquet files")

    cols = [
        "userid", "date", "tweet_type", "stance_bin",
        "political_gen", "reply_userid", "rt_userid", "qtd_userid",
    ]

    chunks = []
    for fp in files:
        df = pd.read_parquet(fp, columns=cols)
        chunks.append(df)
        print(f"    {os.path.basename(fp)}: {len(df):,}")
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    print(f"  Total: {len(df):,}")

    # --- Force all IDs to string for consistent joins ---
    df["userid"] = df["userid"].astype(str)
    df["reply_userid"] = df["reply_userid"].astype(str)
    df["rt_userid"] = df["rt_userid"].astype(str)
    # qtd_userid is float64, convert to string safely
    # Some values are very large ints stored as float (e.g., 7.52e+17)
    df["qtd_userid"] = df["qtd_userid"].astype("object")
    mask = df["qtd_userid"].notna()
    if mask.any():
        df.loc[mask, "qtd_userid"] = df.loc[mask, "qtd_userid"].astype(float).astype(np.int64).astype(str)
    df["qtd_userid"] = df["qtd_userid"].astype(str)

    for c in ["reply_userid", "rt_userid", "qtd_userid"]:
        df.loc[df[c].isin(["nan", "None", "", "NaN"]), c] = np.nan

    # --- Bot filter ---
    if clean_ids is not None:
        n0 = len(df)
        df = df[df["userid"].isin(clean_ids)].copy()
        print(f"  Bot filter: {len(df):,} kept (removed {n0-len(df):,})")

    # --- Datetime + ISO week ---
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)

    iso = df["date"].dt.isocalendar()
    df["year_week"] = (
        iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    )

    print(f"  {df['year_week'].nunique()} weeks, {time.time()-t0:.1f}s")
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
# 3. WEEKLY STANCE DISTRIBUTIONS (S1, S2)
# ====================================================================
def build_weekly_stance_dist(df: pd.DataFrame, all_attrs: pd.DataFrame) -> dict:
    t0 = time.time()
    uw = df[["userid", "year_week"]].drop_duplicates()
    uw = uw.merge(
        all_attrs[["modal_stance", "political_gen"]],
        left_on="userid", right_index=True, how="inner",
    )
    result = uw.groupby(["year_week", "political_gen"])["modal_stance"].mean().to_dict()
    print(f"  {len(result)} entries -- {time.time()-t0:.1f}s")
    return result


# ====================================================================
# 4. ACTION INDEX (replies only)
# ====================================================================
def build_action_index(df: pd.DataFrame, pol_map: pd.Series) -> dict:
    """
    For each (author, week): count outgoing in-group and out-group
    REPLIES + QUOTES. Action = 1 if majority out-group, else 0.
    """
    t0 = time.time()

    # Replies
    m_reply = (df["tweet_type"] == "reply") & df["reply_userid"].notna()
    reply_acts = df.loc[m_reply, ["userid", "reply_userid", "year_week"]].copy()
    reply_acts.columns = ["author", "target", "year_week"]
    print(f"  {len(reply_acts):,} reply actions")

    # Quotes
    m_quote = (df["tweet_type"] == "quoted_tweet") & df["qtd_userid"].notna()
    quote_acts = df.loc[m_quote, ["userid", "qtd_userid", "year_week"]].copy()
    quote_acts.columns = ["author", "target", "year_week"]
    print(f"  {len(quote_acts):,} quote actions")

    # Combine
    acts = pd.concat([reply_acts, quote_acts], ignore_index=True)
    del reply_acts, quote_acts
    print(f"  {len(acts):,} total engagement actions (replies + quotes)")

    acts["a_pol"] = acts["author"].map(pol_map)
    acts["t_pol"] = acts["target"].map(pol_map)
    acts.dropna(subset=["a_pol", "t_pol"], inplace=True)
    print(f"  {len(acts):,} with valid labels on both sides")

    acts["og"] = (acts["a_pol"] != acts["t_pol"]).astype(np.int8)

    grp = acts.groupby(["author", "year_week", "og"]).size().unstack(fill_value=0)
    col_0 = 0 in grp.columns
    col_1 = 1 in grp.columns

    authors = grp.index.get_level_values(0).values
    weeks = grp.index.get_level_values(1).values
    in_vals = grp[0].values if col_0 else np.zeros(len(grp), dtype=int)
    out_vals = grp[1].values if col_1 else np.zeros(len(grp), dtype=int)
    tot_vals = in_vals + out_vals
    act_vals = (out_vals > in_vals).astype(int)

    result = {}
    for i in range(len(grp)):
        result[(authors[i], weeks[i])] = (
            int(in_vals[i]), int(out_vals[i]), int(tot_vals[i]), int(act_vals[i])
        )

    print(f"  {len(result):,} (user, week) entries -- {time.time()-t0:.1f}s")
    return result


# ====================================================================
# 5. COMPUTE PER-USER BASELINE CROSS-PARTISAN RATIO
# ====================================================================
def compute_user_baselines(actions_dict: dict) -> dict:
    """
    For each user, compute overall out-group engagements / total engagements
    across all weeks. Used as fallback when S3 lookback window is empty.
    """
    t0 = time.time()
    user_totals = {}  # uid -> [total_in, total_out]

    for (uid, wk), (n_in, n_out, n_tot, action) in actions_dict.items():
        if uid not in user_totals:
            user_totals[uid] = [0, 0]
        user_totals[uid][0] += n_in
        user_totals[uid][1] += n_out

    result = {}
    for uid, (total_in, total_out) in user_totals.items():
        total = total_in + total_out
        result[uid] = total_out / total if total > 0 else 0.5

    print(f"  {len(result):,} user baselines -- {time.time()-t0:.1f}s")
    return result


# ====================================================================
# 6. TRAJECTORY BUILDER (reads globals via COW)
# ====================================================================
def _build_traj_chunk(user_ids):
    """
    Worker: reads module globals, no pickling of large data.

    For each user, for each week t with sufficient actions:
      - Action: from week t (majority direction of replies)
      - S1: in-group stance agreement from week t-1
      - S2: out-group stance agreement from week t-1
      - S3: cross-partisan engagement ratio over [t-W..t-1]
    """
    out = []

    for uid in user_ids:
        entry = _G_USER_ATTRS.get(uid)
        if entry is None:
            continue
        my_stance, my_pol = entry
        other_pol = 1 - my_pol
        baseline = _G_USER_BASELINE.get(uid, 0.5)

        traj = []
        for wk in _G_SORTED_WEEKS:
            # Action from week t
            act = _G_ACTIONS.get((uid, wk))
            if act is None:
                continue
            n_in, n_out, n_tot, action = act
            if n_tot < _G_MIN_ACT:
                continue

            # Need at least 1 prior week for lag
            wk_idx = _G_WEEK_TO_IDX[wk]
            if wk_idx < 1:
                continue

            # S1, S2: lagged by 1 week
            prev_wk = _G_SORTED_WEEKS[wk_idx - 1]
            ig_frac = _G_STANCE_DIST.get((prev_wk, my_pol), np.nan)
            s1 = ig_frac if my_stance == 1 else (1.0 - ig_frac)

            og_frac = _G_STANCE_DIST.get((prev_wk, other_pol), np.nan)
            s2 = og_frac if my_stance == 1 else (1.0 - og_frac)

            # S3: cross-partisan engagement ratio over [t-W..t-1]
            lb_in = 0
            lb_out = 0
            lookback_start = max(0, wk_idx - _G_S3_WINDOW)
            for j in range(lookback_start, wk_idx):
                past_act = _G_ACTIONS.get((uid, _G_SORTED_WEEKS[j]))
                if past_act is not None:
                    lb_in += past_act[0]
                    lb_out += past_act[1]

            lb_total = lb_in + lb_out
            if lb_total > 0:
                s3 = lb_out / lb_total
            else:
                s3 = baseline  # fallback to user's overall ratio

            traj.append((wk, s1, s2, s3, action))

        if len(traj) >= _G_MIN_WEEKS:
            out.append({
                "user_id": uid,
                "political_gen": int(my_pol),
                "modal_stance": int(my_stance),
                "n_weeks": len(traj),
                "trajectory": traj,
            })
    return out


def parallel_build(user_ids, n_workers):
    t0 = time.time()
    valid = [u for u in user_ids if u in _G_USER_ATTRS]
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

    # Pickle
    with open(os.path.join(out_dir, "trajectories.pkl"), "wb") as f:
        pickle.dump(trajs, f)

    # Flat parquet
    rows = []
    for t in trajs:
        uid, pol, st = t["user_id"], t["political_gen"], t["modal_stance"]
        for (wk, s1, s2, s3, a) in t["trajectory"]:
            rows.append((uid, pol, st, wk, s1, s2, s3, a))
    flat = pd.DataFrame(rows, columns=[
        "user_id", "political_gen", "modal_stance", "year_week",
        "S1_ingroup_stance_agree", "S2_outgroup_stance_agree",
        "S3_cross_partisan_ratio", "action",
    ])
    flat.to_parquet(os.path.join(out_dir, "trajectories_flat.parquet"), index=False)

    # --- Report ---
    n = len(trajs)
    if n == 0:
        print("  WARNING: 0 trajectories produced. Check min_weeks / data.")
        return

    n_lib = sum(1 for t in trajs if t["political_gen"] == 0)
    wks = np.array([t["n_weeks"] for t in trajs])
    out_rates = np.array([np.mean([s[4] for s in t["trajectory"]]) for t in trajs])
    s_all = np.array([[s[1], s[2], s[3]] for t in trajs for s in t["trajectory"]])

    r = []
    r.append("=" * 60)
    r.append("TRAJECTORY SUMMARY")
    r.append("=" * 60)
    r.append(f"Users:          {n:,}  (lib={n_lib:,}, con={n-n_lib:,})")
    r.append(f"Weeks/user:     mean={wks.mean():.1f}  med={np.median(wks):.0f}  "
             f"range=[{wks.min()}, {wks.max()}]")
    r.append(f"User-weeks:     {len(flat):,}")
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

    # Fraction of S3 values that used baseline fallback
    n_baseline = sum(1 for t in trajs for s in t["trajectory"]
                     if s[3] == _G_USER_BASELINE.get(t["user_id"], -1))
    r.append(f"  S3 baseline fallback: {n_baseline:,} / {len(flat):,} "
             f"({100*n_baseline/len(flat):.1f}%)")

    # --- Reference events ---
    shocks = [
        ("CDC mask recommendation", "2020-W14"),
        ("Trump wears mask",        "2020-W30"),
        ("Trump COVID diagnosis",   "2020-W40"),
    ]
    r.append("")
    r.append("--- Reference events (out-group action rate by period) ---")
    boundaries = ["2020-W04"] + [wk for _, wk in shocks] + ["2020-W54"]
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        period = flat[(flat["year_week"] >= start) & (flat["year_week"] < end)]
        if len(period) == 0:
            continue
        label = shocks[i - 1][0] if i > 0 else "Pre-CDC"
        r.append(f"  {label} ({start}..{end}): "
                 f"{len(period):,} rows, out-rate={period['action'].mean():.3f}")

    r.append("")
    r.append("--- By political group ---")
    for pol, lab in [(0, "Liberal"), (1, "Conservative")]:
        sub = flat[flat["political_gen"] == pol]
        if len(sub) > 0:
            r.append(f"  {lab}: {sub['user_id'].nunique():,} users, "
                     f"{len(sub):,} user-wks, out-rate={sub['action'].mean():.3f}, "
                     f"S3 mean={sub['S3_cross_partisan_ratio'].mean():.3f}")

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
    global _G_USER_ATTRS, _G_STANCE_DIST, _G_ACTIONS, _G_USER_BASELINE
    global _G_SORTED_WEEKS, _G_WEEK_TO_IDX, _G_MIN_WEEKS, _G_MIN_ACT, _G_S3_WINDOW

    args = parse_args()
    print("=" * 60)
    print("IRL TRAJECTORY BUILDER -- COVID Masking")
    print(f"  workers={args.n_workers}  min_weeks={args.min_weeks}  s3_window={args.s3_window}")
    print("=" * 60)

    # 1. Load
    print("\n[1/6] Loading...")
    df = load_data(args.data_dir, args.bot_dir)

    # 2. User attrs
    print("\n[2/6] User attributes...")
    all_attrs = compute_user_attrs(df)

    # Load supplementary political labels
    pol_map = all_attrs["political_gen"].copy()
    if os.path.exists(args.supp_labels):
        supp = pd.read_parquet(args.supp_labels)
        supp["userid"] = supp["userid"].astype(str)
        supp = supp.set_index("userid")["political_gen"]
        new_labels = supp[~supp.index.isin(pol_map.index)]
        pol_map = pd.concat([pol_map, new_labels])
        print(f"  Supplementary labels: {len(new_labels):,} new users added")
        print(f"  Total pol_map: {len(pol_map):,} (masking={len(all_attrs):,} + supp={len(new_labels):,})")
        del supp, new_labels
    else:
        print(f"  No supplementary labels at {args.supp_labels}")

    # 3. Stance distributions
    print("\n[3/6] Weekly stance dists (S1, S2) -- masking population only...")
    _G_STANCE_DIST = build_weekly_stance_dist(df, all_attrs)

    # 4. Action index
    print("\n[4/6] Action index -- extended pol_map...")
    _G_ACTIONS = build_action_index(df, pol_map)

    # 5. User baselines (fallback for S3)
    print("\n[5/6] User baseline cross-partisan ratios...")
    _G_USER_BASELINE = compute_user_baselines(_G_ACTIONS)

    # ================================================================
    # EDA: Compare timestep granularities (weekly vs daily vs per-action)
    # ================================================================
    print("\n[EDA] Comparing timestep granularities...")
    eda_lines = []
    eda_lines.append("=" * 70)
    eda_lines.append("EDA: TIMESTEP GRANULARITY COMPARISON")
    eda_lines.append("=" * 70)

    # Get all labeled engagements (replies + quotes with both-side labels)
    m_reply = (df["tweet_type"] == "reply") & df["reply_userid"].notna()
    m_quote = (df["tweet_type"] == "quoted_tweet") & df["qtd_userid"].notna()

    eng_reply = df.loc[m_reply, ["userid", "date", "reply_userid", "year_week"]].copy()
    eng_reply.columns = ["author", "date", "target", "year_week"]
    eng_quote = df.loc[m_quote, ["userid", "date", "qtd_userid", "year_week"]].copy()
    eng_quote.columns = ["author", "date", "target", "year_week"]
    eng = pd.concat([eng_reply, eng_quote], ignore_index=True)
    del eng_reply, eng_quote

    eng["a_pol"] = eng["author"].map(pol_map)
    eng["t_pol"] = eng["target"].map(pol_map)
    eng = eng.dropna(subset=["a_pol", "t_pol"])
    eng["cross"] = (eng["a_pol"] != eng["t_pol"]).astype(int)

    if not pd.api.types.is_datetime64_any_dtype(eng["date"]):
        eng["date"] = pd.to_datetime(eng["date"], errors="coerce")
    eng = eng.dropna(subset=["date"])
    eng["date_str"] = eng["date"].dt.strftime("%Y-%m-%d")

    # Assign political group from modal
    user_pol_eda = {uid: int(row["political_gen"]) for uid, row in all_attrs.iterrows()}
    eng["pol"] = eng["author"].map(user_pol_eda)
    eng = eng.dropna(subset=["pol"])
    eng["pol"] = eng["pol"].astype(int)

    eda_lines.append(f"\nTotal labeled engagements (replies+quotes, both sides labeled): {len(eng):,}")
    eda_lines.append(f"Unique authors: {eng['author'].nunique():,}")

    # --- Per-action (each engagement is one timestep) ---
    eda_lines.append("\n" + "-" * 50)
    eda_lines.append("TIMESTEP = PER ACTION (each engagement)")
    eda_lines.append("-" * 50)
    actions_per_user = eng.groupby("author").size()
    eda_lines.append(f"  Steps/user: mean={actions_per_user.mean():.1f}  median={actions_per_user.median():.0f}  "
                     f"std={actions_per_user.std():.1f}  range=[{actions_per_user.min()}, {actions_per_user.max()}]")
    for min_s in [10, 15, 20, 30, 50]:
        n = (actions_per_user >= min_s).sum()
        n_lib = eng[eng["pol"] == 0].groupby("author").size()
        n_con = eng[eng["pol"] == 1].groupby("author").size()
        nl = (n_lib >= min_s).sum() if len(n_lib) > 0 else 0
        nc = (n_con >= min_s).sum() if len(n_con) > 0 else 0
        eda_lines.append(f"  Min {min_s:>3} steps: {n:>7,} users (lib={nl:>6,}, con={nc:>6,})")

    # --- Daily (group actions within same day) ---
    eda_lines.append("\n" + "-" * 50)
    eda_lines.append("TIMESTEP = DAILY (group by author x date)")
    eda_lines.append("-" * 50)
    daily = eng.groupby(["author", "date_str"]).agg(
        n=("cross", "count"),
        n_cross=("cross", "sum"),
    ).reset_index()
    daily_per_user = daily.groupby("author").size()
    eda_lines.append(f"  Steps/user: mean={daily_per_user.mean():.1f}  median={daily_per_user.median():.0f}  "
                     f"std={daily_per_user.std():.1f}  range=[{daily_per_user.min()}, {daily_per_user.max()}]")

    # Engagements per active day
    eng_per_day = daily["n"]
    eda_lines.append(f"  Engagements/active day: mean={eng_per_day.mean():.1f}  median={eng_per_day.median():.0f}")

    # Merge pol back
    user_pol = eng.groupby("author")["pol"].first().to_dict()
    daily["pol"] = daily["author"].map(user_pol)
    daily_lib = daily[daily["pol"] == 0].groupby("author").size()
    daily_con = daily[daily["pol"] == 1].groupby("author").size()

    for min_s in [10, 15, 20, 30, 50]:
        n = (daily_per_user >= min_s).sum()
        nl = (daily_lib >= min_s).sum() if len(daily_lib) > 0 else 0
        nc = (daily_con >= min_s).sum() if len(daily_con) > 0 else 0
        eda_lines.append(f"  Min {min_s:>3} days:  {n:>7,} users (lib={nl:>6,}, con={nc:>6,})")

    # --- Weekly (current approach) ---
    eda_lines.append("\n" + "-" * 50)
    eda_lines.append("TIMESTEP = WEEKLY (group by author x ISO week)")
    eda_lines.append("-" * 50)
    weekly = eng.groupby(["author", "year_week"]).agg(
        n=("cross", "count"),
        n_cross=("cross", "sum"),
    ).reset_index()
    weekly_per_user = weekly.groupby("author").size()
    eda_lines.append(f"  Steps/user: mean={weekly_per_user.mean():.1f}  median={weekly_per_user.median():.0f}  "
                     f"std={weekly_per_user.std():.1f}  range=[{weekly_per_user.min()}, {weekly_per_user.max()}]")

    eng_per_week = weekly["n"]
    eda_lines.append(f"  Engagements/active week: mean={eng_per_week.mean():.1f}  median={eng_per_week.median():.0f}")

    weekly["pol"] = weekly["author"].map(user_pol)
    weekly_lib = weekly[weekly["pol"] == 0].groupby("author").size()
    weekly_con = weekly[weekly["pol"] == 1].groupby("author").size()

    for min_s in [10, 15, 20, 30, 50]:
        n = (weekly_per_user >= min_s).sum()
        nl = (weekly_lib >= min_s).sum() if len(weekly_lib) > 0 else 0
        nc = (weekly_con >= min_s).sum() if len(weekly_con) > 0 else 0
        eda_lines.append(f"  Min {min_s:>3} weeks: {n:>7,} users (lib={nl:>6,}, con={nc:>6,})")

    # --- Comparison summary ---
    eda_lines.append("\n" + "-" * 50)
    eda_lines.append("COMPARISON AT MIN 10 STEPS")
    eda_lines.append("-" * 50)

    for name, series, lib_s, con_s in [
        ("Per-action", actions_per_user,
         eng[eng["pol"] == 0].groupby("author").size(),
         eng[eng["pol"] == 1].groupby("author").size()),
        ("Daily", daily_per_user, daily_lib, daily_con),
        ("Weekly", weekly_per_user, weekly_lib, weekly_con),
    ]:
        eligible = series[series >= 10]
        el = lib_s[lib_s >= 10] if len(lib_s) > 0 else pd.Series(dtype=int)
        ec = con_s[con_s >= 10] if len(con_s) > 0 else pd.Series(dtype=int)
        if len(eligible) > 0:
            eda_lines.append(f"  {name:<12}: {len(eligible):>7,} users (lib={len(el):>6,}, con={len(ec):>6,})  "
                             f"mean_steps={eligible.mean():.1f}  median={eligible.median():.0f}")

    # --- Gap analysis for weekly ---
    eda_lines.append("\n" + "-" * 50)
    eda_lines.append("WEEK GAP ANALYSIS (weekly timestep, 10+ week users)")
    eda_lines.append("-" * 50)
    eligible_users = set(weekly_per_user[weekly_per_user >= 10].index)
    weekly_eligible = weekly[weekly["author"].isin(eligible_users)]
    
    gaps = []
    for uid, grp in weekly_eligible.groupby("author"):
        weeks_sorted = sorted(grp["year_week"].unique())
        for i in range(1, len(weeks_sorted)):
            w1 = weeks_sorted[i - 1]
            w2 = weeks_sorted[i]
            # Parse year-week to comparable number
            y1, wn1 = int(w1.split("-W")[0]), int(w1.split("-W")[1])
            y2, wn2 = int(w2.split("-W")[0]), int(w2.split("-W")[1])
            gap = (y2 * 52 + wn2) - (y1 * 52 + wn1)
            gaps.append(gap)
    
    if gaps:
        gaps = np.array(gaps)
        eda_lines.append(f"  Total transitions: {len(gaps):,}")
        eda_lines.append(f"  Gap (weeks): mean={gaps.mean():.1f}  median={np.median(gaps):.0f}  max={gaps.max()}")
        eda_lines.append(f"  Consecutive (gap=1): {(gaps == 1).sum():,} ({100 * (gaps == 1).mean():.1f}%)")
        eda_lines.append(f"  Gap <= 2: {(gaps <= 2).sum():,} ({100 * (gaps <= 2).mean():.1f}%)")
        eda_lines.append(f"  Gap > 4:  {(gaps > 4).sum():,} ({100 * (gaps > 4).mean():.1f}%)")
        eda_lines.append(f"  Gap > 8:  {(gaps > 8).sum():,} ({100 * (gaps > 8).mean():.1f}%)")

    # --- Pre/post shock distribution ---
    shock_week = "2020-W29"
    eda_lines.append("\n" + "-" * 50)
    eda_lines.append(f"PRE/POST SHOCK (Trump mask, {shock_week}) -- weekly, 10+ week users")
    eda_lines.append("-" * 50)
    
    for label, pol_val in [("All", None), ("Liberal", 0), ("Conservative", 1)]:
        if pol_val is not None:
            sub = weekly_eligible[weekly_eligible["pol"] == pol_val]
        else:
            sub = weekly_eligible
        
        user_pre_post = []
        for uid, grp in sub.groupby("author"):
            weeks = sorted(grp["year_week"].unique())
            pre = [w for w in weeks if w < shock_week]
            post = [w for w in weeks if w >= shock_week]
            if pre and post:
                user_pre_post.append({"uid": uid, "pre": len(pre), "post": len(post)})
        
        if user_pre_post:
            pp = pd.DataFrame(user_pre_post)
            eda_lines.append(f"\n  {label} ({len(pp)} users with both pre & post):")
            eda_lines.append(f"    Pre weeks:  mean={pp['pre'].mean():.1f}  median={pp['pre'].median():.0f}  "
                             f"range=[{pp['pre'].min()}, {pp['pre'].max()}]")
            eda_lines.append(f"    Post weeks: mean={pp['post'].mean():.1f}  median={pp['post'].median():.0f}  "
                             f"range=[{pp['post'].min()}, {pp['post'].max()}]")
            for min_w in [2, 3, 4, 5]:
                both = pp[(pp["pre"] >= min_w) & (pp["post"] >= min_w)]
                eda_lines.append(f"    Min {min_w} weeks both sides: {len(both):,} users")

    eda_lines.append("\n" + "=" * 70)

    eda_report = "\n".join(eda_lines)
    print(eda_report)

    eda_path = os.path.join(args.out_dir, "eda_timestep_comparison.txt")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(eda_path, "w") as f:
        f.write(eda_report)
    print(f"\n  EDA saved to {eda_path}")

    # 6. Build trajectories
    print("\n[6/6] Building trajectories...")
    _G_USER_ATTRS = {
        uid: (int(row["modal_stance"]), int(row["political_gen"]))
        for uid, row in all_attrs.iterrows()
    }
    _G_SORTED_WEEKS = sorted(df["year_week"].unique())
    _G_WEEK_TO_IDX = {wk: i for i, wk in enumerate(_G_SORTED_WEEKS)}
    _G_MIN_WEEKS = args.min_weeks
    _G_MIN_ACT = args.min_actions_per_week
    _G_S3_WINDOW = args.s3_window

    print(f"  {_G_SORTED_WEEKS[0]} .. {_G_SORTED_WEEKS[-1]} ({len(_G_SORTED_WEEKS)} weeks)")
    print(f"  Lag: S1/S2 from t-1, S3 cross-partisan ratio over t-{_G_S3_WINDOW}..t-1")

    del df
    gc.collect()

    trajs = parallel_build(list(all_attrs.index), args.n_workers)

    print("\n[SAVE]")
    summarize_and_save(trajs, args.out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()