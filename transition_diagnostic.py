#!/usr/bin/env python3
"""
Transition diagnostic: Does P(s'|s,a) actually vary by state?
Updated for 3-feature trajectories: S1, S2, S3.

Tests:
  1. Row homogeneity: do different source states -> different next-state distributions?
  2. Self-transition: are states sticky?
  3. Action effect: does P(s'|s,a=0) differ from P(s'|s,a=1)?

Usage:
  python transition_diagnostic.py --traj_dir /data/dchu/covid_masking_irl --n_bins 2
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traj_dir", type=str, default="/data/dchu/covid_masking_irl")
    p.add_argument("--n_bins", type=int, default=2,
                   help="Bins per feature (total states = n_bins^3 for 3 features)")
    return p.parse_args()


def load_trajectories(traj_dir):
    with open(os.path.join(traj_dir, "trajectories.pkl"), "rb") as f:
        trajs = pickle.load(f)
    print(f"Loaded {len(trajs):,} trajectories")

    # Detect feature count from first trajectory
    sample = trajs[0]["trajectory"][0]
    # Format: (wk, s1, s2, action) or (wk, s1, s2, s3, action)
    n_features = len(sample) - 2  # minus wk and action
    print(f"  Features per step: {n_features}")
    return trajs, n_features


def discretize_states(trajs, n_bins, n_features):
    """
    Discretize all features into bins. State = tuple of bins -> single int.
    """
    # Collect all feature values
    all_vals = [[] for _ in range(n_features)]
    for t in trajs:
        for step in t["trajectory"]:
            for f in range(n_features):
                all_vals[f].append(step[1 + f])  # skip wk at index 0

    # Compute quantile bin edges per feature
    edges = []
    feature_names = ["S1", "S2", "S3"][:n_features]
    actual_bins = []

    for f in range(n_features):
        arr = np.array(all_vals[f])
        e = np.quantile(arr, np.linspace(0, 1, n_bins + 1))
        e = np.unique(e)
        edges.append(e)
        ab = len(e) - 1
        actual_bins.append(ab)
        print(f"  {feature_names[f]} bin edges: {np.round(e, 3)} ({ab} bins)")

    n_states = 1
    for ab in actual_bins:
        n_states *= ab
    print(f"  Discrete states: {'x'.join(str(b) for b in actual_bins)} = {n_states}")

    def to_state(features):
        state_id = 0
        multiplier = 1
        for f in reversed(range(n_features)):
            b = min(np.searchsorted(edges[f], features[f], side="right") - 1,
                    actual_bins[f] - 1)
            b = max(0, b)
            state_id += b * multiplier
            multiplier *= actual_bins[f]
        return state_id

    # Build transitions
    transitions = []
    for t in trajs:
        traj = t["trajectory"]
        for i in range(len(traj) - 1):
            step_t = traj[i]
            step_next = traj[i + 1]
            feats_t = [step_t[1 + f] for f in range(n_features)]
            feats_next = [step_next[1 + f] for f in range(n_features)]
            action = step_t[-1]  # last element is always action
            s = to_state(feats_t)
            s_prime = to_state(feats_next)
            transitions.append((s, action, s_prime))

    transitions = np.array(transitions)
    print(f"  Total transitions: {len(transitions):,}")

    # State labels
    state_labels = {}
    def _label(bins_combo):
        parts = []
        for f, b in enumerate(bins_combo):
            lo = edges[f][b]
            hi = edges[f][b + 1]
            parts.append(f"{feature_names[f]}[{lo:.2f},{hi:.2f}]")
        return "_".join(parts)

    def _enumerate(f_idx, combo):
        if f_idx == n_features:
            sid = to_state([edges[f][(combo[f] + combo[f] + 1) // 2]
                           for f in range(n_features)])
            # Recompute properly
            state_id = 0
            multiplier = 1
            for f in reversed(range(n_features)):
                state_id += combo[f] * multiplier
                multiplier *= actual_bins[f]
            state_labels[state_id] = _label(combo)
            return
        for b in range(actual_bins[f_idx]):
            _enumerate(f_idx + 1, combo + [b])

    _enumerate(0, [])

    return transitions, n_states, state_labels


def compute_transition_matrices(transitions, n_states):
    actions = sorted(set(transitions[:, 1]))
    matrices = {}
    counts = {}

    for a in actions:
        mask = transitions[:, 1] == a
        sub = transitions[mask]

        count_mat = np.zeros((n_states, n_states), dtype=int)
        for s, _, sp in sub:
            count_mat[s, sp] += 1

        row_sums = count_mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        prob_mat = count_mat / row_sums

        matrices[a] = prob_mat
        counts[a] = count_mat
        print(f"  Action {a}: {len(sub):,} transitions")

    return matrices, counts


def test_row_homogeneity(count_mat, state_labels):
    n_states = count_mat.shape[0]
    results = []
    row_totals = count_mat.sum(axis=1)
    valid_rows = np.where(row_totals >= 20)[0]

    if len(valid_rows) < 2:
        print("    Not enough populated states for pairwise tests")
        return results

    for i, j in combinations(valid_rows, 2):
        table = np.array([count_mat[i], count_mat[j]])
        nonzero_cols = table.sum(axis=0) > 0
        table = table[:, nonzero_cols]

        if table.shape[1] < 2:
            continue

        chi2, pval, dof, expected = stats.chi2_contingency(table)
        results.append({
            "state_i": i, "state_j": j,
            "label_i": state_labels.get(i, str(i)),
            "label_j": state_labels.get(j, str(j)),
            "chi2": chi2, "pval": pval, "dof": dof,
            "n_i": int(row_totals[i]), "n_j": int(row_totals[j]),
        })

    return results


def test_action_effect(transitions, n_states, state_labels):
    print("\n--- Action effect test ---")
    print("  For each state: does P(s'|s,a=0) differ from P(s'|s,a=1)?")

    results = []
    for s in range(n_states):
        mask_a0 = (transitions[:, 0] == s) & (transitions[:, 1] == 0)
        mask_a1 = (transitions[:, 0] == s) & (transitions[:, 1] == 1)

        n_a0 = mask_a0.sum()
        n_a1 = mask_a1.sum()

        if n_a0 < 10 or n_a1 < 10:
            continue

        sp_a0 = np.bincount(transitions[mask_a0, 2], minlength=n_states)
        sp_a1 = np.bincount(transitions[mask_a1, 2], minlength=n_states)

        table = np.array([sp_a0, sp_a1])
        nonzero = table.sum(axis=0) > 0
        table = table[:, nonzero]

        if table.shape[1] < 2:
            continue

        chi2, pval, dof, expected = stats.chi2_contingency(table)
        results.append({
            "state": s,
            "label": state_labels.get(s, str(s)),
            "n_a0": int(n_a0), "n_a1": int(n_a1),
            "chi2": chi2, "pval": pval,
        })

    if not results:
        print("  No states with sufficient data for both actions")
        return None

    df = pd.DataFrame(results)
    sig = df[df["pval"] < 0.05]
    print(f"  States tested: {len(df)}")
    print(f"  Significant (p<0.05): {len(sig)} / {len(df)} ({100*len(sig)/len(df):.0f}%)")
    print(f"  Mean chi2: {df['chi2'].mean():.2f}")

    print("\n  Per-state results:")
    for _, row in df.iterrows():
        marker = "***" if row["pval"] < 0.01 else "**" if row["pval"] < 0.05 else ""
        print(f"    State {int(row['state']):2d} ({row['label']}): "
              f"n_a0={int(row['n_a0']):5d}  n_a1={int(row['n_a1']):5d}  "
              f"chi2={row['chi2']:8.2f}  p={row['pval']:.4f} {marker}")

    return df


def self_transition_analysis(matrices, n_states, state_labels):
    print("\n--- Self-transition rates ---")
    uniform_rate = 1.0 / n_states
    print(f"  Uniform baseline: {uniform_rate:.3f}\n")

    for a, mat in matrices.items():
        print(f"  Action {a}:")
        row_totals = mat.sum(axis=1)
        for s in range(n_states):
            if row_totals[s] < 0.01:
                continue
            self_rate = mat[s, s]
            print(f"    State {s:2d} ({state_labels.get(s, '?')}): "
                  f"self={self_rate:.3f}  "
                  f"{'STICKY' if self_rate > 2 * uniform_rate else 'weak'}")


def main():
    args = parse_args()

    print("=" * 60)
    print("TRANSITION DIAGNOSTIC")
    print(f"  n_bins={args.n_bins}")
    print("=" * 60)

    trajs, n_features = load_trajectories(args.traj_dir)

    print(f"  Expected states: {args.n_bins}^{n_features} = {args.n_bins**n_features}")

    lib_trajs = [t for t in trajs if t["political_gen"] == 0]
    con_trajs = [t for t in trajs if t["political_gen"] == 1]

    for label, subset in [("ALL", trajs), ("LIBERAL", lib_trajs), ("CONSERVATIVE", con_trajs)]:
        print(f"\n{'='*60}")
        print(f"  {label} ({len(subset):,} users)")
        print(f"{'='*60}")

        print("\n[1] Discretizing states...")
        transitions, n_states, state_labels = discretize_states(subset, args.n_bins, n_features)

        print("\n[2] Transition matrices...")
        matrices, counts = compute_transition_matrices(transitions, n_states)

        for a in sorted(matrices.keys()):
            print(f"\n[3] Row homogeneity test (action={a}):")
            results = test_row_homogeneity(counts[a], state_labels)

            if results:
                df = pd.DataFrame(results)
                sig = df[df["pval"] < 0.05]
                print(f"  Pairs tested: {len(df)}")
                print(f"  Significant (p<0.05): {len(sig)} / {len(df)} ({100*len(sig)/len(df):.0f}%)")

                top = df.nsmallest(5, "pval")
                print(f"\n  Most different state pairs:")
                for _, row in top.iterrows():
                    print(f"    {row['label_i']}  vs  {row['label_j']}")
                    print(f"      chi2={row['chi2']:.2f}  p={row['pval']:.2e}  "
                          f"n=({row['n_i']}, {row['n_j']})")

        test_action_effect(transitions, n_states, state_labels)
        self_transition_analysis(matrices, n_states, state_labels)

    print("\n" + "=" * 60)
    print("INTERPRETATION GUIDE")
    print("=" * 60)
    print("  Row homogeneity significant -> states constrain transitions -> good")
    print("  Action effect significant -> action changes dynamics -> IRL viable")
    print("  Self-transition >> uniform -> states are sticky -> sequential structure")
    print("  S3 (cross-partisan ratio) should make action effect pass")
    print("=" * 60)


if __name__ == "__main__":
    main()