#!/usr/bin/env python3
"""
Maximum Entropy IRL for COVID masking cross-partisan engagement.

Recovers reward weights w = (w1, w2, w3) such that R(s) = w . phi(s)
where phi(s) = (S1, S2, S3) are the feature values at state s.

Algorithm: MaxEnt IRL (Ziebart et al., 2008)
  1. Discretize continuous features into bins -> discrete states
  2. Estimate transition model P(s'|s,a) from data
  3. Compute empirical feature expectations from demonstrated trajectories
  4. Iterate: value iteration -> policy -> state visitation -> gradient update

Runs separately for liberals and conservatives, then per-user if feasible.

Usage:
  python run_irl.py --traj_dir /data/dchu/covid_masking_irl --n_bins 2
"""

import argparse
import os
import pickle
import time

import numpy as np
from scipy.special import logsumexp


# ====================================================================
# CLI
# ====================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traj_dir", type=str, default="/data/dchu/covid_masking_irl")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--n_bins", type=int, default=2,
                   help="Bins per feature (states = n_bins^3)")
    p.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    p.add_argument("--n_iters", type=int, default=1000, help="MaxEnt iterations")
    p.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    p.add_argument("--l2", type=float, default=0.5, help="L2 regularization weight")
    return p.parse_args()


# ====================================================================
# 1. LOAD & DISCRETIZE
# ====================================================================
def load_trajectories(traj_dir):
    with open(os.path.join(traj_dir, "trajectories.pkl"), "rb") as f:
        trajs = pickle.load(f)
    print(f"Loaded {len(trajs):,} trajectories")
    return trajs


def discretize(trajs, n_bins):
    """
    Discretize S1/S2/S3 into bins.
    Returns:
      - disc_trajs: list of [(state_int, action), ...] per user
      - n_states: total discrete states
      - feature_map: n_states x n_features array (centroid of each bin)
      - edges: list of bin edges per feature
      - meta: list of {user_id, political_gen, modal_stance} per trajectory
    """
    n_features = 3

    # Collect all feature values
    all_vals = [[] for _ in range(n_features)]
    for t in trajs:
        for step in t["trajectory"]:
            for f in range(n_features):
                all_vals[f].append(step[1 + f])

    # Quantile bin edges
    edges = []
    actual_bins = []
    for f in range(n_features):
        arr = np.array(all_vals[f])
        e = np.quantile(arr, np.linspace(0, 1, n_bins + 1))
        e = np.unique(e)
        edges.append(e)
        actual_bins.append(len(e) - 1)

    n_states = 1
    for ab in actual_bins:
        n_states *= ab

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

    # Build feature map: centroid of each discrete state
    feature_map = np.zeros((n_states, n_features))
    state_counts = np.zeros(n_states)

    disc_trajs = []
    meta = []

    for t in trajs:
        traj = t["trajectory"]
        dt = []
        for step in traj:
            feats = [step[1 + f] for f in range(n_features)]
            s = to_state(feats)
            action = step[-1]
            dt.append((s, action))
            feature_map[s] += feats
            state_counts[s] += 1
        disc_trajs.append(dt)
        meta.append({
            "user_id": t["user_id"],
            "political_gen": t["political_gen"],
            "modal_stance": t["modal_stance"],
        })

    # Average to get centroids
    for s in range(n_states):
        if state_counts[s] > 0:
            feature_map[s] /= state_counts[s]

    print(f"  States: {n_states}, Features: {n_features}")
    print(f"  State occupancy: min={int(state_counts.min())}, "
          f"max={int(state_counts.max())}, "
          f"mean={state_counts.mean():.0f}")

    return disc_trajs, n_states, feature_map, edges, meta


# ====================================================================
# 2. ESTIMATE TRANSITION MODEL
# ====================================================================
def estimate_transitions(disc_trajs, n_states, n_actions=2):
    """
    Estimate P(s'|s,a) from observed transitions.
    Laplace smoothing to avoid zeros.
    """
    counts = np.zeros((n_states, n_actions, n_states))

    for dt in disc_trajs:
        for i in range(len(dt) - 1):
            s, a = dt[i]
            s_next, _ = dt[i + 1]
            counts[s, a, s_next] += 1

    # Laplace smoothing
    counts += 0.01
    trans = counts / counts.sum(axis=2, keepdims=True)

    total_transitions = int(counts.sum())
    print(f"  Transition model: {total_transitions:,} transitions observed")

    return trans


# ====================================================================
# 3. EMPIRICAL FEATURE EXPECTATIONS
# ====================================================================
def compute_empirical_feature_expectations(disc_trajs, feature_map, n_states, gamma, verbose=True):
    """
    Compute discounted empirical feature expectations:
      mu_emp = (1/N) sum_trajectories sum_t gamma^t * phi(s_t)
    """
    n_features = feature_map.shape[1]
    mu = np.zeros(n_features)
    total_steps = 0

    for dt in disc_trajs:
        for t_idx, (s, a) in enumerate(dt):
            mu += (gamma ** t_idx) * feature_map[s]
            total_steps += 1

    mu /= len(disc_trajs)
    if verbose:
        print(f"  Empirical feature expectations: {np.round(mu, 4)}")
    return mu


# ====================================================================
# 4. MAXENT IRL CORE
# ====================================================================
def compute_state_visitation_freq(trans, policy, n_states, n_actions,
                                  disc_trajs, gamma, max_steps=50):
    """
    Compute expected state visitation frequency under a policy.

    D[s] = sum_t gamma^t * P(s_t = s | policy)

    Uses forward pass from initial state distribution.
    """
    # Initial state distribution from data
    p0 = np.zeros(n_states)
    for dt in disc_trajs:
        if len(dt) > 0:
            p0[dt[0][0]] += 1
    p0 /= p0.sum()

    # Forward pass
    D = np.zeros(n_states)
    p_t = p0.copy()

    for t in range(max_steps):
        D += (gamma ** t) * p_t

        # Compute next-state distribution
        p_next = np.zeros(n_states)
        for s in range(n_states):
            if p_t[s] < 1e-10:
                continue
            for a in range(n_actions):
                p_next += p_t[s] * policy[s, a] * trans[s, a, :]
        p_t = p_next

        if p_t.sum() < 1e-10:
            break

    return D


def soft_value_iteration(reward, trans, n_states, n_actions, gamma,
                         max_iters=100, tol=1e-6):
    """
    Soft (MaxEnt) value iteration.
    V(s) = softmax_a [R(s) + gamma * sum_s' P(s'|s,a) V(s')]

    Returns soft policy: pi(a|s) = exp(Q(s,a) - V(s))
    """
    V = np.zeros(n_states)

    for i in range(max_iters):
        V_new = np.zeros(n_states)
        Q = np.zeros((n_states, n_actions))

        for a in range(n_actions):
            Q[:, a] = reward + gamma * trans[:, a, :].dot(V)

        V_new = logsumexp(Q, axis=1)

        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new

    # Soft policy
    Q = np.zeros((n_states, n_actions))
    for a in range(n_actions):
        Q[:, a] = reward + gamma * trans[:, a, :].dot(V)

    # Numerically stable softmax
    policy = np.exp(Q - V[:, np.newaxis])
    policy /= policy.sum(axis=1, keepdims=True)

    return policy, V


def maxent_irl(feature_map, trans, disc_trajs, n_states, n_actions,
               gamma, lr, n_iters, l2=0.0, verbose=True):
    """
    MaxEnt IRL main loop with L2 regularization.
    Gradient: (mu_emp - mu_policy) - 2*l2*weights
    Returns: recovered reward weights, training history.
    """
    n_features = feature_map.shape[1]
    weights = np.zeros(n_features)

    # Empirical feature expectations
    mu_emp = compute_empirical_feature_expectations(
        disc_trajs, feature_map, n_states, gamma, verbose=verbose
    )

    history = []

    for i in range(n_iters):
        # Reward from current weights
        reward = feature_map.dot(weights)

        # Soft value iteration -> policy
        policy, V = soft_value_iteration(
            reward, trans, n_states, n_actions, gamma
        )

        # State visitation frequency under policy
        D = compute_state_visitation_freq(
            trans, policy, n_states, n_actions, disc_trajs, gamma
        )

        # Expected feature counts under policy
        mu_policy = feature_map.T.dot(D)

        # Gradient: empirical - expected - L2 penalty
        grad = (mu_emp - mu_policy) - 2.0 * l2 * weights

        # Update weights
        weights += lr * grad

        grad_norm = np.linalg.norm(grad)
        history.append({
            "iter": i,
            "grad_norm": grad_norm,
            "weights": weights.copy(),
        })

        if verbose and (i % 100 == 0 or i == n_iters - 1):
            print(f"    iter {i:4d}: weights={np.round(weights, 4)}  "
                  f"|grad|={grad_norm:.6f}")

        if grad_norm < 1e-6:
            if verbose:
                print(f"    Converged at iter {i}")
            break

    return weights, history


# ====================================================================
# 5. PER-USER IRL (lightweight version)
# ====================================================================
def per_user_irl(feature_map, trans, disc_trajs, meta, n_states, n_actions,
                 gamma, lr, n_iters, l2=0.5, min_steps=15):
    """
    Run MaxEnt IRL per user using shared transition model.
    Only for users with enough data.
    """
    results = []
    n_skipped = 0
    n_total = len(disc_trajs)

    for idx, dt in enumerate(disc_trajs):
        if len(dt) < min_steps:
            n_skipped += 1
            continue

        if (idx + 1) % 500 == 0:
            print(f"    {idx+1}/{n_total} users processed ({len(results)} recovered)...")

        weights, _ = maxent_irl(
            feature_map, trans, [dt], n_states, n_actions,
            gamma, lr, n_iters=100, l2=l2, verbose=False
        )

        results.append({
            "user_id": meta[idx]["user_id"],
            "political_gen": meta[idx]["political_gen"],
            "modal_stance": meta[idx]["modal_stance"],
            "n_steps": len(dt),
            "w1": weights[0],
            "w2": weights[1],
            "w3": weights[2],
        })

    print(f"  Per-user IRL: {len(results):,} users (skipped {n_skipped:,} with <{min_steps} steps)")
    return results


# ====================================================================
# 6. EVALUATION
# ====================================================================
def evaluate_policy(weights, feature_map, trans, disc_trajs,
                    n_states, n_actions, gamma):
    """
    Compute log-likelihood of demonstrated trajectories under recovered policy.
    Higher = better fit.
    """
    reward = feature_map.dot(weights)
    policy, V = soft_value_iteration(reward, trans, n_states, n_actions, gamma)

    total_ll = 0.0
    total_steps = 0

    for dt in disc_trajs:
        for s, a in dt:
            p = max(policy[s, a], 1e-10)
            total_ll += np.log(p)
            total_steps += 1

    # Compare to uniform policy baseline
    uniform_ll = total_steps * np.log(0.5)

    print(f"  Log-likelihood:  {total_ll:.2f}")
    print(f"  Uniform baseline: {uniform_ll:.2f}")
    print(f"  Improvement:      {total_ll - uniform_ll:.2f}")
    print(f"  Per-step avg LL:  {total_ll/total_steps:.4f} "
          f"(uniform: {np.log(0.5):.4f})")

    # Accuracy: does the policy assign higher probability to the demonstrated action?
    correct = 0
    for dt in disc_trajs:
        for s, a in dt:
            if policy[s, a] >= 0.5:
                correct += 1
    accuracy = correct / total_steps
    print(f"  Policy accuracy:  {accuracy:.3f}")

    return total_ll, accuracy


# ====================================================================
# MAIN
# ====================================================================
def main():
    args = parse_args()
    out_dir = args.out_dir or args.traj_dir

    print("=" * 60)
    print("MAXENT IRL -- COVID Masking Cross-Partisan Engagement")
    print(f"  n_bins={args.n_bins}  gamma={args.gamma}  "
          f"lr={args.lr}  n_iters={args.n_iters}  l2={args.l2}")
    print("=" * 60)

    # 1. Load & discretize
    print("\n[1] Loading trajectories...")
    trajs = load_trajectories(args.traj_dir)

    lib_trajs = [t for t in trajs if t["political_gen"] == 0]
    con_trajs = [t for t in trajs if t["political_gen"] == 1]

    # 2. Run per-group IRL
    n_actions = 2
    all_results = {}

    for label, subset in [("ALL", trajs), ("LIBERAL", lib_trajs), ("CONSERVATIVE", con_trajs)]:
        print(f"\n{'='*60}")
        print(f"  {label} ({len(subset):,} users)")
        print(f"{'='*60}")

        print("\n[2] Discretizing...")
        disc_trajs, n_states, feature_map, edges, meta = discretize(subset, args.n_bins)

        print("\n[3] Estimating transitions...")
        trans = estimate_transitions(disc_trajs, n_states, n_actions)

        print("\n[4] Running MaxEnt IRL...")
        t0 = time.time()
        weights, history = maxent_irl(
            feature_map, trans, disc_trajs, n_states, n_actions,
            args.gamma, args.lr, args.n_iters, l2=args.l2
        )
        print(f"  Completed in {time.time()-t0:.1f}s")

        print(f"\n  RECOVERED WEIGHTS:")
        print(f"    w1 (in-group stance agree):  {weights[0]:+.4f}")
        print(f"    w2 (out-group stance agree): {weights[1]:+.4f}")
        print(f"    w3 (cross-partisan ratio):   {weights[2]:+.4f}")

        # Interpret
        dominant = ["S1 (in-grp agree)", "S2 (out-grp agree)", "S3 (cross-partisan)"]
        abs_w = np.abs(weights)
        ranked = np.argsort(-abs_w)
        print(f"    Dominant feature: {dominant[ranked[0]]} "
              f"(|w|={abs_w[ranked[0]]:.4f})")

        print("\n[5] Evaluating policy fit...")
        evaluate_policy(weights, feature_map, trans, disc_trajs,
                       n_states, n_actions, args.gamma)

        all_results[label] = {
            "weights": weights,
            "history": history,
            "n_users": len(subset),
            "n_states": n_states,
            "feature_map": feature_map,
            "edges": edges,
        }

    # 3. Compare groups
    print(f"\n{'='*60}")
    print("GROUP COMPARISON")
    print(f"{'='*60}")
    features = ["S1 (in-grp agree)", "S2 (out-grp agree)", "S3 (cross-partisan)"]
    print(f"  {'Feature':<25} {'Liberal':>10} {'Conservative':>14}")
    print(f"  {'-'*25} {'-'*10} {'-'*14}")
    for i, feat in enumerate(features):
        wl = all_results["LIBERAL"]["weights"][i]
        wc = all_results["CONSERVATIVE"]["weights"][i]
        print(f"  {feat:<25} {wl:>+10.4f} {wc:>+14.4f}")

    # 4. Per-user IRL
    print(f"\n{'='*60}")
    print("PER-USER IRL")
    print(f"{'='*60}")

    print("\n[6] Discretizing full population...")
    disc_trajs_all, n_states, feature_map, edges, meta = discretize(trajs, args.n_bins)

    print("\n[7] Estimating shared transition model...")
    trans = estimate_transitions(disc_trajs_all, n_states, n_actions)

    print("\n[8] Running per-user IRL (users with 15+ steps)...")
    t0 = time.time()
    user_results = per_user_irl(
        feature_map, trans, disc_trajs_all, meta, n_states, n_actions,
        args.gamma, lr=0.05, n_iters=100, l2=args.l2, min_steps=15
    )
    print(f"  Completed in {time.time()-t0:.1f}s")

    if user_results:
        import pandas as pd
        user_df = pd.DataFrame(user_results)

        print(f"\n  Per-user weight distributions:")
        for col, label in [("w1", "S1"), ("w2", "S2"), ("w3", "S3")]:
            vals = user_df[col]
            print(f"    {label}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
                  f"[{vals.min():.4f}, {vals.max():.4f}]")

        # Group comparison from per-user weights
        print(f"\n  Per-user weights by group:")
        for pol, label in [(0, "Liberal"), (1, "Conservative")]:
            sub = user_df[user_df["political_gen"] == pol]
            print(f"    {label} ({len(sub)} users):")
            for col, feat in [("w1", "S1"), ("w2", "S2"), ("w3", "S3")]:
                print(f"      {feat}: mean={sub[col].mean():.4f}  std={sub[col].std():.4f}")

        # Save
        user_df.to_parquet(os.path.join(out_dir, "irl_per_user_weights.parquet"), index=False)
        print(f"\n  Saved per-user weights to {out_dir}/irl_per_user_weights.parquet")

    # Save group results
    save_path = os.path.join(out_dir, "irl_results.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\n  Saved group results to {save_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()