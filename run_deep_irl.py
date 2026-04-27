#!/usr/bin/env python3
"""
Deep MaxEnt IRL for COVID masking cross-partisan engagement.
Adapted from Frankie's (Lanqin Yuan) DeepMaximumEntropy implementation.

Key difference from linear MaxEnt:
  - Reward is R(s) = NeuralNet(phi(s)), not R(s) = w . phi(s)
  - Network automatically learns feature interactions
  - Output is per-state reward values, not interpretable weights

Uses per-user transition probabilities (Frankie's approach).
Parallel execution via joblib.

Usage:
  python run_deep_irl.py --traj_dir /data/dchu/covid_masking_irl \
                         --n_bins 2 --epochs 1500 --n_jobs 32
"""

import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import Parallel, delayed


# ====================================================================
# CLI
# ====================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traj_dir", type=str, default="/data/dchu/covid_masking_irl")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--n_bins", type=int, default=2)
    p.add_argument("--custom_edges", type=str, default=None,
                   help="Custom bin edges as semicolon-separated lists, "
                        "e.g. '0,0.2,0.8,1;0,0.2,0.8,1;0,0.2,1'")
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--l1", type=float, default=0.0)
    p.add_argument("--l2", type=float, default=0.5)
    p.add_argument("--layers", type=str, default="3,3",
                   help="Hidden layer sizes, comma-separated")
    p.add_argument("--n_jobs", type=int, default=32)
    p.add_argument("--min_steps", type=int, default=10)
    return p.parse_args()


# ====================================================================
# FRANKIE'S IRL CLASSES (adapted)
# ====================================================================
class IRLEnv:
    def __init__(self, n_actions, n_states, dynamics):
        self.dynamics = dynamics
        self.n_states = n_states
        self.n_actions = n_actions


class DeepMaximumEntropy:
    def __init__(self, env, trajectories, features,
                 layers=(3, 3), lr=0.01, discount=0.9, l1=0.0, l2=0.0):
        self.env = env
        self.trajectories = torch.LongTensor(trajectories)
        self.features = torch.FloatTensor(features)
        self.discount = discount
        self.lr = lr
        self.l1 = l1
        self.l2 = l2
        self._eps = 1e-6

        self.dynamics = torch.FloatTensor(env.dynamics)

        # Build network: features -> hidden layers -> scalar reward
        modules = []
        last = features.shape[1]
        for h in layers:
            modules += [nn.Linear(last, h), nn.Sigmoid()]
            last = h
        self.net = nn.Sequential(*modules)
        self.alpha = nn.Linear(last, 1, bias=False)

        # Init weights
        torch.manual_seed(12345)
        nn.init.normal_(self.alpha.weight, mean=0.0, std=1.0)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1.0)
                nn.init.normal_(m.bias, mean=0.0, std=1.0)

        self.param_list = list(self.net.parameters()) + list(self.alpha.parameters())
        self.hist = [torch.zeros_like(p.data) for p in self.param_list]

    def forward(self, features):
        phi = self.net(features)
        r = self.alpha(phi).view(-1)
        return (r - r.mean()) / (r.std() + 1e-8)

    def state_visitation_frequency_vec(self):
        device = self.dynamics.device
        states = self.trajectories[..., 0]
        N, L = states.shape
        S = self.env.n_states

        # Mask out sentinel -1 entries
        valid = (states >= 0)
        gamma_pows = self.discount ** torch.arange(L, device=device, dtype=torch.float32)
        weights = gamma_pows.unsqueeze(0).expand(N, L)

        # Zero out weights for padded positions
        weights = weights * valid.float()

        flat_states = states.clamp(min=0).reshape(-1)  # clamp -1 to 0 (won't matter, weight=0)
        flat_weights = weights.reshape(-1)

        svf = torch.zeros(S, device=device)
        svf.scatter_add_(0, flat_states, flat_weights)
        denom = flat_weights.sum().clamp(min=1e-8)
        return svf / denom

    def expected_svf_vec(self, policy):
        device = self.dynamics.device
        S = self.env.n_states

        start_states = self.trajectories[:, 0, 0]
        prob0 = torch.zeros(S, device=device)
        prob0.scatter_add_(0, start_states,
                          torch.ones(start_states.size(0), device=device))
        prob0 /= self.trajectories.size(0)

        x = (policy[:, :, None] * self.dynamics).sum(dim=1)

        states = self.trajectories[:, :, 0]
        valid_mask = (states >= 0).float()
        lengths = valid_mask.sum(dim=1).long()

        N, L_max, _ = self.trajectories.shape
        idx = torch.arange(L_max, device=device).unsqueeze(0)
        mask = (idx < lengths.unsqueeze(1)).float()
        M_t = mask.sum(dim=0)

        gamma_pows = self.discount ** torch.arange(L_max, device=device, dtype=torch.float32)
        w_t = gamma_pows * M_t

        exp_counts = torch.zeros(S, device=device)
        mu = prob0.clone()
        for t in range(L_max):
            exp_counts += w_t[t] * mu
            mu = mu @ x

        denom = w_t.sum().clamp(min=1e-8)
        return exp_counts / denom

    @torch.no_grad()
    def value_iteration(self, rewards, threshold=1e-2):
        r = rewards.to(self.dynamics.device)
        V = torch.zeros(self.env.n_states, device=self.dynamics.device)
        for _ in range(1000):
            V_prev = V
            Q = torch.matmul(self.dynamics, (r + self.discount * V))
            V = Q.max(dim=1)[0]
            if (V - V_prev).abs().max() < threshold:
                break
        Q = torch.matmul(self.dynamics, (r + self.discount * V))
        Q = Q - Q.max(dim=1, keepdim=True)[0]
        return torch.softmax(Q, dim=1)

    def train(self, n_epochs, verbose=False):
        svf = self.state_visitation_frequency_vec().to(self.dynamics.device)
        prev_rewards = None

        for e in range(n_epochs):
            rewards = self.forward(self.features.to(self.dynamics.device))
            if verbose and (e % 200 == 0 or e == n_epochs - 1):
                r_np = rewards.detach().cpu().numpy().round(3)
                if prev_rewards is not None:
                    delta = np.max(np.abs(r_np - prev_rewards))
                    print(f"    epoch {e:4d}: rewards={r_np}  max_delta={delta:.4f}")
                else:
                    print(f"    epoch {e:4d}: rewards={r_np}")
                prev_rewards = r_np.copy()

            policy = self.value_iteration(rewards)
            exp_svf = self.expected_svf_vec(policy)

            for p in self.param_list:
                if p.grad is not None:
                    p.grad.zero_()

            loss = (svf - exp_svf).dot(rewards)
            loss.backward()

            for idx, p in enumerate(self.param_list):
                g = p.grad.data
                if idx < len(self.param_list) - 1:
                    penalty_grad = self.l1 * torch.sign(p.data) + 2.0 * self.l2 * p.data
                    w_grad = g - penalty_grad
                else:
                    w_grad = g

                self.hist[idx] += w_grad.pow(2)
                p.data.add_((w_grad / (self.hist[idx].sqrt() + self._eps)) * self.lr)

        final_rewards = self.forward(self.features.to(self.dynamics.device)).detach()
        final_policy = self.value_iteration(final_rewards)
        return final_rewards.cpu().numpy(), final_policy.cpu().numpy()


# ====================================================================
# TRANSITION PROBABILITY UTILITIES (from Frankie's irl_utils.py)
# ====================================================================
def compute_tp(state_sequence, n_states, n_actions):
    tp = np.zeros([n_states, n_actions, n_states])
    for i in range(len(state_sequence) - 1):
        s = state_sequence[i][0]
        a = state_sequence[i][1]
        ns = state_sequence[i + 1][0]
        tp[s, a, ns] += 1

    for s in range(n_states):
        row_sums = tp[s].sum(axis=1)
        row_sums[row_sums == 0] = 1
        tp[s] = tp[s] / row_sums[:, None]
    return tp


def legalise_tp(tp, legal_transitions):
    legal_tp = tp.copy()
    for s in range(tp.shape[0]):
        for a in range(tp.shape[1]):
            if np.sum(tp[s, a]) == 0:
                denom = np.sum(legal_transitions[s, a])
                if denom > 0:
                    legal_tp[s, a] = legal_transitions[s, a] / denom
                else:
                    legal_tp[s, a] = 1.0 / tp.shape[2]
    return legal_tp


# ====================================================================
# DISCRETIZATION
# ====================================================================
def discretize(trajs, n_bins=None, custom_edges=None):
    """
    Discretize S1/S2/S3 into bins.
    If custom_edges provided, use those. Otherwise quantile bins.
    custom_edges: list of lists, e.g. [[0, 0.2, 0.8, 1], [0, 0.2, 0.8, 1], [0, 0.2, 1]]
    """
    n_features = 3
    all_vals = [[] for _ in range(n_features)]
    for t in trajs:
        for step in t["trajectory"]:
            for f in range(n_features):
                all_vals[f].append(step[1 + f])

    if custom_edges is not None:
        edges = [np.array(e) for e in custom_edges]
    else:
        edges = []
        for f in range(n_features):
            arr = np.array(all_vals[f])
            e = np.quantile(arr, np.linspace(0, 1, n_bins + 1))
            e = np.unique(e)
            edges.append(e)

    actual_bins = [len(e) - 1 for e in edges]
    feature_names = ["S1", "S2", "S3"]
    for f in range(n_features):
        print(f"  {feature_names[f]} edges: {np.round(edges[f], 3)} ({actual_bins[f]} bins)")

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

    # Feature map: one-hot identity (like Frankie's approach)
    feature_map = np.eye(n_states)

    # Build discretized trajectories as (state, action) pairs per user
    disc_trajs = []
    meta = []

    for t in trajs:
        traj = t["trajectory"]
        dt = []
        for step in traj:
            feats = [step[1 + f] for f in range(n_features)]
            s = to_state(feats)
            action = step[-1]
            dt.append([s, action])
        disc_trajs.append(dt)
        meta.append({
            "user_id": t["user_id"],
            "political_gen": t["political_gen"],
            "modal_stance": t["modal_stance"],
        })

    print(f"  States: {n_states}, Features: {n_states} (one-hot)")
    return disc_trajs, n_states, feature_map, meta


# ====================================================================
# PER-USER IRL WORKER
# ====================================================================
def run_single_user(dt, tp_legal, n_actions, n_states, feature_map,
                    layers, lr, discount, l1, l2, epochs):
    """Run Deep MaxEnt IRL for a single user. Returns rewards and policy."""
    try:
        env = IRLEnv(n_actions, n_states, tp_legal)
        traj_array = np.array([dt])
        dme = DeepMaximumEntropy(
            env, traj_array, feature_map, layers,
            lr, discount, l1=l1, l2=l2
        )
        rewards, policy = dme.train(epochs)
        return {"rewards": rewards, "policy": policy, "success": True}
    except Exception as e:
        return {"rewards": None, "policy": None, "success": False, "error": str(e)}


def _eval_policy_detailed(policy, dt):
    """Compute detailed metrics for a policy on a trajectory."""
    correct = {0: 0, 1: 0}
    total = {0: 0, 1: 0}
    ll = 0.0

    for s, a in dt:
        p = max(policy[s, a], 1e-10)
        ll += np.log(p)
        total[a] += 1
        if policy[s, a] >= 0.5:
            correct[a] += 1

    n = len(dt)
    acc = sum(correct.values()) / n if n > 0 else 0.0

    # Per-class accuracy
    acc_0 = correct[0] / total[0] if total[0] > 0 else float('nan')
    acc_1 = correct[1] / total[1] if total[1] > 0 else float('nan')

    # Balanced accuracy
    accs = []
    if total[0] > 0:
        accs.append(acc_0)
    if total[1] > 0:
        accs.append(acc_1)
    balanced = np.mean(accs) if accs else 0.0

    # Majority baseline
    majority_class = 0 if total[0] >= total[1] else 1
    majority_acc = total[majority_class] / n if n > 0 else 0.0

    # LL baselines
    ll_uniform = n * np.log(0.5)
    # Majority LL: assign P=1 to majority class
    ll_majority = 0.0
    for s, a in dt:
        if a == majority_class:
            ll_majority += np.log(0.99)  # near-1 to avoid -inf
        else:
            ll_majority += np.log(0.01)

    return {
        "acc": acc,
        "balanced": balanced,
        "acc_0": acc_0,
        "acc_1": acc_1,
        "majority_acc": majority_acc,
        "ll": ll,
        "ll_uniform": ll_uniform,
        "ll_majority": ll_majority,
        "n": n,
        "n_0": total[0],
        "n_1": total[1],
    }


def run_single_user_traintest(dt, tp_legal, n_actions, n_states, feature_map,
                               layers, lr, discount, l1, l2, epochs, train_frac=0.7):
    """
    Train/test split evaluation for a single user.
    Train on first 70% of trajectory, test on last 30%.
    Returns detailed metrics for both train and test.
    """
    try:
        split = int(len(dt) * train_frac)
        if split < 5 or len(dt) - split < 3:
            return {"success": False, "error": "too short for split"}

        train_dt = dt[:split]
        test_dt = dt[split:]

        # Train on first portion
        env = IRLEnv(n_actions, n_states, tp_legal)
        traj_array = np.array([train_dt])
        dme = DeepMaximumEntropy(
            env, traj_array, feature_map, layers,
            lr, discount, l1=l1, l2=l2
        )
        rewards, policy = dme.train(epochs)

        train_metrics = _eval_policy_detailed(policy, train_dt)
        test_metrics = _eval_policy_detailed(policy, test_dt)

        return {
            "success": True,
            "rewards": rewards,
            "policy": policy,
            "train": train_metrics,
            "test": test_metrics,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ====================================================================
# EVALUATION
# ====================================================================
def evaluate(disc_trajs, rewards, trans, n_states, n_actions, discount):
    """Evaluate policy accuracy and log-likelihood."""
    r = torch.FloatTensor(rewards)
    V = torch.zeros(n_states)
    dynamics = torch.FloatTensor(trans)

    for _ in range(1000):
        V_prev = V
        Q = torch.matmul(dynamics, (r + discount * V))
        V = Q.max(dim=1)[0]
        if (V - V_prev).abs().max() < 1e-2:
            break
    Q = torch.matmul(dynamics, (r + discount * V))
    Q = Q - Q.max(dim=1, keepdim=True)[0]
    policy = torch.softmax(Q, dim=1).numpy()

    total_ll = 0.0
    correct = 0
    total = 0

    for dt in disc_trajs:
        for s, a in dt:
            p = max(policy[s, a], 1e-10)
            total_ll += np.log(p)
            if policy[s, a] >= 0.5:
                correct += 1
            total += 1

    uniform_ll = total * np.log(0.5)
    accuracy = correct / total

    return total_ll, uniform_ll, accuracy


# ====================================================================
# MAIN
# ====================================================================
def main():
    args = parse_args()
    out_dir = args.out_dir or args.traj_dir
    layers = tuple(int(x) for x in args.layers.split(","))

    # Parse custom edges if provided
    custom_edges = None
    if args.custom_edges:
        custom_edges = []
        for part in args.custom_edges.split(";"):
            custom_edges.append([float(x) for x in part.split(",")])

    print("=" * 60)
    print("DEEP MAXENT IRL -- COVID Masking Cross-Partisan Engagement")
    if custom_edges:
        print(f"  custom edges: {custom_edges}")
    else:
        print(f"  n_bins={args.n_bins}")
    print(f"  gamma={args.gamma}  lr={args.lr}")
    print(f"  epochs={args.epochs}  l1={args.l1}  l2={args.l2}  layers={layers}")
    print(f"  n_jobs={args.n_jobs}  min_steps={args.min_steps}")
    print("=" * 60)

    # 1. Load
    print("\n[1] Loading trajectories...")
    with open(os.path.join(args.traj_dir, "trajectories.pkl"), "rb") as f:
        trajs = pickle.load(f)
    print(f"  {len(trajs):,} trajectories")

    lib_trajs = [t for t in trajs if t["political_gen"] == 0]
    con_trajs = [t for t in trajs if t["political_gen"] == 1]

    n_actions = 2

    # 2. Group-level Deep MaxEnt IRL
    for label, subset in [("ALL", trajs), ("LIBERAL", lib_trajs), ("CONSERVATIVE", con_trajs)]:
        print(f"\n{'='*60}")
        print(f"  {label} ({len(subset):,} users)")
        print(f"{'='*60}")

        print("\n[2] Discretizing...")
        disc_trajs, n_states, feature_map, meta = discretize(
            subset, n_bins=args.n_bins, custom_edges=custom_edges)

        # Population-level TP for group model
        all_sa = []
        for dt in disc_trajs:
            all_sa.extend(dt)
        pop_tp = compute_tp(all_sa, n_states, n_actions)

        print("\n[3] Running group-level Deep MaxEnt IRL...")
        t0 = time.time()

        # Pad all trajectories to same length with sentinel -1
        max_len = max(len(dt) for dt in disc_trajs)
        padded = []
        for dt in disc_trajs:
            pad = dt + [[-1, -1]] * (max_len - len(dt))
            padded.append(pad)
        traj_array = np.array(padded)

        env = IRLEnv(n_actions, n_states, pop_tp)
        dme = DeepMaximumEntropy(
            env, traj_array, feature_map, layers,
            args.lr, args.gamma, l1=args.l1, l2=args.l2
        )
        rewards, policy = dme.train(args.epochs, verbose=True)
        print(f"  Completed in {time.time()-t0:.1f}s")

        print(f"\n  RECOVERED REWARDS (per state):")
        for s in range(n_states):
            print(f"    State {s}: reward={rewards[s]:+.4f}")

        print(f"\n  RECOVERED POLICY:")
        for s in range(n_states):
            print(f"    State {s}: P(in-grp)={policy[s,0]:.3f}  P(out-grp)={policy[s,1]:.3f}")

        # Evaluate
        print(f"\n[4] Evaluating...")
        total_ll, uniform_ll, accuracy = evaluate(
            disc_trajs, rewards, pop_tp, n_states, n_actions, args.gamma
        )
        print(f"  Log-likelihood:  {total_ll:.2f}")
        print(f"  Uniform baseline: {uniform_ll:.2f}")
        print(f"  Improvement:      {total_ll - uniform_ll:.2f}")
        print(f"  Policy accuracy:  {accuracy:.3f}")

    # 3. Per-user Deep MaxEnt IRL
    print(f"\n{'='*60}")
    print("PER-USER DEEP MAXENT IRL")
    print(f"{'='*60}")

    print("\n[5] Discretizing full population...")
    disc_trajs_all, n_states, feature_map, meta = discretize(
        trajs, n_bins=args.n_bins, custom_edges=custom_edges)

    # Population-level TP as legal transitions fallback
    all_sa = []
    for dt in disc_trajs_all:
        all_sa.extend(dt)
    pop_tp = compute_tp(all_sa, n_states, n_actions)

    # Filter users with enough steps
    valid_indices = [i for i, dt in enumerate(disc_trajs_all) if len(dt) >= args.min_steps]
    print(f"  Users with {args.min_steps}+ steps: {len(valid_indices):,} / {len(disc_trajs_all):,}")

    # Compute per-user TPs (as dict for reuse in train/test)
    print("\n[6] Computing per-user transition probabilities...")
    user_tps_map = {}
    for i in range(len(disc_trajs_all)):
        dt = disc_trajs_all[i]
        if len(dt) >= args.min_steps:
            tp = compute_tp(dt, n_states, n_actions)
            tp = legalise_tp(tp, pop_tp)
            user_tps_map[i] = tp

    # Run in parallel
    print(f"\n[7] Running per-user Deep MaxEnt IRL ({len(valid_indices)} users, {args.n_jobs} jobs)...")
    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(run_single_user)(
            disc_trajs_all[idx], user_tps_map[idx],
            n_actions, n_states, feature_map,
            layers, args.lr, args.gamma, args.l1, args.l2, args.epochs
        )
        for idx in valid_indices
    )
    print(f"  Completed in {time.time()-t0:.1f}s")

    # Collect results
    rows = []
    n_success = 0
    n_fail = 0
    for j, idx in enumerate(valid_indices):
        r = results[j]
        if r["success"]:
            n_success += 1
            row = {
                "user_id": meta[idx]["user_id"],
                "political_gen": meta[idx]["political_gen"],
                "modal_stance": meta[idx]["modal_stance"],
                "n_steps": len(disc_trajs_all[idx]),
            }
            for s in range(n_states):
                row[f"reward_s{s}"] = r["rewards"][s]
                row[f"policy_s{s}_a0"] = r["policy"][s, 0]
                row[f"policy_s{s}_a1"] = r["policy"][s, 1]
            rows.append(row)
        else:
            n_fail += 1

    print(f"  Success: {n_success:,}  Failed: {n_fail:,}")

    if rows:
        user_df = pd.DataFrame(rows)

        # Summary: per-state reward distributions
        print(f"\n  Per-state reward distributions:")
        for s in range(n_states):
            col = f"reward_s{s}"
            vals = user_df[col]
            print(f"    State {s}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
                  f"[{vals.min():.4f}, {vals.max():.4f}]")

        # By group
        print(f"\n  Mean rewards by group:")
        for pol, lab in [(0, "Liberal"), (1, "Conservative")]:
            sub = user_df[user_df["political_gen"] == pol]
            if len(sub) == 0:
                continue
            print(f"    {lab} ({len(sub)} users):")
            for s in range(n_states):
                col = f"reward_s{s}"
                print(f"      State {s}: mean={sub[col].mean():.4f}  std={sub[col].std():.4f}")

        # Policy accuracy from per-user results, broken down by group
        group_correct = {0: 0, 1: 0}
        group_total = {0: 0, 1: 0}
        group_correct_c0 = {0: 0, 1: 0}  # class 0 correct
        group_total_c0 = {0: 0, 1: 0}
        group_correct_c1 = {0: 0, 1: 0}  # class 1 correct
        group_total_c1 = {0: 0, 1: 0}

        for j, idx in enumerate(valid_indices):
            r = results[j]
            if not r["success"]:
                continue
            policy = r["policy"]
            pol = meta[idx]["political_gen"]
            for s, a in disc_trajs_all[idx]:
                group_total[pol] += 1
                if a == 0:
                    group_total_c0[pol] += 1
                else:
                    group_total_c1[pol] += 1
                if policy[s, a] >= 0.5:
                    group_correct[pol] += 1
                    if a == 0:
                        group_correct_c0[pol] += 1
                    else:
                        group_correct_c1[pol] += 1

        print(f"\n  --- TRAIN SET (full trajectory) ---")
        print(f"  {'Group':<16} {'N':>6} {'Majority':>10} {'IRL acc':>10} {'Balanced':>10} {'Class0':>10} {'Class1':>10}")
        print(f"  {'-'*16} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

        for label, pols in [("Pooled", [0, 1]), ("Liberal", [0]), ("Conservative", [1])]:
            tot = sum(group_total[p] for p in pols)
            cor = sum(group_correct[p] for p in pols)
            t0 = sum(group_total_c0[p] for p in pols)
            c0 = sum(group_correct_c0[p] for p in pols)
            t1 = sum(group_total_c1[p] for p in pols)
            c1 = sum(group_correct_c1[p] for p in pols)
            if tot == 0:
                continue
            acc = cor / tot
            maj = max(t0, t1) / tot
            acc_0 = c0 / t0 if t0 > 0 else float('nan')
            acc_1 = c1 / t1 if t1 > 0 else float('nan')
            bal_parts = []
            if t0 > 0: bal_parts.append(acc_0)
            if t1 > 0: bal_parts.append(acc_1)
            bal = np.mean(bal_parts)
            n_users = sum(len([i for i in valid_indices if meta[i]["political_gen"] == p]) for p in pols) if len(pols) > 1 else len([i for i in valid_indices if meta[i]["political_gen"] == pols[0]])
            print(f"  {label:<16} {n_users:>6} {maj:>10.3f} {acc:>10.3f} {bal:>10.3f} {acc_0:>10.3f} {acc_1:>10.3f}")

        # Save
        user_df.to_parquet(os.path.join(out_dir, "deep_irl_per_user.parquet"), index=False)
        print(f"\n  Saved to {out_dir}/deep_irl_per_user.parquet")

    # 4. Train/test split evaluation (the honest test)
    print(f"\n{'='*60}")
    print("TRAIN/TEST SPLIT EVALUATION (train on first 70%, test on last 30%)")
    print(f"{'='*60}")

    # Need users with enough steps for meaningful split (10+ total -> 7 train, 3 test)
    tt_indices = [i for i, dt in enumerate(disc_trajs_all)
                  if len(dt) >= 10 and i in user_tps_map]
    print(f"\n  Users with 10+ steps and TPs: {len(tt_indices):,}")

    print(f"\n[9] Running train/test per-user IRL ({len(tt_indices)} users, {args.n_jobs} jobs)...")
    t0 = time.time()
    tt_results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(run_single_user_traintest)(
            disc_trajs_all[idx], user_tps_map[idx],
            n_actions, n_states, feature_map,
            layers, args.lr, args.gamma, args.l1, args.l2, args.epochs
        )
        for idx in tt_indices
    )
    print(f"  Completed in {time.time()-t0:.1f}s")

    # Aggregate results by group
    group_results = {0: [], 1: []}
    n_success = 0

    for j, idx in enumerate(tt_indices):
        r = tt_results[j]
        if not r["success"]:
            continue
        n_success += 1
        pol = meta[idx]["political_gen"]
        group_results[pol].append(r)

    all_results = group_results[0] + group_results[1]

    print(f"\n  Successful: {n_success:,} / {len(tt_indices):,}")

    def _report_group(label, results_list):
        if not results_list:
            return
        n = len(results_list)
        train_metrics = [r["train"] for r in results_list]
        test_metrics = [r["test"] for r in results_list]

        # Accuracy
        maj = np.mean([m["majority_acc"] for m in test_metrics])
        tr_acc = np.mean([m["acc"] for m in train_metrics])
        te_acc = np.mean([m["acc"] for m in test_metrics])
        te_bal = np.mean([m["balanced"] for m in test_metrics])
        te_c0 = np.nanmean([m["acc_0"] for m in test_metrics])
        te_c1 = np.nanmean([m["acc_1"] for m in test_metrics])

        # LL
        te_ll = np.mean([m["ll"] for m in test_metrics])
        te_ll_u = np.mean([m["ll_uniform"] for m in test_metrics])
        te_ll_m = np.mean([m["ll_majority"] for m in test_metrics])

        print(f"\n  {label} ({n} users):")
        print(f"    {'Metric':<24} {'Value':>10}")
        print(f"    {'-'*24} {'-'*10}")
        print(f"    {'Majority baseline':.<24} {maj:>10.3f}")
        print(f"    {'IRL train accuracy':.<24} {tr_acc:>10.3f}")
        print(f"    {'IRL test accuracy':.<24} {te_acc:>10.3f}")
        print(f"    {'IRL test balanced acc':.<24} {te_bal:>10.3f}")
        print(f"    {'IRL test class 0 (in)':.<24} {te_c0:>10.3f}")
        print(f"    {'IRL test class 1 (out)':.<24} {te_c1:>10.3f}")
        print(f"    {'LL: IRL vs uniform':.<24} {te_ll - te_ll_u:>+10.2f}")
        print(f"    {'LL: IRL vs majority':.<24} {te_ll - te_ll_m:>+10.2f}")

    # Summary table first
    print(f"\n  {'Group':<16} {'N':>6} {'Majority':>10} {'IRL acc':>10} {'Balanced':>10} {'Class0':>10} {'Class1':>10}")
    print(f"  {'-'*16} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for label, rlist in [("Pooled", all_results), ("Liberal", group_results[0]), ("Conservative", group_results[1])]:
        if not rlist:
            continue
        test_m = [r["test"] for r in rlist]
        maj = np.mean([m["majority_acc"] for m in test_m])
        acc = np.mean([m["acc"] for m in test_m])
        bal = np.mean([m["balanced"] for m in test_m])
        c0 = np.nanmean([m["acc_0"] for m in test_m])
        c1 = np.nanmean([m["acc_1"] for m in test_m])
        print(f"  {label:<16} {len(rlist):>6} {maj:>10.3f} {acc:>10.3f} {bal:>10.3f} {c0:>10.3f} {c1:>10.3f}")

    print(f"\n  {'Group':<16} {'LL IRL':>12} {'LL uniform':>12} {'LL majority':>12} {'IRL vs unif':>12} {'IRL vs maj':>12}")
    print(f"  {'-'*16} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    for label, rlist in [("Pooled", all_results), ("Liberal", group_results[0]), ("Conservative", group_results[1])]:
        if not rlist:
            continue
        test_m = [r["test"] for r in rlist]
        ll = np.mean([m["ll"] for m in test_m])
        ll_u = np.mean([m["ll_uniform"] for m in test_m])
        ll_m = np.mean([m["ll_majority"] for m in test_m])
        print(f"  {label:<16} {ll:>12.2f} {ll_u:>12.2f} {ll_m:>12.2f} {ll-ll_u:>+12.2f} {ll-ll_m:>+12.2f}")

    # Detailed per-group
    _report_group("Pooled", all_results)
    _report_group("Liberal", group_results[0])
    _report_group("Conservative", group_results[1])

    print(f"\n  INTERPRETATION:")
    all_test_accs = [r["test"]["acc"] for r in all_results if r["success"]]
    if all_test_accs:
        te_mean = np.mean(all_test_accs)
        if te_mean > 0.6:
            print(f"    Test accuracy {te_mean:.3f} > 0.6: rewards generalize, not just memorization")
        elif te_mean > 0.55:
            print(f"    Test accuracy {te_mean:.3f}: weak generalization, partial memorization")
        else:
            print(f"    Test accuracy {te_mean:.3f} near chance: rewards are overfit, memorization")

    print("\nDone.")


if __name__ == "__main__":
    main()