#!/usr/bin/env python3
"""
Rolling-Window Deep MaxEnt IRL around Trump COVID shock (2020-10-02).

Cohort: users with >=7 actions in every 30-day window ending at T-3, T+0, T+3, T+6, T+9, T+12.
Per-user IRL trained on the actions inside each window.
Discretization: per-group (lib/con) median splits, computed from the cohort's full trajectories.
Features: S1, S2 (no S3). 4 states.

Outputs: one parquet per window with rewards + policy per user.
"""

import os, pickle, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import Parallel, delayed

# ================================================================
# CONFIG
# ================================================================
COHORT_PKL = "/nas/home/dchu/IDP4CSS/Ashwin/Trajectories/pre_post_shock/rolling_window_around_shock/trajectories_cohort_thr7.pkl"
OUT_DIR    = "/nas/home/dchu/IDP4CSS/Ashwin/Trajectories/pre_post_shock/rolling_window_around_shock"

SHOCK_DATE       = pd.Timestamp("2020-10-02")
WINDOW_LEN       = 30
STEP_SIZE        = 3
N_WINDOWS        = 7
END_OFFSET_FIRST = -6

n_bins      = 2
n_actions   = 2
n_features  = 2
epochs      = 3000
lr_rate     = 0.01
gamma       = 0.95
l1          = 0.0
l2          = 0.5
layers      = (3, 3)
n_jobs      = 32

os.makedirs(OUT_DIR, exist_ok=True)


# ================================================================
# IRL CLASSES (Deep MaxEnt, identical to existing scripts)
# ================================================================
class IRLEnv:
    def __init__(self, n_actions, n_states, dynamics):
        self.dynamics = dynamics
        self.n_states = n_states
        self.n_actions = n_actions


class DeepMaximumEntropy:
    def __init__(self, env, trajectories, features, layers=(3, 3), lr=0.01,
                 discount=0.9, l1=0.0, l2=0.0):
        self.env = env
        self.trajectories = torch.LongTensor(trajectories)
        self.features = torch.FloatTensor(features)
        self.discount = discount
        self.lr = lr
        self.l1 = l1
        self.l2 = l2
        self._eps = 1e-6
        self.dynamics = torch.FloatTensor(env.dynamics)
        modules = []
        last = features.shape[1]
        for h in layers:
            modules += [nn.Linear(last, h), nn.Sigmoid()]
            last = h
        self.net = nn.Sequential(*modules)
        self.alpha = nn.Linear(last, 1, bias=False)
        torch.manual_seed(12345)
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
        states = self.trajectories[..., 0]
        N, L = states.shape
        S = self.env.n_states
        valid = (states >= 0)
        gamma_pows = self.discount ** torch.arange(L, dtype=torch.float32)
        weights = gamma_pows.unsqueeze(0).expand(N, L) * valid.float()
        flat_states = states.clamp(min=0).reshape(-1)
        flat_weights = weights.reshape(-1)
        svf = torch.zeros(S)
        svf.scatter_add_(0, flat_states, flat_weights)
        return svf / flat_weights.sum().clamp(min=1e-8)

    def expected_svf_vec(self, policy):
        S = self.env.n_states
        start_states = self.trajectories[:, 0, 0]
        prob0 = torch.zeros(S)
        prob0.scatter_add_(0, start_states, torch.ones(start_states.size(0)))
        prob0 /= self.trajectories.size(0)
        x = (policy[:, :, None] * self.dynamics).sum(dim=1)
        states = self.trajectories[:, :, 0]
        valid_mask = (states >= 0).float()
        lengths = valid_mask.sum(dim=1).long()
        N, L_max, _ = self.trajectories.shape
        idx = torch.arange(L_max).unsqueeze(0)
        mask = (idx < lengths.unsqueeze(1)).float()
        M_t = mask.sum(dim=0)
        gamma_pows = self.discount ** torch.arange(L_max, dtype=torch.float32)
        w_t = gamma_pows * M_t
        exp_counts = torch.zeros(S)
        mu = prob0.clone()
        for t in range(L_max):
            exp_counts += w_t[t] * mu
            mu = mu @ x
        return exp_counts / w_t.sum().clamp(min=1e-8)

    @torch.no_grad()
    def value_iteration(self, rewards, threshold=1e-2):
        r = rewards
        V = torch.zeros(self.env.n_states)
        for _ in range(1000):
            V_prev = V
            Q = torch.matmul(self.dynamics, (r + self.discount * V))
            V = Q.max(dim=1)[0]
            if (V - V_prev).abs().max() < threshold:
                break
        Q = torch.matmul(self.dynamics, (r + self.discount * V))
        Q = Q - Q.max(dim=1, keepdim=True)[0]
        return torch.softmax(Q, dim=1)

    def train(self, n_epochs):
        svf = self.state_visitation_frequency_vec()
        for e in range(n_epochs):
            rewards = self.forward(self.features)
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
                    w_grad = g - (self.l1 * torch.sign(p.data) + 2.0 * self.l2 * p.data)
                else:
                    w_grad = g
                self.hist[idx] += w_grad.pow(2)
                p.data.add_((w_grad / (self.hist[idx].sqrt() + self._eps)) * self.lr)
        final_rewards = self.forward(self.features).detach()
        final_policy = self.value_iteration(final_rewards)
        return final_rewards.numpy(), final_policy.numpy()


# ================================================================
# TP HELPERS
# ================================================================
def compute_tp(state_sequence, n_states, n_actions):
    tp = np.zeros([n_states, n_actions, n_states])
    for i in range(len(state_sequence) - 1):
        s, a = state_sequence[i]
        ns = state_sequence[i + 1][0]
        tp[s, a, ns] += 1
    for s in range(n_states):
        row_sums = tp[s].sum(axis=1)
        row_sums[row_sums == 0] = 1
        tp[s] = tp[s] / row_sums[:, None]
    return tp


def legalise_tp(tp, legal):
    out = tp.copy()
    for s in range(tp.shape[0]):
        for a in range(tp.shape[1]):
            if np.sum(tp[s, a]) == 0:
                denom = np.sum(legal[s, a])
                out[s, a] = legal[s, a] / denom if denom > 0 else 1.0 / tp.shape[2]
    return out


def run_irl_on_traj(dt, tp_legal, feature_map, n_states):
    try:
        tp = legalise_tp(compute_tp(dt, n_states, n_actions), tp_legal)
        env = IRLEnv(n_actions, n_states, tp)
        dme = DeepMaximumEntropy(env, np.array([dt]), feature_map,
                                 layers, lr_rate, gamma, l1, l2)
        rewards, policy = dme.train(epochs)
        correct = sum(1 for s, a in dt if policy[s, a] >= 0.5)
        acc = correct / len(dt)
        c0_total = sum(1 for s, a in dt if a == 0)
        c1_total = sum(1 for s, a in dt if a == 1)
        c0_corr = sum(1 for s, a in dt if a == 0 and policy[s, 0] >= 0.5)
        c1_corr = sum(1 for s, a in dt if a == 1 and policy[s, 1] >= 0.5)
        return {
            "success": True,
            "rewards": rewards,
            "policy": policy,
            "acc": acc,
            "c0": c0_corr / c0_total if c0_total > 0 else np.nan,
            "c1": c1_corr / c1_total if c1_total > 0 else np.nan,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ================================================================
# LOAD COHORT
# ================================================================
print(f"Loading cohort from {COHORT_PKL}")
with open(COHORT_PKL, "rb") as f:
    trajs = pickle.load(f)
print(f"  {len(trajs):,} users in cohort")
print(f"  Liberal:      {sum(1 for t in trajs if t['political_gen']==0):,}")
print(f"  Conservative: {sum(1 for t in trajs if t['political_gen']==1):,}")


# ================================================================
# DISCRETIZATION (per-group medians, recomputed PER WINDOW)
# ================================================================
n_states = n_bins ** n_features
feature_map = np.eye(n_states)


def compute_group_edges(window_trajs_by_pol):
    """window_trajs_by_pol: dict pol -> list of raw step tuples."""
    edges_by_pol = {}
    for pol in [0, 1]:
        steps = window_trajs_by_pol[pol]
        edges = []
        for f in range(n_features):
            vals = np.array([s[1 + f] for s in steps])
            if len(vals) == 0:
                edges.append(np.array([0.0, 0.5, 1.0]))
                continue
            e = np.unique(np.quantile(vals, np.linspace(0, 1, n_bins + 1)))
            edges.append(e)
        edges_by_pol[pol] = edges
    return edges_by_pol


def to_state(features, pol, edges_by_pol):
    edges = edges_by_pol[pol]
    ab = [len(e) - 1 for e in edges]
    state_id, multiplier = 0, 1
    for f in reversed(range(n_features)):
        b = max(0, min(np.searchsorted(edges[f], features[f], side="right") - 1, ab[f] - 1))
        state_id += b * multiplier
        multiplier *= ab[f]
    return state_id


# ================================================================
# ROLLING WINDOWS
# ================================================================
end_offsets = [END_OFFSET_FIRST + w * STEP_SIZE for w in range(N_WINDOWS)]
print(f"\nWindows (length={WINDOW_LEN}d, ends at offsets {end_offsets} from shock)")
for w, off in enumerate(end_offsets):
    w_end   = SHOCK_DATE + pd.Timedelta(days=off)
    w_start = w_end - pd.Timedelta(days=WINDOW_LEN)
    print(f"  W{w}: {w_start.date()} to {w_end.date()}  (end T{off:+d})")


# ================================================================
# MAIN LOOP
# ================================================================
for w_idx, off in enumerate(end_offsets):
    w_end   = SHOCK_DATE + pd.Timedelta(days=off)
    w_start = w_end - pd.Timedelta(days=WINDOW_LEN)

    print(f"\n{'='*80}")
    print(f"WINDOW {w_idx} (end T{off:+d}): {w_start.date()} to {w_end.date()}")
    print(f"{'='*80}")

    # First pass: collect raw slices and compute per-window per-group medians
    raw_slices = []
    pol_steps = {0: [], 1: []}
    for t in trajs:
        pol = t["political_gen"]
        raw_slice = [s for s in t["trajectory"]
                     if w_start <= pd.Timestamp(s[0]) < w_end]
        if len(raw_slice) < 2:
            continue
        raw_slices.append((t["user_id"], pol, raw_slice))
        pol_steps[pol].extend(raw_slice)

    edges_by_pol = compute_group_edges(pol_steps)
    print(f"  Per-window edges:")
    for pol, lab in [(0, "Liberal"), (1, "Conservative")]:
        print(f"    {lab}: S1 {np.round(edges_by_pol[pol][0], 3)}  "
              f"S2 {np.round(edges_by_pol[pol][1], 3)}")

    # Second pass: discretize using THIS window's edges
    users_in_window = []
    for uid, pol, raw_slice in raw_slices:
        disc = [[to_state([s[1], s[2]], pol, edges_by_pol), s[3]] for s in raw_slice]
        out_rate = float(np.mean([s[3] for s in raw_slice]))
        users_in_window.append({
            "uid": uid,
            "pol": pol,
            "dt": disc,
            "n_actions": len(disc),
            "out_rate": out_rate,
        })

    print(f"  Users with >=2 actions in window: {len(users_in_window):,}")

    # Population TP from all users in this window
    all_sa = []
    for u in users_in_window:
        all_sa.extend(u["dt"])
    pop_tp = compute_tp(all_sa, n_states, n_actions)

    # Per-user IRL in parallel
    print(f"  Training per-user IRL ({n_jobs} workers)...")
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(run_irl_on_traj)(u["dt"], pop_tp, feature_map, n_states)
        for u in users_in_window
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # Collect
    rows = []
    n_ok = 0
    for u, r in zip(users_in_window, results):
        if not r["success"]:
            continue
        n_ok += 1
        row = {
            "window_id":    w_idx,
            "end_offset":   off,
            "window_start": str(w_start.date()),
            "window_end":   str(w_end.date()),
            "uid":          u["uid"],
            "pol":          u["pol"],
            "n_actions":    u["n_actions"],
            "out_rate":     u["out_rate"],
            "acc":          r["acc"],
            "c0_acc":       r["c0"],
            "c1_acc":       r["c1"],
        }
        for s in range(n_states):
            row[f"reward_s{s}"]     = float(r["rewards"][s])
            row[f"policy_out_s{s}"] = float(r["policy"][s, 1])
        rows.append(row)

    out_path = os.path.join(OUT_DIR, f"rolling_irl_thr7_window_{w_idx}_T{off:+d}.parquet")
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"  Saved {n_ok:,} successful records -> {out_path}")

print("\nAll windows complete.")