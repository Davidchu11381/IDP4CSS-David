#!/usr/bin/env python3
"""
Action-Based Rolling Window IRL (Cohort > 60 Actions)
Windows 0-4: Standard 20-action sliding window.
Window 5 (Terminal): Actions 50 to the end of the user's trajectory.
"""

import os, pickle, numpy as np, pandas as pd, time
import torch
import torch.nn as nn
from joblib import Parallel, delayed

# ================================================================
# Hyperparameters & Config
# ================================================================
window_size = 20
step_size = 10
action_cutoff = 60 # ONLY KEEP USERS WITH > 60 ACTIONS

n_bins = 2
n_actions = 2
epochs = 3000
lr_rate = 0.01
gamma = 0.95
l1 = 0.0
l2 = 0.5
layers = (3, 3)
n_jobs = 32
n_features = 2

out_dir = "/nas/home/dchu/IDP4CSS/Ashwin/Trajectories/pre_post_shock/rolling_window_gt60"
os.makedirs(out_dir, exist_ok=True)

# ================================================================
# IRL Classes & Helpers (Standard Deep MaxEnt)
# ================================================================
class IRLEnv:
    def __init__(self, n_actions, n_states, dynamics):
        self.dynamics = dynamics
        self.n_states = n_states
        self.n_actions = n_actions

class DeepMaximumEntropy:
    def __init__(self, env, trajectories, features, layers=(3,3), lr=0.01, discount=0.9, l1=0.0, l2=0.0):
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
                w_grad = g - (self.l1 * torch.sign(p.data) + 2.0 * self.l2 * p.data) if idx < len(self.param_list) - 1 else g
                self.hist[idx] += w_grad.pow(2)
                p.data.add_((w_grad / (self.hist[idx].sqrt() + self._eps)) * self.lr)
        final_rewards = self.forward(self.features).detach()
        final_policy = self.value_iteration(final_rewards)
        return final_rewards.numpy(), final_policy.numpy()

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
        dme = DeepMaximumEntropy(IRLEnv(n_actions, n_states, tp), np.array([dt]), feature_map, layers, lr_rate, gamma, l1, l2)
        rewards, policy = dme.train(epochs)
        correct = sum(1 for s, a in dt if policy[s, a] >= 0.5)
        acc = correct / len(dt)
        return {"success": True, "rewards": rewards, "policy": policy, "acc": acc}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ================================================================
# Load Data & Discretize Full Trajectory
# ================================================================
print("Loading trajectories...")
with open("/nas/home/dchu/IDP4CSS/Ashwin/Trajectories/subset_50/trajectories.pkl", "rb") as f:
    raw_trajs = pickle.load(f)

# --- APPLY THE COHORT FILTER ---
trajs = [t for t in raw_trajs if len(t["trajectory"]) > action_cutoff]
print(f"Filtered dataset: {len(trajs)} users remaining (all have > {action_cutoff} actions).")

group_vals = {0: [[], []], 1: [[], []]}
for t in trajs:
    for step in t["trajectory"]:
        group_vals[t["political_gen"]][0].append(step[1])
        group_vals[t["political_gen"]][1].append(step[2])

group_edges = {}
for pol in [0, 1]:
    edges = []
    for f in range(n_features):
        e = np.unique(np.quantile(np.array(group_vals[pol][f]), np.linspace(0, 1, n_bins + 1)))
        edges.append(e)
    group_edges[pol] = edges

n_states = n_bins ** n_features
feature_map = np.eye(n_states)
print(f"Discretization complete. Total States: {n_states}")

def to_state(features, pol):
    edges = group_edges[pol]
    ab = [len(e) - 1 for e in edges]
    state_id, multiplier = 0, 1
    for f in reversed(range(n_features)):
        b = max(0, min(np.searchsorted(edges[f], features[f], side="right") - 1, ab[f] - 1))
        state_id += b * multiplier
        multiplier *= ab[f]
    return state_id

# ================================================================
# Main Rolling Execution Loop
# ================================================================
start_idx = 0
window_id = 0

while True:
    is_terminal_window = (window_id == 5)

    print(f"\n{'='*80}")
    if is_terminal_window:
        print(f"WINDOW {window_id} (Terminal): Actions {start_idx} to END of trajectory")
    else:
        end_idx = start_idx + window_size
        print(f"WINDOW {window_id}: Actions {start_idx} to {end_idx - 1}")
    print(f"{'='*80}")

    users_in_window = []
    for t in trajs:
        if is_terminal_window:
            # Grab everything from start_idx (50) to the very end
            raw_slice = t["trajectory"][start_idx:]
            actual_end_idx = len(t["trajectory"]) - 1
        else:
            raw_slice = t["trajectory"][start_idx:end_idx]
            actual_end_idx = end_idx - 1

        pol = t["political_gen"]
        disc_slice = [[to_state([s[1], s[2]], pol), s[3]] for s in raw_slice]
        out_rate = np.mean([s[3] for s in raw_slice])
        
        users_in_window.append({
            "uid": t["user_id"],
            "pol": pol,
            "dt": disc_slice,
            "out_rate": out_rate,
            "actual_end_idx": actual_end_idx
        })

    print(f"Users active in this window: {len(users_in_window):,}")

    all_sa = []
    for u in users_in_window:
        all_sa.extend(u["dt"])
    pop_tp = compute_tp(all_sa, n_states, n_actions)

    print(f"Training Per-User IRL ({n_jobs} cores)...")
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(run_irl_on_traj)(u["dt"], pop_tp, feature_map, n_states) for u in users_in_window
    )
    print(f"Done in {time.time()-t0:.1f}s")

    result_rows = []
    for i, u in enumerate(users_in_window):
        r = results[i]
        if not r["success"]: continue
        
        row = {
            "window_id": window_id, 
            "start_action": start_idx, 
            "end_action": u["actual_end_idx"], # Dynamically recorded for plotting
            "uid": u["uid"], "pol": u["pol"], "out_rate": u["out_rate"], "acc": r["acc"]
        }
        for s in range(n_states):
            row[f"reward_s{s}"] = r["rewards"][s]
            row[f"policy_out_s{s}"] = r["policy"][s, 1]
        result_rows.append(row)

    out_file = os.path.join(out_dir, f"rolling_irl_window_{window_id}.parquet")
    pd.DataFrame(result_rows).to_parquet(out_file, index=False)
    print(f"Saved {len(result_rows)} successful records to: {out_file}")

    # Stop the loop after Window 5 finishes
    if is_terminal_window:
        break
    
    start_idx += step_size
    window_id += 1

print("\nRolling Window Execution Complete for > 60 Cohort!")