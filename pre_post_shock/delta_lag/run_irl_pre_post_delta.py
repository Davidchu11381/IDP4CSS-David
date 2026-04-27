#!/usr/bin/env python3
"""
Train separate IRL models on PRE and POST shock trajectories.
Shock Date: October 2, 2020 (Trump COVID Diagnosis).

Time-Based Delta Lag: [0, 1, 3, 5, 10]
LOCKED COHORT: Users must survive up to Delta=10 to be included at all.
FROZEN ENVIRONMENT: pop_tp is calculated once and shared across all models.
"""

import os, pickle, numpy as np, pandas as pd, time
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from joblib import Parallel, delayed

# ================================================================
# Hyperparameters & Config
# ================================================================
shock_date_str = "2020-10-02"
action_cutoff = 60
min_steps = 10
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

# Time-based deltas capped at 10 days
deltas_in_days = [0, 1, 3, 5, 10]
shock_date_obj = datetime.strptime(shock_date_str, "%Y-%m-%d")

out_dir = "/nas/home/dchu/IDP4CSS/Ashwin/Trajectories/pre_post_shock/time_delta_oct2_locked"
os.makedirs(out_dir, exist_ok=True)

# ================================================================
# IRL Classes & Helpers 
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
        torch.manual_seed(12345) # Seed locks the weights perfectly
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
        acc = sum(1 for s, a in dt if policy[s, a] >= 0.5) / len(dt)
        return {"success": True, "rewards": rewards, "policy": policy, "acc": acc}
    except Exception as e:
        return {"success": False, "error": str(e)}

def run_pair(pre_dt, post_dt, tp_legal, feature_map, n_states):
    return run_irl_on_traj(pre_dt, tp_legal, feature_map, n_states), run_irl_on_traj(post_dt, tp_legal, feature_map, n_states)

# ================================================================
# Load Data & Setup States
# ================================================================
print("Loading trajectories...")
with open("/nas/home/dchu/IDP4CSS/Ashwin/Trajectories/subset_50/trajectories.pkl", "rb") as f:
    raw_trajs = pickle.load(f)

trajs = [t for t in raw_trajs if len(t["trajectory"]) > action_cutoff]

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
# Phase 1: Lock the Universal Cohort & Environment Matrix
# ================================================================
print("\nEstablishing Universal Cohort (Users surviving up to Delta = 10)...")
max_delta = max(deltas_in_days)
max_start_str = (shock_date_obj + timedelta(days=max_delta)).strftime("%Y-%m-%d")

universal_users = []
all_sa_for_tp = []

for t in trajs:
    pol = t["political_gen"]
    
    # Pre-Shock is strictly before Oct 2
    pre_raw = [s for s in t["trajectory"] if str(s[0]) < shock_date_str]
    
    # We check if they survive the STRICTEST condition (Delta=10)
    post_raw_strict = [s for s in t["trajectory"] if str(s[0]) >= max_start_str]
    
    if len(pre_raw) >= min_steps and len(post_raw_strict) >= min_steps:
        pre_disc = [[to_state([s[1], s[2]], pol), s[3]] for s in pre_raw]
        
        # To build the transition matrix, we use their full Delta=0 reality
        post_raw_full = [s for s in t["trajectory"] if str(s[0]) >= shock_date_str]
        post_disc_full = [[to_state([s[1], s[2]], pol), s[3]] for s in post_raw_full]
        
        universal_users.append({
            "uid": t["user_id"],
            "pol": pol,
            "pre_disc": pre_disc,
            "pre_out_rate": np.mean([s[3] for s in pre_raw]),
            "trajectory": t["trajectory"]
        })
        
        all_sa_for_tp.extend(pre_disc)
        all_sa_for_tp.extend(post_disc_full)

print(f"LOCKED Cohort Size: {len(universal_users):,} strictly aligned users.")

print("Computing FROZEN Population Transition Matrix (pop_tp)...")
frozen_pop_tp = compute_tp(all_sa_for_tp, n_states, n_actions)

# ================================================================
# Phase 2: Execution Loop
# ================================================================
for delta_days in deltas_in_days:
    post_start_str = (shock_date_obj + timedelta(days=delta_days)).strftime("%Y-%m-%d")
    
    print(f"\n{'='*80}")
    print(f"PIPELINE: DELTA = {delta_days} DAYS")
    print(f"{'='*80}")

    loop_users = []
    for u in universal_users:
        pol = u["pol"]
        
        # Grab post-data based on current delta
        post_raw = [s for s in u["trajectory"] if str(s[0]) >= post_start_str]
        post_disc = [[to_state([s[1], s[2]], pol), s[3]] for s in post_raw]
        post_out = np.mean([s[3] for s in post_raw])

        loop_users.append({
            "uid": u["uid"], "pol": pol,
            "pre_dt": u["pre_disc"], "post_dt": post_disc, # Pre is frozen!
            "pre_out_rate": u["pre_out_rate"], "post_out_rate": post_out,
            "delta_behavior": post_out - u["pre_out_rate"], 
        })

    print(f"Training Per-User IRL ({n_jobs} cores)...")
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(run_pair)(u["pre_dt"], u["post_dt"], frozen_pop_tp, feature_map, n_states) for u in loop_users
    )
    print(f"Done in {time.time()-t0:.1f}s")

    result_rows = []
    for i, u in enumerate(loop_users):
        pre_r, post_r = results[i]
        if not pre_r["success"] or not post_r["success"]: continue
        
        row = {"uid": u["uid"], "pol": u["pol"], "delta_behavior": u["delta_behavior"],
               "pre_out_rate": u["pre_out_rate"], "post_out_rate": u["post_out_rate"],
               "pre_acc": pre_r["acc"], "post_acc": post_r["acc"]}
        for s in range(n_states):
            row[f"pre_reward_s{s}"] = pre_r["rewards"][s]
            row[f"post_reward_s{s}"] = post_r["rewards"][s]
            row[f"pre_policy_out_s{s}"] = pre_r["policy"][s, 1]
            row[f"post_policy_out_s{s}"] = post_r["policy"][s, 1]
        result_rows.append(row)

    out_file = os.path.join(out_dir, f"time_delta_oct2_lag_{delta_days}days.parquet")
    pd.DataFrame(result_rows).to_parquet(out_file, index=False)
    print(f"Saved {len(result_rows)} records to: {out_file}\n")