#!/usr/bin/env python3
"""
Train separate IRL models on PRE and POST shock trajectories.
Each user gets two reward functions and two policies.
Compare how rewards/policies shift after the shock.

S1/S2 only, 4 states, per-group median splits.
"""

import pickle, numpy as np, pandas as pd, time
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from scipy.stats import wilcoxon, mannwhitneyu, spearmanr

# ================================================================
# Load trajectories
# ================================================================
with open("/nas/home/dchu/IDP4CSS/Ashwin/Trajectories/subset_50/trajectories.pkl", "rb") as f:
    trajs = pickle.load(f)

shock_date = "2020-07-11"
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

# ================================================================
# Discretize using per-group medians from FULL trajectory
# ================================================================
n_features = 2
group_vals = {0: [[], []], 1: [[], []]}
for t in trajs:
    pol = t["political_gen"]
    for step in t["trajectory"]:
        group_vals[pol][0].append(step[1])
        group_vals[pol][1].append(step[2])

group_edges = {}
for pol in [0, 1]:
    edges = []
    for f in range(n_features):
        arr = np.array(group_vals[pol][f])
        e = np.quantile(arr, np.linspace(0, 1, n_bins + 1))
        e = np.unique(e)
        edges.append(e)
    group_edges[pol] = edges

n_states = n_bins ** n_features

for pol, lab in [(0, "Liberal"), (1, "Conservative")]:
    print(f"{lab}:")
    for f, name in enumerate(["S1", "S2"]):
        print(f"  {name} edges: {np.round(group_edges[pol][f], 3)}")
print(f"States: {n_states}")

def to_state(features, pol):
    edges = group_edges[pol]
    ab = [len(e) - 1 for e in edges]
    state_id = 0
    multiplier = 1
    for f in reversed(range(n_features)):
        b = min(np.searchsorted(edges[f], features[f], side="right") - 1, ab[f] - 1)
        b = max(0, b)
        state_id += b * multiplier
        multiplier *= ab[f]
    return state_id

feature_map = np.eye(n_states)

# ================================================================
# Split into pre/post, discretize, filter
# ================================================================
users = []
for t in trajs:
    pol = t["political_gen"]
    pre_raw = [s for s in t["trajectory"] if s[0] < shock_date]
    post_raw = [s for s in t["trajectory"] if s[0] >= shock_date]

    if len(pre_raw) < min_steps or len(post_raw) < min_steps:
        continue

    pre_disc = [[to_state([s[1], s[2]], pol), s[3]] for s in pre_raw]
    post_disc = [[to_state([s[1], s[2]], pol), s[3]] for s in post_raw]

    pre_out = np.mean([s[3] for s in pre_raw])
    post_out = np.mean([s[3] for s in post_raw])

    users.append({
        "uid": t["user_id"],
        "pol": pol,
        "pre_dt": pre_disc,
        "post_dt": post_disc,
        "pre_n": len(pre_disc),
        "post_n": len(post_disc),
        "pre_out_rate": pre_out,
        "post_out_rate": post_out,
        "delta": post_out - pre_out,
    })

print(f"\nUsers with {min_steps}+ pre and {min_steps}+ post: {len(users):,}")
print(f"  Liberal: {sum(1 for u in users if u['pol']==0):,}")
print(f"  Conservative: {sum(1 for u in users if u['pol']==1):,}")
print(f"  Pre steps:  mean={np.mean([u['pre_n'] for u in users]):.1f}, median={np.median([u['pre_n'] for u in users]):.0f}")
print(f"  Post steps: mean={np.mean([u['post_n'] for u in users]):.1f}, median={np.median([u['post_n'] for u in users]):.0f}")

# ================================================================
# Transition probability helpers
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
                if denom > 0:
                    out[s, a] = legal[s, a] / denom
                else:
                    out[s, a] = 1.0 / tp.shape[2]
    return out

# Population TP from all data
all_sa = []
for u in users:
    all_sa.extend(u["pre_dt"])
    all_sa.extend(u["post_dt"])
pop_tp = compute_tp(all_sa, n_states, n_actions)

# ================================================================
# IRL classes (same as before)
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
# Worker: train IRL on a single trajectory
# ================================================================
def run_irl_on_traj(dt, tp_legal):
    try:
        tp = compute_tp(dt, n_states, n_actions)
        tp = legalise_tp(tp, tp_legal)
        env = IRLEnv(n_actions, n_states, tp)
        traj_array = np.array([dt])
        dme = DeepMaximumEntropy(env, traj_array, feature_map, layers, lr_rate, gamma, l1=l1, l2=l2)
        rewards, policy = dme.train(epochs)

        correct = sum(1 for s, a in dt if policy[s, a] >= 0.5)
        acc = correct / len(dt)
        c0_total = sum(1 for s, a in dt if a == 0)
        c1_total = sum(1 for s, a in dt if a == 1)
        c0_correct = sum(1 for s, a in dt if a == 0 and policy[s, 0] >= 0.5)
        c1_correct = sum(1 for s, a in dt if a == 1 and policy[s, 1] >= 0.5)
        ll = np.mean([np.log(max(policy[s, a], 1e-10)) for s, a in dt])

        return {
            "success": True,
            "rewards": rewards,
            "policy": policy,
            "acc": acc,
            "c0": c0_correct / c0_total if c0_total > 0 else np.nan,
            "c1": c1_correct / c1_total if c1_total > 0 else np.nan,
            "ll": ll,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def run_pair(pre_dt, post_dt, tp_legal):
    pre_result = run_irl_on_traj(pre_dt, tp_legal)
    post_result = run_irl_on_traj(post_dt, tp_legal)
    return pre_result, post_result

# ================================================================
# GROUP-LEVEL: train on pooled pre and pooled post per group
# ================================================================
print(f"\n{'='*70}")
print("GROUP-LEVEL IRL: SEPARATE PRE vs POST SHOCK MODELS")
print(f"{'='*70}")

for label, pol_filter in [("All", None), ("Liberal", 0), ("Conservative", 1)]:
    sub = [u for u in users if pol_filter is None or u["pol"] == pol_filter]
    if not sub:
        continue

    pre_all = []
    post_all = []
    for u in sub:
        pre_all.extend(u["pre_dt"])
        post_all.extend(u["post_dt"])

    # Pre group model
    pre_tp = compute_tp(pre_all, n_states, n_actions)
    pre_tp = legalise_tp(pre_tp, pop_tp)
    env_pre = IRLEnv(n_actions, n_states, pre_tp)

    # Need to pad trajectories to same length for batch
    max_pre_len = max(len(u["pre_dt"]) for u in sub)
    pre_batch = []
    for u in sub:
        padded = u["pre_dt"] + [[-1, -1]] * (max_pre_len - len(u["pre_dt"]))
        pre_batch.append(padded)
    pre_batch = np.array(pre_batch)

    dme_pre = DeepMaximumEntropy(env_pre, pre_batch, feature_map, layers, lr_rate, gamma, l1=l1, l2=l2)
    pre_rewards, pre_policy = dme_pre.train(epochs)

    # Post group model
    post_tp = compute_tp(post_all, n_states, n_actions)
    post_tp = legalise_tp(post_tp, pop_tp)
    env_post = IRLEnv(n_actions, n_states, post_tp)

    max_post_len = max(len(u["post_dt"]) for u in sub)
    post_batch = []
    for u in sub:
        padded = u["post_dt"] + [[-1, -1]] * (max_post_len - len(u["post_dt"]))
        post_batch.append(padded)
    post_batch = np.array(post_batch)

    dme_post = DeepMaximumEntropy(env_post, post_batch, feature_map, layers, lr_rate, gamma, l1=l1, l2=l2)
    post_rewards, post_policy = dme_post.train(epochs)

    print(f"\n--- {label} ({len(sub):,} users) ---")
    print(f"\n  {'State':<10} {'Pre reward':>12} {'Post reward':>12} {'Delta':>10}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*10}")
    for s in range(n_states):
        d = post_rewards[s] - pre_rewards[s]
        print(f"  State {s:<3} {pre_rewards[s]:>+12.4f} {post_rewards[s]:>+12.4f} {d:>+10.4f}")

    print(f"\n  {'State':<10} {'Pre P(out)':>12} {'Post P(out)':>12} {'Delta':>10}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*10}")
    for s in range(n_states):
        d = post_policy[s, 1] - pre_policy[s, 1]
        print(f"  State {s:<3} {pre_policy[s,1]:>12.4f} {post_policy[s,1]:>12.4f} {d:>+10.4f}")

    # Accuracy on own data
    pre_correct = sum(1 for s, a in pre_all if pre_policy[s, a] >= 0.5)
    post_correct = sum(1 for s, a in post_all if post_policy[s, a] >= 0.5)
    pre_acc = pre_correct / len(pre_all)
    post_acc = post_correct / len(post_all)
    print(f"\n  Pre policy accuracy on pre data:   {pre_acc:.3f}")
    print(f"  Post policy accuracy on post data: {post_acc:.3f}")

# ================================================================
# PER-USER: train separate pre and post IRL for each user
# ================================================================
print(f"\n{'='*70}")
print(f"PER-USER IRL: SEPARATE PRE vs POST SHOCK MODELS ({len(users)} users, {n_jobs} jobs)")
print(f"{'='*70}")

t0 = time.time()
results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(run_pair)(u["pre_dt"], u["post_dt"], pop_tp) for u in users
)
print(f"Done in {time.time()-t0:.1f}s")

# Collect
group_data = {0: [], 1: []}
for i, u in enumerate(users):
    pre_r, post_r = results[i]
    if not pre_r["success"] or not post_r["success"]:
        continue
    group_data[u["pol"]].append({
        "uid": u["uid"],
        "pol": u["pol"],
        "delta": u["delta"],
        "pre_out_rate": u["pre_out_rate"],
        "post_out_rate": u["post_out_rate"],
        "pre_rewards": pre_r["rewards"],
        "post_rewards": post_r["rewards"],
        "pre_policy": pre_r["policy"],
        "post_policy": post_r["policy"],
        "pre_acc": pre_r["acc"],
        "post_acc": post_r["acc"],
        "pre_c0": pre_r["c0"],
        "pre_c1": pre_r["c1"],
        "post_c0": post_r["c0"],
        "post_c1": post_r["c1"],
        "pre_ll": pre_r["ll"],
        "post_ll": post_r["ll"],
    })

all_data = group_data[0] + group_data[1]
n_success = len(all_data)
print(f"\nSuccessful: {n_success} / {len(users)}")

# ================================================================
# Per-user accuracy table
# ================================================================
print(f"\n--- Per-User Accuracy (trained on own period) ---")
print(f"\n{'Group':<15} {'N':>6} {'Pre acc':>10} {'Post acc':>10} {'Pre C0':>10} {'Pre C1':>10} {'Post C0':>10} {'Post C1':>10}")
print("-" * 85)

for label, rlist in [("All", all_data), ("Liberal", group_data[0]), ("Conservative", group_data[1])]:
    if not rlist:
        continue
    n = len(rlist)
    print(f"{label:<15} {n:>6,} "
          f"{np.mean([r['pre_acc'] for r in rlist]):>10.3f} "
          f"{np.mean([r['post_acc'] for r in rlist]):>10.3f} "
          f"{np.nanmean([r['pre_c0'] for r in rlist]):>10.3f} "
          f"{np.nanmean([r['pre_c1'] for r in rlist]):>10.3f} "
          f"{np.nanmean([r['post_c0'] for r in rlist]):>10.3f} "
          f"{np.nanmean([r['post_c1'] for r in rlist]):>10.3f}")

# ================================================================
# Reward shift analysis: did rewards change after shock?
# ================================================================
print(f"\n{'='*70}")
print("REWARD SHIFT: PRE vs POST SHOCK (per-user paired comparison)")
print(f"{'='*70}")

for label, rlist in [("Liberal", group_data[0]), ("Conservative", group_data[1])]:
    if not rlist:
        continue
    print(f"\n{label} ({len(rlist)} users):")
    print(f"  {'State':<10} {'Pre mean':>10} {'Post mean':>10} {'Delta':>10} {'Wilcoxon p':>12} {'Sig':>5}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*5}")

    for s in range(n_states):
        pre_vals = np.array([r["pre_rewards"][s] for r in rlist])
        post_vals = np.array([r["post_rewards"][s] for r in rlist])
        diff = post_vals - pre_vals
        try:
            stat, p = wilcoxon(pre_vals, post_vals)
        except:
            p = 1.0
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  State {s:<3} {pre_vals.mean():>+10.4f} {post_vals.mean():>+10.4f} "
              f"{diff.mean():>+10.4f} {p:>12.6f} {sig:>5}")

# ================================================================
# Policy shift: did P(out-group) change after shock?
# ================================================================
print(f"\n{'='*70}")
print("POLICY SHIFT: P(out-group) PRE vs POST SHOCK")
print(f"{'='*70}")

for label, rlist in [("Liberal", group_data[0]), ("Conservative", group_data[1])]:
    if not rlist:
        continue
    print(f"\n{label} ({len(rlist)} users):")
    print(f"  {'State':<10} {'Pre P(out)':>12} {'Post P(out)':>12} {'Delta':>10} {'Wilcoxon p':>12} {'Sig':>5}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*10} {'-'*12} {'-'*5}")

    for s in range(n_states):
        pre_vals = np.array([r["pre_policy"][s, 1] for r in rlist])
        post_vals = np.array([r["post_policy"][s, 1] for r in rlist])
        diff = post_vals - pre_vals
        try:
            stat, p = wilcoxon(pre_vals, post_vals)
        except:
            p = 1.0
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  State {s:<3} {pre_vals.mean():>12.4f} {post_vals.mean():>12.4f} "
              f"{diff.mean():>+10.4f} {p:>12.6f} {sig:>5}")

# ================================================================
# Does reward SHIFT predict behavioral delta?
# ================================================================
print(f"\n{'='*70}")
print("DOES REWARD SHIFT PREDICT BEHAVIORAL CHANGE?")
print(f"{'='*70}")

for label, rlist in [("Liberal", group_data[0]), ("Conservative", group_data[1])]:
    if not rlist:
        continue
    deltas = np.array([r["delta"] for r in rlist])
    print(f"\n{label} ({len(rlist)} users):")
    print(f"  {'State':<10} {'r(shift,delta)':>15} {'p-value':>12} {'Sig':>5}")
    print(f"  {'-'*10} {'-'*15} {'-'*12} {'-'*5}")

    for s in range(n_states):
        shifts = np.array([r["post_rewards"][s] - r["pre_rewards"][s] for r in rlist])
        r_val, p_val = spearmanr(shifts, deltas)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  State {s:<3} {r_val:>+15.4f} {p_val:>12.6f} {sig:>5}")

# ================================================================
# Save
# ================================================================
result_rows = []
for r in all_data:
    row = {"uid": r["uid"], "pol": r["pol"], "delta": r["delta"],
           "pre_out_rate": r["pre_out_rate"], "post_out_rate": r["post_out_rate"],
           "pre_acc": r["pre_acc"], "post_acc": r["post_acc"]}
    for s in range(n_states):
        row[f"pre_reward_s{s}"] = r["pre_rewards"][s]
        row[f"post_reward_s{s}"] = r["post_rewards"][s]
        row[f"pre_policy_out_s{s}"] = r["pre_policy"][s, 1]
        row[f"post_policy_out_s{s}"] = r["post_policy"][s, 1]
    result_rows.append(row)

out_path = "/nas/home/dchu/IDP4CSS/Ashwin/Trajectories/pre_post_shock/pre_vs_post_shock_irl.parquet"
pd.DataFrame(result_rows).to_parquet(out_path, index=False)
print(f"\nSaved to {out_path}")
print("Done.")