import pickle, numpy as np, pandas as pd, time
import torch
import torch.nn as nn
from joblib import Parallel, delayed

# ================================================================
# Load trajectories
# ================================================================
with open("/nas/home/dchu/IDP4CSS/Ashwin/Trajectories/subset_50/trajectories.pkl", "rb") as f:
    trajs = pickle.load(f)

shock_date = "2020-07-11"
min_pre = 10
min_post = 10
n_bins = 2
n_actions = 2
epochs = 3000
lr = 0.01
gamma = 0.95
l1 = 0.0
l2 = 0.5
layers = (3, 3)
n_jobs = 32

# ================================================================
# Discretize using per-group medians from FULL trajectory
# (same bins for pre and post so states are comparable)
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
    
    if len(pre_raw) < min_pre or len(post_raw) < min_post:
        continue
    
    # v3 tuples: (date, s1, s2, action)
    pre_disc = [[to_state([s[1], s[2]], pol), s[3]] for s in pre_raw]
    post_disc = [[to_state([s[1], s[2]], pol), s[3]] for s in post_raw]
    
    # Pre/post out-group rates
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

print(f"\nUsers with {min_pre}+ pre and {min_post}+ post: {len(users):,}")
print(f"  Liberal: {sum(1 for u in users if u['pol']==0):,}")
print(f"  Conservative: {sum(1 for u in users if u['pol']==1):,}")

# ================================================================
# Population TPs
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

all_sa = []
for u in users:
    all_sa.extend(u["pre_dt"])
    all_sa.extend(u["post_dt"])
pop_tp = compute_tp(all_sa, n_states, n_actions)

# ================================================================
# IRL classes
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
# Worker: train on pre, evaluate on both
# ================================================================
def run_single(pre_dt, post_dt, tp_legal, n_actions, n_states, feature_map, layers, lr, gamma, l1, l2, epochs):
    try:
        tp = compute_tp(pre_dt, n_states, n_actions)
        tp = legalise_tp(tp, tp_legal)
        env = IRLEnv(n_actions, n_states, tp)
        traj_array = np.array([pre_dt])
        dme = DeepMaximumEntropy(env, traj_array, feature_map, layers, lr, gamma, l1=l1, l2=l2)
        rewards, policy = dme.train(epochs)
        
        # Evaluate on pre (train)
        pre_correct = sum(1 for s, a in pre_dt if policy[s, a] >= 0.5)
        pre_acc = pre_correct / len(pre_dt)
        pre_c0 = sum(1 for s, a in pre_dt if a == 0 and policy[s, 0] >= 0.5)
        pre_c0_total = sum(1 for s, a in pre_dt if a == 0)
        pre_c1 = sum(1 for s, a in pre_dt if a == 1 and policy[s, 1] >= 0.5)
        pre_c1_total = sum(1 for s, a in pre_dt if a == 1)
        
        # Evaluate on post (test)
        post_correct = sum(1 for s, a in post_dt if policy[s, a] >= 0.5)
        post_acc = post_correct / len(post_dt)
        post_c0 = sum(1 for s, a in post_dt if a == 0 and policy[s, 0] >= 0.5)
        post_c0_total = sum(1 for s, a in post_dt if a == 0)
        post_c1 = sum(1 for s, a in post_dt if a == 1 and policy[s, 1] >= 0.5)
        post_c1_total = sum(1 for s, a in post_dt if a == 1)
        
        # Log-likelihood
        pre_ll = sum(np.log(max(policy[s, a], 1e-10)) for s, a in pre_dt)
        post_ll = sum(np.log(max(policy[s, a], 1e-10)) for s, a in post_dt)
        
        return {
            "success": True,
            "rewards": rewards,
            "policy": policy,
            "pre_acc": pre_acc,
            "pre_c0": pre_c0 / pre_c0_total if pre_c0_total > 0 else np.nan,
            "pre_c1": pre_c1 / pre_c1_total if pre_c1_total > 0 else np.nan,
            "post_acc": post_acc,
            "post_c0": post_c0 / post_c0_total if post_c0_total > 0 else np.nan,
            "post_c1": post_c1 / post_c1_total if post_c1_total > 0 else np.nan,
            "pre_ll": pre_ll / len(pre_dt),
            "post_ll": post_ll / len(post_dt),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ================================================================
# Run
# ================================================================
print(f"\nTraining IRL on PRE-shock, testing on POST-shock ({len(users)} users, {n_jobs} jobs)...")
t0 = time.time()
results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(run_single)(
        u["pre_dt"], u["post_dt"], pop_tp, n_actions, n_states, feature_map,
        layers, lr, gamma, l1, l2, epochs
    ) for u in users
)
print(f"Done in {time.time()-t0:.1f}s")

# ================================================================
# Collect and report
# ================================================================
group_results = {0: [], 1: []}
for i, u in enumerate(users):
    r = results[i]
    if not r["success"]:
        continue
    r["pol"] = u["pol"]
    r["delta"] = u["delta"]
    r["pre_out_rate"] = u["pre_out_rate"]
    group_results[u["pol"]].append(r)

all_r = group_results[0] + group_results[1]

print(f"\n{'='*70}")
print("TRAIN ON PRE-SHOCK, TEST ON POST-SHOCK")
print(f"{'='*70}")

print(f"\n{'Group':<15} {'N':>6} {'Pre acc':>10} {'Post acc':>10} {'Pre C0':>10} {'Pre C1':>10} {'Post C0':>10} {'Post C1':>10}")
print("-" * 85)

for label, rlist in [("All", all_r), ("Liberal", group_results[0]), ("Conservative", group_results[1])]:
    if not rlist:
        continue
    n = len(rlist)
    pre_acc = np.mean([r["pre_acc"] for r in rlist])
    post_acc = np.mean([r["post_acc"] for r in rlist])
    pre_c0 = np.nanmean([r["pre_c0"] for r in rlist])
    pre_c1 = np.nanmean([r["pre_c1"] for r in rlist])
    post_c0 = np.nanmean([r["post_c0"] for r in rlist])
    post_c1 = np.nanmean([r["post_c1"] for r in rlist])
    print(f"{label:<15} {n:>6,} {pre_acc:>10.3f} {post_acc:>10.3f} {pre_c0:>10.3f} {pre_c1:>10.3f} {post_c0:>10.3f} {post_c1:>10.3f}")

print(f"\n{'Group':<15} {'Pre LL':>10} {'Post LL':>10} {'Uniform LL':>10} {'Pre vs unif':>12} {'Post vs unif':>12}")
print("-" * 70)

for label, rlist in [("All", all_r), ("Liberal", group_results[0]), ("Conservative", group_results[1])]:
    if not rlist:
        continue
    pre_ll = np.mean([r["pre_ll"] for r in rlist])
    post_ll = np.mean([r["post_ll"] for r in rlist])
    unif = np.log(0.5)
    print(f"{label:<15} {pre_ll:>10.3f} {post_ll:>10.3f} {unif:>10.3f} {pre_ll-unif:>+12.3f} {post_ll-unif:>+12.3f}")

# ================================================================
# Do pre-shock rewards predict post-shock delta?
# ================================================================
from scipy.stats import spearmanr

print(f"\n{'='*70}")
print("DO PRE-SHOCK REWARDS PREDICT POST-SHOCK BEHAVIORAL CHANGE?")
print("(No circularity: rewards trained on pre-shock only)")
print(f"{'='*70}")

for label, rlist in [("Liberal", group_results[0]), ("Conservative", group_results[1])]:
    if not rlist:
        continue
    print(f"\n{label} ({len(rlist)} users):")
    print(f"  {'State':<12} {'Spearman r':>12} {'p-value':>12} {'Sig':>5}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*5}")
    
    deltas = np.array([r["delta"] for r in rlist])
    for s in range(n_states):
        rewards_s = np.array([r["rewards"][s] for r in rlist])
        r_val, p_val = spearmanr(rewards_s, deltas)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  State {s:<5} {r_val:>12.4f} {p_val:>12.6f} {sig:>5}")

# Control for regression to mean
from sklearn.linear_model import LinearRegression

print(f"\n{'='*70}")
print("CONTROLLING FOR REGRESSION TO MEAN")
print(f"{'='*70}")

for label, rlist in [("Liberal", group_results[0]), ("Conservative", group_results[1])]:
    if not rlist:
        continue
    
    deltas = np.array([r["delta"] for r in rlist])
    pre_rates = np.array([r["pre_out_rate"] for r in rlist])
    
    # Baseline R2
    lr_base = LinearRegression()
    lr_base.fit(pre_rates.reshape(-1, 1), deltas)
    r2_base = lr_base.score(pre_rates.reshape(-1, 1), deltas)
    
    # With rewards
    reward_cols = np.array([[r["rewards"][s] for s in range(n_states)] for r in rlist])
    X_full = np.column_stack([pre_rates, reward_cols])
    lr_full = LinearRegression()
    lr_full.fit(X_full, deltas)
    r2_full = lr_full.score(X_full, deltas)
    
    # Partial correlations
    residuals = deltas - lr_base.predict(pre_rates.reshape(-1, 1))
    
    print(f"\n{label} ({len(rlist)} users):")
    print(f"  R2 (pre_rate only):     {r2_base:.4f}")
    print(f"  R2 (pre_rate + rewards): {r2_full:.4f}")
    print(f"  R2 gain from rewards:    {r2_full - r2_base:.4f}")
    print(f"\n  Partial correlations (controlling for pre_rate):")
    print(f"  {'State':<12} {'Partial r':>12} {'p-value':>12} {'Sig':>5}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*5}")
    for s in range(n_states):
        rewards_s = np.array([r["rewards"][s] for r in rlist])
        r_val, p_val = spearmanr(rewards_s, residuals)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  State {s:<5} {r_val:>12.4f} {p_val:>12.6f} {sig:>5}")

# Save results
result_rows = []
for i, u in enumerate(users):
    r = results[i]
    if not r["success"]:
        continue
    row = {"uid": u["uid"], "pol": u["pol"], "delta": u["delta"], "pre_out_rate": u["pre_out_rate"]}
    for s in range(n_states):
        row[f"pre_reward_s{s}"] = r["rewards"][s]
    result_rows.append(row)

out_path = "/nas/home/dchu/IDP4CSS/Ashwin/Trajectories/subset_50/shock_pre_train_post_test.parquet"
pd.DataFrame(result_rows).to_parquet(out_path, index=False)
print(f"\nSaved to {out_path}")