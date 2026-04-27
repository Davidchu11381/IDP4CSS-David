#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==============================================================================
# 1. Setup Directories & Load Data
# ==============================================================================
out_dir = "/nas/home/dchu/IDP4CSS/Ashwin/Trajectories/pre_post_shock/cluster_users"
os.makedirs(out_dir, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.2)

# Using the Day 0 file from your previous request. 
# (You can change this to 10days if you want to look at the stabilized lag)
input_file = "/nas/home/dchu/IDP4CSS/Ashwin/Trajectories/pre_post_shock/time_delta_oct2_locked/time_delta_oct2_lag_1days.parquet"
print(f"Loading data from: {input_file}", flush=True)

df = pd.read_parquet(input_file)
df['pol_label'] = df['pol'].map({0: 'Liberal', 1: 'Conservative'})

# ==============================================================================
# 2. Pipeline Definition (K-Means + PCA)
# ==============================================================================
def get_clusters_and_pca(df, feature_cols):
    """Automatically finds best K, scales, clusters, and runs PCA."""
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Prove Optimal K
    K_range = range(2, 8)
    sil_scores = []
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
        sil_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        
    best_k = K_range[np.argmax(sil_scores)]
    
    # Final Fit
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = final_kmeans.fit_predict(X_scaled)
    
    # PCA for 2D visualization
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(X_scaled)
    var_ratio = pca.explained_variance_ratio_ * 100
    
    return labels, pca_coords, best_k, var_ratio

# ==============================================================================
# 3. Process Pre-Shock and Post-Shock Separately
# ==============================================================================
print("Clustering Pre-Shock Policy Space...", flush=True)
pre_features = [f"pre_policy_out_s{s}" for s in range(4)]
pre_labels, pre_pca, pre_k, pre_var = get_clusters_and_pca(df, pre_features)
df['Pre_Cluster'] = [f"Archetype {x+1}" for x in pre_labels]
df['Pre_PCA_1'] = pre_pca[:, 0]
df['Pre_PCA_2'] = pre_pca[:, 1]

print("Clustering Post-Shock Policy Space...", flush=True)
post_features = [f"post_policy_out_s{s}" for s in range(4)]
post_labels, post_pca, post_k, post_var = get_clusters_and_pca(df, post_features)
df['Post_Cluster'] = [f"Archetype {x+1}" for x in post_labels]
df['Post_PCA_1'] = post_pca[:, 0]
df['Post_PCA_2'] = post_pca[:, 1]

print(f"--> Pre-Shock Optimal K: {pre_k} | Post-Shock Optimal K: {post_k}", flush=True)

# ==============================================================================
# 4. Generate the 2x2 Visualization Grid
# ==============================================================================
print("Generating visualization...", flush=True)
fig, axes = plt.subplots(2, 2, figsize=(18, 14), sharex=False, sharey=False)

palette_pre = sns.color_palette("Set1", pre_k)
palette_post = sns.color_palette("Set2", post_k)

# ---- ROW 1: PRE-SHOCK (Left: Liberal, Right: Conservative) ----
sns.scatterplot(
    data=df[df['pol_label'] == 'Liberal'], x='Pre_PCA_1', y='Pre_PCA_2', 
    hue='Pre_Cluster', palette=palette_pre, s=90, alpha=0.8, edgecolor='w', ax=axes[0, 0]
)
axes[0, 0].set_title(f"PRE-SHOCK: Liberal Algorithmic View (k={pre_k})", fontsize=15, fontweight='bold')
axes[0, 0].set_xlabel(f"PCA Dim 1 ({pre_var[0]:.1f}%)")
axes[0, 0].set_ylabel(f"PCA Dim 2 ({pre_var[1]:.1f}%)")

sns.scatterplot(
    data=df[df['pol_label'] == 'Conservative'], x='Pre_PCA_1', y='Pre_PCA_2', 
    hue='Pre_Cluster', palette=palette_pre, s=90, alpha=0.8, edgecolor='w', ax=axes[0, 1]
)
axes[0, 1].set_title(f"PRE-SHOCK: Conservative Algorithmic View (k={pre_k})", fontsize=15, fontweight='bold')
axes[0, 1].set_xlabel(f"PCA Dim 1 ({pre_var[0]:.1f}%)")
axes[0, 1].set_ylabel(f"PCA Dim 2 ({pre_var[1]:.1f}%)")

# ---- ROW 2: POST-SHOCK (Left: Liberal, Right: Conservative) ----
sns.scatterplot(
    data=df[df['pol_label'] == 'Liberal'], x='Post_PCA_1', y='Post_PCA_2', 
    hue='Post_Cluster', palette=palette_post, s=90, alpha=0.8, edgecolor='w', ax=axes[1, 0]
)
axes[1, 0].set_title(f"POST-SHOCK: Liberal Algorithmic View (k={post_k})", fontsize=15, fontweight='bold')
axes[1, 0].set_xlabel(f"PCA Dim 1 ({post_var[0]:.1f}%)")
axes[1, 0].set_ylabel(f"PCA Dim 2 ({post_var[1]:.1f}%)")

sns.scatterplot(
    data=df[df['pol_label'] == 'Conservative'], x='Post_PCA_1', y='Post_PCA_2', 
    hue='Post_Cluster', palette=palette_post, s=90, alpha=0.8, edgecolor='w', ax=axes[1, 1]
)
axes[1, 1].set_title(f"POST-SHOCK: Conservative Algorithmic View (k={post_k})", fontsize=15, fontweight='bold')
axes[1, 1].set_xlabel(f"PCA Dim 1 ({post_var[0]:.1f}%)")
axes[1, 1].set_ylabel(f"PCA Dim 2 ({post_var[1]:.1f}%)")

# Formatting Legends
for ax in axes.flat:
    ax.legend(loc='best', fontsize=11)

plt.suptitle("Algorithmic Blindness: Pre-Shock vs. Post-Shock Policy Embeddings", y=1.03, fontsize=20, fontweight='bold')
plt.tight_layout()

# Save
plot_out_path = os.path.join(out_dir, "irl_policy_pre_vs_post_clusters.png")
plt.savefig(plot_out_path, dpi=300, bbox_inches='tight')
print(f"--> Saved high-res plot to: {plot_out_path}", flush=True)

# Save the dataframe
data_out_path = os.path.join(out_dir, "irl_policy_pre_vs_post_clusters.parquet")
df.to_parquet(data_out_path, index=False)
print(f"--> Saved clustered dataframe to: {data_out_path}", flush=True)
print("Pipeline complete!", flush=True)