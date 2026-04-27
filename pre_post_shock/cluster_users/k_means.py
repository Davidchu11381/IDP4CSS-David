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
# 1. Setup Directories & Load Day 0 Data
# ==============================================================================
out_dir = "/nas/home/dchu/IDP4CSS/Ashwin/Trajectories/pre_post_shock/cluster_users"
os.makedirs(out_dir, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.2)

# Load the exact Day 0 dataset
input_file = "/nas/home/dchu/IDP4CSS/Ashwin/Trajectories/pre_post_shock/time_delta_oct2_locked/time_delta_oct2_lag_1days.parquet"
print(f"Loading data from: {input_file}", flush=True)

df = pd.read_parquet(input_file)
df['pol_label'] = df['pol'].map({0: 'Liberal', 1: 'Conservative'})

# ==============================================================================
# 2. Build the 8-Dimensional Feature Space (Pre-Policy + Delta Policy)
# ==============================================================================
print("Extracting IRL Policy and Delta features...", flush=True)

features = []
for s in range(4):
    # Calculate the algorithmic shift
    df[f'delta_policy_s{s}'] = df[f'post_policy_out_s{s}'] - df[f'pre_policy_out_s{s}']
    # Add to feature list
    features.extend([f'pre_policy_out_s{s}', f'delta_policy_s{s}'])

clustering_data = df[features].dropna()

# Scale features (Critical for K-Means to treat all 8 dimensions equally)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(clustering_data)

# ==============================================================================
# 3. Prove Optimal 'k'
# ==============================================================================
print("Calculating optimal number of clusters...", flush=True)
K_range = range(2, 8)
inertia = []
silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

best_k = K_range[np.argmax(silhouette_scores)]
print(f"--> Mathematical Optimal Number of Clusters (k): {best_k}", flush=True)

# ==============================================================================
# 4. Fit K-Means & PCA for Visualization
# ==============================================================================
# Fit the final clusters
final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['Cluster'] = final_kmeans.fit_predict(X_scaled)
df['Algorithmic Archetype'] = df['Cluster'].apply(lambda x: f"Cluster {x+1}")

# Run PCA to squash the 8 dimensions down to 2 for plotting
print("Running PCA to visualize 8D space...", flush=True)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df['PCA_Dim_1'] = X_pca[:, 0]
df['PCA_Dim_2'] = X_pca[:, 1]
var_d1 = pca.explained_variance_ratio_[0] * 100
var_d2 = pca.explained_variance_ratio_[1] * 100

# Save Dataframe
data_out_path = os.path.join(out_dir, f"irl_policy_clusters_day0_k{best_k}.parquet")
df.to_parquet(data_out_path, index=False)
print(f"--> Saved clustered dataframe to: {data_out_path}", flush=True)

# ==============================================================================
# 5. Generate and Save Visualization
# ==============================================================================
print("Generating visualization...", flush=True)
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5])

# ---- TOP LEFT: Elbow Method ----
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(K_range, inertia, marker='o', linewidth=2.5, color='#2c3e50')
ax1.set_title("The Elbow Method (Inertia on 8 IRL Features)", fontweight='bold')
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Within-Cluster Variance")

# ---- TOP RIGHT: Silhouette Score ----
ax2 = fig.add_subplot(gs[0, 1])
sns.barplot(x=list(K_range), y=silhouette_scores, color='#3498db', ax=ax2)
ax2.patches[np.argmax(silhouette_scores)].set_facecolor('#e74c3c')
ax2.set_title("Silhouette Score", fontweight='bold')
ax2.set_xlabel("Number of Clusters (k)")
ax2.set_ylabel("Separation Quality")

# ---- BOTTOM ROW: PCA Scatter Plots (Split by Political Group) ----
cluster_palette = sns.color_palette("Set1", best_k)

# Plot Liberals
ax3 = fig.add_subplot(gs[1, 0])
sns.scatterplot(
    data=df[df['pol_label'] == 'Liberal'], 
    x='PCA_Dim_1', y='PCA_Dim_2', 
    hue='Algorithmic Archetype', palette=cluster_palette, 
    s=100, alpha=0.8, edgecolor='w', ax=ax3
)
ax3.set_title("Liberal IRL Policy Clusters (Day 0)", fontsize=16, fontweight='bold')
ax3.set_xlabel(f"PCA Dimension 1 ({var_d1:.1f}% Variance)")
ax3.set_ylabel(f"PCA Dimension 2 ({var_d2:.1f}% Variance)")
ax3.get_legend().remove()

# Plot Conservatives
ax4 = fig.add_subplot(gs[1, 1], sharex=ax3, sharey=ax3)
sns.scatterplot(
    data=df[df['pol_label'] == 'Conservative'], 
    x='PCA_Dim_1', y='PCA_Dim_2', 
    hue='Algorithmic Archetype', palette=cluster_palette, 
    s=100, alpha=0.8, edgecolor='w', ax=ax4
)
ax4.set_title("Conservative IRL Policy Clusters (Day 0)", fontsize=16, fontweight='bold')
ax4.set_xlabel(f"PCA Dimension 1 ({var_d1:.1f}% Variance)")
ax4.set_ylabel(f"PCA Dimension 2 ({var_d2:.1f}% Variance)")

# Shared Legend
handles, labels = ax4.get_legend_handles_labels()
ax4.get_legend().remove()
fig.legend(handles, labels, loc='lower center', ncol=best_k, fontsize=14, title="Algorithmic Archetype", title_fontsize=16, bbox_to_anchor=(0.5, -0.02))

plt.suptitle(f"IRL Policy Clustering Analysis (8 Features, k={best_k})", y=1.03, fontsize=22, fontweight='bold')
plt.tight_layout()

# Save
plot_out_path = os.path.join(out_dir, f"irl_policy_clusters_pca_k{best_k}.png")
plt.savefig(plot_out_path, dpi=300, bbox_inches='tight')
print(f"--> Saved high-res plot to: {plot_out_path}", flush=True)
print("Pipeline complete!", flush=True)