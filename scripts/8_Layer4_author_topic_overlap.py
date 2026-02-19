# 8_Layer4_author_topic_overlap.py
# Layer 4B – Semantic Proximity Matrix
# ------------------------------------
# Reads:  Layer4_topic_distribution_per_author.csv
# Outputs:
#   - Layer4_semantic_similarity_matrix.csv
#   - Layer4_semantic_similarity_long.csv
#   - Layer4_semantic_linkage.npy (for dendrograms)
#   - Layer4_semantic_PCA_coordinates.csv (optional PCA projection)

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

print("Loading per-author topic fingerprints...")
df = pd.read_csv("Layer4_topic_distribution_per_author_col_renamed.csv")
# -----------------------------------------------------------
# 2. Extract author IDs + topic matrix
# -----------------------------------------------------------
author_ids = df["author_id"]
topic_cols = df.columns[1:]  # all topic columns after author_id

if len(topic_cols) == 0:
    raise ValueError("No topic columns found! Check your CSV formatting.")

X = df[topic_cols].values

print(f"Found {len(author_ids)} authors, {len(topic_cols)} topics.")

# -----------------------------------------------------------
# 3. Standardize (optional but improves clustering)
# -----------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------------------------
# 4. Compute cosine similarity between authors
# -----------------------------------------------------------
sim_matrix = cosine_similarity(X_scaled)

sim_df = pd.DataFrame(sim_matrix, index=author_ids, columns=author_ids)
sim_df.to_csv("Layer4_author_similarity_matrix.csv")
print("Saved: Layer4_author_similarity_matrix.csv")

# -----------------------------------------------------------
# 5. Hierarchical clustering dendrogram
# -----------------------------------------------------------
plt.figure(figsize=(12, 6))
link = linkage(X_scaled, method="ward")
dendrogram(link, labels=author_ids.tolist(), leaf_rotation=90)
plt.title("Layer 4B: Author Semantic Clustering (Ward linkage)")
plt.tight_layout()
plt.savefig("Layer4B_author_dendrogram.png", dpi=300)
plt.close()
print("Saved: Layer4B_author_dendrogram.png")

# -----------------------------------------------------------
# 6. Heatmap of similarities
# -----------------------------------------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(sim_df, cmap="viridis", square=True)
plt.title("Layer 4B: Author-to-Author Semantic Similarity (Cosine)")
plt.tight_layout()
plt.savefig("Layer4B_similarity_heatmap.png", dpi=300)
plt.close()
print("Saved: Layer4B_similarity_heatmap.png")

# -----------------------------------------------------------
print("Done. Layer 4B semantic overlap analysis complete.")
