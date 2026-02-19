# 10C_Layer5_disentanglement_tests.py

import pandas as pd
from scipy.stats import chi2_contingency, pearsonr, ttest_ind

# Load datasets
affect = pd.read_csv("Layer3_100_affect_clusters.csv")  # includes author_id + cluster
topics = pd.read_csv("Layer5B_author_topic_proportions.csv")

dem = pd.read_csv("authors100_full_records.csv")[["id", "age", "gender"]]
dem.rename(columns={"id": "author_id"}, inplace=True)
dem = dem[["author_id", "gender", "age"]].drop_duplicates("author_id")
# Merge
df = affect.merge(topics, on="author_id").merge(dem, on="author_id")

# Extract variables
clusters = df["cluster"]
ages = df["age"]
gender = df["gender"]

topic_cols = [c for c in df.columns if isinstance(c, str) and c.isdigit()]

# ---------------------------
# 1. Are affect clusters related to topic distribution?
# ---------------------------
contingency = pd.crosstab(df["cluster"], df[topic_cols].idxmax(axis=1))
chi2, p_chi, dof, expected = chi2_contingency(contingency)

# ---------------------------
# 2. Age correlation with emotional variance
# ---------------------------
r_age, p_age = pearsonr(df["age"], df["emotion_variance"])

# ---------------------------
# 3. Gender × cluster independence test
# ---------------------------
cont_gender = pd.crosstab(df["gender"], df["cluster"])
chi2_g, p_g, dof_g, exp_g = chi2_contingency(cont_gender)

# ---------------------------
# Save report
# ---------------------------
with open("Layer5_disentanglement_summary.txt", "w") as f:
    f.write("Layer 5 — Disentanglement Analysis\n\n")
    f.write("1. Topic-Affect Relationship (Chi-square)\n")
    f.write(f"Chi² = {chi2:.3f}, p = {p_chi:.4f}\n\n")
    f.write("2. Age–Emotion Variance Correlation\n")
    f.write(f"r = {r_age:.3f}, p = {p_age:.4f}\n\n")
    f.write("3. Gender–Affect Independence Test\n")
    f.write(f"Chi² = {chi2_g:.3f}, p = {p_g:.4f}\n")

print("Saved: Layer5_disentanglement_summary.txt")
