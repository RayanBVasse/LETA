import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------------------
# Load datasets
# ---------------------------

print("Loading author posts...")
df = pd.read_csv("Layer0_cleaned_100authors.csv")
print("Rows:", df.shape)

print("Loading topic clusters...")
clusters = pd.read_csv("Layer4_clusters_clean.csv")

# Prepare cluster dictionary
topic_keywords = (
    clusters.groupby("topic_id")["keyword"]
    .apply(list)
    .to_dict()
)

print(f"Loaded {len(topic_keywords)} topics with keyword lists.")


# ---------------------------
# Topic assignment via keyword matching
# ---------------------------

def assign_topic(text):
    if not isinstance(text, str):
        return -1

    words = text.lower().split()
    scores = {}

    for topic_id, keywords in topic_keywords.items():
        score = sum(w in words for w in keywords)
        scores[topic_id] = score

    # If all zero, no match
    if max(scores.values()) == 0:
        return -1

    return max(scores, key=scores.get)


print("Assigning topics...")
df["topic_id"] = df["text"].apply(assign_topic)

print("Topic assignment complete.")


# ---------------------------
# Sort posts per author
# ---------------------------

df = df.sort_values(["id", "post_index"])

# compute topic numeric sequence per author
results = []

# ---------------------------
# Drift detection functions
# ---------------------------

def compute_slope(x, y):
    if len(x) < 2:
        return 0
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    model = LinearRegression().fit(x, y)
    return float(model.coef_[0])

def detect_inflections(series):
    # inflection point = change in direction of slope sign
    inflections = []
    for i in range(2, len(series)):
        prev_diff = series[i-1] - series[i-2]
        curr_diff = series[i] - series[i-1]
        if prev_diff == 0:
            continue
        # sign flip means direction change
        if np.sign(prev_diff) != np.sign(curr_diff):
            inflections.append(i)
    return inflections


# ---------------------------
# Per-author drift analysis
# ---------------------------

for author, sub in df.groupby("id"):
    seq = sub["topic_id"].tolist()
    idx = list(range(len(seq)))

    # Filter invalid topic_ids
    valid_idx = [i for i, t in enumerate(seq) if t >= 0]
    if len(valid_idx) < 3:
        results.append([author, None, None, None, []])
        continue

    x = [i for i in valid_idx]
    y = [seq[i] for i in valid_idx]

    slope = compute_slope(x, y)
    volatility = np.std(np.diff(y))
    infl = detect_inflections(y)

    results.append([author, slope, volatility, len(infl), infl])


# ---------------------------
# Save results
# ---------------------------

out = pd.DataFrame(results, columns=[
    "author_id",
    "topic_slope",
    "topic_volatility",
    "num_inflections",
    "inflection_points"
])

out.to_csv("Layer4C_topic_drift_results.csv", index=False)
print("Saved: Layer4C_topic_drift_results.csv")

print("Done. Topic drift analysis completed WITHOUT embeddings.")
