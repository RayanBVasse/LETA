import pandas as pd

# ---------------------------------------------------
# INPUT FILES
# ---------------------------------------------------
KEYWORDS_FILE = "Layer4_keywords_per_author.csv"
CLUSTERS_FILE = "Layer4_clusters_clean.csv"

# ---------------------------------------------------
# LOAD FILES
# ---------------------------------------------------
print("Loading files...")

df_kw = pd.read_csv(KEYWORDS_FILE)
df_cl = pd.read_csv(CLUSTERS_FILE)

print("Keywords shape:", df_kw.shape)
print("Clusters shape:", df_cl.shape)

# Clean up possible whitespace issues
df_kw['keyword'] = df_kw['keyword'].astype(str).str.strip()
df_cl['keyword'] = df_cl['keyword'].astype(str).str.strip()

# ---------------------------------------------------
# MERGE: Assign each keyword its topic_id
# ---------------------------------------------------
print("Merging keyword → topic_id...")

df = df_kw.merge(df_cl[['keyword', 'topic_id']], on='keyword', how='left')

missing = df['topic_id'].isna().sum()
if missing > 0:
    print(f"⚠ WARNING: {missing} keywords had no topic assignment")
else:
    print("All keywords assigned to a topic.")

# ---------------------------------------------------
# COUNT: Keywords per author per topic
# ---------------------------------------------------
print("Counting topic frequencies per author...")

topic_counts = df.groupby(['author_id', 'topic_id']).size().reset_index(name='count')

# Pivot to wide format: authors × 12 topics
topic_matrix = topic_counts.pivot(index='author_id',
                                  columns='topic_id',
                                  values='count').fillna(0)

# Ensure topic columns in order
topic_matrix = topic_matrix.reindex(sorted(topic_matrix.columns), axis=1)

# ---------------------------------------------------
# NORMALIZE to distributions (sum to 1)
# ---------------------------------------------------
print("Normalizing...")
topic_distribution = topic_matrix.div(topic_matrix.sum(axis=1), axis=0)

# ---------------------------------------------------
# SAVE OUTPUTS
# ---------------------------------------------------
topic_distribution.to_csv("Layer4_topic_distribution_per_author.csv")
topic_matrix.to_csv("Layer4_topic_raw_counts_per_author.csv")

print("\nSaved:")
print(" - Layer4_topic_distribution_per_author.csv")
print(" - Layer4_topic_raw_counts_per_author.csv")

print("\nDone. Each author now has a 12-topic semantic fingerprint.")
