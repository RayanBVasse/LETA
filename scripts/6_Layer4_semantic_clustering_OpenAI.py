# 6_Layer4_semantic_clustering_OpenAI.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from openai import OpenAI

client = OpenAI(api_key="YOUR KEY HERE")


INPUT_FILE = "Layer4_keywords_masterlist.csv"
OUTPUT_FILE = "Layer4_keyword_clusters.csv"

# -------------------------------
# 1. Load data
# -------------------------------

df = pd.read_csv(INPUT_FILE)
print("Loaded keywords:", df.shape)

keywords = df["keyword"].tolist()

# -------------------------------
# 2. Embed keywords using OpenAI
# -------------------------------

def embed(text):
    """Get OpenAI embedding for a single keyword."""
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return emb.data[0].embedding

print("Embedding keywords...")
df["embedding"] = df["keyword"].apply(embed)

# Convert list of embeddings into matrix
import numpy as np
X = np.vstack(df["embedding"].values)

# Optional scaling
X_scaled = StandardScaler().fit_transform(X)

# -------------------------------
# 3. Cluster (Choose number of topics)
# -------------------------------

K = 12   # You can adjust this after inspection
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
df["topic_id"] = kmeans.fit_predict(X_scaled)

# -------------------------------
# 4. Save results
# -------------------------------

df.to_csv(OUTPUT_FILE, index=False)
print("Saved:", OUTPUT_FILE)
print("Unique topics:", df.topic_id.nunique())
