# ============================================
# PILLAR 3: BLOG CORPUS CLUSTERING & TRAJECTORY VERIFICATION
# ============================================
# This script verifies the claimed statistics:
# - k=4 archetypes with silhouette scores
# - 73% stable trajectories
# - 46% unexplained variance
# ============================================

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================
# STEP 1: LOAD DATA
# ============================================

print("="*70)
print("PILLAR 3: CLUSTERING & TRAJECTORY VERIFICATION")
print("="*70)

# Load post-level emotions
posts = pd.read_csv("pillar4_post_emotions.csv")
print(f"\nLoaded post-level emotions: {len(posts)} posts")

# Load author-level aggregates (if available)
try:
    authors = pd.read_csv("pillar4_layer1_40results.csv")
    print(f"Loaded author-level aggregates: {len(authors)} authors")
    HAS_AUTHOR_LEVEL = True
except:
    print("Author-level file not found - will compute from posts")
    HAS_AUTHOR_LEVEL = False

print(f"\nPost columns: {posts.columns.tolist()}")

# ============================================
# STEP 2: COMPUTE AUTHOR-LEVEL EMOTION PROFILES
# ============================================

print("\n" + "="*70)
print("STEP 2: AUTHOR-LEVEL EMOTION PROFILES")
print("="*70)

emotion_cols = ['score_anger', 'score_anticipation', 'score_disgust', 'score_fear', 
                'score_joy', 'score_sadness', 'score_surprise', 'score_trust']

# Check which columns exist
available_emotions = [col for col in emotion_cols if col in posts.columns]
print(f"Available emotion columns: {available_emotions}")

# Compute author-level means
author_emotions = posts.groupby('author_id')[available_emotions].mean()
print(f"\nComputed profiles for {len(author_emotions)} authors")

# Add metadata
if 'gender' in posts.columns:
    author_meta = posts.groupby('author_id').agg({
        'gender': 'first',
        'age': 'first',
        'topic': 'first'
    })
    author_emotions = author_emotions.join(author_meta)

print("\nAuthor emotion profile statistics:")
print(author_emotions[available_emotions].describe())

# ============================================
# STEP 3: CLUSTERING ANALYSIS
# ============================================

print("\n" + "="*70)
print("STEP 3: CLUSTERING ANALYSIS")
print("="*70)

# Prepare data for clustering
X = author_emotions[available_emotions].values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Test multiple k values
print("\nSilhouette scores for different k values:")
print("-" * 40)

silhouette_scores = {}
for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels)
    silhouette_scores[k] = sil_score
    print(f"  k={k}: silhouette = {sil_score:.3f}")

# Find optimal k
optimal_k = max(silhouette_scores, key=silhouette_scores.get)
print(f"\nOptimal k by silhouette: k={optimal_k} (score={silhouette_scores[optimal_k]:.3f})")

# Final clustering with k=4 (as claimed in manuscript)
print("\n" + "-"*40)
print("FINAL CLUSTERING WITH k=4:")
print("-"*40)

kmeans_4 = KMeans(n_clusters=4, random_state=42, n_init=10)
author_emotions['cluster'] = kmeans_4.fit_predict(X_scaled)
sil_4 = silhouette_score(X_scaled, author_emotions['cluster'])

print(f"Silhouette score (k=4): {sil_4:.3f}")
print(f"\nCluster sizes:")
print(author_emotions['cluster'].value_counts().sort_index())

# Cluster profiles
print("\nCluster emotion profiles (mean):")
cluster_profiles = author_emotions.groupby('cluster')[available_emotions].mean()
print(cluster_profiles.round(3))

# ============================================
# STEP 4: ARCHETYPE INTERPRETATION
# ============================================

print("\n" + "="*70)
print("STEP 4: ARCHETYPE INTERPRETATION")
print("="*70)

# Identify dominant emotions for each cluster
for cluster in sorted(author_emotions['cluster'].unique()):
    profile = cluster_profiles.loc[cluster]
    top_emotions = profile.nlargest(3)
    n_authors = (author_emotions['cluster'] == cluster).sum()
    pct = n_authors / len(author_emotions) * 100
    
    print(f"\nCluster {cluster} ({n_authors} authors, {pct:.0f}%):")
    top_emotion_strs = []
    for e, v in top_emotions.items():
        emotion_name = e.replace("score_", "")
        top_emotion_strs.append(f"{emotion_name}: {v:.3f}")
    print(f"  Top emotions: {', '.join(top_emotion_strs)}")
# ============================================
# STEP 5: TRAJECTORY ANALYSIS
# ============================================

print("\n" + "="*70)
print("STEP 5: TRAJECTORY ANALYSIS")
print("="*70)

# Sort posts by author and date
if 'post_idx' in posts.columns:
    posts_sorted = posts.sort_values(['author_id', 'post_idx'])
else:
    posts_sorted = posts.sort_values('author_id')

# Compute trajectory metrics per author
def compute_trajectory_metrics(author_posts, emotion_cols):
    """Compute trajectory stability metrics for an author."""
    if len(author_posts) < 5:
        return None
    
    metrics = {}
    for col in emotion_cols:
        if col not in author_posts.columns:
            continue
        values = author_posts[col].dropna().values
        if len(values) < 5:
            continue
        
        # Coefficient of variation (stability measure)
        cv = np.std(values) / (np.mean(values) + 1e-10)
        metrics[f'{col}_cv'] = cv
        
        # Trend (slope of linear fit)
        x = np.arange(len(values))
        slope, _, r_value, _, _ = stats.linregress(x, values)
        metrics[f'{col}_slope'] = slope
        metrics[f'{col}_r2'] = r_value**2
    
    return metrics

print("Computing trajectory metrics per author...")

trajectory_results = []
for author_id, author_posts in posts.groupby('author_id'):
    metrics = compute_trajectory_metrics(author_posts, available_emotions)
    if metrics:
        metrics['author_id'] = author_id
        trajectory_results.append(metrics)

trajectory_df = pd.DataFrame(trajectory_results)
print(f"Computed trajectories for {len(trajectory_df)} authors")

# Classify trajectory types
def classify_trajectory(row, emotion_cols):
    """Classify trajectory as stable, increasing, decreasing, or volatile."""
    slopes = [row.get(f'{col}_slope', 0) for col in emotion_cols if f'{col}_slope' in row]
    cvs = [row.get(f'{col}_cv', 0) for col in emotion_cols if f'{col}_cv' in row]
    
    if not slopes or not cvs:
        return 'unknown'
    
    mean_abs_slope = np.mean(np.abs(slopes))
    mean_cv = np.mean(cvs)
    
    # Thresholds (adjustable)
    if mean_cv < 0.5 and mean_abs_slope < 0.001:
        return 'stable'
    elif mean_cv > 1.0:
        return 'volatile'
    elif np.mean(slopes) > 0.0005:
        return 'increasing'
    elif np.mean(slopes) < -0.0005:
        return 'decreasing'
    else:
        return 'stable'

trajectory_df['trajectory_type'] = trajectory_df.apply(
    lambda row: classify_trajectory(row, available_emotions), axis=1)

print("\nTrajectory type distribution:")
trajectory_dist = trajectory_df['trajectory_type'].value_counts()
print(trajectory_dist)

stable_pct = trajectory_dist.get('stable', 0) / len(trajectory_df) * 100
print(f"\n% Stable trajectories: {stable_pct:.1f}%")

# ============================================
# STEP 6: VARIANCE DECOMPOSITION
# ============================================

print("\n" + "="*70)
print("STEP 6: VARIANCE DECOMPOSITION")
print("="*70)

# Check if we have demographic data
has_demographics = all(col in author_emotions.columns for col in ['gender', 'age'])
has_topic = 'topic' in author_emotions.columns

if has_demographics or has_topic:
    from sklearn.preprocessing import LabelEncoder
    
    # Prepare predictors
    predictors = []
    predictor_names = []
    
    if 'gender' in author_emotions.columns:
        le_gender = LabelEncoder()
        author_emotions['gender_encoded'] = le_gender.fit_transform(author_emotions['gender'].fillna('unknown'))
        predictors.append('gender_encoded')
        predictor_names.append('gender')
    
    if 'age' in author_emotions.columns:
        author_emotions['age_clean'] = pd.to_numeric(author_emotions['age'], errors='coerce').fillna(author_emotions['age'].median() if 'age' in author_emotions else 25)
        predictors.append('age_clean')
        predictor_names.append('age')
    
    if 'topic' in author_emotions.columns:
        le_topic = LabelEncoder()
        author_emotions['topic_encoded'] = le_topic.fit_transform(author_emotions['topic'].fillna('unknown'))
        predictors.append('topic_encoded')
        predictor_names.append('topic')
    
    # Compute R² for each emotion
    print("\nVariance explained by demographics/topic:")
    print("-" * 50)
    
    r2_results = []
    for emotion in available_emotions:
        y = author_emotions[emotion].values
        X_pred = author_emotions[predictors].values
        
        # Simple OLS
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_pred, y)
        r2 = model.score(X_pred, y)
        
        r2_results.append({'emotion': emotion, 'r2_explained': r2, 'r2_unexplained': 1 - r2})
        print(f"  {emotion.replace('score_', ''):15}: R² = {r2:.3f} (unexplained: {1-r2:.3f})")
    
    r2_df = pd.DataFrame(r2_results)
    mean_r2 = r2_df['r2_explained'].mean()
    mean_unexplained = r2_df['r2_unexplained'].mean()
    
    print(f"\n  Mean R² explained: {mean_r2:.3f}")
    print(f"  Mean unexplained variance: {mean_unexplained:.3f} ({mean_unexplained*100:.1f}%)")
else:
    print("Demographic/topic data not available for variance decomposition")
    mean_unexplained = None

# ============================================
# STEP 7: MANUSCRIPT VERIFICATION
# ============================================

print("\n" + "="*70)
print("STEP 7: MANUSCRIPT CLAIM VERIFICATION")
print("="*70)

print("\nMANUSCRIPT CLAIMS:")
print("  - k=4 archetypes")
print("  - Silhouette score ~0.205")
print("  - 73% stable trajectories")
print("  - 46% unexplained variance")

print(f"\nCOMPUTED VALUES:")
print(f"  - Optimal k by silhouette: k={optimal_k}")
print(f"  - Silhouette score (k=4): {sil_4:.3f}")
print(f"  - Stable trajectories: {stable_pct:.1f}%")
if mean_unexplained:
    print(f"  - Unexplained variance: {mean_unexplained*100:.1f}%")

print(f"\nVERIFICATION:")

# k=4 check
if optimal_k == 4:
    print(f"  ✓ k=4 is optimal by silhouette")
else:
    print(f"  ⚠ Optimal k={optimal_k}, but k=4 claimed - silhouette for k=4 is {sil_4:.3f}")

# Silhouette check
if abs(sil_4 - 0.205) < 0.05:
    print(f"  ✓ Silhouette ~0.205 verified")
else:
    print(f"  ✗ Silhouette ({sil_4:.3f}) differs from claimed 0.205 - UPDATE MANUSCRIPT")

# Stable trajectories check
if abs(stable_pct - 73) < 10:
    print(f"  ✓ Stable trajectories ~73% verified")
else:
    print(f"  ✗ Stable trajectories ({stable_pct:.1f}%) differs from claimed 73% - UPDATE MANUSCRIPT")

# Unexplained variance check
if mean_unexplained and abs(mean_unexplained*100 - 46) < 10:
    print(f"  ✓ Unexplained variance ~46% verified")
elif mean_unexplained:
    print(f"  ✗ Unexplained variance ({mean_unexplained*100:.1f}%) differs from claimed 46% - UPDATE MANUSCRIPT")

# ============================================
# STEP 8: SAVE RESULTS
# ============================================

print("\n" + "="*70)
print("STEP 8: SAVING RESULTS")
print("="*70)

# Save clustering results
author_emotions.to_csv("pillar3_author_clusters.csv", index=True)

# Save silhouette scores
sil_df = pd.DataFrame({'k': list(silhouette_scores.keys()), 
                       'silhouette': list(silhouette_scores.values())})
sil_df.to_csv("pillar3_silhouette_scores.csv", index=False)

# Save trajectory results
trajectory_df.to_csv("pillar3_trajectories.csv", index=False)

# Save variance decomposition
if 'r2_df' in dir():
    r2_df.to_csv("pillar3_variance_decomposition.csv", index=False)

# Save summary
summary = {
    'n_authors': len(author_emotions),
    'n_posts': len(posts),
    'optimal_k': optimal_k,
    'silhouette_k4': sil_4,
    'pct_stable_trajectories': stable_pct,
    'mean_unexplained_variance': mean_unexplained * 100 if mean_unexplained else None
}
summary_df = pd.DataFrame([summary])
summary_df.to_csv("pillar3_verification_summary.csv", index=False)

print("\nResults saved to:")
print("  - pillar3_author_clusters.csv")
print("  - pillar3_silhouette_scores.csv")
print("  - pillar3_trajectories.csv")
print("  - pillar3_variance_decomposition.csv")
print("  - pillar3_verification_summary.csv")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)
