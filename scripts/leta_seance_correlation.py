import pandas as pd
import numpy as np
from scipy import stats

# ============================================
# LOAD DATA
# ============================================

print("Loading data...")

# Load SEANCE results
seance = pd.read_csv("seance_results.csv")
print(f"SEANCE: {len(seance)} rows")

# Load LETA post-level emotions
leta = pd.read_csv("pillar4_post_emotions.csv")
print(f"LETA: {len(leta)} rows")

# ============================================
# ALIGN DATA BY POST INDEX
# ============================================

# SEANCE filename is like "post_00000.txt" - extract index
seance['post_idx'] = seance['filename'].str.extract(r'post_(\d+)').astype(int)

# Merge on post_idx
merged = leta.merge(seance, on='post_idx', how='inner')
print(f"Matched: {len(merged)} rows")

# ============================================
# CORRELATION ANALYSIS 1: LETA vs SEANCE (Same Lexicon - NRC/EmoLex)
# ============================================

print("\n" + "="*70)
print("PART 1: LETA vs SEANCE EmoLex (Same NRC Lexicon)")
print("="*70)

# Direct mappings - BOTH use NRC lexicon
emolex_pairs = [
    ('score_anger', 'Anger_EmoLex'),
    ('score_anticipation', 'Anticipation_EmoLex'),
    ('score_disgust', 'Disgust_EmoLex'),
    ('score_fear', 'Fear_EmoLex'),
    ('score_joy', 'Joy_EmoLex'),
    ('score_sadness', 'Sadness_EmoLex'),
    ('score_surprise', 'Surprise_EmoLex'),
    ('score_trust', 'Trust_EmoLex')
]

emolex_results = []
for leta_col, seance_col in emolex_pairs:
    mask = merged[leta_col].notna() & merged[seance_col].notna()
    if mask.sum() > 100:
        r, p = stats.pearsonr(merged.loc[mask, leta_col], merged.loc[mask, seance_col])
        emotion = leta_col.replace('score_', '')
        emolex_results.append({
            'emotion': emotion,
            'pearson_r': round(r, 3),
            'p_value': f"{p:.2e}",
            'n': mask.sum()
        })
        print(f"{emotion:15} r = {r:.3f}  (p = {p:.2e}, n = {mask.sum()})")

emolex_df = pd.DataFrame(emolex_results)
print(f"\nMean correlation (EmoLex): r = {emolex_df['pearson_r'].mean():.3f}")

# ============================================
# CORRELATION ANALYSIS 2: LETA vs SEANCE GALC (Different Lexicon)
# ============================================

print("\n" + "="*70)
print("PART 2: LETA vs SEANCE GALC (Different Lexicon - Geneva)")
print("="*70)

galc_pairs = [
    ('score_anger', 'Anger_GALC'),
    ('score_disgust', 'Disgust_GALC'),
    ('score_fear', 'Fear_GALC'),
    ('score_joy', 'Joy_GALC'),
    ('score_sadness', 'Sadness_GALC'),
    ('score_surprise', 'Surprise_GALC')
]

galc_results = []
for leta_col, seance_col in galc_pairs:
    mask = merged[leta_col].notna() & merged[seance_col].notna()
    if mask.sum() > 100:
        r, p = stats.pearsonr(merged.loc[mask, leta_col], merged.loc[mask, seance_col])
        emotion = leta_col.replace('score_', '')
        galc_results.append({
            'emotion': emotion,
            'pearson_r': round(r, 3),
            'p_value': f"{p:.2e}",
            'n': mask.sum()
        })
        print(f"{emotion:15} r = {r:.3f}  (p = {p:.2e}, n = {mask.sum()})")

galc_df = pd.DataFrame(galc_results)
print(f"\nMean correlation (GALC): r = {galc_df['pearson_r'].mean():.3f}")

# ============================================
# CORRELATION ANALYSIS 3: LETA vs VADER Sentiment
# ============================================

print("\n" + "="*70)
print("PART 3: LETA Valence vs VADER Sentiment")
print("="*70)

# Compute LETA valence (positive - negative)
merged['leta_valence'] = (merged['score_joy'] + merged['score_trust'] + merged['score_anticipation']) - \
                         (merged['score_anger'] + merged['score_fear'] + merged['score_sadness'] + merged['score_disgust'])

mask = merged['leta_valence'].notna() & merged['vader_compound'].notna()
r, p = stats.pearsonr(merged.loc[mask, 'leta_valence'], merged.loc[mask, 'vader_compound'])
print(f"LETA valence vs VADER compound: r = {r:.3f} (p = {p:.2e}, n = {mask.sum()})")

# ============================================
# SUMMARY
# ============================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

summary = {
    'comparison': ['LETA vs EmoLex (same lexicon)', 'LETA vs GALC (different lexicon)', 'LETA vs VADER (sentiment)'],
    'mean_r': [emolex_df['pearson_r'].mean(), galc_df['pearson_r'].mean(), round(r, 3)],
    'interpretation': ['', '', '']
}

for i, mean_r in enumerate(summary['mean_r']):
    if mean_r > 0.7:
        summary['interpretation'][i] = 'Strong'
    elif mean_r > 0.5:
        summary['interpretation'][i] = 'Moderate'
    elif mean_r > 0.3:
        summary['interpretation'][i] = 'Weak-Moderate'
    else:
        summary['interpretation'][i] = 'Weak'

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))

# ============================================
# SAVE RESULTS
# ============================================

emolex_df.to_csv("leta_seance_emolex_correlations.csv", index=False)
galc_df.to_csv("leta_seance_galc_correlations.csv", index=False)
summary_df.to_csv("leta_seance_summary.csv", index=False)

print("\n" + "="*70)
print("FILES SAVED:")
print("  - leta_seance_emolex_correlations.csv")
print("  - leta_seance_galc_correlations.csv")
print("  - leta_seance_summary.csv")
print("="*70)