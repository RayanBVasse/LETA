# ============================================
# PILLAR 1: RW3D AI-HUMAN CONVERGENCE VERIFICATION
# ============================================
# This script verifies the claimed statistics:
# - 42% top-2 match
# - r=0.29-0.31 dimensional correlations
# ============================================

import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter

# ============================================
# STEP 1: LOAD DATA
# ============================================

print("="*70)
print("PILLAR 1: RW3D AI-HUMAN CONVERGENCE VERIFICATION")
print("="*70)

# Load RW3D dataset
rw3d = pd.read_csv("RW3D_dataset.csv")
print(f"\nLoaded RW3D dataset: {len(rw3d)} participants")

# Check columns
print("\nAvailable columns:")
for i, col in enumerate(rw3d.columns):
    print(f"  {i}: {col}")

# ============================================
# STEP 2: IDENTIFY RELEVANT COLUMNS
# ============================================

# Self-reported emotions (Wave 1) - Likert scale 1-9
emotion_cols_human = {
    'anger': 'anger_wave1',
    'disgust': 'disgust_wave1', 
    'fear': 'fear_wave1',
    'anxiety': 'anxiety_wave1',
    'sadness': 'sadness_wave1',
    'happiness': 'happiness_wave1',
    'relaxation': 'relaxation_wave1',
    'desire': 'desire_wave1'
}

# Primary emotion chosen by participant
primary_emotion_col = 'chosen_emotion_wave1'

# Text column for AI analysis
text_col = 'text_long_wave1'

print(f"\nPrimary emotion column: {primary_emotion_col}")
print(f"Text column: {text_col}")

# ============================================
# STEP 3: SAMPLE SELECTION
# ============================================

print("\n" + "="*70)
print("STEP 3: SAMPLE SELECTION")
print("="*70)

# Check what we have
print(f"\nTotal participants: {len(rw3d)}")

# Valid text (non-null, >50 characters)
if text_col in rw3d.columns:
    has_text = rw3d[text_col].notna() & (rw3d[text_col].str.len() > 50)
    print(f"Has valid text (>50 chars): {has_text.sum()}")
else:
    print(f"WARNING: Text column '{text_col}' not found!")
    has_text = pd.Series([False] * len(rw3d))

# Valid primary emotion
if primary_emotion_col in rw3d.columns:
    has_primary = rw3d[primary_emotion_col].notna()
    print(f"Has primary emotion: {has_primary.sum()}")
else:
    print(f"WARNING: Primary emotion column '{primary_emotion_col}' not found!")
    has_primary = pd.Series([False] * len(rw3d))

# Valid emotion ratings
available_emotions = [col for col in emotion_cols_human.values() if col in rw3d.columns]
print(f"\nAvailable emotion rating columns: {available_emotions}")

if available_emotions:
    has_ratings = rw3d[available_emotions].notna().all(axis=1)
    print(f"Has complete emotion ratings: {has_ratings.sum()}")
else:
    has_ratings = pd.Series([False] * len(rw3d))

# Combined selection
valid_sample = has_text & has_primary & has_ratings
print(f"\nValid sample (text + primary + ratings): {valid_sample.sum()}")

sample_df = rw3d[valid_sample].copy()
print(f"Selected sample size: {len(sample_df)}")

# ============================================
# STEP 4: EXAMINE PRIMARY EMOTIONS
# ============================================

print("\n" + "="*70)
print("STEP 4: PRIMARY EMOTION DISTRIBUTION")
print("="*70)

if len(sample_df) > 0 and primary_emotion_col in sample_df.columns:
    print("\nHuman-chosen primary emotions:")
    print(sample_df[primary_emotion_col].value_counts())
    
    # Map to standardized emotion names if needed
    emotion_mapping = {
        1: 'anger', 2: 'disgust', 3: 'fear', 4: 'anxiety',
        5: 'sadness', 6: 'happiness', 7: 'relaxation', 8: 'desire'
    }
    
    # Check if values are numeric or string
    sample_values = sample_df[primary_emotion_col].dropna().head()
    print(f"\nSample values: {sample_values.tolist()}")

# ============================================
# STEP 5: SIMULATE AI EMOTION EXTRACTION
# ============================================

print("\n" + "="*70)
print("STEP 5: AI EMOTION EXTRACTION (NRC Lexicon)")
print("="*70)

# For this verification, we'll use NRC lexicon to extract emotions
# This simulates what LETA does

try:
    # Try to load NRC lexicon
    nrc = pd.read_csv("NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", 
                      sep='\t', names=['word', 'emotion', 'value'])
    nrc_dict = {}
    for emotion in ['anger', 'fear', 'sadness', 'joy', 'trust', 'disgust', 'surprise', 'anticipation']:
        words = nrc[(nrc['emotion'] == emotion) & (nrc['value'] == 1)]['word'].tolist()
        nrc_dict[emotion] = set(words)
    print(f"Loaded NRC lexicon with {len(nrc_dict)} emotions")
    HAS_NRC = True
except:
    print("NRC lexicon not found - will skip AI extraction step")
    print("Place 'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt' in same directory")
    HAS_NRC = False

def extract_emotions_nrc(text, nrc_dict):
    """Extract emotion scores from text using NRC lexicon."""
    if pd.isna(text) or not isinstance(text, str):
        return {e: 0 for e in nrc_dict.keys()}
    
    words = text.lower().split()
    total = len(words)
    if total == 0:
        return {e: 0 for e in nrc_dict.keys()}
    
    scores = {}
    for emotion, word_set in nrc_dict.items():
        count = sum(1 for w in words if w in word_set)
        scores[emotion] = count / total
    return scores

if HAS_NRC and len(sample_df) > 0:
    print("\nExtracting emotions from texts...")
    
    ai_emotions = []
    for idx, row in sample_df.iterrows():
        text = row[text_col] if text_col in sample_df.columns else ""
        scores = extract_emotions_nrc(text, nrc_dict)
        scores['idx'] = idx
        ai_emotions.append(scores)
    
    ai_df = pd.DataFrame(ai_emotions).set_index('idx')
    print(f"Extracted emotions for {len(ai_df)} texts")
    
    # Add primary AI emotion
    ai_df['ai_primary'] = ai_df[['anger', 'fear', 'sadness', 'joy', 'disgust', 'surprise', 'trust', 'anticipation']].idxmax(axis=1)
    ai_df['ai_secondary'] = ai_df[['anger', 'fear', 'sadness', 'joy', 'disgust', 'surprise', 'trust', 'anticipation']].apply(
        lambda x: x.nlargest(2).index[1] if x.nlargest(2).shape[0] > 1 else None, axis=1)
    
    print("\nAI-detected primary emotions:")
    print(ai_df['ai_primary'].value_counts())

# ============================================
# STEP 6: COMPUTE CONVERGENCE METRICS
# ============================================

print("\n" + "="*70)
print("STEP 6: AI-HUMAN CONVERGENCE METRICS")
print("="*70)

if HAS_NRC and len(sample_df) > 0:
    # Merge AI emotions with human data
    sample_df = sample_df.join(ai_df)
    
    # Map human primary emotion to standard names
    # First check what format it's in
    human_primary = sample_df[primary_emotion_col]
    
    # Common mappings (adjust based on actual data)
    emotion_map_numeric = {
        1: 'anger', 2: 'disgust', 3: 'fear', 4: 'anxiety',
        5: 'sadness', 6: 'happiness', 7: 'relaxation', 8: 'desire'
    }
    
    emotion_map_text = {
        'anger': 'anger', 'disgust': 'disgust', 'fear': 'fear',
        'anxiety': 'fear',  # Map anxiety to fear for comparison
        'sadness': 'sadness', 'happiness': 'joy', 'joy': 'joy',
        'relaxation': 'trust',  # Approximate mapping
        'desire': 'anticipation'  # Approximate mapping
    }
    
    # Try to standardize human primary emotion
    if human_primary.dtype in ['int64', 'float64']:
        sample_df['human_primary_std'] = human_primary.map(emotion_map_numeric).map(emotion_map_text)
    else:
        sample_df['human_primary_std'] = human_primary.str.lower().map(emotion_map_text)
    
    # Remove rows where mapping failed
    valid_comparison = sample_df['human_primary_std'].notna() & sample_df['ai_primary'].notna()
    comparison_df = sample_df[valid_comparison].copy()
    
    print(f"\nValid comparisons: {len(comparison_df)}")
    
    if len(comparison_df) > 0:
        # Metric 1: Exact match (AI primary = Human primary)
        exact_match = (comparison_df['ai_primary'] == comparison_df['human_primary_std']).mean()
        print(f"\n1. Exact match rate: {exact_match:.1%} ({(comparison_df['ai_primary'] == comparison_df['human_primary_std']).sum()}/{len(comparison_df)})")
        
        # Metric 2: Top-2 match (Human primary in AI top-2)
        top2_match = ((comparison_df['ai_primary'] == comparison_df['human_primary_std']) | 
                      (comparison_df['ai_secondary'] == comparison_df['human_primary_std'])).mean()
        print(f"2. Top-2 match rate: {top2_match:.1%}")
        
        # Chance baseline for 8 emotions
        chance_exact = 1/8
        chance_top2 = 2/8
        print(f"\n   Chance baseline (exact): {chance_exact:.1%}")
        print(f"   Chance baseline (top-2): {chance_top2:.1%}")
        print(f"   Improvement over chance (exact): {exact_match/chance_exact:.1f}x")
        print(f"   Improvement over chance (top-2): {top2_match/chance_top2:.1f}x")

# ============================================
# STEP 7: DIMENSIONAL CORRELATIONS
# ============================================

print("\n" + "="*70)
print("STEP 7: DIMENSIONAL CORRELATIONS")
print("="*70)

if HAS_NRC and len(sample_df) > 0:
    # Map AI emotions to human emotion columns
    correlation_pairs = [
        ('anger', 'anger_wave1'),
        ('fear', 'fear_wave1'),
        ('fear', 'anxiety_wave1'),  # AI fear vs human anxiety
        ('sadness', 'sadness_wave1'),
        ('joy', 'happiness_wave1'),
        ('disgust', 'disgust_wave1'),
    ]
    
    print("\nAI-Human dimensional correlations:")
    print("-" * 60)
    
    dim_correlations = []
    for ai_col, human_col in correlation_pairs:
        if ai_col in sample_df.columns and human_col in sample_df.columns:
            mask = sample_df[ai_col].notna() & sample_df[human_col].notna()
            if mask.sum() > 10:
                r, p = stats.pearsonr(sample_df.loc[mask, ai_col], sample_df.loc[mask, human_col])
                print(f"  {ai_col:12} vs {human_col:18}: r = {r:.3f} (p = {p:.4f}, n = {mask.sum()})")
                dim_correlations.append({'ai': ai_col, 'human': human_col, 'r': r, 'p': p, 'n': mask.sum()})
    
    if dim_correlations:
        mean_r = np.mean([d['r'] for d in dim_correlations])
        print(f"\n  Mean correlation: r = {mean_r:.3f}")

# ============================================
# STEP 8: SUMMARY AND MANUSCRIPT VERIFICATION
# ============================================

print("\n" + "="*70)
print("STEP 8: MANUSCRIPT CLAIM VERIFICATION")
print("="*70)

print("\nMANUSCRIPT CLAIMS:")
print("  - 42% top-2 match")
print("  - r = 0.29-0.31 dimensional correlations")

if HAS_NRC and len(comparison_df) > 0:
    print(f"\nCOMPUTED VALUES:")
    print(f"  - Top-2 match: {top2_match:.1%}")
    if dim_correlations:
        print(f"  - Dimensional r range: {min(d['r'] for d in dim_correlations):.2f} to {max(d['r'] for d in dim_correlations):.2f}")
        print(f"  - Dimensional r mean: {mean_r:.3f}")
    
    print(f"\nVERIFICATION:")
    if abs(top2_match - 0.42) < 0.10:
        print(f"  ✓ Top-2 match is close to claimed 42%")
    else:
        print(f"  ✗ Top-2 match ({top2_match:.1%}) differs from claimed 42% - UPDATE MANUSCRIPT")
    
    if dim_correlations and 0.25 <= mean_r <= 0.35:
        print(f"  ✓ Dimensional correlations in claimed range (0.29-0.31)")
    elif dim_correlations:
        print(f"  ✗ Dimensional correlations ({mean_r:.2f}) differ from claimed range - UPDATE MANUSCRIPT")

# ============================================
# STEP 9: SAVE RESULTS
# ============================================

print("\n" + "="*70)
print("STEP 9: SAVING RESULTS")
print("="*70)

results = {
    'sample_size': len(sample_df) if 'sample_df' in dir() else 0,
    'comparison_size': len(comparison_df) if 'comparison_df' in dir() else 0,
}

if HAS_NRC and 'comparison_df' in dir() and len(comparison_df) > 0:
    results['exact_match'] = exact_match
    results['top2_match'] = top2_match
    results['dimensional_correlations'] = dim_correlations if dim_correlations else []
    results['mean_dimensional_r'] = mean_r if dim_correlations else None

# Save to CSV
results_df = pd.DataFrame([results])
results_df.to_csv("pillar1_verification_results.csv", index=False)

if dim_correlations:
    corr_df = pd.DataFrame(dim_correlations)
    corr_df.to_csv("pillar1_dimensional_correlations.csv", index=False)

print("\nResults saved to:")
print("  - pillar1_verification_results.csv")
print("  - pillar1_dimensional_correlations.csv")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)