# LETA: Longitudinal Emotion Trajectory Analysis

Validation framework for LLM-based emotion extraction from text.

# LETA: Longitudinal Emotion Trajectory Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Validation framework for LLM-based emotion extraction from longitudinal text.

## Key Finding

AI emotion extraction measures **expressed emotion** (what appears in text), which is related to but distinct from **felt emotion** (what humans consciously report).

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Categorical overlap | 58.3% | 4.7× chance baseline |
| Dimensional correlation | r = .18 | ~3% shared variance |
| Trajectory stability | 75% | Trait-like characteristic |
| Demographic independence | 94% unexplained | Not reducible to age/gender |

## Paper

> **Expressed versus Felt: Validating Large Language Model Emotion Extraction in Longitudinal Text**
> 
> [Author]. *Behavior Research Methods* (submitted).

## Three Validations

### Validation 1: AI-Human Convergence
Compares AI-extracted emotions against self-reported emotions using the RW3D pandemic narrative dataset (n=1,152).

### Validation 2: Methodological Validity  
Tests convergent validity against SEANCE (same lexicon: r=.87) and discriminant validity against transformer classifiers (r=.12).

### Validation 3: Naturalistic Application
Examines longitudinal blog data (100 authors, 32,799 posts) for archetype stability, trajectory patterns, and demographic independence.

---

## Repository Structure

```
LETA-Validation/
├── data/
│   ├── validation1_rw3d/           # AI-human convergence results
│   ├── validation2_convergent/     # SEANCE & transformer comparisons
│   └── validation3_longitudinal/   # Blog corpus analysis
├── scripts/                        # Analysis scripts (see mapping below)
├── NRC-Emotion-Lexicon-Wordlevel-v0.92.txt
├── LETA-transformers.ipynb
└── README.md
```

## Script → Validation Mapping

Scripts retain original layer numbering from development. Here's how they map to the paper's validation structure:

| Script | Validation | Purpose |
|--------|------------|---------|
| `pillar1_rw3d_verification.py` | **Validation 1** | AI-human emotion convergence analysis |
| `leta_seance_correlation.py` | **Validation 2** | SEANCE convergent/discriminant validity |
| `pillar3_clustering_verification.py` | **Validation 3** | Emotional archetype clustering |
| `6_Layer4_semantic_clustering_OpenAI.py` | **Validation 3** | Topic modeling (LDA) |
| `7_Layer4_author_topic_distribution.py` | **Validation 3** | Topic-emotion relationships |
| `8_Layer4_author_topic_overlap.py` | **Validation 3** | Cross-author topic analysis |
| `9_Layer4_TopicDrift_no.embedding.py` | **Validation 3** | Trajectory stability analysis |
| `10_Layer5C_disentanglement_tests.py` | **Validation 3** | Demographic independence tests |

## Data Sources

| Dataset | Availability | URL |
|---------|--------------|-----|
| RW3D (Real World Worry Dataset) | Public | https://osf.io/9b85r/ |
| Blog Authorship Corpus | Public | http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm |
| NRC Emotion Lexicon | Public | https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm |

## Requirements

```
python >= 3.10
pandas >= 2.0
numpy >= 1.24
scikit-learn >= 1.3
scipy >= 1.11
gensim >= 4.3
nltk >= 3.8
```

## Quick Start

```python
# Basic emotion extraction using NRC lexicon
from scripts.utils.emotion_extraction import extract_emotions

text = "I feel worried about the future but hopeful things will improve."
emotions = extract_emotions(text)
# Returns: {'anger': 0.0, 'anticipation': 0.08, 'fear': 0.08, 'joy': 0.08, ...}
```

## Reproducibility

- All analyses use fixed random seeds (seed=42)
- Statistical tests use α = .05
- Bonferroni correction applied for multiple comparisons

## Citation
TBC

## License
MIT License - see [LICENSE](LICENSE) for details.

## Contact
rayan@living-literature.org
