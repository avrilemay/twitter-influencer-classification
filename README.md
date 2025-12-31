# Tweets Classification: Influencer vs Observer -- 1st place

A stacking ensemble approach for predicting Twitter user social roles. This pipeline combines **LightGBM** for structured features, fine-tuned **CamemBERTav2** for French text analysis, and **HistGradientBoosting** as a meta-learner.

> **Final Score:** 88.0% accuracy on Kaggle public leaderboard

## Overview

This project classifies Twitter users as either **Influencers** (content creators reaching large audiences) or **Observers** (users who engage through replies and balanced interactions).

### Key Insights

| Model | OOF Accuracy | Kaggle Score |
|-------|--------------|--------------|
| LightGBM (structural features) | 87.1% | 87.3% |
| CamemBERTav2 (text analysis) | 87.5% | 87.7% |
| **Meta-Model (HGB stacking)** | **87.7%** | **88.0%** |

## Methodology

### User-Level Aggregation

Since no explicit user ID is provided, we use `user.created_at` as a unique identifier. This enables:
- Aggregation of behavioral features across tweets
- Proper stratified cross-validation without data leakage
- Construction of comprehensive user profiles

### Feature Engineering

**1,373 user-level features** derived from 343 tweet-level features:

| Category | Features | Examples |
|----------|----------|----------|
| Temporal | 8 | Account age, hour, weekday |
| Text Content | 4 | Length, hashtags, mentions |
| User Engagement | 12 | Followers, friends, statuses |
| Profile Metadata | 23 | Verified flag, RGB colors |
| Entity Counts | 25 | Hashtags, URLs in entities |
| Quoted Status | 22 | Quoted user metrics |
| + more | 249 | Generic extractions |

### Transductive Feature Engineering

We bridge train/test sets by matching users appearing in `quoted_status` metadata, recovering valuable engagement metrics like `followers_count` and `friends_count`.

### User Cards for Text Analysis

Instead of classifying individual tweets, we build **user cards**: text summaries combining bio, stats, and sampled tweets. Users with 5+ tweets get 2 cards for data augmentation.

## Project Structure

```
Gameover/
├── README.md
├── Report_Gameover.pdf
└── code/
    ├── train.jsonl                 # ← training data (not included)
    ├── test_kaggle.jsonl           # ← test data (not included)
    ├── requirements.txt
    ├── 1_lightgbm.ipynb            # Structural features + LightGBM
    ├── 2_1_user_cards.ipynb        # User card generation
    ├── 2_2_encoder.ipynb           # CamemBERTav2 fine-tuning (GPU required)
    ├── 3_meta_hgb.ipynb            # Meta-model stacking
    ├── intermediate/               # Auto-generated files
    │   ├── oof_lgbm.csv
    │   ├── test_lgbm.csv
    │   ├── lgbm_features_*.csv
    │   ├── user_cards_*.csv
    │   ├── oof_camembert.csv
    │   └── test_camembert.csv
    └── submission.csv              # Final predictions
```

## Requirements

- Python 3.10+
- **GPU with CUDA** (for notebook `2_2_encoder.ipynb`)

```bash
pip install -r requirements.txt
```

Dependencies: `lightgbm`, `pandas`, `scikit-learn`, `torch`, `transformers`

## Usage

### Step 1: Prepare Data

Place `train.jsonl` and `test_kaggle.jsonl` in the `code/` folder.

### Step 2: Run Notebooks in Order

```bash
cd code
jupyter notebook
```

| Notebook | Description | Output |
|----------|-------------|--------|
| `1_lightgbm.ipynb` | Trains LightGBM on 1,373 features | `oof_lgbm.csv`, `test_lgbm.csv` |
| `2_1_user_cards.ipynb` | Generates user text summaries | `user_cards_*.csv` |
| `2_2_encoder.ipynb` | Fine-tunes CamemBERTav2 (**~90 min on A100**) | `oof_camembert.csv`, `test_camembert.csv` |
| `3_meta_hgb.ipynb` | Stacks models with HGB | `submission.csv` |

## Key Findings

### Behavioral Differences

**Influencers:**
- Older accounts (median: 7.7 years vs 4.2 years)
- Higher listed count (median: 27 vs 1)
- Low reply rate (median: 0% replies)

**Observers:**
- Higher reply engagement (median: 40% replies)
- More balanced follower/friend ratios
- Newer accounts on average

### Model Complementarity

LightGBM captures **behavioral patterns** while CamemBERTav2 captures **linguistic signatures**. The stacking approach leverages both, achieving gains over individual models.

## Authors

- **Avrile Floro** — Institut Polytechnique de Paris
- **Falguny Barua Ema** — Institut Polytechnique de Paris  
- **Saurabh Mishra** — Institut Polytechnique de Paris

---

*Project developed for CSC_51054_EP Deep Learning Data Challenge, December 2025*
