# Slogan Effectiveness Analyzer

> A Python framework for analyzing and scoring marketing slogans using NLP, sentiment analysis, and unsupervised machine learning.

## Overview

This project provides a data-driven approach to analyze marketing slogans across 6 quantitative dimensions, producing a composite effectiveness score and thematic clustering across 19 industries.

**Dataset:** 3,239 unique slogans scraped from sloganlist.com. Categories with fewer than 30 slogans are excluded from cross-industry comparisons to avoid statistically unreliable results.

### Scoring dimensions

| Dimension | Weight | Description |
|-----------|--------|-------------|
| **Memorability** | 25% | Alliteration, rhyme, syllabic rhythm, word repetition |
| **Emotion** | 20% | Sentiment intensity × subjectivity (VADER + TextBlob) |
| **Length** | 15% | Optimal range: 3–6 words, 20–40 characters |
| **Action** | 15% | Presence and position of power verbs (imperative form) |
| **Originality** | 15% | Penalty system for corporate buzzwords and clichés |
| **Personal Engagement** | 10% | Use of "you/your" or "we/our" |

Thematic clustering (TF-IDF + K-Means, 10 groups) is computed separately and does not affect the composite score.

---

## Quick Start

### Installation

```bash
git clone https://github.com/FoulquesA/slogan-analyzer.git
cd slogan-analyzer

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
```

### Run the analysis

```python
# scrape_new_data=False loads the existing all_slogans.csv — skip the scraping
df = analyze_slogans(
    scrape_new_data=False,
    data_path='all_slogans.csv',
    output_path='slogans_analyzed.csv',
    visual_path='slogan_analysis_visuals.png',
)
```

### Score a single slogan

```python
from slogan_analyzer import score_length, score_memorability, score_originality

slogan = "Just Do It"

print(f"Length:       {score_length(slogan)}/10")
print(f"Memorability: {score_memorability(slogan)}/10")
print(f"Originality:  {score_originality(slogan)}/10")
```

---

## Key Features

### Cliché detection
Automatically flags 20+ overused marketing terms with weighted penalties:
- **Heavy penalty (3pts):** best, ultimate, innovative, revolutionary, solutions, world-class, leader, trusted, quality
- **Moderate penalty (2pts):** premium, superior, advanced, value, unique, authentic, expert
- **Light penalty (1pt):** international, experience

Terms like "love", "good", or "new" are intentionally excluded — they are common in effective slogans and carry no meaningful signal for cliché detection.

### Statistical reliability flag
Each row includes `cat_reliable` (True/False). Categories with fewer than 30 slogans are flagged and excluded from cross-industry comparisons. Affected categories in this dataset: health-medicine (n=5), campaign (n=11), education (n=14).

### Thematic clustering
TF-IDF vectorization (100 features, bi-grams, min_df=5) followed by K-Means (10 clusters). Results are stored in `theme_cluster` and `theme_keywords` columns. Cluster labels are keyword-based, not manually assigned.

---

## Sample Output

```
========================================================================
  RAPPORT D'ANALYSE — SLOGAN EFFECTIVENESS
========================================================================

  DATASET
  Total slogans analysés  : 3,239
  Catégories fiables (≥30) : 16
  Score moyen global       : 4.72 / 10
  Médiane                  : 4.60 / 10
  Slogans "Excellent" (≥8) : 13 (0.4%)

  DISTRIBUTION DU SENTIMENT
  Positive  : 39.7%
  Neutral   : 56.1%
  Negative  :  4.2%

========================================================================
  TOP 10 SLOGANS
========================================================================
  8.78  Dettol               "We Protect What We Love"
  8.76  Cornetto             "Enjoy the ride, love the ending"
  8.76  Estee Lauder         "Enjoy the stay. Love the shine."
  8.66  Mahindra Scorpio     "Live Young Live Free"
  8.64  eBay                 "Buy it. Sell it. Love it."
```

---

## Methodology

### Data collection
- **Source:** sloganlist.com via BeautifulSoup4
- **Coverage:** 19 industry categories, all paginated pages
- **Deduplication:** automatic on Company + Slogan pair

### Analysis pipeline
1. Scraping and deduplication
2. Data cleaning — NaN removal, string normalization
3. Basic metrics — word count, character count
4. Sentiment analysis — VADER compound score + TextBlob subjectivity
5. Six independent scores computed per slogan
6. Cliché detection — pattern matching with weighted penalty
7. Composite score — weighted average, clipped to 0–10
8. Thematic clustering — TF-IDF + K-Means
9. Report generation and visualizations

### Technologies
- **pandas / numpy** — data manipulation
- **spaCy** — POS tagging for verb detection
- **VADER** — sentiment analysis optimized for short text
- **TextBlob** — subjectivity scoring
- **scikit-learn** — TF-IDF vectorization + K-Means clustering
- **BeautifulSoup4** — web scraping
- **matplotlib / adjustText** — visualizations

---

## Key Insights

### Industry patterns

**Most emotional** (avg emotion score):
- Cosmetics (3.43)
- Restaurants (3.25)
- Household products (3.22)

**Most factual** (highest % neutral sentiment):
- Television channels (86% neutral)
- Sports & Games (65% neutral)
- Financial services (64% neutral)

**Most clichéd** (highest cliché rate):
- Company slogans (9.1%)
- Financial services (8.6%)
- Household products (8.4%)

**Most original** (zero clichés):
- Sports & Games (0%)
- Cosmetics (2%)
- Television channels (2%)

### What the model measures — and what it doesn't

The top-scoring slogans (Dettol, Cornetto, Estée Lauder) score highly because they stack structural signals: rhyme, rhythm, personal pronouns, action verbs, no buzzwords. They are formally correct slogans.

Culturally durable slogans like "Just Do It" (Nike) or "Think Different" (Apple) score lower because they violate at least one conventional rule — too short, grammatically incorrect, no personal pronoun. This is a known limitation of rule-based NLP scoring: it measures conformity to formal criteria, not cultural distinctiveness. The two things are often in tension in effective marketing.

Slogans without clichés score on average 0.26 points higher than those containing them (4.74 vs 4.48), which is consistent but modest — the originality dimension alone does not drive the composite.

---

## Advanced Usage

### Custom weighting

```python
custom_weights = {
    'score_length':       0.10,
    'score_memorability': 0.35,
    'score_emotion':      0.15,
    'score_action':       0.10,
    'score_originality':  0.25,
    'score_personal':     0.05,
}

df = add_effectiveness_score(df)  # uses WEIGHTS dict defined at top of file
```

Modify the `WEIGHTS` constant directly at the top of the script to apply globally.

### Filter by category

```python
tech = df[df['Category'] == 'technology-slogans']
print(f"Average effectiveness: {tech['effectiveness_score'].mean():.2f}")
```

### Export

```python
# Top 100
df.nlargest(100, 'effectiveness_score').to_csv('top_100_slogans.csv', index=False)

# By category
for cat in df['Category'].unique():
    df[df['Category'] == cat].to_csv(f'slogans_{cat}.csv', index=False)
```

---

## Future Enhancements

- [ ] External validation — correlate scores with consumer recall or brand tracking data
- [ ] Empirical weighting — calibrate composite weights on real performance data rather than assumed values
- [ ] Temporal analysis — track how slogan patterns shift by decade or campaign cycle
- [ ] BERT-based scoring — replace rule-based memorability with semantic similarity
- [ ] Multilingual support — extend pipeline to French, Spanish, German
- [ ] Interactive dashboard — Streamlit interface for real-time slogan scoring

---

## Author

**Foulques Arbaretier** — Data Analyst  
Master's in Marketing Insight & Data Analytics Strategy, Paris School of Business  
[LinkedIn](https://www.linkedin.com/in/foulques-arbaretier/) · [GitHub](https://github.com/FoulquesA)

## Acknowledgments

- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment) by C.J. Hutto & Eric Gilbert
- [SloganList.com](https://www.sloganlist.com) for the dataset
- spaCy, scikit-learn, pandas, and matplotlib communities
