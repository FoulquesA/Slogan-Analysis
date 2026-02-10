# Slogan Effectiveness Analyzer

> A comprehensive Python framework for analyzing and scoring marketing slogans using NLP, sentiment analysis, and machine learning.

## 📋 Overview

This project provides a data-driven approach to analyze marketing slogans across 9 quantitative dimensions:

1. **Sentiment Analysis** (VADER) - Emotional tone and positivity
2. **Length Optimization** - Adherence to marketing best practices (3-6 words)
3. **Memorability** - Alliteration, rhyme, and rhythm detection
4. **Emotional Impact** - Sentiment intensity × subjectivity
5. **Action Verbs** - Presence of powerful call-to-action verbs
6. **Personal Engagement** - Use of "you/your" or "we/our"
7. **Originality** - Detection of marketing clichés
8. **Thematic Clustering** - TF-IDF + K-Means grouping
9. **Composite Effectiveness Score** - Weighted aggregate (0-10)

**Dataset:** 3,338 unique slogans across 19 industries scraped from sloganlist.com

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/slogan-analyzer.git
cd slogan-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Basic Usage

```python
from slogan_analyzer import analyze_slogans

# Run full analysis pipeline
df = analyze_slogans(
    scrape_new_data=False,  # Use existing data
    data_path='all_slogans.csv',
    output_path='slogans_analyzed.csv'
)

# View top 10 slogans
print(df.nlargest(10, 'effectiveness_score')[
    ['Company', 'Slogan', 'effectiveness_score']
])
```

### Analyze a Single Slogan

```python
from slogan_analyzer import (
    score_length, score_memorability, score_originality
)

slogan = "Just Do It"

print(f"Length: {score_length(slogan)}/10")
print(f"Memorability: {score_memorability(slogan)}/10")
print(f"Originality: {score_originality(slogan)}/10")
```

## 📊 Key Features

### 1. Sentiment Analysis (VADER)
- **Compound score** normalized between -1 (negative) and +1 (positive)
- Classification: Positive (≥0.05), Neutral, Negative (≤-0.05)
- **Subjectivity** measurement via TextBlob (0=factual, 1=subjective)

### 2. Effectiveness Scoring System
Each slogan receives scores (0-10) across 6 dimensions:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| **Memorability** | 25% | Alliteration, rhyme, rhythm, repetition |
| **Emotion** | 20% | Sentiment intensity × subjectivity |
| **Length** | 15% | Optimal: 3-6 words, 20-40 characters |
| **Action** | 15% | Presence of power verbs (make, create, discover) |
| **Originality** | 15% | Penalty for 45+ marketing clichés |
| **Personal** | 10% | Use of "you/your" or "we/our" |

### 3. Cliché Detection
Automatically detects 45+ overused marketing terms:
- **Superlatives:** best, ultimate, perfect, premier
- **Innovation buzzwords:** innovative, revolutionary, cutting-edge
- **Business speak:** solutions, trusted, leader, world-class

### 4. Thematic Clustering
- **TF-IDF vectorization** (100 features, bi-grams)
- **K-Means clustering** (10 thematic groups)
- Automatic title assignment (e.g., "Innovation & Novelty", "Taste & Freshness")

## 📈 Sample Output

```
================================================================================
📊 SLOGAN EFFECTIVENESS ANALYSIS - SUMMARY REPORT
================================================================================

📈 DATASET OVERVIEW
  Total slogans analyzed: 3,338
  Number of categories: 19

⭐ EFFECTIVENESS SCORES
  Average score: 5.23/10
  Median score: 5.10/10
  Excellent slogans (≥8/10): 127 (3.8%)

💭 SENTIMENT DISTRIBUTION
  Positive: 39.7%
  Neutral: 56.1%
  Negative: 4.2%

================================================================================
🏆 TOP 10 MOST EFFECTIVE SLOGANS
================================================================================

8.75/10 - Nike
  "Just Do It"
  Category: sports-games-slogans

8.50/10 - Apple
  "Think Different"
  Category: technology-slogans
```

## 🔬 Methodology

### Data Collection
- **Source:** Web scraping from [sloganlist.com](https://www.sloganlist.com)
- **Technique:** BeautifulSoup4 for HTML parsing
- **Coverage:** 19 industry categories
- **Deduplication:** Automatic removal of duplicate entries

### Analysis Pipeline
1. **Data Cleaning** - Remove NaN, ensure string types
2. **Basic Metrics** - Calculate character/word counts
3. **Sentiment Analysis** - VADER + TextBlob
4. **Effectiveness Scoring** - 6 independent scores
5. **Cliché Detection** - Pattern matching with penalty system
6. **Thematic Clustering** - TF-IDF + K-Means
7. **Composite Score** - Weighted average
8. **Report Generation** - Statistics + top/bottom slogans

### Technologies
- **pandas** - Data manipulation
- **spaCy** - NLP (POS tagging for verb detection)
- **VADER** - Sentiment analysis optimized for short text
- **TextBlob** - Subjectivity measurement
- **scikit-learn** - TF-IDF vectorization + K-Means clustering
- **BeautifulSoup4** - Web scraping

## 📊 Key Insights

### Industry Patterns

**Most Creative** (high originality):
- Sports & Games
- Food & Restaurants
- Cosmetics

**Most Clichéd** (low originality):
- Business slogans
- Financial services
- Technology

**Most Emotional**:
- Cosmetics (3.43/10)
- Restaurants (3.25/10)
- Household products (3.22/10)

**Most Factual**:
- Television channels (0.88/10)
- Automotive (1.84/10)
- Financial services (2.05/10)

### Effectiveness Formula

**Characteristics of high-scoring slogans:**
- 3-6 words (optimal length)
- Alliteration or rhyme (memorability)
- Positive sentiment (not excessive)
- Action verb in imperative form
- Zero marketing clichés
- Simple, concrete vocabulary

**Examples:**
- "Just Do It" (Nike) - 
- "Think Different" (Apple) - 
- "I'm Lovin' It" (McDonald's) - 

## 🛠️ Advanced Usage

### Custom Weighting

```python
from slogan_analyzer import calculate_effectiveness_score

# Define custom weights (must sum to 1.0)
custom_weights = {
    'score_length': 0.10,
    'score_memorability': 0.35,  # Prioritize memorability
    'score_emotion': 0.15,
    'score_action': 0.10,
    'score_originality': 0.25,   # Heavily penalize clichés
    'score_personal': 0.05
}

df = calculate_effectiveness_score(df, weights=custom_weights)
```

### Filter by Category

```python
# Analyze only technology slogans
tech_slogans = df[df['Category'] == 'technology-slogans']

print("Tech Industry Statistics:")
print(f"Average effectiveness: {tech_slogans['effectiveness_score'].mean():.2f}")
print(f"Top clichés: {tech_slogans['cliches_detected'].explode().value_counts().head(5)}")
```

### Export Results

```python
# Save top 100 slogans
top_100 = df.nlargest(100, 'effectiveness_score')
top_100.to_csv('top_100_slogans.csv', index=False)

# Export by category
for category in df['Category'].unique():
    cat_df = df[df['Category'] == category]
    cat_df.to_csv(f'slogans_{category}.csv', index=False)
```

## 🔮 Future Enhancements

- [ ] **Temporal Analysis** - Track slogan trends over time
- [ ] **Performance Correlation** - Link to actual brand success metrics
- [ ] **Deep Learning** - BERT-based semantic similarity
- [ ] **Multilingual Support** - Analyze non-English slogans
- [ ] **API Endpoint** - Real-time slogan scoring service
- [ ] **Interactive Dashboard** - Streamlit/Dash visualization
- [ ] **Slogan Generator** - GPT-based creative generation

## 📚 Documentation

- **[VADER Technical Docs](docs/VADER_Documentation.md)** - How VADER analyzes sentiment
- **[Full Analysis Report](docs/analysis_summary.md)** - Complete insights & patterns
- **[API Reference](#)** - Function documentation (coming soon)

## 🤝 Contributing

Contributions welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

**Code Style:** PEP 8 compliance required. Use `black` for formatting.

## Author

**Foulques Arbaretier**
- Data Analyst
- Master's in Marketing Insight & Data Analytics Strategy (Paris School of Business)
- LinkedIn: [https://www.linkedin.com/in/foulques-arbaretier/]
- Portfolio: [https://github.com/FoulquesA]

## 🙏 Acknowledgments

- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment) by C.J. Hutto & Eric Gilbert
- [SloganList.com](https://www.sloganlist.com) for the slogan data
- spaCy, scikit-learn, and pandas communities
