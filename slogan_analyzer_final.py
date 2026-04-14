"""
Slogan Effectiveness Analyzer
==============================

Analyse de 3 239 slogans marketing sur 19 industries via NLP,
analyse de sentiment et clustering thématique non supervisé.

Dataset : sloganlist.com — scrapé via BeautifulSoup
Auteur  : Foulques Arbaretier
"""

import re, time, ast, warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

CATEGORIES = [
    'drinking-slogans', 'food-slogans', 'restaurant-slogans', 'car-slogan',
    'apparel-slogans', 'technology-slogans', 'business-slogans', 'company-slogans',
    'cosmetics-slogans', 'household-slogans', 'financial-slogans', 'tours-slogans',
    'airlines-slogans', 'television-channels-slogan', 'health-medicine-slogans',
    'sports-games-slogans', 'education-slogans', 'campaign-slogans', 'uncategorized'
]

# Seuil de fiabilité statistique : catégories < 30 slogans exclues des comparaisons
MIN_CAT_SIZE = 30

# Clichés marketing — distinction entre buzzwords corporate (pénalité forte)
# et termes génériques (pénalité légère). "love", "good", "new" volontairement exclus.
MARKETING_CLICHES = {
    # Superlatives corporate
    'best': 3, 'ultimate': 3, 'perfect': 2, 'premier': 3,
    'leading': 3, 'number one': 3, '#1': 3,
    # Buzzwords innovation
    'innovative': 3, 'innovation': 3, 'cutting-edge': 3,
    'revolutionary': 3, 'breakthrough': 3, 'next generation': 3,
    'state of the art': 3, 'advanced': 2,
    # Claims qualité vagues
    'quality': 3, 'excellence': 3, 'premium': 2, 'superior': 3,
    'finest': 2, 'exceptional': 2,
    # Business speak
    'solutions': 3, 'solution': 3, 'trusted': 3,
    'leader': 3, 'leadership': 3, 'world-class': 3, 'world class': 3,
    'global': 2, 'international': 1,
    # Promesses floues
    'experience': 1, 'value': 2, 'expert': 2, 'professional': 2,
    'authentic': 2, 'unique': 2,
}

POWER_VERBS = [
    'make', 'create', 'build', 'discover', 'explore', 'get', 'find',
    'start', 'go', 'do', 'think', 'imagine', 'love', 'live', 'feel',
]

# Poids du score composite — somme = 1.0
WEIGHTS = {
    'score_length':       0.15,
    'score_memorability': 0.25,
    'score_emotion':      0.20,
    'score_action':       0.15,
    'score_originality':  0.15,
    'score_personal':     0.10,
}

# Palette visuelle (fond sombre)
COLORS = {
    'bg':    '#FAFAFA', 'panel': '#FFFFFF',
    'gold':  '#E8A020', 'gold2': '#C47A15',
    'text':  '#1a1a2e', 'dim':   '#888899',
    'cats':  ['#E8A020','#7C5CBF','#2196F3','#E53935','#43A047',
              '#FB8C00','#00ACC1','#D81B60','#6DB33F','#5C6BC0',
              '#FF7043','#26A69A','#AB47BC'],
}


# =============================================================================
# 1. SCRAPING
# =============================================================================

def scrape_slogans(base_url='https://www.sloganlist.com/', delay=0.3):
    """
    Scrape sloganlist.com sur toutes les catégories.
    Retourne un DataFrame Company / Slogan / Category.
    """
    all_slogans = []

    for cat in CATEGORIES:
        url = f'{base_url}{cat}/'
        try:
            r    = requests.get(url, timeout=10)
            soup = BeautifulSoup(r.content, 'html.parser')
        except Exception as e:
            print(f'  [skip] {cat} : {e}')
            continue

        max_page = _get_max_page(soup)
        print(f'{cat}: {max_page} pages')

        for page in range(1, max_page + 1):
            page_url = url if page == 1 else f'{base_url}{cat}/index_{page}.html'
            try:
                slogans = _scrape_page(page_url, cat)
                all_slogans.extend(slogans)
                time.sleep(delay)
            except Exception as e:
                print(f'  page {page} error: {e}')

    df = pd.DataFrame(all_slogans).drop_duplicates()
    print(f'\nTotal brut : {len(df)} slogans')
    return df


def _get_max_page(soup):
    max_p = 1
    for link in soup.find_all('a', href=True):
        href = link['href']
        if 'index_' in href and '.html' in href:
            try:
                n    = int(href.split('index_')[1].split('.html')[0])
                max_p = max(max_p, n)
            except ValueError:
                pass
    return max_p


def _scrape_page(url, category):
    r    = requests.get(url, timeout=10)
    soup = BeautifulSoup(r.content, 'html.parser')
    links = soup.find_all('a', href=lambda x: x and (
        x.endswith('-slogan.html') or x.endswith('-slogans.html')
    ))
    slogans = []
    for link in links:
        text = link.get_text()
        if '- ' in text:
            parts = text.split('- ', 1)
            slogans.append({
                'Company': parts[0].strip(),
                'Slogan':  parts[1].strip(),
                'Category': category,
            })
    return slogans


# =============================================================================
# 2. NETTOYAGE
# =============================================================================

def clean_data(df):
    before = len(df)
    df = df.dropna()
    df['Slogan'] = df['Slogan'].astype(str)
    df = df[df['Slogan'].str.len() > 0].reset_index(drop=True)
    print(f'Nettoyage : {before} → {len(df)} slogans')
    return df


# =============================================================================
# 3. MÉTRIQUES DE BASE
# =============================================================================

def add_basic_metrics(df):
    df['char_count'] = df['Slogan'].str.len()
    df['word_count'] = df['Slogan'].str.split().str.len()
    # Flag fiabilité catégorie (exclut les catégories trop petites des comparaisons)
    cat_n = df['Category'].value_counts()
    df['cat_n']        = df['Category'].map(cat_n)
    df['cat_reliable'] = df['cat_n'] >= MIN_CAT_SIZE
    return df


# =============================================================================
# 4. SENTIMENT
# =============================================================================

def add_sentiment(df):
    vader = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df['Slogan'].apply(
        lambda x: vader.polarity_scores(str(x))['compound']
    )
    df['sentiment'] = df['sentiment_score'].apply(
        lambda s: 'Positive' if s >= 0.05 else ('Negative' if s <= -0.05 else 'Neutral')
    )
    df['subjectivity'] = df['Slogan'].apply(
        lambda x: TextBlob(str(x)).sentiment.subjectivity
    )
    return df


# =============================================================================
# 5. SCORES
# =============================================================================

def score_length(slogan):
    words = len(slogan.split())
    chars = len(slogan)
    w_score = (10 if 3 <= words <= 6
               else 7 if 2 <= words <= 8
               else max(0, 10 - abs(words - 4.5)))
    c_score = (10 if 20 <= chars <= 40
               else 7 if 15 <= chars <= 50
               else max(0, 10 - abs(chars - 30) / 5))
    return (w_score + c_score) / 2


def score_memorability(slogan):
    words = slogan.lower().split()
    if not words:
        return 0
    score = 0
    # Allitération
    letters = [w[0] for w in words if w]
    if len(words) >= 2 and len(set(letters)) < len(letters):
        score += 3
    # Rime
    endings = [w[-2:] for w in words if len(w) >= 2]
    if len(words) >= 2 and len(set(endings)) < len(endings):
        score += 3
    # Rythme syllabique
    sylls = [len(re.findall(r'[aeiou]+', w)) for w in words]
    if len(sylls) >= 2 and max(sylls) - min(sylls) <= 1:
        score += 2
    # Répétition de mots
    if len(words) != len(set(words)):
        score += 2
    return min(score, 10)


def score_emotion(row):
    s = abs(row['sentiment_score']) * 5 + row['subjectivity'] * 5
    if row['sentiment_score'] > 0.5:
        s += 2
    return min(s, 10)


def score_originality(slogan):
    sl      = slogan.lower()
    penalty = sum(pts for cliche, pts in MARKETING_CLICHES.items() if cliche in sl)
    return max(0, 10 - penalty)


def score_personal(slogan):
    sl    = slogan.lower()
    score = 0
    if 'you' in sl or 'your' in sl: score += 5
    if 'we' in sl or 'our' in sl:   score += 3
    return min(score, 10)


def score_action(slogan, nlp):
    doc   = nlp(slogan.lower())
    verbs = [t for t in doc if t.pos_ == 'VERB']
    score = 0
    if verbs: score += 5
    if any(v.lemma_ in POWER_VERBS for v in verbs): score += 3
    if doc and doc[0].pos_ == 'VERB': score += 2
    return min(score, 10)


def detect_cliches(slogan):
    sl = slogan.lower()
    return [c for c in MARKETING_CLICHES if c in sl]


def add_all_scores(df):
    nlp = spacy.load('en_core_web_sm')

    df['score_length']       = df['Slogan'].apply(score_length)
    df['score_memorability'] = df['Slogan'].apply(score_memorability)
    df['score_emotion']      = df.apply(score_emotion, axis=1)
    df['score_action']       = df['Slogan'].apply(lambda x: score_action(x, nlp))
    df['score_originality']  = df['Slogan'].apply(score_originality)
    df['score_personal']     = df['Slogan'].apply(score_personal)
    df['cliches_detected']   = df['Slogan'].apply(detect_cliches)
    df['cliche_count']       = df['cliches_detected'].apply(len)
    return df


# =============================================================================
# 6. SCORE COMPOSITE
# =============================================================================

def add_effectiveness_score(df):
    df['effectiveness_score'] = sum(
        df[col] * w for col, w in WEIGHTS.items()
    ).clip(0, 10)

    df['effectiveness_category'] = pd.cut(
        df['effectiveness_score'],
        bins=[0, 3.5, 5, 6.5, 8, 10],
        labels=['Faible', 'Moyen', 'Bon', 'Très bon', 'Excellent'],
    )
    return df


# =============================================================================
# 7. CLUSTERING THÉMATIQUE
# =============================================================================

def add_clustering(df, n_clusters=10):
    df_clean = df[df['Slogan'].str.len() > 0].copy()

    vectorizer = TfidfVectorizer(
        max_features=100, stop_words='english',
        ngram_range=(1, 2), min_df=5,
    )
    X = vectorizer.fit_transform(df_clean['Slogan'])

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_clean['theme_cluster'] = km.fit_predict(X)

    feature_names = vectorizer.get_feature_names_out()
    cluster_keywords = {}
    for i in range(n_clusters):
        top_idx  = km.cluster_centers_[i].argsort()[-5:][::-1]
        cluster_keywords[i] = ', '.join(feature_names[j] for j in top_idx)

    df['theme_cluster']  = -1
    df.loc[df_clean.index, 'theme_cluster'] = df_clean['theme_cluster']
    df['theme_keywords'] = df['theme_cluster'].map(
        lambda x: cluster_keywords.get(x, '') if x != -1 else ''
    )
    return df


# =============================================================================
# 8. VISUALISATIONS
# =============================================================================

def _setup_style():
    plt.rcParams.update({
        'figure.facecolor': COLORS['bg'],
        'axes.facecolor':   COLORS['panel'],
        'text.color':       COLORS['text'],
        'axes.labelcolor':  COLORS['text'],
        'xtick.color':      COLORS['dim'],
        'ytick.color':      COLORS['dim'],
        'axes.edgecolor':   '#2a2540',
        'grid.color':       '#1e1c2e',
        'font.family':      'monospace',
        'font.size':        9,
        'axes.spines.top':  False,
        'axes.spines.right':False,
    })


def generate_visuals(df, output_path='slogan_analysis_visuals.png'):
    BG    = COLORS['bg'];    PANEL = COLORS['panel']
    GOLD  = COLORS['gold'];  GOLD2 = COLORS['gold2']
    TEXT  = COLORS['text'];  DIM   = COLORS['dim']
    COLS  = COLORS['cats']

    plt.rcParams.update({
        'figure.facecolor': BG, 'axes.facecolor': PANEL,
        'text.color': TEXT, 'axes.labelcolor': TEXT,
        'xtick.color': TEXT, 'ytick.color': TEXT,
        'axes.edgecolor': '#CCCCDD', 'grid.color': '#EEEEEE',
        'font.family': 'monospace', 'font.size': 11,
        'axes.spines.top': False, 'axes.spines.right': False,
    })

    df_r = df[df['cat_reliable']].copy()
    cat_order = (df_r.groupby('Category')['effectiveness_score']
                 .median().sort_values(ascending=False).index.tolist())
    cat_labels = [c.replace('-slogans','').replace('-slogan','').replace('-',' ')
                  for c in cat_order]
    COLS_CAT = [COLS[i % len(COLS)] for i in range(len(cat_order))]

    # ═══════════════════════════════════════════════════════════
    # 1. VIOLIN — distribution par catégorie
    # ═══════════════════════════════════════════════════════════
    fig1, ax = plt.subplots(figsize=(20, 8), facecolor=BG)
    ax.set_facecolor(PANEL)
    data_per = [df_r[df_r['Category'] == c]['effectiveness_score'].values for c in cat_order]

    parts = ax.violinplot(data_per, positions=range(len(cat_order)),
                          showmedians=True, showextrema=False, widths=0.7)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(COLS_CAT[i]); pc.set_alpha(0.45); pc.set_edgecolor('white')
    parts['cmedians'].set_color('#1a1a2e'); parts['cmedians'].set_linewidth(2.5)

    for i, (cat, vals) in enumerate(zip(cat_order, data_per)):
        median = np.median(vals)
        ax.text(i, median + 0.25, f'{median:.2f}',
                ha='center', fontsize=8.5, color=TEXT, fontweight='bold')
        ax.text(i, -0.7, f'n={len(vals)}',
                ha='center', fontsize=8, color=DIM)

    ax.set_xticks(range(len(cat_order)))
    ax.set_xticklabels(cat_labels, rotation=38, ha='right', fontsize=10)
    ax.set_ylabel('Score d\'efficacité (0–10)', fontsize=12, labelpad=10)
    ax.set_ylim(-1, 12)
    ax.axhline(df_r['effectiveness_score'].mean(), color=GOLD, ls='--', lw=1.5, alpha=0.8)
    ax.text(len(cat_order) - 0.5, df_r['effectiveness_score'].mean() + 0.2,
            f'moyenne globale : {df_r["effectiveness_score"].mean():.2f}',
            fontsize=9, color=GOLD, fontstyle='italic')
    ax.grid(axis='y', alpha=0.5, lw=0.8)
    ax.set_title('Distribution des scores d\'efficacité par catégorie\n(catégories ≥ 30 slogans uniquement)',
                 fontsize=14, fontweight='bold', pad=16, color=TEXT)
    fig1.tight_layout()
    fig1.savefig('visual_1_violin.png', dpi=150, bbox_inches='tight', facecolor=BG)
    plt.show()

    # ═══════════════════════════════════════════════════════════
    # 2. SCATTER — sentiment vs efficacité
    # ═══════════════════════════════════════════════════════════
    from adjustText import adjust_text

    fig2, ax = plt.subplots(figsize=(13, 9), facecolor=BG)
    ax.set_facecolor(PANEL)

    reliable_cats = df_r['Category'].unique()
    for i, cat in enumerate(reliable_cats):
        sub = df_r[df_r['Category'] == cat]
        label = cat.replace('-slogans','').replace('-slogan','').replace('-',' ')
        ax.scatter(sub['sentiment_score'], sub['effectiveness_score'],
                   c=COLS[i % len(COLS)], alpha=0.35, s=22, linewidths=0, label=label)

    top_rows = df_r.nlargest(8, 'effectiveness_score')
    texts = []
    for _, row in top_rows.iterrows():
        t = ax.text(row['sentiment_score'], row['effectiveness_score'],
                    f"{row['Company']} ({row['effectiveness_score']:.1f})",
                    fontsize=8.5, color=TEXT, fontweight='bold')
        texts.append(t)
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color=DIM, lw=0.8))

    ax.axvline(0, color=DIM, ls='--', lw=1, alpha=0.6)
    ax.axhline(df_r['effectiveness_score'].mean(), color=GOLD, ls='--', lw=1.2, alpha=0.7)
    ax.set_xlabel('Score sentiment VADER  (← négatif · positif →)', fontsize=12, labelpad=10)
    ax.set_ylabel('Score d\'efficacité composite (0–10)', fontsize=12, labelpad=10)
    ax.set_title('Sentiment vs Efficacité\nchaque point = un slogan, coloré par industrie',
                 fontsize=14, fontweight='bold', pad=14, color=TEXT)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.6,
              ncol=2, markerscale=1.4, labelcolor=TEXT, facecolor=PANEL)
    ax.grid(alpha=0.4, lw=0.8)
    fig2.tight_layout()
    fig2.savefig('visual_2_scatter.png', dpi=150, bbox_inches='tight', facecolor=BG)
    plt.show()

    # ═══════════════════════════════════════════════════════════
    # 3a. HEATMAP — corrélation entre dimensions
    # ═══════════════════════════════════════════════════════════
    fig3, ax = plt.subplots(figsize=(9, 7), facecolor=BG)
    ax.set_facecolor(PANEL)

    score_cols   = ['score_length','score_memorability','score_emotion',
                    'score_action','score_originality','score_personal']
    labels_short = ['Longueur','Mémorabilité','Émotion','Action','Originalité','Engagement']
    corr = df_r[score_cols].corr()

    cmap = mcolors.LinearSegmentedColormap.from_list(
        'heatmap', ['#2196F3', '#FFFFFF', '#E8A020'])
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

    for i in range(len(score_cols)):
        for j in range(len(score_cols)):
            v = corr.values[i, j]
            color = TEXT if abs(v) < 0.6 else 'white'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=11, color=color, fontweight='bold')

    ax.set_xticks(range(len(score_cols))); ax.set_yticks(range(len(score_cols)))
    ax.set_xticklabels(labels_short, rotation=30, ha='right', fontsize=11)
    ax.set_yticklabels(labels_short, fontsize=11)
    cbar = fig3.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label('Corrélation de Pearson', fontsize=10, color=TEXT)
    ax.set_title('Corrélations entre les 6 dimensions de score\n1 = corrélation parfaite · 0 = indépendance · -1 = opposition',
                 fontsize=13, fontweight='bold', pad=14, color=TEXT)
    fig3.tight_layout()
    fig3.savefig('visual_3_heatmap.png', dpi=150, bbox_inches='tight', facecolor=BG)
    plt.show()

    # ═══════════════════════════════════════════════════════════
    # 3b. BAR COMP — profil moyen top 5 vs bottom 5 catégories
    # ═══════════════════════════════════════════════════════════
    fig3b, ax = plt.subplots(figsize=(13, 7), facecolor=BG)
    ax.set_facecolor(PANEL)

    dims       = ['score_length','score_memorability','score_emotion',
                  'score_action','score_originality','score_personal']
    dim_labels = ['Longueur','Mémorabilité','Émotion','Action','Originalité','Engagement']

    cat_avg   = df_r.groupby('Category')[dims].mean()
    cat_score = df_r.groupby('Category')['effectiveness_score'].mean().sort_values(ascending=False)
    top5_cats = cat_score.head(5).index.tolist()
    bot5_cats = cat_score.tail(5).index.tolist()

    top5_avg = cat_avg.loc[top5_cats].mean()
    bot5_avg = cat_avg.loc[bot5_cats].mean()

    x     = np.arange(len(dims))
    width = 0.35

    ax.bar(x - width/2, top5_avg.values, width, label='Top 5 catégories',    color=GOLD,    alpha=0.85)
    ax.bar(x + width/2, bot5_avg.values, width, label='Bottom 5 catégories', color=COLS[2], alpha=0.85)

    for i, (t, b) in enumerate(zip(top5_avg.values, bot5_avg.values)):
        ax.text(i - width/2, t + 0.1, f'{t:.1f}', ha='center', fontsize=9, color=TEXT, fontweight='bold')
        ax.text(i + width/2, b + 0.1, f'{b:.1f}', ha='center', fontsize=9, color=TEXT, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, fontsize=11)
    ax.set_ylabel('Score moyen (0–10)', fontsize=11)
    ax.set_ylim(0, 12)
    ax.legend(fontsize=11, framealpha=0.6, facecolor=PANEL)
    ax.grid(axis='y', alpha=0.4)
    ax.set_title('Profil moyen par dimension\nTop 5 catégories vs Bottom 5 catégories',
                 fontsize=13, fontweight='bold', pad=14, color=TEXT)
    fig3b.tight_layout()
    fig3b.savefig('visual_3b_barcomp.png', dpi=150, bbox_inches='tight', facecolor=BG)
    plt.show()

    # ═══════════════════════════════════════════════════════════
    # 4. RADAR — profil des slogans
    # ═══════════════════════════════════════════════════════════
    fig4, ax = plt.subplots(figsize=(10, 10), facecolor=BG, subplot_kw=dict(polar=True))
    ax.set_facecolor('#F5F5FF')

    dims    = ['score_length','score_memorability','score_emotion',
            'score_action','score_originality','score_personal']
    dim_lbl = ['Longueur','Mémorabilité','Émotion','Action','Originalité','Engagement']
    angles  = np.linspace(0, 2*np.pi, len(dims), endpoint=False).tolist() + [0]

    iconic  = df_r[df_r['Company'].str.lower().isin(['pepsi','coca-cola','bmw','adidas','nike','apple'])]
    top1    = df_r.nlargest(1, 'effectiveness_score')
    bot1    = df_r.nsmallest(1, 'effectiveness_score')
    radar_df= pd.concat([iconic.head(1), top1, bot1]).drop_duplicates('Company').head(3)

    palette = [GOLD, COLS[3], COLS[2]]
    labels  = []

    for i, (_, row) in enumerate(radar_df.iterrows()):
        vals = [row[d] for d in dims] + [row[dims[0]]]
        ax.plot(angles, vals, color=palette[i], lw=2.5, alpha=0.95, zorder=3)
        ax.fill(angles, vals, color=palette[i], alpha=0.15)
        labels.append(f"{row['Company']}  ·  \"{row['Slogan'][:28]}\"  →  {row['effectiveness_score']:.1f}/10")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_lbl, size=12, color=TEXT, fontweight='bold')
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2','4','6','8','10'], size=9, color=DIM)
    ax.grid(color='#BBBBCC', alpha=0.7, lw=0.9)

    handles = [plt.Line2D([0],[0], color=c, lw=3) for c in palette]
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.10),
            fontsize=10, framealpha=0.7, facecolor=PANEL, labelcolor=TEXT, ncol=1)
    ax.set_title('Profil radar : un slogan iconique · un top scoreur · un bottom scoreur',
                fontsize=12, fontweight='bold', pad=24, color=TEXT)
    fig4.tight_layout()
    fig4.savefig('visual_4_radar.png', dpi=150, bbox_inches='tight', facecolor=BG)
    plt.show()
    # ═══════════════════════════════════════════════════════════
    # 5. BAR — top 10 clichés
    # ═══════════════════════════════════════════════════════════
    fig5, ax = plt.subplots(figsize=(11, 7), facecolor=BG)
    ax.set_facecolor(PANEL)

    def safe_parse(x):
        if isinstance(x, list): return x
        try: return ast.literal_eval(x)
        except: return []

    all_c  = [c for sub in df['cliches_detected'].apply(safe_parse) for c in sub]
    top_c  = pd.Series(all_c).value_counts().head(10)
    bar_colors = [GOLD if i == 0 else COLS[i % len(COLS)] for i in range(len(top_c))]

    bars = ax.barh(range(len(top_c)), top_c.values,
                   color=bar_colors, alpha=0.82, height=0.62)
    ax.set_yticks(range(len(top_c)))
    ax.set_yticklabels(top_c.index, fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlabel('Nombre d\'occurrences dans le dataset', fontsize=11, labelpad=10)
    ax.set_title('Top 10 des clichés marketing détectés\n(présence = pénalité sur le score d\'originalité)',
                 fontsize=14, fontweight='bold', pad=14, color=TEXT)
    ax.grid(axis='x', alpha=0.4, lw=0.8)

    for i, (bar, val) in enumerate(zip(bars, top_c.values)):
        pct = val / len(df) * 100
        ax.text(val + 1, i, f'{val}  ({pct:.1f}% des slogans)',
                va='center', fontsize=10, color=TEXT)

    ax.set_xlim(0, top_c.values[0] * 1.3)
    fig5.tight_layout()
    fig5.savefig('visual_5_cliches.png', dpi=150, bbox_inches='tight', facecolor=BG)
    plt.show()

    print('5 visuels sauvegardés dans le dossier courant.')

# =============================================================================
# 9. RAPPORT + CONCLUSION STRATÉGIQUE
# =============================================================================

def generate_report(df):
    df_r = df[df['cat_reliable']].copy()

    cat_stats = df_r.groupby('Category').agg(
        score_moyen   = ('effectiveness_score', 'mean'),
        originality   = ('score_originality', 'mean'),
        emotion       = ('score_emotion', 'mean'),
        memorability  = ('score_memorability', 'mean'),
        pct_neutral   = ('sentiment', lambda x: (x == 'Neutral').mean()),
        n             = ('Slogan', 'count'),
    ).sort_values('score_moyen', ascending=False)

    def safe_parse(x):
        if isinstance(x, list): return x
        try: return ast.literal_eval(x)
        except: return []

    df['cliches_detected'] = df['cliches_detected'].apply(safe_parse)
    cliche_rate = df_r.copy()
    cliche_rate['has_cliche'] = cliche_rate['cliche_count'] > 0
    cliche_by_cat = cliche_rate.groupby('Category')['has_cliche'].mean().sort_values(ascending=False)

    with_c    = df_r[df_r['cliche_count'] > 0]['effectiveness_score'].mean()
    without_c = df_r[df_r['cliche_count'] == 0]['effectiveness_score'].mean()

    sep = '=' * 72

    print(f'\n{sep}')
    print('  RAPPORT D\'ANALYSE — SLOGAN EFFECTIVENESS')
    print(sep)

    print(f'\n  DATASET')
    print(f'  Total slogans analysés  : {len(df):,}')
    print(f'  Catégories fiables (≥{MIN_CAT_SIZE}) : {df_r["Category"].nunique()}')
    print(f'  Score moyen global      : {df["effectiveness_score"].mean():.2f} / 10')
    print(f'  Médiane                 : {df["effectiveness_score"].median():.2f} / 10')
    print(f'  Slogans "Excellent" (≥8): {(df["effectiveness_score"] >= 8).sum()} ({(df["effectiveness_score"] >= 8).mean()*100:.1f}%)')

    print(f'\n  DISTRIBUTION DU SENTIMENT')
    for s in ['Positive', 'Neutral', 'Negative']:
        pct = (df['sentiment'] == s).mean() * 100
        print(f'  {s:10s}: {pct:.1f}%')

    print(f'\n  CATÉGORIES — SCORE MOYEN (fiables uniquement)')
    print(cat_stats[['score_moyen','emotion','memorability','originality','n']].round(2).to_string())

    print(f'\n  TOP 10 SLOGANS')
    top10 = df.nlargest(10, 'effectiveness_score')[
        ['Company','Slogan','effectiveness_score','score_memorability','cliche_count']
    ]
    for _, r in top10.iterrows():
        print(f'  {r["effectiveness_score"]:.2f}  {r["Company"]:<20} "{r["Slogan"]}"')

    print(f'\n{sep}')
    print('  CONCLUSION STRATÉGIQUE')
    print(sep)

    print("""
  1. LE MODÈLE MESURE LA CONFORMITÉ, PAS LA DISTINCTIVITÉ
  ─────────────────────────────────────────────────────────
  Les 10 slogans les mieux notés (Dettol, Cornetto, Estée Lauder, eBay...)
  ont un score de mémorabilité maximal (10/10) parce qu'ils cumulent
  allitérations, rimes et répétitions. Ce sont des slogans structurellement
  corrects — pas nécessairement les plus culturellement durables.

  Des slogans comme "Just Do It" (Nike) ou "Think Different" (Apple)
  n'apparaissent pas dans ce top car ils violent au moins une règle
  conventionnelle : longueur trop courte, grammaire incorrecte, absence
  de pronom personnel. C'est précisément ce qui les rend distinctifs.

  Interprétation : un bon score NLP indique un slogan qui joue selon les
  règles. Un slogan réellement mémorable les transgresse souvent.

  2. LES CLICHÉS CORPORATE PÉNALISENT EFFECTIVEMENT LE SCORE
  ────────────────────────────────────────────────────────────
  Slogans sans cliché : score moyen {:.2f}/10
  Slogans avec cliché : score moyen {:.2f}/10

  L'écart est faible mais cohérent. Les industries les plus touchées
  par les buzzwords sont le B2B (company-slogans, financial-slogans)
  avec un taux de clichés autour de 9%, contre 0% pour le sport
  et 2% pour la cosmétique et la télévision.

  Implication pratique : "solutions", "trusted", "world-class" et "leading"
  sont les quatre termes les plus répandus. Leur fréquence d'usage
  indique précisément leur absence de valeur différenciante.

  3. LE SENTIMENT NEUTRE EST LA NORME, PAS L'EXCEPTION
  ──────────────────────────────────────────────────────
  56% des slogans analysés sont neutres sur VADER. Ce taux monte
  à 86% pour les chaînes TV, 65% pour le sport, 64% pour la finance.
  La cosmétique et la restauration sont les seules industries à basculer
  vers le positif (57% et 50% de slogans émotionnels).

  Cela invalide l'hypothèse marketing courante selon laquelle "la copy
  émotionnelle surperforme toujours". La neutralité est une stratégie
  délibérée dans les secteurs où la crédibilité prime sur l'affect.

  4. LA MÉMORABILITÉ DISCRIMINE PLUS QUE LE SENTIMENT
  ─────────────────────────────────────────────────────
  Tous les slogans du top 10 ont un score de mémorabilité de 8 à 10.
  C'est la dimension la plus discriminante du modèle. La présence d'une
  structure sonore (rime, allitération, rythme syllabique régulier)
  est le prédicteur le plus fort d'un score élevé — ce qui confirme
  les travaux en psychologie cognitive sur la "fluency" et la rétention.

  LIMITES DU MODÈLE
  ─────────────────
  Les scores sont construits sur des critères formels (structure, longueur,
  présence de verbes d'action). Ils ne capturent pas : la pertinence
  culturelle, la cohérence avec le positionnement de la marque, l'impact
  réel sur les indicateurs de recall ou de conversion. Ce modèle est un
  outil de présélection et de benchmark structurel, pas un prédicteur
  de performance commerciale.
""".format(without_c, with_c))

    print(sep + '\n')


# =============================================================================
# 10. PIPELINE PRINCIPAL
# =============================================================================

def analyze_slogans(
    scrape_new_data=False,
    data_path='all_slogans.csv',
    output_path='slogans_analyzed.csv',
    visual_path='slogan_analysis_visuals.png',
):
    print('\n' + '=' * 72)
    print('  SLOGAN EFFECTIVENESS ANALYZER')
    print('=' * 72 + '\n')

    # Données
    if scrape_new_data:
        df = scrape_slogans()
        df.to_csv(data_path, index=False, encoding='utf-8')
    else:
        print(f'Chargement : {data_path}')
        df = pd.read_csv(data_path)

    # Pipeline
    df = clean_data(df)
    df = add_basic_metrics(df)
    df = add_sentiment(df)
    df = add_all_scores(df)
    df = add_effectiveness_score(df)
    df = add_clustering(df)

    # Sauvegarde
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f'CSV sauvegardé : {output_path}')

    # Rapport
    generate_report(df)

    # Visuels
    generate_visuals(df, visual_path)

    return df


# =============================================================================
# ENTRÉE
# =============================================================================

if __name__ == '__main__':
    df = analyze_slogans(
        scrape_new_data=False,
        data_path='all_slogans.csv',
        output_path='slogans_analyzed.csv',
        visual_path='slogan_analysis_visuals.png',
    )
