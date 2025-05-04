"""
Plots histograms of features, barcharts and word clouds for fake and real news articles.
"""

from collections import Counter
import re
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
import pandas as pd
from nltk.corpus import stopwords


def plot_feature_histogram(news, column, title, xlabel, ylabel="Frequency"):
    """Plots histograms of a specified column for fake and real news."""
    bins = get_bin_width_fd(news[column])

    plt.hist(
        news[news["is_fake"] == 1][column],
        bins=bins,
        alpha=0.5,
        label="Fake",
        stacked=True,
        color="#FF5733",
        edgecolor="black",
    )

    plt.hist(
        news[news["is_fake"] == 0][column],
        bins=bins,
        alpha=0.5,
        label="Real",
        stacked=True,
        color="#33FFB8",
        edgecolor="black",
    )

    plt.title(title)
    plt.xlabel(xlabel + " (5th–95th percentile)")
    plt.ylabel(ylabel)
    plt.legend(title="Article is")
    plt.tight_layout()
    plt.show()


def get_bin_width_fd(data):
    """Calculates the bin width using Freedman-Diaconis rule."""
    low = data.quantile(0.05)
    high = data.quantile(0.95)
    iqr = data.quantile(0.75) - data.quantile(0.25)
    bin_width = 2 * iqr / np.power(len(data), 1 / 3)
    n_bins = int(np.ceil((high - low) / bin_width))
    return np.linspace(low, high, n_bins)


def plot_wordcloud(text):
    """Plots word cloud for given text."""
    wordcloud = WordCloud(
        background_color="white",
        stopwords=set(STOPWORDS),
        max_words=300,
        width=800,
        height=400,
    )

    wordcloud = wordcloud.generate(str(text))

    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


STOPWORDS = set(stopwords.words("english"))
SYMBOL_RE = re.compile(r"^[^A-Za-z0-9]+$")


def plot_top_words_barchart(text: pd.Series, top_n: int = 20):
    """Plots a horizontal bar chart of the top_n most common non-stopword tokens."""

    # Split and flatten the series into a single list of tokens
    tokens = text.dropna().astype(str).str.split().explode().str.strip().str.lower()

    is_symbol = tokens.str.match(SYMBOL_RE).infer_objects(copy=False)
    is_stop = tokens.isin(STOPWORDS)
    mask = ~(is_symbol | is_stop)

    filtered = tokens[mask]

    most_common = Counter(filtered).most_common(top_n)
    if not most_common:
        return  # nothing to plot

    words, freqs = zip(*most_common)

    # 4) Plot
    plt.figure(figsize=(8, top_n * 0.2))
    sns.barplot(x=list(freqs), y=list(words), edgecolor="k")
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.title(f"Top {top_n} words")
    plt.tight_layout()
    plt.show()
