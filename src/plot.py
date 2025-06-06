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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import RocCurveDisplay
import torch
from torch.amp import autocast
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt


def plot_class_distribution(news, target="is_fake"):
    """Plots the distribution of fake and real news articles."""
    news[target].\
    value_counts(normalize=True).\
    rename({0: "Real", 1: "Fake"}).\
    plot(
        kind="bar",
        title="Distribution of Fake/Real News",
        color=["#33FFB8", "#FF5733"],
        alpha=0.5,
        edgecolor="black",
        xlabel="News Type",
        ylabel="Proportion",
    )

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
    plt.figure(figsize=(10, 5))
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

    plt.figure(figsize=(8, top_n * 0.2))
    sns.barplot(x=list(freqs), y=list(words), edgecolor="k", alpha=0.5)
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.title(f"Top {top_n} words")
    plt.tight_layout()
    plt.show()

def plot_corr_matrix(
    df,
    features,
    figsize=(10,8),
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    mask_upper=True
):
    """ Plot a correlation matrix for the given features."""
    corr = df[features].corr()
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .8},
    )
    plt.title("Feature Correlation Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_pairwise(
    df,
    features,
    hue=None,
    sample_size=None,
    diag_kind="kde",
    corner=True,
    height=2.5
):
    """ Plot pairwise relationships for the given features, optionally colored by a class. """
    plot_df = df[features + ([hue] if hue else [])].dropna()
    if sample_size and len(plot_df) > sample_size:
        plot_df = plot_df.sample(sample_size, random_state=42)

    g = sns.pairplot(
        plot_df,
        vars=features,
        hue=hue,
        diag_kind=diag_kind,
        corner=corner,
        height=height
    )
    # tighten title spacing
    plt.suptitle("Pairwise Feature Relationships", y=1.02)
    plt.tight_layout()
    plt.show()

def plot_logreg_evaluation(split_name, model, X, y):
    """ Prints classification report, ROC-AUC, ROC curve, and confusion matrix for a given data split. """
    y_preds = model.predict(X)
    y_probs = model.predict_proba(X)[:, 1]

    print(f"\n--- {split_name} Results ---")
    print(classification_report(y, y_preds, target_names=["real", "fake"]))
    auc = roc_auc_score(y, y_probs)
    print(f"ROC-AUC: {auc:.4f}")

    RocCurveDisplay.from_estimator(model, X, y)
    plt.title(f"{split_name} ROC Curve")
    plt.show()

    cm = confusion_matrix(y, y_preds)
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["real", "fake"],
        yticklabels=["real", "fake"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{split_name} Confusion Matrix")
    plt.show()

def plot_lstm_evaluation(split_name, model, X, y_true):
    """ Prints classification report, ROC-AUC, ROC curve, and confusion matrix for a given data split. """
    y_proba = model.predict(X).flatten()
    y_pred  = (y_proba >= 0.5).astype(int)

    print(f"--- {split_name} Results ---")
    print(classification_report(y_true, y_pred, target_names=["real","fake"]))
    auc = roc_auc_score(y_true, y_proba)
    print(f"ROC-AUC: {auc:.4f}\n")

    # ROC Curve
    RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.title(f"{split_name} ROC Curve")
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["real","fake"], yticklabels=["real","fake"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{split_name} Confusion Matrix")
    plt.show()
    
def plot_bert_evaluation(split_name, model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        with torch.no_grad(), autocast(device_type=device.type):
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = np.argmax(logits.cpu().numpy(), axis=1)

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    # Metrics
    acc  = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")
    auc  = roc_auc_score(all_labels, all_probs)
    print(f"\n--- {split_name} Metrics ---")
    print(f"Accuracy : {acc: .4f}")
    print(f"Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print(f"ROC‐AUC  : {auc:.4f}\n")

    # ROC Curve
    from sklearn.metrics import RocCurveDisplay
    RocCurveDisplay.from_predictions(all_labels, all_probs)
    plt.title(f"{split_name} ROC Curve")
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["real", "fake"], yticklabels=["real", "fake"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{split_name} Confusion Matrix")
    plt.show()
