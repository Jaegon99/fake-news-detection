import re
import pandas as pd
from textstat import textstat
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob

URL_RE = re.compile(r"https?://\S+|www\.\S+")
NONALPHANUM = re.compile(r"[^a-z0-9\s]|\n|\r|\t")
MULTISPACE = re.compile(r"\s{2,}")

def basic_clean(txt):
    """
    Clean the text by removing URLs, HTML tags, and non-alphanumeric characters.
    Also converts to lowercase and removes extra spaces.
    """
    txt = txt.lower()
    txt = BeautifulSoup(txt, "html.parser").get_text()
    txt = re.sub(URL_RE, "", txt)
    txt = re.sub(NONALPHANUM, " ", txt)
    txt = re.sub(MULTISPACE, " ", txt).strip()
    return txt


LEM = WordNetLemmatizer()
STOP = set(stopwords.words("english"))

def preprocess_text(txt):
    """Lemmatize the text by tokenizing, removing stopwords, and lemmatizing."""
    tokens = word_tokenize(txt)
    return " ".join(
        LEM.lemmatize(tok) for tok in tokens
        if tok.isalpha() and tok not in STOP
    )


def get_sentiment(text):
    """Get the sentiment polarity and subjectivity of a text."""
    sent = TextBlob(text).sentiment
    return sent.polarity, sent.subjectivity



def get_features(news: pd.DataFrame) -> pd.DataFrame:
    """ Compute basic text features for the news articles. """
    features = pd.DataFrame()

    features['word_count'] = news['clean_text'].str.split().map(len)
    features['char_count'] = news['clean_text'].str.len()
    features['avg_word_length'] = features['char_count'] / features['word_count']

    features['polarity'], features['subjectivity'] = zip(*news["clean_text"].apply(get_sentiment))

    features['reading_ease'] = news['raw_text'].map(textstat.flesch_reading_ease)
    features['smog_index'] = news['raw_text'].map(textstat.smog_index)
    return features
