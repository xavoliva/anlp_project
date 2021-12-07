import os
import string
import re
import time

import nltk
from nltk.tokenize import word_tokenize
# import pandas as pd
import dask.dataframe as dd
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from constants import COLUMNS, EVENTS_DIR, INPUT_DIR

sno = nltk.stem.LancasterStemmer()  # ("english")


def txt_to_list(data_path):
    with open(data_path, "r") as f:
        data = f.read().splitlines()

        return data


STOPWORDS = txt_to_list(f"{INPUT_DIR}/stopwords.txt")


def get_files_from_folder(folder_name, compression="bz2"):
    # return all files as a list
    files = []
    for file in os.listdir(folder_name):
        if file.endswith(f".{compression}"):
            files.append(f"{folder_name}/{file}")

    return sorted(files)


def load_event(event):
    return pd.read_csv(f"{EVENTS_DIR}/{event}.csv", usecols=["author", "body", "created_utc", "affiliation"])


def load_data(data_path, year, months=None, tokenize=False, comp="bz2", dev=False):
    files = get_files_from_folder(
        f"{data_path}/{year}", compression=comp)

    if months:
        files = [f for f in files if int(f[-10:-8]) in months]

    print(f"Loading data of {year}...")

    if dev:
        files = files[:1]

    if comp == "bz2":
        data = dd.read_csv(files,
                           blocksize=None,  # 500e6 = 500MB
                           usecols=COLUMNS,
                           dtype={
                               "subreddit": "string",
                               "author": "string",
                               "body": "string",
                           })

        # keep only day
        data["created_utc"] = dd.to_datetime(
            data["created_utc"], unit="s").dt.date

    elif comp == "parquet":
        data = dd.read_parquet(files,
                               engine="pyarrow",
                               gather_statistics=True)
    else:
        raise NotImplementedError("Compression not allowed.")

    if tokenize:
        print(f"Tokenizing body... (nr_rows = {len(data)})")

        tic = time.perf_counter()
        data["tokens"] = data["body"].apply(
            lambda x: tokenize_post(x, STOPWORDS, stemmer=True))
        toc = time.perf_counter()

        print(f"\tTokenized dataframe in {toc - tic:0.4f} seconds")
    if dev:
        return data.sample(frac=0.01)
    return data


def process_post(text):
    # lower case
    text = text.lower()
    # eliminate urls
    text = re.sub(r"http\S*|\S*\.com\S*|\S*www\S*", " ", text)
    # replace all whitespace with a single space
    text = re.sub(r"\s+", " ", text)
    # strip off spaces on either end
    text = text.strip()

    return text


def tokenize_post(text, keep_stopwords=False, stemmer=True):
    p_text = process_post(text)

    tokens = word_tokenize(p_text)
    # filter punctuation and digits
    tokens = filter(lambda token: token not in string.punctuation and not token.isdigit(), tokens)

    if not keep_stopwords:
        # filter stopwords
        tokens = [t for t in tokens if t not in STOPWORDS]
    # stem words
    if stemmer:
        tokens = [sno.stem(t) for t in tokens]

    return tokens


# Create a SentimentIntensityAnalyzer object.
sia = SentimentIntensityAnalyzer()


def get_sentiment_score(post):
    post = process_post(post)
    return sia.polarity_scores(post)['compound']
