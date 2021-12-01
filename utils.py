import os
import string
import re
import time

import nltk
from nltk.tokenize import word_tokenize
# import pandas as pd
import dask.dataframe as dd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from constants import COLUMNS

sno = nltk.stem.SnowballStemmer("english")


def get_files_from_folder(folder_name, compression="bz2"):
    # return all files as a list
    files = []
    for file in os.listdir(folder_name):
        if file.endswith(f".{compression}"):
            files.append(f"{folder_name}/{file}")

    return sorted(files)


def load_data(data_path, year, tokenize=False, comp="bz2", dev=False):
    files = get_files_from_folder(
        f"{data_path}/{year}", compression=comp)

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
            lambda x: tokenize_post(x, STOPWORDS, stem=True))
        toc = time.perf_counter()

        print(f"\tTokenized dataframe in {toc - tic:0.4f} seconds")
    if dev:
        return data.sample(frac=0.01)
    return data


def txt_to_list(data_path):
    with open(data_path, "r") as f:
        stopwords = f.read().splitlines()

        return stopwords


STOPWORDS = txt_to_list("data/stopwords.txt")


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


def tokenize_post(text, stopwords, stem=True):
    p_text = process_post(text)

    tokens = word_tokenize(p_text)
    # filter punctuation
    tokens = filter(lambda token: token not in string.punctuation, tokens)
    # filter stopwords
    tokens = [t for t in tokens if t not in stopwords]
    # stem words
    if stem:
        tokens = [sno.stem(t) for t in tokens]

    return tokens


# Create a SentimentIntensityAnalyzer object.
sia = SentimentIntensityAnalyzer()


def get_sentiment_score(post):
    post = process_post(post)
    return sia.polarity_scores(post)['compound']
