import os
import string
import re
import time

import nltk
from nltk.tokenize import word_tokenize
import pandas as pd

sno = nltk.stem.SnowballStemmer('english')


def get_files_from_folder(folder_name, compression="bz2"):
    # return all files as a list
    files = []
    for file in os.listdir(folder_name):
        if file.endswith(f".{compression}"):
            files.append(f"{folder_name}/{file}")
            
    return files


def load_data(data_path, year, tokenize=False, frac=None):
    files = get_files_from_folder(f"{data_path}/{year}")
    dfs = []
    print("Loading data...")
    # TODO: ALL FILES
    for f in files:
            print(f)
            df = pd.read_csv(f)
            dfs.append(df)
            
    data = pd.concat(dfs)
    
    if frac:
        data = data.sample(frac=frac)
    
    data = data.set_index(["subreddit_id"])
    
    data['created_utc'] = pd.to_datetime(data['created_utc'], unit='s')
    
    with open("data/stopwords.txt", "r") as f:
        stopwords = f.read().splitlines()
    
    print(f"Tokenizing body... (nr_rows = {len(data)})")
    if tokenize:
        tic = time.perf_counter()
        data['tokens'] = data['body'].apply(lambda x: get_tokens(x, stopwords, stem=True))
        toc = time.perf_counter()
        
        print(f"\tTokenized dataframe in {toc - tic:0.4f} seconds")

    
    return data


def get_tokens(text, stopwords, stem=True):
    # lower case
    text = text.lower()
    # eliminate urls
    text = re.sub(r'http\S*|\S*\.com\S*|\S*www\S*', ' ', text)
    # replace all whitespace with a single space
    text = re.sub(r'\s+', ' ', text)
    # strip off spaces on either end
    text = text.strip()
    tokens = word_tokenize(text)
    # filter punctuation
    tokens = filter(lambda token: token not in string.punctuation, tokens)
    # filter stopwords
    tokens = [t for t in tokens if t not in stopwords]
    # stem words
    if stem:
        tokens = [sno.stem(t) for t in tokens]
        
    return tokens
