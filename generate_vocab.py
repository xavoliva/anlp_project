import operator
from collections import Counter

import pandas as pd

from utils import tokenize_post
from constants import EVENTS_DIR, MIN_OCCURENCE_FOR_VOCAB, EVENTS


def get_all_vocabs(seed_val):
    vocabs = []
    for event in EVENTS:
        data = pd.read_csv(f"{EVENTS_DIR}/{event}.csv", usecols=["body"])

        # print(e, len(data))
        # sample a (quasi-)equal number of tweets from each event
        # this has to be done to eliminate words that are too specific to a particular event
        data = data.sample(min(len(data), 10000), random_state=seed_val)
        word_counts = Counter(
            tokenize_post(" ".join(data["body"]), keep_stopwords=True)
        )
        vocab = []
        for word, cnt in word_counts.items():
            if cnt >= 10:  # keep words that occur at least 10 times
                vocab.append(word)
        vocabs.append(set(vocab))
    return vocabs


def word_event_count(vocabs):
    word_event_cnt = {}
    for vocab in vocabs:
        for word in vocab:
            if word in word_event_cnt:
                word_event_cnt[word] += 1
            else:
                word_event_cnt[word] = 1
    # Keep all words that occur in at least three events' tweets. Note that we keep stopwords.
    keep = [
        word
        for word, cnt in sorted(
            word_event_cnt.items(), key=operator.itemgetter(1), reverse=True
        )
        if (cnt > 2 and not word.isdigit())
    ]
    print(len(keep))
    return set(keep)


def build_vocab(corpus):
    uni_and_bigrams = Counter(corpus)

    for i in range(1, len(corpus)):
        word = corpus[i]
        prev_word = corpus[i - 1]
        uni_and_bigrams.update([" ".join([prev_word, word])])

    vocab = [
        word
        for word, cnt in sorted(
            uni_and_bigrams.items(), key=operator.itemgetter(1), reverse=True
        )
        if cnt > MIN_OCCURENCE_FOR_VOCAB
    ]

    return vocab


def build_event_vocab(event):
    data = pd.read_csv(f"{EVENTS_DIR}/{event}.csv", usecols=["body"])

    corpus = tokenize_post(" ".join(data["body"]), stemmer=False, keep_stopwords=False)

    vocab = build_vocab(corpus)
    print("Vocab length:", len(vocab))

    return vocab
