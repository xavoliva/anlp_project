import operator
from collections import Counter

import pandas as pd

from utils import tokenize_post, txt_to_list
from constants import INPUT_DIR, EVENTS_DIR

# mass shooting event names
events = txt_to_list(INPUT_DIR + "event_names.txt")


def get_all_vocabs(seed_val):
    vocabs = []
    for e in events:
        # TODO
        data = pd.read_csv(f"{EVENTS_DIR}/{e}/{e}.csv",
                           usecols=['text', 'remove', 'isRT'])

        # print(e, len(data))
        # sample a (quasi-)equal number of tweets from each event
        # this has to be done to eliminate words that are too specific to a particular event
        data = data.sample(min(len(data), 10000), random_state=seed_val)
        word_counts = Counter(tokenize_post(
            ' '.join(data['text']), keep_stopwords=True))
        vocab = []
        for k, v in word_counts.items():
            if v >= 10:  # keep words that occur at least 10 times
                vocab.append(k)
        vocabs.append(set(vocab))
    return vocabs


def word_event_count(vocabs):
    word_event_count = {}
    for vocab in vocabs:
        for w in vocab:
            if w in word_event_count:
                word_event_count[w] += 1
            else:
                word_event_count[w] = 1
    # Keep all words that occur in at least three events' tweets. Note that we keep stopwords.
    keep = [k for k, v in sorted(word_event_count.items(), key=operator.itemgetter(
        1), reverse=True) if (v > 2 and not k.isdigit())]
    print(len(keep))
    return set(keep)


def build_vocab(corpus, cutoff=50):
    freq = {}
    for words in corpus:
        prev = ""
        count = 0
        for i, w in enumerate(words):
            if w in freq:
                freq[w] += 1
            else:
                freq[w] = 1
            if count > 0:
                bigram = prev + " " + w
                if bigram in freq:
                    freq[bigram] += 1
                else:
                    freq[bigram] = 1
            count += 1
            prev = w
    # keep unigrams / bigrams that occur at least cutoff times
    vocab = [k for k, v in sorted(
        freq.items(), key=operator.itemgetter(1), reverse=True) if v > cutoff]
    return vocab


def build_event_vocabs(events):
    for i, e in enumerate(events):
        print(e)
        data = pd.read_csv(f"{EVENTS_DIR}/{e}/{e}.csv",
                           usecols=['text', 'remove', 'isRT'])

        cleaned = data['text'].apply(
            tokenize_post, args=(False, e))  # don't keep stopwords
        vocab = build_vocab(cleaned)
        print(len(vocab))
        with open(f"{EVENTS_DIR}/{e}/{e}_vocab.txt", 'w') as f:
            f.write('\n'.join(vocab))
