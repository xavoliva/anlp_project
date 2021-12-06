from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp

from constants import EVENTS_DIR, RNG


def get_party_q(party_counts, exclude_author=None):
    author_sum = party_counts.sum(axis=0)

    if exclude_author:
        author_sum -= party_counts[exclude_author, :]

    total_sum = author_sum.sum()

    return author_sum / total_sum


def get_rho(left_q, right_q):
    return (right_q / (left_q + right_q)).transpose()


def calculate_polarization(left_counts, right_counts):
    left_author_total = left_counts.sum(axis=1)
    right_author_total = right_counts.sum(axis=1)

    # get row-wise distributions
    left_author_distr = (
        sp.diags(1 / left_author_total.A.ravel())).dot(left_counts)
    right_author_distr = (
        sp.diags(1 / right_author_total.A.ravel())).dot(right_counts)

    nr_left_authors = left_counts.shape[0]
    nr_right_authors = right_counts.shape[0]

    # make sure there are no zero rows
    assert set(left_author_total.nonzero()[0]) == set(range(nr_left_authors))
    # make sure there are no zero rows
    assert set(right_author_total.nonzero()[0]) == set(range(nr_right_authors))

    left_q = get_party_q(left_counts)
    right_q = get_party_q(right_counts)

    # apply measures via leave-out
    left_sum = 0
    right_sum = 0

    left_author_pols = []
    right_author_pols = []

    for i in range(nr_left_authors):
        left_leaveout_q = get_party_q(left_counts, i)
        token_scores_left = get_rho(left_leaveout_q, right_q)

        left_author_pol = left_author_distr[i, :].dot(token_scores_left)[0, 0]
        left_author_pols.append(left_author_pol)

        left_sum += left_author_pol

    for i in range(nr_right_authors):
        right_leaveout_q = get_party_q(right_counts, i)
        token_scores_right = 1. - get_rho(left_q, right_leaveout_q)

        right_author_pol = right_author_distr[i, :].dot(token_scores_right)[
            0, 0]
        right_author_pols.append(right_author_pol)

        right_sum += right_author_pol

    left_val = 1 / nr_left_authors * left_sum
    right_val = 1 / nr_right_authors * right_sum

    total_pol = 1/2 * (left_val + right_val)

    return total_pol, left_author_pols, right_author_pols


def split_political_affiliation(data):
    return data[data["affiliation"] == "D"], data[data["affiliation"] == "R"]


def get_author_token_counts(posts, vocab):
    authors = posts.groupby('author')
    row_idx = []
    col_idx = []
    data = []

    for author_idx, (author, author_group), in enumerate(authors):
        word_indices = []
        for post in author_group['post']:
            count = 0
            prev_w = ''
            for w in post:
                if w == '':
                    continue
                if w in vocab:
                    word_indices.append(vocab[w])
                if count > 0:
                    bigram = prev_w + ' ' + w
                    if bigram in vocab:
                        word_indices.append(vocab[bigram])
                count += 1
                prev_w = w

        for word_idx, count in Counter(word_indices).items():
            row_idx.append(author_idx)
            col_idx.append(word_idx)
            data.append(count)

    return sp.csr_matrix((data, (row_idx, col_idx)), shape=(len(authors), len(vocab)))


def get_polarization(event, data, default_score=0.5):
    """
    Measure polarization.
    event: name of the event
    data: dataframe with 'post' and 'author'
    default_score: default token partisanship score
    """
    # get partisan posts
    left_posts, right_posts = split_political_affiliation(data)

    # get vocab
    vocab = {w: i for i, w in
             enumerate(open(f"{EVENTS_DIR}/{event}_tokens.txt", "r").read().splitlines())}

    left_counts = get_author_token_counts(left_posts, vocab)
    right_counts = get_author_token_counts(right_posts, vocab)

    left_author_len = left_counts.shape[0]
    right_author_len = right_counts.shape[0]

    author_len = left_author_len + right_author_len

    if left_author_len < 10 or right_author_len < 10:
        # return these values when there is not enough data to make predictions on
        return (default_score, default_score, author_len), ([], [])

    # make the prior neutral (i.e. make sure there are the same number of left and right authors)
    left_author_len = left_counts.shape[0]
    right_author_len = right_counts.shape[0]

    if left_author_len > right_author_len:
        left_subset = np.array(RNG.sample(
            range(left_author_len), right_author_len))
        left_counts = left_counts[left_subset, :]
        left_author_len = left_counts.shape[0]
    elif right_author_len > left_author_len:
        right_subset = np.array(RNG.sample(
            range(right_author_len), left_author_len))
        right_counts = right_counts[right_subset, :]
        right_author_len = right_counts.shape[0]

    assert (left_author_len == right_author_len)

    all_counts = sp.vstack([left_counts, right_counts])

    wordcounts = all_counts.nonzero()[1]

    # filter words used by fewer than 2 people
    all_counts = all_counts[:, np.array(
        [(np.count_nonzero(wordcounts == i) > 1) for i in range(all_counts.shape[1])])]

    left_counts = all_counts[:left_author_len, :]
    right_counts = all_counts[left_author_len:, :]

    left_nonzero = set(left_counts.nonzero()[0])
    right_nonzero = set(right_counts.nonzero()[0])

    # filter authors who did not use words from vocab
    left_counts = left_counts[np.array([(i in left_nonzero) for i in range(
        left_counts.shape[0])]), :]
    right_counts = right_counts[np.array(
        [(i in right_nonzero) for i in range(right_counts.shape[0])]), :]

    pol_val, left_pol_vals, right_pol_vals = calculate_polarization(
        left_counts, right_counts)

    all_counts = sp.vstack([left_counts, right_counts])
    index = np.arange(all_counts.shape[0])
    RNG.shuffle(index)
    shuffled_all_counts = all_counts[index, :]

    # Calculate polarization with random assignment of users
    random_val, _, _ = calculate_polarization(shuffled_all_counts[:left_counts.shape[0], :],
                                              shuffled_all_counts[left_counts.shape[0]:, :])

    return (pol_val, random_val, author_len), (left_pol_vals, right_pol_vals)


def split_by_day(data):
    return [(v, k) for k, v in data.groupby("time")]

def split_by_week(data):
    data.time = pd.to_datetime(data['time'])

    return [(v, k) for k, v in data.groupby(pd.Grouper(key="time", freq="W"))]


def get_polarization_by_time(event, data, freq="day"):
    if freq == "day":
        data_time = split_by_day(data)
    elif freq == "week":
        data_time = split_by_week(data)

    pol = []
    for d, date in data_time:
        pol_data, _ = get_polarization(event, d)
        pol_val, random_val, author_len = pol_data
        pol.append((pol_val, random_val, author_len, pd.to_datetime(date)))

    return pd.DataFrame(pol, columns=["pol", "random_pol", "author_len", "time"])
