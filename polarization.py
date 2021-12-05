from collections import Counter
import sys
import gc

import numpy as np
import scipy.sparse as sp

from constants import EVENTS_DIR, RNG


def get_token_user_counts(party_counts):
    nr_tokens = party_counts.shape[1]
    nonzero = sp.find(party_counts)[:2]

    # number of users using each term
    user_t_counts = Counter(nonzero[1])

    # add one smoothing
    party_t = np.ones(nr_tokens)
    for k, v in user_t_counts.items():
        party_t[k] += v

    return party_t


def get_party_q(party_counts, exclude_author=None):
    user_sum = party_counts.sum(axis=0)

    if exclude_author:
        user_sum -= party_counts[exclude_author, :]

    total_sum = user_sum.sum()

    return user_sum / total_sum


def get_rho(left_q, right_q):
    return (right_q / (left_q + right_q)).transpose()


def mutual_information(left_t, right_t, left_not_t, right_not_t, left_no, right_no):
    nr_users = left_no + right_no

    all_t = left_t + right_t
    all_not_t = nr_users - all_t + 4

    mi_left_t = left_t * np.log2(nr_users * (left_t / (all_t * left_no)))
    mi_left_not_t = left_not_t * \
        np.log2(nr_users * (left_not_t / (all_not_t * left_no)))

    mi_right_t = right_t * np.log2(nr_users * (right_t / (all_t * right_no)))
    mi_right_not_t = right_not_t * \
        np.log2(nr_users * (right_not_t / (all_not_t * right_no)))

    return (1 / nr_users * (mi_left_t + mi_left_not_t + mi_right_t + mi_right_not_t)).transpose()[:, np.newaxis]


def chi_square(left_t, right_t, left_not_t, right_not_t, left_no, right_no):
    nr_users = left_no + right_no

    all_t = left_t + right_t
    all_not_t = nr_users - all_t + 4

    chi_enum = nr_users * (left_t * right_not_t - left_not_t * right_t) ** 2
    chi_denom = all_t * all_not_t * \
        (left_t + left_not_t) * (right_t + right_not_t)

    return (chi_enum / chi_denom).transpose()[:, np.newaxis]


def calculate_polarization(left_counts, right_counts, measure="posterior"):
    left_user_total = left_counts.sum(axis=1)
    right_user_total = right_counts.sum(axis=1)

    # get row-wise distributions
    left_user_distr = (
        sp.diags(1 / left_user_total.A.ravel())).dot(left_counts)
    right_user_distr = (
        sp.diags(1 / right_user_total.A.ravel())).dot(right_counts)
    left_no = left_counts.shape[0]
    right_no = right_counts.shape[0]

    # make sure there are no zero rows
    assert set(left_user_total.nonzero()[0]) == set(range(left_no))
    # make sure there are no zero rows
    assert set(right_user_total.nonzero()[0]) == set(range(right_no))

    if measure not in ('posterior', 'mutual_information', 'chi_square'):
        raise ValueError("Invalid Method")

    left_q = get_party_q(left_counts)
    right_q = get_party_q(right_counts)
    left_t = get_token_user_counts(left_counts)
    right_t = get_token_user_counts(right_counts)

    # add one smoothing
    left_not_t = left_no - left_t + 2
    right_not_t = right_no - right_t + 2

    func = mutual_information if measure == 'mutual_information' else chi_square

    # apply measures via leave-out
    left_addup = 0
    right_addup = 0

    left_leaveout_no = left_no - 1
    right_leaveout_no = right_no - 1

    for i in range(left_no):
        if measure == 'posterior':
            left_leaveout_q = get_party_q(left_counts, i)
            token_scores_left = 1. - get_rho(left_leaveout_q, right_q)
        else:
            left_leaveout_t = left_t.copy()
            excl_user_terms = sp.find(left_counts[i, :])[1]
            for term_idx in excl_user_terms:
                left_leaveout_t[term_idx] -= 1
            left_leaveout_not_t = left_leaveout_no - left_leaveout_t + 2
            token_scores_left = func(
                left_leaveout_t, right_t, left_leaveout_not_t, right_not_t,
                left_leaveout_no, right_no)
        left_addup += left_user_distr[i, :].dot(token_scores_left)[0, 0]

    for i in range(right_no):
        if measure == 'posterior':
            right_leaveout_q = get_party_q(right_counts, i)
            token_scores_right = get_rho(left_q, right_leaveout_q)
        else:
            right_leaveout_t = right_t.copy()
            excl_user_terms = sp.find(right_counts[i, :])[1]

            for term_idx in excl_user_terms:
                right_leaveout_t[term_idx] -= 1

            right_leaveout_not_t = right_leaveout_no - right_leaveout_t + 2

            token_scores_right = func(
                left_t, right_leaveout_t, left_not_t, right_leaveout_not_t,
                left_no, right_leaveout_no)

        right_addup += right_user_distr[i, :].dot(token_scores_right)[0, 0]

    right_val = 1 / right_no * right_addup
    left_val = 1 / left_no * left_addup
    return 1/2 * (left_val + right_val)


def split_political_orientation(data):
    return data[data["politics"] == "D"], data[data["politics"] == "R"]


def get_user_token_counts(posts, vocab):
    users = posts.groupby('author')
    row_idx = []
    col_idx = []
    data = []

    for group_idx, (_, group), in enumerate(users):
        word_indices = []
        for split in group['post']:
            count = 0
            prev = ''
            for w in split:
                if w == '':
                    continue
                if w in vocab:
                    word_indices.append(vocab[w])
                if count > 0:
                    bigram = prev + ' ' + w
                    if bigram in vocab:
                        word_indices.append(vocab[bigram])
                count += 1
                prev = w
        for k, v in Counter(word_indices).items():
            col_idx.append(group_idx)
            row_idx.append(k)
            data.append(v)
    return sp.csr_matrix((data, (col_idx, row_idx)), shape=(len(users), len(vocab)))


def get_polarization(event, data, token_partisanship_measure='posterior', default_score=0.5):
    """
    Measure polarization.
    event: name of the event
    data: dataframe with 'post' and 'author'
    token_partisanship_measure: type of measure for calculating token partisanship based on user-token counts
    between_topic: whether the estimate is between topics or tokens
    default_score: default token partisanship score
    """
    # get partisan posts
    left_posts, right_posts = split_political_orientation(data)

    # get vocab
    vocab = {w: i for i, w in
             enumerate(open(f"{EVENTS_DIR}/{event}_vocab.txt", "r").read().splitlines())}
    left_counts = get_user_token_counts(left_posts, vocab)
    right_counts = get_user_token_counts(right_posts, vocab)

    left_user_len = left_counts.shape[0]
    right_user_len = right_counts.shape[0]
    if left_user_len < 10 or right_user_len < 10:
        # return these values when there is not enough data to make predictions on
        return default_score, default_score, left_user_len + right_user_len
    del left_posts
    del right_posts
    del data
    gc.collect()

    # make the prior neutral (i.e. make sure there are the same number of Rep and Dem users)
    left_user_len = left_counts.shape[0]
    right_user_len = right_counts.shape[0]
    if left_user_len > right_user_len:
        left_subset = np.array(RNG.sample(
            range(left_user_len), right_user_len))
        left_counts = left_counts[left_subset, :]
        left_user_len = left_counts.shape[0]
    elif right_user_len > left_user_len:
        right_subset = np.array(RNG.sample(
            range(right_user_len), left_user_len))
        right_counts = right_counts[right_subset, :]
        right_user_len = right_counts.shape[0]
    assert (left_user_len == right_user_len)

    all_counts = sp.vstack([left_counts, right_counts])

    wordcounts = all_counts.nonzero()[1]

    # filter words used by fewer than 2 people
    all_counts = all_counts[:, np.array(
        [(np.count_nonzero(wordcounts == i) > 1) for i in range(all_counts.shape[1])])]

    left_counts = all_counts[:left_user_len, :]
    right_counts = all_counts[left_user_len:, :]
    del wordcounts
    del all_counts
    gc.collect()

    left_nonzero = set(left_counts.nonzero()[0])
    right_nonzero = set(right_counts.nonzero()[0])

    # filter users who did not use words from vocab
    left_counts = left_counts[np.array([(i in left_nonzero) for i in range(
        left_counts.shape[0])]), :]
    right_counts = right_counts[np.array(
        [(i in right_nonzero) for i in range(right_counts.shape[0])]), :]

    del left_nonzero
    del right_nonzero
    gc.collect()

    actual_val = calculate_polarization(
        left_counts, right_counts, token_partisanship_measure)

    all_counts = sp.vstack([left_counts, right_counts])
    del left_counts
    del right_counts
    gc.collect()

    index = np.arange(all_counts.shape[0])
    RNG.shuffle(index)
    all_counts = all_counts[index, :]

    random_val = calculate_polarization(all_counts[:left_user_len, :],
                                        all_counts[left_user_len:, :],
                                        token_partisanship_measure)

    print(actual_val, random_val, left_user_len + right_user_len)
    sys.stdout.flush()
    del all_counts
    gc.collect()

    return actual_val, random_val, left_user_len + right_user_len


def split_by_day(data):
    return [(v, int(k.replace("-", ""))) for k, v in data.groupby("time")]


def get_polarization_by_day(event, data):
    method = "posterior"

    data_days = split_by_day(data)

    nr_splits = len(data_days)

    # TODO: CHANGE
    pol = np.zeros((nr_splits, 4))
    for i, (d, day) in enumerate(data_days):
        pol[i, :3] = get_polarization(event, d, method)
        pol[i, 3] = day

    return pol
