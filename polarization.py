from collections import Counter
import scipy.sparse as sp
import numpy as np


def get_token_user_counts(party_counts):
    no_tokens = party_counts.shape[1]
    nonzero = sp.find(party_counts)[:2]
    user_t_counts = Counter(nonzero[1])  # number of users using each term
    party_t = np.ones(no_tokens)  # add one smoothing
    for k, v in user_t_counts.items():
        party_t[k] += v
    return party_t


def get_party_q(party_counts, exclude_user_id=None):
    user_sum = party_counts.sum(axis=0)
    if exclude_user_id:
        user_sum -= party_counts[exclude_user_id, :]
    total_sum = user_sum.sum()
    return user_sum / total_sum


def get_rho(left_q, right_q):
    return (right_q / (left_q + right_q)).transpose()


def mutual_information(left_t, right_t, left_not_t, right_not_t, left_no, right_no):
    no_users = left_no + right_no
    all_t = left_t + right_t
    all_not_t = no_users - all_t + 4
    mi_left_t = left_t * np.log2(no_users * (left_t / (all_t * left_no)))
    mi_left_not_t = left_not_t * \
        np.log2(no_users * (left_not_t / (all_not_t * left_no)))
    mi_right_t = right_t * np.log2(no_users * (right_t / (all_t * right_no)))
    mi_right_not_t = right_not_t * \
        np.log2(no_users * (right_not_t / (all_not_t * right_no)))
    return (1 / no_users * (mi_left_t + mi_left_not_t + mi_right_t + mi_right_not_t)).transpose()[:, np.newaxis]


def chi_square(left_t, right_t, left_not_t, right_not_t, left_no, right_no):
    no_users = left_no + right_no
    all_t = left_t + right_t
    all_not_t = no_users - all_t + 4
    chi_enum = no_users * (left_t * right_not_t - left_not_t * right_t) ** 2
    chi_denom = all_t * all_not_t * \
        (left_t + left_not_t) * (right_t + right_not_t)
    return (chi_enum / chi_denom).transpose()[:, np.newaxis]


def calculate_polarization(left_counts, right_counts, measure="posterior", leaveout=True):
    left_user_total = left_counts.sum(axis=1)
    right_user_total = right_counts.sum(axis=1)

    left_user_distr = (sp.diags(1 / left_user_total.A.ravel())
                       ).dot(left_counts)  # get row-wise distributions
    right_user_distr = (
        sp.diags(1 / right_user_total.A.ravel())).dot(right_counts)
    left_no = left_counts.shape[0]
    right_no = right_counts.shape[0]
    assert (set(left_user_total.nonzero()[0]) == set(
        range(left_no)))  # make sure there are no zero rows
    assert (set(right_user_total.nonzero()[0]) == set(
        range(right_no)))  # make sure there are no zero rows
    if measure not in ('posterior', 'mutual_information', 'chi_square'):
        print('invalid method')
        return
    left_q = get_party_q(left_counts)
    right_q = get_party_q(right_counts)
    left_t = get_token_user_counts(left_counts)
    right_t = get_token_user_counts(right_counts)
    left_not_t = left_no - left_t + 2  # because of add one smoothing
    right_not_t = right_no - right_t + 2  # because of add one smoothing
    func = mutual_information if measure == 'mutual_information' else chi_square

    # apply measure without leave-out
    if not leaveout:
        if measure == 'posterior':
            token_scores_rep = get_rho(left_q, right_q)
            token_scores_dem = 1. - token_scores_rep
        else:
            token_scores_dem = func(
                left_t, right_t, left_not_t, right_not_t, left_no, right_no)
            token_scores_rep = token_scores_dem
        left_val = 1 / left_no * left_user_distr.dot(token_scores_dem).sum()
        right_val = 1 / right_no * right_user_distr.dot(token_scores_rep).sum()
        return 1/2 * (left_val + right_val)

    # apply measures via leave-out
    left_addup = 0
    right_addup = 0
    left_leaveout_no = left_no - 1
    right_leaveout_no = right_no - 1
    for i in range(left_no):
        if measure == 'posterior':
            left_leaveout_q = get_party_q(left_counts, i)
            token_scores_dem = 1. - get_rho(left_leaveout_q, right_q)
        else:
            left_leaveout_t = left_t.copy()
            excl_user_terms = sp.find(left_counts[i, :])[1]
            for term_idx in excl_user_terms:
                left_leaveout_t[term_idx] -= 1
            left_leaveout_not_t = left_leaveout_no - left_leaveout_t + 2
            token_scores_dem = func(
                left_leaveout_t, right_t, left_leaveout_not_t, right_not_t, left_leaveout_no, right_no)
        left_addup += left_user_distr[i, :].dot(token_scores_dem)[0, 0]
    for i in range(right_no):
        if measure == 'posterior':
            right_leaveout_q = get_party_q(right_counts, i)
            token_scores_rep = get_rho(left_q, right_leaveout_q)
        else:
            right_leaveout_t = right_t.copy()
            excl_user_terms = sp.find(right_counts[i, :])[1]
            for term_idx in excl_user_terms:
                right_leaveout_t[term_idx] -= 1
            right_leaveout_not_t = right_leaveout_no - right_leaveout_t + 2
            token_scores_rep = func(
                left_t, right_leaveout_t, left_not_t, right_leaveout_not_t,
                left_no, right_leaveout_no)
        right_addup += right_user_distr[i, :].dot(token_scores_rep)[0, 0]
    right_val = 1 / right_no * right_addup
    left_val = 1 / left_no * left_addup
    return 1/2 * (left_val + right_val)
