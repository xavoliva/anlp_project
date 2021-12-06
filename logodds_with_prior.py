import math
from collections import Counter
from heapq import nlargest, nsmallest


def logodds_with_prior(one_tokens, two_tokens, display=25, priors=None):
    types = set(one_tokens + two_tokens)

    V = len(types)

    n_one = len(one_tokens)
    n_two = len(two_tokens)

    y_one = Counter(one_tokens)
    y_two = Counter(two_tokens)

    xi = {}

    if priors:
        alpha_0 = 1000
    else:
        alpha_w = 0.01
        alpha_0 = V * alpha_w

    for w in types:
        if priors:
            alpha_w = priors[w] * alpha_0

        d_hat = math.log((y_one[w] + alpha_w) / (n_one + alpha_0 - y_one[w] - alpha_w)) - \
            math.log((y_two[w] + alpha_w) /
                     (n_two + alpha_0 - y_two[w] - alpha_w))
        sigma_squared = 1 / (y_one[w] + alpha_w) + 1 / (y_two[w] + alpha_w)

        xi[w] = d_hat / math.sqrt(sigma_squared)

    return nlargest(display, xi, key=xi.get), nsmallest(display, xi, key=xi.get)


def get_priors(tokens):
    counts = Counter(tokens)

    freqs = {}
    total = len(tokens)

    for w in counts:
        freqs[w] = counts[w] / total

    return freqs
