from helper_function import *
import pandas as pd
import numpy as np

def alpha1(o ,h ,l , c , vwap , v , r):
    """(-1 * ts_corr(rank(ts_delta(log(volume), 2)), rank(((close - open) / open)), 6))"""
    s1 = rank(ts_delta(log(v), 2))
    s2 = rank((c / o) - 1)
    alpha = -ts_corr(s1, s2, 6)
    return alpha


def alpha2(o ,h ,l , c , vwap ,v , r):
    """(-1 * ts_corr(rank(open), rank(volume), 10))"""
    return -ts_corr(o,v, 10)


def alpha3(o ,h ,l , c , vwap ,v , r):
    """(rank((open - ts_mean(vwap, 10))) * (-1 * abs(rank((close - vwap)))))"""
    return normalize(rank(o.sub(ts_mean(vwap, 10))).mul(rank(c.sub(vwap)).mul(-1).abs()))


def alpha4(o ,h ,l , c , vwap ,v , r):
    """(-ts_corr(open, volume, 10))"""
    return  -ts_corr(o, v, 10)


def alpha5(o ,h ,l , c , vwap ,v , r):
    return normalize(ts_delta(c,2))


def alpha6(o ,h ,l , c , vwap ,v , r):
    """-rank(((ts_sum(open, 5) * ts_sum(returns, 5)) -
        ts_lag((ts_sum(open, 5) * ts_sum(returns, 5)),10)))
    """
    return -normalize((rank(((ts_sum(o, 5) * ts_sum(r, 5)) -
                       ts_lag((ts_sum(o, 5) * ts_sum(r, 5)), 10)))))
    
    
def alpha7(o ,h ,l , c , vwap ,v , r):
    """(sign(ts_delta(volume, 1)) *
            (-1 * ts_delta(close, 1)))
        """
    return -normalize(sign(ts_delta(v, 1)).mul(ts_delta(c, 1)))



def alpha8(o ,h ,l , c , vwap ,v , r):
    """-rank(ts_cov(rank(close), rank(volume), 5))"""
    return -normalize(rank(ts_cov(rank(c), rank(v), 5)))



def alpha9(o ,h ,l , c , vwap ,v , r):
    """
    (rank(ts_delta(returns, 3))) * ts_corr(open, volume, 10))
    """

    alpha = rank(ts_delta(r, 3)).mul(ts_corr(o, v, 10)
                                      .replace([-np.inf,
                                                np.inf],
                                               np.nan))
    return normalize(alpha)


def alpha10(o ,h ,l , c , vwap ,v , r):
    """-rank(open - ts_lag(high, 1)) *
        rank(open - ts_lag(close, 1)) *
        rank(open -ts_lag(low, 1))"""
    return normalize(rank(o - ts_lag(h, 1)).mul(rank(o - ts_lag(c, 1))).mul(rank(o - ts_lag(l, 1))).mul(-1))


