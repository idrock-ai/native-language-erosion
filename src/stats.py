"""Exact paired statistics for the erosion analyses. Pure stdlib, unit-tested."""
from __future__ import annotations
from math import comb, erfc, sqrt


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p for discordant counts (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * p, 100 * (centre - half), 100 * (centre + half)


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def cochran_armitage(ks: list[int], ns: list[int], scores: list[float] | None = None):
    """Trend test for proportions ks/ns over ordered groups. Returns (z, two-sided p)."""
    s = scores or list(range(len(ks)))
    N, K = sum(ns), sum(ks)
    if N == 0 or K in (0, N):
        return 0.0, 1.0
    pbar = K / N
    t = sum(si * (ki - ni * pbar) for si, ki, ni in zip(s, ks, ns))
    var = pbar * (1 - pbar) * (sum(ni * si * si for si, ni in zip(s, ns))
                               - sum(ni * si for si, ni in zip(s, ns)) ** 2 / N)
    if var <= 0:
        return 0.0, 1.0
    z = t / sqrt(var)
    return z, erfc(abs(z) / sqrt(2))


def flips(pairs: list[tuple[int, int]]) -> tuple[int, int]:
    """(b, c): b = first right & second wrong; c = first wrong & second right."""
    b = sum(1 for a, x in pairs if a and not x)
    c = sum(1 for a, x in pairs if not a and x)
    return b, c


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]] by summing every hypergeometric
    table at most as probable as the observed one."""
    n, r1, c1 = a + b + c + d, a + b, a + c
    if n == 0 or r1 in (0, n) or c1 in (0, n):
        return 1.0

    def prob(x):
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)

    p_obs = prob(a)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    return min(1.0, sum(p for x in range(lo, hi + 1)
                        if (p := prob(x)) <= p_obs * (1 + 1e-9)))


def mantel_haenszel_or(strata: list[tuple[int, int, int, int]]) -> float:
    """Mantel-Haenszel common odds ratio over strata of 2x2 tables [[a, b], [c, d]].
    Pools the association while allowing each stratum its own baseline rate. Returns
    inf when no stratum contributes to the denominator."""
    num = den = 0.0
    for a, b, c, d in strata:
        n = a + b + c + d
        if n:
            num += a * d / n
            den += b * c / n
    return num / den if den else float("inf")
