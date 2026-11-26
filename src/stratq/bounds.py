"""
Finite-sample anytime-valid confidence sequences and bounds for StratQ.

The confidence sequence uses a fixed-population, uniform without-replacement
reveal law and a control variate fixed before each batch.  ``safe_rounding``
adds directed binary64 enclosures for the statistical accumulators and final
endpoints.  ``select_optimal_shrinkage_theta`` defaults to a predictable,
analytically cross-validated Ridge gate; its ``legacy`` mode preserves the
previous in-sample-variance selector for ablation reproduction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


#: IEEE-754 binary64 unit roundoff (half the distance from 1 to its successor).
UNIT_ROUNDOFF = 2.0 ** -53
DEFAULT_KAPPA_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


_INF = math.inf
_NEG_INF = -math.inf
_ISFINITE = math.isfinite
_NEXTAFTER = math.nextafter


def _finite(value):
    if isinstance(value, (float, int)):
        fval = float(value)
        if not _ISFINITE(fval):
            raise FloatingPointError("Non-finite value in a directed-rounding enclosure")
        return fval
    if np.isscalar(value):
        fval = float(value)
        if not _ISFINITE(fval):
            raise FloatingPointError("Non-finite value in a directed-rounding enclosure")
        return fval
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise FloatingPointError("Non-finite value in a directed-rounding enclosure")
    return array


def _down(value):
    """One binary64 step downward from a correctly rounded result."""
    if isinstance(value, float):
        if not _ISFINITE(value):
            raise FloatingPointError("Non-finite value in a directed-rounding enclosure")
        return _NEXTAFTER(value, _NEG_INF)
    value = _finite(value)
    if isinstance(value, float):
        return _NEXTAFTER(value, _NEG_INF)
    return np.nextafter(value, -np.inf)


def _up(value):
    """One binary64 step upward from a correctly rounded result."""
    if isinstance(value, float):
        if not _ISFINITE(value):
            raise FloatingPointError("Non-finite value in a directed-rounding enclosure")
        return _NEXTAFTER(value, _INF)
    value = _finite(value)
    if isinstance(value, float):
        return _NEXTAFTER(value, _INF)
    return np.nextafter(value, math.inf)


def _add_interval(a, b):
    if isinstance(a, tuple) and isinstance(b, tuple):
        a0, a1 = a
        b0, b1 = b
        if isinstance(a0, float) and isinstance(b0, float):
            if b0 == 0.0 and b1 == 0.0:
                return a
            if a0 == 0.0 and a1 == 0.0:
                return b
            s0 = a0 + b0
            s1 = a1 + b1
            if not (_ISFINITE(s0) and _ISFINITE(s1)):
                raise FloatingPointError("Non-finite value in a directed-rounding enclosure")
            return _NEXTAFTER(s0, _NEG_INF), _NEXTAFTER(s1, _INF)
    if np.isscalar(a):
        a = (a, a)
    if np.isscalar(b):
        b = (b, b)
    # Adding exact zero is exact and needs no artificial widening.
    if np.all(np.asarray(b) == 0.0):
        return a
    if np.all(np.asarray(a) == 0.0):
        return b
    return _down(a[0] + b[0]), _up(a[1] + b[1])


def _sub_interval(a, b):
    if isinstance(a, tuple) and isinstance(b, tuple):
        a0, a1 = a
        b0, b1 = b
        if isinstance(a0, float) and isinstance(b0, float):
            if b0 == 0.0 and b1 == 0.0:
                return a
            d0 = a0 - b1
            d1 = a1 - b0
            if not (_ISFINITE(d0) and _ISFINITE(d1)):
                raise FloatingPointError("Non-finite value in a directed-rounding enclosure")
            return _NEXTAFTER(d0, _NEG_INF), _NEXTAFTER(d1, _INF)
    if np.isscalar(a):
        a = (a, a)
    if np.isscalar(b):
        b = (b, b)
    if np.all(np.asarray(b) == 0.0):
        return a
    return _down(a[0] - b[1]), _up(a[1] - b[0])


def _mul_interval(a, b):
    # Preserve exact 0/1 identities (notably normalized feature extrema).
    if isinstance(a, tuple) and isinstance(b, tuple):
        a0, a1 = a
        b0, b1 = b
        if isinstance(a0, float) and isinstance(b0, float):
            if (a0 == 0.0 and a1 == 0.0) or (b0 == 0.0 and b1 == 0.0):
                return 0.0, 0.0
            if a0 == 1.0 and a1 == 1.0:
                return b
            if b0 == 1.0 and b1 == 1.0:
                return a
            p0 = a0 * b0
            p1 = a0 * b1
            p2 = a1 * b0
            p3 = a1 * b1
            lo = p0 if p0 <= p1 else p1
            if p2 < lo:
                lo = p2
            if p3 < lo:
                lo = p3
            hi = p0 if p0 >= p1 else p1
            if p2 > hi:
                hi = p2
            if p3 > hi:
                hi = p3
            if not (_ISFINITE(lo) and _ISFINITE(hi)):
                raise FloatingPointError("Non-finite value in a directed-rounding enclosure")
            return _NEXTAFTER(lo, _NEG_INF), _NEXTAFTER(hi, _INF)
    if np.all(np.asarray(a) == 0.0) or np.all(np.asarray(b) == 0.0):
        zero = 0.0
        return zero, zero
    if np.all(np.asarray(a) == 1.0):
        return b
    if np.all(np.asarray(b) == 1.0):
        return a
    products = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return _down(min(products)), _up(max(products))


def _div_interval(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    b0, b1 = b
    if b0 <= 0.0:
        raise FloatingPointError("Directed division requires a positive denominator")
    a0, a1 = a
    if isinstance(a0, float) and isinstance(a1, float) and isinstance(b0, float) and isinstance(b1, float):
        q0 = a0 / b1
        q1 = a1 / b0
        if not (_ISFINITE(q0) and _ISFINITE(q1)):
            raise FloatingPointError("Non-finite value in a directed-rounding enclosure")
        return _NEXTAFTER(q0, _NEG_INF), _NEXTAFTER(q1, _INF)
    return _down(a0 / b1), _up(a1 / b0)


def _clip_interval(interval: Tuple[float, float]) -> Tuple[float, float]:
    return max(0.0, interval[0]), min(1.0, interval[1])


def nonnegative_sum_interval(values: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Outward enclosure of sums of finite binary64 values in ``[0, 1]``.

    A standard floating-point summation error bound is
    ``gamma_n * sum(abs(x)) <= gamma_n * n`` with
    ``gamma_n = n*u/(1-n*u)``.  This also covers NumPy's pairwise reduction.
    """
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("nonnegative_sum_interval requires finite values in [0, 1]")
    n = int(values.shape[axis])
    if n <= 0:
        raise ValueError("cannot sum an empty axis")
    numerator = _up(n * UNIT_ROUNDOFF)
    denominator = _down(1.0 - n * UNIT_ROUNDOFF)
    gamma = _up(numerator / denominator)
    error = _up(n * gamma)
    total = np.sum(values, axis=axis)
    lower = np.maximum(0.0, _down(total - error))
    upper = np.minimum(float(n), _up(total + error))
    return lower, upper


def _square_interval(interval: Tuple[float, float]) -> Tuple[float, float]:
    product = _mul_interval(interval, interval)
    if interval[0] <= 0.0 <= interval[1]:
        low = 0.0
    else:
        low = product[0]
    return low, product[1]


def _dot_interval(theta: np.ndarray, phi: np.ndarray) -> Tuple[float, float]:
    """Directed dot product of two binary64 vectors (inputs are exact reals)."""
    acc = (0.0, 0.0)
    for coefficient, feature in zip(theta, phi):
        acc = _add_interval(acc, _mul_interval((float(coefficient),) * 2,
                                               (float(feature),) * 2))
    return acc


def psi_e(lam: float) -> float:
    """Empirical-Bernstein cumulant ``-log(1-lam)-lam`` for ``0 <= lam < 1``."""
    if not 0.0 <= lam < 1.0:
        raise ValueError("lam must be in [0, 1)")
    # log1p is accurate near zero; psi's public scalar API is not the safe path.
    return -math.log1p(-lam) - lam


@lru_cache(maxsize=8192)
def _psi_e_enclosure_cached(lam: float) -> Tuple[float, float]:
    if lam == 0.0:
        return 0.0, 0.0
    low = 0.0
    high = 0.0
    p0 = _NEXTAFTER(lam * lam, _NEG_INF)
    p1 = _NEXTAFTER(lam * lam, _INF)
    for exponent in range(2, 98):
        if exponent > 2:
            p0 = _NEXTAFTER(p0 * lam, _NEG_INF)
            p1 = _NEXTAFTER(p1 * lam, _INF)
        denom = float(exponent)
        t0 = _NEXTAFTER(p0 / denom, _NEG_INF)
        t1 = _NEXTAFTER(p1 / denom, _INF)
        low = _NEXTAFTER(low + t0, _NEG_INF)
        high = _NEXTAFTER(high + t1, _INF)
    # sum_{k=98}^inf lam^k/k <= lam^98 / (98*(1-lam)).
    p98_0 = _NEXTAFTER(p0 * lam, _NEG_INF)
    p98_1 = _NEXTAFTER(p1 * lam, _INF)
    oml_0 = _NEXTAFTER(1.0 - lam, _NEG_INF)
    oml_1 = _NEXTAFTER(1.0 - lam, _INF)
    den_0 = _NEXTAFTER(98.0 * oml_0, _NEG_INF)
    den_1 = _NEXTAFTER(98.0 * oml_1, _INF)
    tail_0 = _NEXTAFTER(p98_0 / den_1, _NEG_INF)
    tail_1 = _NEXTAFTER(p98_1 / den_0, _INF)
    return _NEXTAFTER(low + tail_0, _NEG_INF), _NEXTAFTER(high + tail_1, _INF)


def psi_e_enclosure(lam: float) -> Tuple[float, float]:
    """Rigorous binary64 enclosure of ``psi(lam)`` using its positive series.

    For the supported betting range ``0 <= lam <= 0.5``, 96 terms leave a
    geometric tail below 2**-100.  Every multiplication, division and addition
    is rounded outward; the tail is also bounded outward.  This avoids relying
    on an unspecified error bound for a platform's transcendental ``log``.
    """
    lam = _finite(lam)
    if not 0.0 <= lam <= 0.5:
        raise ValueError("safe psi enclosure supports 0 <= lam <= 0.5")
    return _psi_e_enclosure_cached(lam)


def _log_threshold_enclosure(alpha: float) -> Tuple[float, float]:
    """Outward binary64 enclosure of ``log(2/alpha)`` via 80-digit Decimal.

    ``Decimal.ln`` is correctly rounded to the active decimal precision.  We
    therefore bound its rounding error by one adjacent Decimal value *before*
    converting to binary64; a single ``nextafter`` around the directly rounded
    float would not by itself prove enclosure if the exact value were unusually
    close to a binary64 rounding boundary.
    """
    with localcontext() as ctx:
        ctx.prec = 80
        exact_alpha = Decimal.from_float(float(alpha))
        value = (Decimal(2) / exact_alpha).ln()
        lower_decimal = value.next_minus()
        upper_decimal = value.next_plus()
        lower = float(lower_decimal)
        upper = float(upper_decimal)
    return _down(lower), _up(upper)


def hoeffding_radius(n: int, alpha: float, value_range: float = 1.0) -> float:
    """Standard fixed-sample two-sided Hoeffding half-width."""
    if n <= 0:
        return value_range
    return value_range * math.sqrt(math.log(2.0 / alpha) / (2.0 * n))


def empirical_bernstein_radius(
    sample_var: float, n: int, alpha: float, value_range: float = 1.0
) -> float:
    """Audibert-Munos-Szepesvari / Maurer-Pontil fixed-sample EB half-width."""
    if n <= 1:
        return value_range
    log_term = math.log(3.0 / alpha)
    return math.sqrt(2.0 * max(sample_var, 0.0) * log_term / n) + (
        3.0 * value_range * log_term / n
    )


@dataclass
class FixedSampleEB:
    """Simple fixed-sample Empirical-Bernstein bound on ``[lo, hi]``."""

    lo: float = 0.0
    hi: float = 1.0

    def interval(self, values: np.ndarray, alpha: float) -> Tuple[float, float, float]:
        n = len(values)
        if n == 0:
            return (0.5 * (self.lo + self.hi), self.lo, self.hi)
        mean = float(np.mean(values))
        if n == 1:
            return (mean, self.lo, self.hi)
        var = float(np.var(values, ddof=1))
        rad = empirical_bernstein_radius(var, n, alpha, self.hi - self.lo)
        return (mean, max(self.lo, mean - rad), min(self.hi, mean + rad))


class WoRControlVariateCS:
    """Predictable control-variate EB confidence sequence for uniform WoR draws.

    Parameters
    ----------
    safe_rounding:
        If true, all statistical arithmetic is enclosed with directed binary64
        operations.  An empty floating-point intersection is widened to the
        convex hull of its conservative components rather than restarted.  See
        ``PROOF_PACKAGE.md`` for the resulting coverage argument.
    """

    def __init__(
        self,
        n_total: int,
        alpha: float,
        phi_mean_init: np.ndarray,
        phi_min_g: Optional[np.ndarray] = None,
        phi_max_g: Optional[np.ndarray] = None,
        y_min: float = 0.0,
        y_max: float = 1.0,
        target_eps: Optional[float] = None,
        use_wor: bool = True,
        lam_cap: float = 0.5,
        safe_rounding: bool = True,
        phi_sum_interval: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        if n_total <= 0:
            raise ValueError("n_total must be positive")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        if not 0.0 < lam_cap <= 0.5:
            raise ValueError("lam_cap must be in (0, 0.5]")
        self.n_total = int(n_total)
        self.alpha = float(alpha)
        self.log_thresh = math.log(2.0 / self.alpha)
        self._log_thresh_interval = (
            _log_threshold_enclosure(self.alpha) if safe_rounding
            else (self.log_thresh, self.log_thresh)
        )
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        span_raw = float(y_max) - float(y_min)
        if span_raw <= 0.0:
            raise ValueError("y_max must exceed y_min")
        self.y_span = span_raw
        self._y_span_interval = (
            (_down(span_raw), _up(span_raw))
            if safe_rounding and self.y_min != 0.0
            else (span_raw, span_raw)
        )
        self.target_eps_norm = (
            None if target_eps is None else float(target_eps) / self.y_span
        )
        self.use_wor = bool(use_wor)
        self.lam_cap = float(lam_cap)
        self.safe_rounding = bool(safe_rounding)

        phi_mean_init = np.asarray(phi_mean_init, dtype=np.float64)
        self.phi_sum_total = phi_mean_init * self.n_total
        if self.safe_rounding:
            if phi_sum_interval is None:
                # Backward-compatible contract: normalized features lie in
                # [0,1], so the mean reduction has an explicit n*u envelope.
                self._phi_total_lo, self._phi_total_hi = self._mean_times_total_interval(
                    phi_mean_init
                )
            else:
                low, high = (np.asarray(v, dtype=np.float64) for v in phi_sum_interval)
                if low.shape != phi_mean_init.shape or high.shape != low.shape or np.any(low > high):
                    raise ValueError("phi_sum_interval has incompatible dimensions")
                self._phi_total_lo, self._phi_total_hi = low.copy(), high.copy()
        else:
            self._phi_total_lo = self._phi_total_hi = self.phi_sum_total
        self.phi_sum_seen = np.zeros_like(self.phi_sum_total)
        self._phi_seen_lo = np.zeros_like(self.phi_sum_total)
        self._phi_seen_hi = np.zeros_like(self.phi_sum_total)
        self.d = int(self.phi_sum_total.shape[0])

        self.phi_min_g = (
            np.zeros(self.d, dtype=np.float64)
            if phi_min_g is None
            else np.asarray(phi_min_g, dtype=np.float64).copy()
        )
        self.phi_max_g = (
            np.ones(self.d, dtype=np.float64)
            if phi_max_g is None
            else np.asarray(phi_max_g, dtype=np.float64).copy()
        )
        self.phi_min_g[0] = 1.0
        self.phi_max_g[0] = 1.0

        self.n_seen = 0
        self.y_sum_seen = 0.0
        self._y_sum_interval = (0.0, 0.0)
        self.observed_count: Optional[int] = 0

        # A_t is rounded down.  B_t has a lower/upper enclosure; V_t rounds up.
        self.sum_gamma = 0.0
        self.sum_c = 0.0
        self.sum_v = 0.0
        self._b_interval = (0.0, 0.0)

        self.e_sq_sum = 0.12
        self.e_count = 1.0
        self.resid_sum = 0.0
        self._resid_interval = (0.0, 0.0)

        self.lo_norm = 0.0
        self.hi_norm = 1.0
        self.intersection_collisions = 0
        # Safe mode deliberately does not restart: convex-hull widening is a
        # superset of every component and therefore cannot lose a covered mean.
        self.empty_intersection_restarts = 0
        self.lambda_history: List[float] = []

    def _mean_times_total_interval(self, phi_mean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Enclose ``N*mean(phi)`` when only a normalized mean is supplied."""
        n = self.n_total
        numerator = _up(n * UNIT_ROUNDOFF)
        denominator = _down(1.0 - n * UNIT_ROUNDOFF)
        gamma = _up(numerator / denominator)
        total_error = _up(n * gamma)
        base_low = np.array([_down(float(v) * n) for v in phi_mean])
        base_high = np.array([_up(float(v) * n) for v in phi_mean])
        return np.maximum(0.0, _down(base_low - total_error)), np.minimum(
            float(n), _up(base_high + total_error)
        )

    @property
    def A_t(self) -> float:
        """Conservative (downward-rounded) betting-weight accumulator."""
        return self.sum_gamma

    @property
    def B_t(self) -> Tuple[float, float]:
        """Outward enclosure of the centered betting accumulator."""
        return self._b_interval if self.safe_rounding else (self.sum_c, self.sum_c)

    @property
    def V_t(self) -> float:
        """Conservative (upward-rounded) variance-penalty accumulator."""
        return self.sum_v

    @property
    def deterministic_count_bounds(self) -> Optional[Tuple[int, int]]:
        """Exact integer count endpoints when every revealed label is binary."""
        if self.observed_count is None:
            return None
        return self.observed_count, self.observed_count + self.n_total - self.n_seen

    @staticmethod
    def theta_box_bounds(
        theta: np.ndarray,
        phi_min_g: Optional[np.ndarray] = None,
        phi_max_g: Optional[np.ndarray] = None,
        *,
        safe_rounding: bool = False,
    ) -> Tuple[float, float]:
        """Closed-form O(d) min/max of ``theta^T phi`` over a feature box.

        In safe mode each signed extremal sum is enclosed, so the returned
        interval contains the exact real range of the stored binary64 inputs.
        """
        theta = np.asarray(theta, dtype=np.float64)
        if len(theta) <= 1:
            value = float(theta[0])
            return (_down(value), _up(value)) if safe_rounding else (value, value)
        rest = theta[1:]
        if phi_min_g is None or phi_max_g is None:
            pmin, pmax = None, None
        else:
            pmin = np.asarray(phi_min_g, dtype=np.float64)[1:]
            pmax = np.asarray(phi_max_g, dtype=np.float64)[1:]
        if not safe_rounding:
            t0 = float(theta[0])
            if pmin is None:
                return (t0 + float(np.sum(np.minimum(rest, 0.0))),
                        t0 + float(np.sum(np.maximum(rest, 0.0))))
            lo_terms = np.where(rest >= 0.0, rest * pmin, rest * pmax)
            hi_terms = np.where(rest >= 0.0, rest * pmax, rest * pmin)
            return t0 + float(np.sum(lo_terms)), t0 + float(np.sum(hi_terms))

        t0 = (float(theta[0]),) * 2
        low_terms, high_terms = [], []
        for j, coefficient in enumerate(rest):
            coefficient = float(coefficient)
            if coefficient >= 0.0:
                lo_feature = 0.0 if pmin is None else float(pmin[j])
                hi_feature = 1.0 if pmax is None else float(pmax[j])
            else:
                lo_feature = 1.0 if pmax is None else float(pmax[j])
                hi_feature = 0.0 if pmin is None else float(pmin[j])
            low_terms.append(_down(coefficient * lo_feature))
            high_terms.append(_up(coefficient * hi_feature))
        low, high = t0
        for term in low_terms:
            low = _down(low + term)
        for term in high_terms:
            high = _up(high + term)
        return low, high

    @staticmethod
    def project_theta_span(
        theta: np.ndarray,
        max_span: float = 1.0,
        phi_min_g: Optional[np.ndarray] = None,
        phi_max_g: Optional[np.ndarray] = None,
        exact_bounds: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """Project linear weights so ``f_max-f_min <= max_span``."""
        out = np.array(theta, dtype=np.float64, copy=True)
        if len(out) <= 1:
            return out
        if exact_bounds is not None:
            f_min, f_max = exact_bounds
        else:
            f_min, f_max = WoRControlVariateCS.theta_box_bounds(out, phi_min_g, phi_max_g)
        span = f_max - f_min
        if span > max_span and span > 1e-12:
            scale = max_span / span
            out[1:] *= scale
            out[0] = float(np.clip(out[0], -1.0, 2.0))
        return out

    def _safe_f_bounds(
        self, theta: np.ndarray, exact_f_bounds: Optional[Tuple[float, float]]
    ) -> Tuple[float, float]:
        if exact_f_bounds is not None:
            # The safe engine path supplies an outward enclosure.  The name
            # ``exact`` is the caller's contract; do not widen a certified
            # enclosure a second time.
            low, high = float(exact_f_bounds[0]), float(exact_f_bounds[1])
            if not low <= high:
                raise ValueError("exact_f_bounds must be ordered")
            return low, high
        return self.theta_box_bounds(theta, self.phi_min_g, self.phi_max_g,
                                     safe_rounding=True)

    def _normalize_y(self, raw: float) -> Tuple[float, float]:
        if not self.safe_rounding:
            return float(np.clip((raw - self.y_min) / self.y_span, 0.0, 1.0)), None
        numerator = _sub_interval((float(raw),) * 2, (self.y_min,) * 2)
        return _clip_interval(_div_interval(numerator, self._y_span_interval))

    def _deterministic_interval(self) -> Tuple[float, float]:
        count_bounds = self.deterministic_count_bounds
        if count_bounds is not None and self.safe_rounding and self.y_min == 0.0 and self.y_max == 1.0:
            endpoints = []
            for index, integer in enumerate(count_bounds):
                value = integer / self.n_total
                endpoints.append(value if Fraction(value) == Fraction(integer, self.n_total)
                                 else (_down(value) if index == 0 else _up(value)))
            return endpoints[0], endpoints[1]
        interval = _div_interval(self._y_sum_interval, (float(self.n_total),) * 2)
        if self.use_wor:
            remaining = (float(self.n_total - self.n_seen),) * 2
            upper = _add_interval((interval[1],) * 2,
                                  _div_interval(remaining, (float(self.n_total),) * 2))[1]
            return interval[0], upper
        return 0.0, 1.0

    def update_batch(
        self,
        y_raw: np.ndarray,
        phi_batch: np.ndarray,
        theta_pred: np.ndarray,
        exact_f_bounds: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Reveal a batch in order using one predictor fixed before the batch.

        No label or row-dependent statistic from the current batch enters the
        predictor, betting parameter, center, or range before that row's step.
        """
        y_raw = np.asarray(y_raw, dtype=np.float64)
        phi_b = np.asarray(phi_batch, dtype=np.float64)
        theta = np.asarray(theta_pred, dtype=np.float64)
        if y_raw.shape != (len(phi_b),) or len(theta) != self.d:
            raise ValueError("Incompatible label, feature, or predictor dimensions")
        if self.safe_rounding:
            self._update_batch_safe(y_raw, phi_b, theta, exact_f_bounds)
        else:
            self._update_batch_legacy(y_raw, phi_b, theta, exact_f_bounds)

    def _update_batch_legacy(self, y_raw, phi_b, theta, exact_f_bounds) -> None:
        """Original arithmetic path retained for bit-level legacy comparisons."""
        y_norm = np.clip((y_raw - self.y_min) / self.y_span, 0.0, 1.0)
        if exact_f_bounds is not None:
            f_min, f_max = exact_f_bounds
        else:
            f_min, f_max = self.theta_box_bounds(theta, self.phi_min_g, self.phi_max_g)
        R_s = 1.0 + max(f_max - f_min, 0.0)
        for idx in range(len(y_norm)):
            if self.n_seen >= self.n_total:
                break
            y_i, phi_i = float(y_norm[idx]), phi_b[idx]
            rem = self.n_total - self.n_seen
            if self.use_wor:
                phi_rem = (self.phi_sum_total - self.phi_sum_seen) / rem
                w_s = self.n_total / rem
                seen_shift = self.y_sum_seen / rem
            else:
                phi_rem = self.phi_sum_total / self.n_total
                w_s, seen_shift = 1.0, 0.0
            f_rem, f_i = float(np.dot(theta, phi_rem)), float(np.dot(theta, phi_i))
            a_s = f_rem - f_max
            Z_s = f_rem + (y_i - f_i)
            tz_s = min(1.0, max(0.0, (Z_s - a_s) / R_s))
            if np.any(theta[1:] != 0.0):
                avg_resid = self.resid_sum / self.n_seen if self.n_seen > 0 else 0.0
                m_rem_pred = min(1.0, max(0.0, f_rem + 0.5 * avg_resid))
            else:
                m_rem_pred = (self.y_sum_seen + 0.5) / (self.n_seen + 1.0)
            tz_pred = min(1.0, max(0.0, (m_rem_pred - a_s) / R_s))
            s_step = self.n_seen + 1
            sig2_e = max(self.e_sq_sum / self.e_count, 1e-4)
            if self.target_eps_norm is not None and self.target_eps_norm > 0.0:
                lam_s = self.target_eps_norm * R_s / sig2_e
            else:
                lam_s = R_s * math.sqrt(2.0 * self.log_thresh /
                    (sig2_e * s_step * math.log(s_step + math.e)))
            lam_s = min(self.lam_cap, max(1e-5, lam_s))
            self.lambda_history.append(float(lam_s))
            scale = lam_s / R_s
            gamma_s = scale * w_s
            c_s = scale * (Z_s + seen_shift)
            v_s = psi_e(lam_s) * ((tz_s - tz_pred) ** 2)
            self.sum_gamma += gamma_s
            self.sum_c += c_s
            self.sum_v += v_s
            self.e_sq_sum += (Z_s - m_rem_pred) ** 2
            self.e_count += 1.0
            self.resid_sum += y_i - f_i
            self.n_seen += 1
            self.y_sum_seen += y_i
            self.phi_sum_seen += phi_i
            if self.observed_count is not None:
                if float(y_raw[idx]) in (0.0, 1.0):
                    self.observed_count += int(y_raw[idx])
                else:
                    self.observed_count = None
            self._intersect_legacy()

    def _intersect_legacy(self) -> None:
        if self.sum_gamma > 0.0:
            mu_hat = self.sum_c / self.sum_gamma
            rad = (self.log_thresh + self.sum_v) / self.sum_gamma
            lo_eb, hi_eb = mu_hat - rad, mu_hat + rad
        else:
            lo_eb, hi_eb = 0.0, 1.0
        if self.use_wor:
            det_lo = self.y_sum_seen / self.n_total
            det_hi = (self.y_sum_seen + (self.n_total - self.n_seen)) / self.n_total
        else:
            det_lo, det_hi = 0.0, 1.0
        if self.exhausted:
            self.lo_norm, self.hi_norm = det_lo, det_hi
        else:
            new_lo = max(self.lo_norm, lo_eb, det_lo)
            new_hi = min(self.hi_norm, hi_eb, det_hi)
            if new_lo > new_hi:
                self.empty_intersection_restarts += 1
                self.lo_norm, self.hi_norm = det_lo, det_hi
            else:
                self.lo_norm, self.hi_norm = new_lo, new_hi

    def _update_batch_safe(self, y_raw, phi_b, theta, exact_f_bounds) -> None:
        f_min, f_max = self._safe_f_bounds(theta, exact_f_bounds)
        one = (1.0, 1.0)
        R_interval = _add_interval(one, _sub_interval((f_max,) * 2, (f_min,) * 2))
        R_s = max(R_interval[1], 1e-12)
        for idx in range(len(y_raw)):
            if self.n_seen >= self.n_total:
                break
            phi_i = phi_b[idx]
            y_i = self._normalize_y(float(y_raw[idx]))
            rem = self.n_total - self.n_seen
            if self.use_wor:
                remaining_phi = _sub_interval(
                    (self._phi_total_lo, self._phi_total_hi),
                    (self._phi_seen_lo, self._phi_seen_hi),
                )
                phi_rem = _div_interval(remaining_phi, (float(rem),) * 2)
                w_interval = _div_interval((float(self.n_total),) * 2,
                                           (float(rem),) * 2)
                w_s = self.n_total / rem
                seen_shift = _div_interval(self._y_sum_interval, (float(rem),) * 2)
            else:
                phi_rem = _div_interval((self._phi_total_lo, self._phi_total_hi),
                                        (float(self.n_total),) * 2)
                w_s, seen_shift = 1.0, (0.0, 0.0)
            # Componentwise enclosure of a vector dot product.
            f_rem = (0.0, 0.0)
            for j, coefficient in enumerate(theta):
                feature = (float(phi_rem[0][j]), float(phi_rem[1][j]))
                f_rem = _add_interval(f_rem, _mul_interval((float(coefficient),) * 2,
                                                            feature))
            f_i = _dot_interval(theta, phi_i)
            a_s = _sub_interval(f_rem, (f_max,) * 2)
            Z_s = _sub_interval(_add_interval(f_rem, y_i), f_i)
            tz_s = _clip_interval(_div_interval(_sub_interval(Z_s, a_s), R_interval))

            if np.any(theta[1:] != 0.0):
                if self.n_seen > 0:
                    avg_resid = _div_interval(self._resid_interval,
                                              (float(self.n_seen),) * 2)
                    residual_term = _mul_interval((0.5,) * 2, avg_resid)
                else:
                    residual_term = (0.0, 0.0)
                m_rem_pred = _clip_interval(_add_interval(f_rem, residual_term))
            else:
                numerator = _add_interval(self._y_sum_interval, (0.5,) * 2)
                m_rem_pred = _clip_interval(_div_interval(numerator,
                                                          (float(self.n_seen + 1),) * 2))
            tz_pred = _clip_interval(_div_interval(_sub_interval(m_rem_pred, a_s), R_interval))

            s_step = self.n_seen + 1
            sig2_e = max(self.e_sq_sum / self.e_count, 1e-4)
            if self.target_eps_norm is not None and self.target_eps_norm > 0.0:
                lam_s = self.target_eps_norm * R_s / sig2_e
            else:
                lam_s = R_s * math.sqrt(2.0 * self.log_thresh /
                    (sig2_e * s_step * math.log(s_step + math.e)))
            lam_s = min(self.lam_cap, max(1e-5, lam_s))
            self.lambda_history.append(float(lam_s))
            lam_interval = (lam_s, lam_s)
            scale = _div_interval(lam_interval, R_interval)
            gamma = _mul_interval(scale, w_interval)
            c_term = _mul_interval(scale, _add_interval(Z_s, seen_shift))
            centered = _sub_interval(tz_s, tz_pred)
            v_term = _mul_interval(psi_e_enclosure(lam_s), _square_interval(centered))

            self.sum_gamma = _down(self.sum_gamma + gamma[0])
            self._b_interval = _add_interval(self._b_interval, c_term)
            self.sum_c = 0.5 * (self._b_interval[0] + self._b_interval[1])
            self.sum_v = _up(self.sum_v + v_term[1])
            residual = _sub_interval(Z_s, m_rem_pred)
            self.e_sq_sum += 0.5 * (residual[0] ** 2 + residual[1] ** 2)
            self.e_count += 1.0
            row_residual = _sub_interval(y_i, f_i)
            self._resid_interval = _add_interval(self._resid_interval, row_residual)
            self.resid_sum = 0.5 * (self._resid_interval[0] + self._resid_interval[1])

            self.n_seen += 1
            self._y_sum_interval = _add_interval(self._y_sum_interval, y_i)
            self.y_sum_seen = 0.5 * (self._y_sum_interval[0] + self._y_sum_interval[1])
            self._phi_seen_lo = np.array([_down(a + float(b)) for a, b in
                                          zip(self._phi_seen_lo, phi_i)])
            self._phi_seen_hi = np.array([_up(a + float(b)) for a, b in
                                          zip(self._phi_seen_hi, phi_i)])
            self.phi_sum_seen += phi_i
            if self.observed_count is not None:
                if float(y_raw[idx]) in (0.0, 1.0):
                    self.observed_count += int(y_raw[idx])
                else:
                    self.observed_count = None
            self._intersect_safe()

    def _intersect_safe(self) -> None:
        det_lo, det_hi = self._deterministic_interval()
        if self.sum_gamma > 0.0:
            penalty = _add_interval(self._log_thresh_interval, (self.sum_v,) * 2)
            lower_num = _sub_interval((self._b_interval[0],) * 2, penalty)
            upper_num = _add_interval((self._b_interval[1],) * 2, penalty)
            denominator = (self.sum_gamma,) * 2
            lo_eb = _div_interval(lower_num, denominator)[0]
            hi_eb = _div_interval(upper_num, denominator)[1]
        else:
            lo_eb, hi_eb = 0.0, 1.0
        if self.exhausted:
            self.lo_norm, self.hi_norm = det_lo, det_hi
            return
        if det_lo > self.hi_norm or det_hi < self.lo_norm:
            # Conservative disjointness proves that the exact previous and
            # deterministic intervals are disjoint.  The true mean cannot be
            # in both, so this restart lies outside the joint coverage event.
            self.empty_intersection_restarts += 1
            self.lo_norm, self.hi_norm = det_lo, det_hi
            return
        new_lo = max(self.lo_norm, lo_eb, det_lo)
        new_hi = min(self.hi_norm, hi_eb, det_hi)
        if new_lo <= new_hi:
            self.lo_norm, self.hi_norm = new_lo, new_hi
        else:
            # Conservative widening can create a three-way collision even when
            # the exact-real intersection is nonempty.  Do not claim that this
            # proves an exact empty intersection: retain a convex superset so
            # that no possibly covered mean is discarded.
            self.intersection_collisions += 1
            self.lo_norm = min(self.lo_norm, lo_eb, det_lo)
            self.hi_norm = max(self.hi_norm, hi_eb, det_hi)

    @property
    def bounds(self) -> Tuple[float, float]:
        if not self.safe_rounding:
            lo = self.y_min + self.lo_norm * self.y_span
            hi = self.y_min + self.hi_norm * self.y_span
            return lo, hi
        lo = _add_interval((self.y_min,) * 2,
                           _mul_interval((self.lo_norm,) * 2, self._y_span_interval))[0]
        hi = _add_interval((self.y_min,) * 2,
                           _mul_interval((self.hi_norm,) * 2, self._y_span_interval))[1]
        return lo, hi

    @property
    def estimate(self) -> float:
        lo, hi = self.bounds
        return lo + 0.5 * (hi - lo)

    @property
    def half_width(self) -> float:
        lo, hi = self.bounds
        width = 0.5 * (hi - lo)
        return _up(width) if self.safe_rounding else width

    @property
    def exhausted(self) -> bool:
        return self.use_wor and (self.n_seen >= self.n_total)


def _dot_rows_interval(theta: np.ndarray, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized outward dot products for every row of a feature matrix."""
    theta = np.asarray(theta, dtype=np.float64)
    features = np.asarray(features, dtype=np.float64)
    low = np.full(len(features), float(theta[0]), dtype=np.float64)
    high = low.copy()
    for j, coefficient in enumerate(theta[1:]):
        coefficient = float(coefficient)
        if coefficient == 0.0:
            continue
        column = features[:, j]
        if coefficient == 1.0:
            term_low = term_high = column
        else:
            product = coefficient * column
            term_low = np.nextafter(product, -np.inf)
            term_high = np.nextafter(product, np.inf)
        low = np.nextafter(low + term_low, -np.inf)
        high = np.nextafter(high + term_high, np.inf)
    return low, high


def linear_prediction_bounds(
    theta: np.ndarray,
    phi_group_eval: Optional[np.ndarray] = None,
    phi_min_g: Optional[np.ndarray] = None,
    phi_max_g: Optional[np.ndarray] = None,
    *,
    safe_rounding: bool = True,
) -> Tuple[float, float]:
    """Outward range of a predictable linear score on a row slice or box."""
    theta = np.asarray(theta, dtype=np.float64)
    if not np.any(theta[1:] != 0.0):
        value = float(theta[0])
        return value, value
    if phi_group_eval is not None and len(phi_group_eval) <= 25000:
        features = np.asarray(phi_group_eval, dtype=np.float64)
        if not safe_rounding:
            values = features @ theta
            return float(np.min(values)), float(np.max(values))
        row_low, row_high = _dot_rows_interval(theta, features)
        return float(np.min(row_low)), float(np.max(row_high))
    return WoRControlVariateCS.theta_box_bounds(
        theta, phi_min_g, phi_max_g, safe_rounding=safe_rounding
    )


def ridge_loo_residuals(
    phi_seen: np.ndarray,
    residual: np.ndarray,
    kappa: float,
    ridge_lambda: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Analytic leave-one-out residuals for a centered, shrunken Ridge fit.

    The intercept is unpenalized and slopes start from zero correction.  For
    ``kappa>0`` the scaled design is ``Z=kappa*Xc`` and the smoother is
    ``J/n + Z*(Z'Z+lambda*I)^-1*Z'``; its hat diagonal gives every exact Ridge
    LOO residual without refitting ``n`` times.
    ``kappa=0`` is the pure no-proxy intercept-only baseline.
    """
    X = np.asarray(phi_seen, dtype=np.float64)
    r = np.asarray(residual, dtype=np.float64)
    n = len(r)
    if X.shape[0] != n or X.ndim != 2:
        raise ValueError("phi_seen and residual must have matching rows")
    if ridge_lambda <= 0.0 or not 0.0 <= kappa <= 1.0:
        raise ValueError("ridge_lambda must be positive and kappa in [0, 1]")
    centered_r = r - float(np.mean(r))
    if X.shape[1] <= 1 or kappa == 0.0:
        return centered_r / (1.0 - 1.0 / n), np.full(n, 1.0 / n), n - 1.0
    Xc = X[:, 1:] - np.mean(X[:, 1:], axis=0)
    scaled = kappa * Xc
    gram = scaled.T @ scaled + float(ridge_lambda) * np.eye(Xc.shape[1])
    try:
        solved = np.linalg.solve(gram, scaled.T)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(gram) @ scaled.T
    correction = scaled @ np.linalg.solve(gram, scaled.T @ centered_r)
    leverage = 1.0 / n + np.einsum("ij,ji->i", scaled, solved)
    denominator = 1.0 - leverage
    if np.any(denominator <= 1e-8):
        raise FloatingPointError("Ridge LOO leverage is numerically singular")
    loo = (centered_r - correction) / denominator
    effective_df = float(n - np.sum(leverage))
    return loo, leverage, effective_df


def _betting_cost(eps: float, variance: float, range_factor: float, alpha: float) -> float:
    """Existing exact betting-width sample-complexity surrogate, with range R."""
    log_term = math.log(2.0 / alpha)
    lam = min(0.5, max(1e-4, eps * range_factor / variance))
    psi = psi_e(lam)
    asymptotic_floor = psi * variance / (lam * range_factor)
    margin = eps - asymptotic_floor
    if margin <= 1e-6:
        return 2.5 * variance * log_term / (eps * eps)
    return log_term * range_factor / (lam * margin)


def _legacy_select_optimal_shrinkage_theta(
    phi_seen_g: np.ndarray,
    y_seen_g: np.ndarray,
    theta_global: np.ndarray,
    alpha: float,
    phi_group_eval: Optional[np.ndarray],
    phi_min_g: Optional[np.ndarray],
    phi_max_g: Optional[np.ndarray],
    target_eps: Optional[float],
    ridge_lambda: float,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Previous selector, preserved unchanged for legacy ablation runs."""
    d = len(theta_global)
    n_g = len(y_seen_g)

    def bounds(th):
        if not np.any(th[1:] != 0.0):
            return float(th[0]), float(th[0])
        if phi_group_eval is not None and len(phi_group_eval) <= 25000:
            values = phi_group_eval @ th
            return float(np.min(values)), float(np.max(values))
        return WoRControlVariateCS.theta_box_bounds(th, phi_min_g, phi_max_g)

    if n_g < 4:
        fb = bounds(theta_global)
        projected = WoRControlVariateCS.project_theta_span(theta_global, 1.0, exact_bounds=fb)
        return projected, bounds(projected)
    r = y_seen_g - phi_seen_g @ theta_global
    A = phi_seen_g.T @ phi_seen_g + ridge_lambda * np.eye(d)
    A[0, 0] -= 0.9 * ridge_lambda
    b = phi_seen_g.T @ r
    try:
        delta_w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        delta_w = np.zeros(d)
    theta_raw = theta_global + delta_w
    fb_raw = bounds(theta_raw)
    theta_proj = WoRControlVariateCS.project_theta_span(theta_raw, 1.0, exact_bounds=fb_raw)
    if n_g < 8:
        return theta_proj, bounds(theta_proj)
    log_term = math.log(2.0 / alpha)
    eps_use = target_eps if target_eps is not None and target_eps > 0.0 else 0.05
    df_factor = n_g / max(n_g - min(d, n_g // 2), 1.0)
    best_th = np.zeros(d)
    best_th[0] = float(np.mean(y_seen_g))
    best_fb = bounds(best_th)
    best_cost = float("inf")
    for kappa in (0.0, 0.25, 0.5, 0.75, 1.0):
        th_k = theta_proj.copy()
        th_k[1:] *= kappa
        th_k[0] = float(np.mean(y_seen_g - phi_seen_g[:, 1:] @ th_k[1:]))
        resid = y_seen_g - phi_seen_g @ th_k
        raw_var = float(np.var(resid, ddof=1)) if n_g > 1 else 0.25
        var_k = max(raw_var * (df_factor if kappa > 0.0 else 1.0), 1e-4)
        fb_k = bounds(th_k)
        R_k = 1.0 + max(fb_k[1] - fb_k[0], 0.0)
        lam_k = min(0.5, max(1e-4, eps_use * R_k / var_k))
        psi_k = psi_e(lam_k)
        asymp_floor = psi_k * var_k / (lam_k * R_k)
        margin = eps_use - asymp_floor
        cost_k = (2.5 * var_k * log_term / (eps_use * eps_use) if margin <= 1e-6
                  else log_term * R_k / (lam_k * margin))
        if cost_k < best_cost:
            best_cost, best_th, best_fb = cost_k, th_k, fb_k
    return best_th, best_fb


def select_optimal_shrinkage_theta(
    phi_seen_g: np.ndarray,
    y_seen_g: np.ndarray,
    theta_global: np.ndarray,
    alpha: float,
    phi_group_eval: Optional[np.ndarray] = None,
    phi_min_g: Optional[np.ndarray] = None,
    phi_max_g: Optional[np.ndarray] = None,
    target_eps: Optional[float] = None,
    ridge_lambda: float = 10.0,
    *,
    gating_mode: str = "safe_loo",
    min_group_size: int = 12,
    min_effective_df: float = 3.0,
    safety_margin: float = 0.05,
    kappa_grid: Sequence[float] = DEFAULT_KAPPA_GRID,
    return_diagnostics: bool = False,
):
    """Choose a predictable shrinkage weight using safe analytic Ridge LOO.

    Only ``phi_seen_g``, ``y_seen_g`` and pre-known feature statistics enter.
    ``phi_group_eval`` contains cheap features, never reference labels.  A
    nonzero proxy must reduce predicted betting cost by ``safety_margin`` over
    the LOO intercept-only baseline; otherwise the returned range is exactly
    zero and the betting range factor is one.

    ``gating_mode='legacy'`` reproduces the old selector.  The default return
    remains ``(theta, (f_min, f_max))``; diagnostics are opt-in.
    """
    phi_seen_g = np.asarray(phi_seen_g, dtype=np.float64)
    y_seen_g = np.asarray(y_seen_g, dtype=np.float64)
    theta_global = np.asarray(theta_global, dtype=np.float64)
    if phi_seen_g.ndim != 2 or phi_seen_g.shape[1] != len(theta_global):
        raise ValueError("phi_seen_g has incompatible dimensions")
    if len(y_seen_g) != len(phi_seen_g):
        raise ValueError("phi_seen_g and y_seen_g must have matching rows")
    if not 0.0 <= safety_margin < 1.0:
        raise ValueError("safety_margin must be in [0, 1)")
    if gating_mode not in ("safe_loo", "legacy"):
        raise ValueError("gating_mode must be 'safe_loo' or 'legacy'")
    if gating_mode == "legacy":
        result = _legacy_select_optimal_shrinkage_theta(
            phi_seen_g, y_seen_g, theta_global, alpha, phi_group_eval,
            phi_min_g, phi_max_g, target_eps, ridge_lambda,
        )
        if return_diagnostics:
            return result[0], result[1], {
                "gating_mode": "legacy", "n_g": len(y_seen_g),
                "selected_kappa": None, "gate_reason": "legacy_in_sample_selector",
            }
        return result

    n_g, d = len(y_seen_g), len(theta_global)

    def bounds(th):
        if not np.any(th[1:] != 0.0):
            value = float(th[0])
            return value, value
        return linear_prediction_bounds(
            th, phi_group_eval, phi_min_g, phi_max_g, safe_rounding=True
        )

    diagnostics: Dict[str, object] = {
        "gating_mode": "safe_loo", "n_g": n_g, "selected_kappa": 0.0,
        "effective_kappa": 0.0, "effective_df": float(max(n_g - 1, 0)),
        "safe_margin": safety_margin, "baseline_cost": None,
        "selected_cost": None, "loo_costs": {}, "loo_variances": {},
        "range_factors": {}, "gate_reason": None,
    }

    def baseline():
        th = np.zeros(d, dtype=np.float64)
        th[0] = float(np.mean(y_seen_g)) if n_g else float(theta_global[0])
        return th, bounds(th)

    if n_g < min_group_size:
        th, fb = baseline()
        diagnostics["gate_reason"] = "insufficient_group_size"
        return (th, fb, diagnostics) if return_diagnostics else (th, fb)

    residual = y_seen_g - phi_seen_g @ theta_global
    base_loo, base_hat, base_df = ridge_loo_residuals(
        phi_seen_g, y_seen_g, 0.0, ridge_lambda
    )
    base_var = max(float(np.mean(base_loo ** 2)), 1e-4)
    # Effective epsilon is supplied by decision operators.  For direct callers
    # without a target, use a history-scale target, never the old fixed 0.05.
    eps_use = (float(target_eps) if target_eps is not None and target_eps > 0.0
               else max(float(np.std(y_seen_g, ddof=1)), 1e-3))
    base_cost = _betting_cost(eps_use, base_var, 1.0, alpha)
    diagnostics.update(baseline_cost=base_cost, effective_df=base_df)

    best_th, best_fb = baseline()
    best_cost, best_kappa = base_cost, 0.0
    effective_df = base_df
    gate_reason = "no_cost_improvement"
    range_rejected = False
    df_rejected = False
    for requested_kappa in kappa_grid:
        kappa = float(requested_kappa)
        if not 0.0 <= kappa <= 1.0:
            raise ValueError("kappa grid values must be in [0, 1]")
        if kappa == 0.0:
            diagnostics["loo_costs"][kappa] = base_cost
            diagnostics["loo_variances"][kappa] = base_var
            diagnostics["range_factors"][kappa] = 1.0
            continue
        loo, _, df = ridge_loo_residuals(phi_seen_g, residual, kappa, ridge_lambda)
        var_k = max(float(np.mean(loo ** 2)), 1e-4)
        th = theta_global.copy()
        th[1:] += _ridge_slope_correction(phi_seen_g, residual, ridge_lambda, kappa)
        th[0] = float(np.mean(y_seen_g - phi_seen_g[:, 1:] @ th[1:]))
        fb = bounds(th)
        R_k = 1.0 + max(fb[1] - fb[0], 0.0)
        # Reject, rather than project, a candidate whose range exceeds one:
        # projection would invalidate the analytic LOO residual correspondence.
        cost = _betting_cost(eps_use, var_k, R_k, alpha) if fb[1] - fb[0] <= 1.0 else float("inf")
        diagnostics["loo_costs"][kappa] = cost
        diagnostics["loo_variances"][kappa] = var_k
        diagnostics["range_factors"][kappa] = R_k
        if kappa > 0.0 and not math.isfinite(cost):
            range_rejected = True
        if kappa > 0.0 and df < min_effective_df:
            df_rejected = True
        if (kappa > 0.0 and df >= min_effective_df and math.isfinite(cost)
                and cost <= (1.0 - safety_margin) * base_cost and cost < best_cost):
            best_th, best_fb, best_cost = th, fb, cost
            best_kappa, effective_df, gate_reason = kappa, df, "selected"

    if best_kappa == 0.0:
        best_th, best_fb = baseline()
        if df_rejected:
            gate_reason = "insufficient_effective_df"
        elif range_rejected:
            gate_reason = "range_constraint"
    diagnostics.update(selected_kappa=best_kappa, effective_kappa=best_kappa,
                       effective_df=effective_df, selected_cost=best_cost,
                       gate_reason=gate_reason)
    return (best_th, best_fb, diagnostics) if return_diagnostics else (best_th, best_fb)


def _ridge_slope_correction(phi_seen: np.ndarray, residual: np.ndarray,
                            ridge_lambda: float, kappa: float) -> np.ndarray:
    """Ridge coefficient on the scaled design ``kappa * X``."""
    if phi_seen.shape[1] <= 1 or kappa == 0.0:
        return np.zeros(max(phi_seen.shape[1] - 1, 0), dtype=np.float64)
    Xc = phi_seen[:, 1:] - np.mean(phi_seen[:, 1:], axis=0)
    rc = residual - float(np.mean(residual))
    scaled = float(kappa) * Xc
    gram = scaled.T @ scaled + float(ridge_lambda) * np.eye(Xc.shape[1])
    try:
        return np.linalg.solve(gram, scaled.T @ rc)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram) @ scaled.T @ rc
