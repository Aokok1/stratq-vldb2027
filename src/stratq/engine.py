"""
StratQ Core Execution Engine over DuckDB + Apache Arrow / Parquet.

Supports four analytical SQL query operators with simultaneous per-group
finite-sample statistical guarantees (joint probability >= 1 - delta):
  - Q1: execute_grouped_agg      (GROUP BY + COUNT / SUM / AVG_INDICATOR)
  - Q2: execute_grouped_selection (GROUP BY + Per-Group Recall / Precision Target)
  - Q3a: execute_grouped_having  (GROUP BY ... HAVING agg >= tau with early pruning)
  - Q3b: execute_grouped_topk    (GROUP BY ... ORDER BY agg DESC LIMIT K with LUCB)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .bounds import (
    WoRControlVariateCS,
    _down,
    _up,
    linear_prediction_bounds,
    nonnegative_sum_interval,
    select_optimal_shrinkage_theta,
)


@dataclass
class GroupAggResult:
    groups: List[str]
    n_total_by_group: Dict[str, int]
    estimates: Dict[str, float]
    lower: Dict[str, float]
    upper: Dict[str, float]
    half_widths: Dict[str, float]
    oracle_calls_by_group: Dict[str, int]
    total_oracle_calls: int
    total_rows: int
    duckdb_scan_ms: float
    optimizer_ms: float
    oracle_ms: float
    wall_ms: float
    converged: bool
    agg_type: str = "mean"  # "mean" | "count" | "sum"


@dataclass
class GroupSelectResult:
    groups: List[str]
    selected_indices: np.ndarray
    selected_count_by_group: Dict[str, int]
    thresholds_by_group: Dict[str, float]
    oracle_calls_by_group: Dict[str, int]
    total_oracle_calls: int
    total_rows: int
    duckdb_scan_ms: float
    optimizer_ms: float
    wall_ms: float
    target_mode: str  # "recall" | "precision"
    target_gamma: float


@dataclass
class GroupHavingResult:
    groups: List[str]
    accepted_groups: List[str]
    rejected_groups: List[str]
    uncertain_groups: List[str]
    estimates: Dict[str, float]
    lower: Dict[str, float]
    upper: Dict[str, float]
    oracle_calls_by_group: Dict[str, int]
    total_oracle_calls: int
    total_rows: int
    duckdb_scan_ms: float
    optimizer_ms: float
    wall_ms: float
    tau: float


@dataclass
class GroupTopKResult:
    groups: List[str]
    topk_groups: List[str]
    estimates: Dict[str, float]
    lower: Dict[str, float]
    upper: Dict[str, float]
    oracle_calls_by_group: Dict[str, int]
    total_oracle_calls: int
    total_rows: int
    duckdb_scan_ms: float
    optimizer_ms: float
    wall_ms: float
    k: int
    certified: bool


@dataclass
class _TableProfile:
    """In-memory Arrow/NumPy view built from a single DuckDB profiling scan."""

    groups: List[str]
    group_to_idx: Dict[str, int]
    group_row_indices: List[np.ndarray]  # list of row-id arrays per group
    n_by_group: np.ndarray
    n_total: int
    phi_all: np.ndarray  # shape (N, d): [1, proxy_norm, cov_1_norm, ...] in [0, 1]
    phi_sum_lo_by_group: np.ndarray  # shape (G, d)
    phi_sum_hi_by_group: np.ndarray  # shape (G, d)
    phi_mean_by_group: np.ndarray  # shape (G, d)
    phi_min_by_group: np.ndarray  # shape (G, d)
    phi_max_by_group: np.ndarray  # shape (G, d)
    proxy_raw: np.ndarray  # shape (N,)
    measure_raw: np.ndarray  # shape (N,)
    measure_min: float
    measure_max: float
    oracle_true: Optional[np.ndarray]  # optional pre-materialized oracle column
    duckdb_scan_ms: float


class StratQEngine:
    """
    DuckDB + Apache Arrow query engine for Group-Guaranteed Proxy Query Processing.
    """

    def __init__(
        self,
        table_source: Union[str, Path, pa.Table],
        group_col: str,
        proxy_col: str,
        covariate_cols: Optional[Sequence[str]] = None,
        measure_col: Optional[str] = None,
        oracle_col: Optional[str] = None,
        oracle_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        include_interactions: bool = False,
        seed: int = 42,
        gating_mode: str = "safe_loo",
        safe_rounding: bool = True,
    ) -> None:
        """
        Args:
            table_source: Path to Parquet file or in-memory PyArrow Table.
            group_col: Grouping column name (GROUP BY g).
            proxy_col: Cheap proxy score column (in [0, 1] or normalized automatically).
            covariate_cols: Zero-cost structured relational columns in the table.
            measure_col: Optional numeric measure column for SUM/AVG; None => COUNT/rate (a_i = 1).
            oracle_col: Optional ground-truth / pre-evaluated oracle column for offline replay.
            oracle_fn: Optional live oracle callable taking row indices -> oracle values in [0, 1].
            include_interactions: Include proxy * covariate interaction features.
            seed: Deterministic PRNG seed.
            gating_mode: ``safe_loo`` (default) or reproducible ``legacy`` selector.
            safe_rounding: Use outward binary64 statistical intervals (default).
        """
        if gating_mode not in ("safe_loo", "legacy"):
            raise ValueError("gating_mode must be 'safe_loo' or 'legacy'")
        if oracle_col is None and oracle_fn is None:
            raise ValueError("Provide either oracle_col or oracle_fn")
        self.table_source = table_source
        self.group_col = group_col
        self.proxy_col = proxy_col
        self.covariate_cols = list(covariate_cols or [])
        self.measure_col = measure_col
        self.oracle_col = oracle_col
        self.oracle_fn = oracle_fn
        self.include_interactions = include_interactions
        self.seed = int(seed)
        self.gating_mode = gating_mode
        self.safe_rounding = bool(safe_rounding)

        self.profile: _TableProfile = self._profile_with_duckdb()

    def _profile_with_duckdb(self) -> _TableProfile:
        """Initialize a reusable profile from one DuckDB profiling stage.

        The stage issues an aggregate query for group counts/extrema followed
        by a column-pruned Arrow projection.  Arrow buffers may be zero-copy at
        the DuckDB boundary; conversion to Python lists/NumPy and the feature
        matrix materialize new arrays.
        """
        t0 = time.perf_counter()
        con = duckdb.connect(database=":memory:")
        try:
            if isinstance(self.table_source, pa.Table):
                con.register("stratq_tbl", self.table_source)
                src_sql = "stratq_tbl"
            else:
                path_str = str(self.table_source).replace("\\", "/")
                src_sql = f"read_parquet('{path_str}')"

            # Build single-pass SQL profiling query for group counts + min/max/means
            feat_cols = [self.proxy_col] + self.covariate_cols
            agg_exprs = ["COUNT(*) AS _n_g"]
            for idx, col in enumerate(feat_cols):
                agg_exprs.append(f'AVG(CAST("{col}" AS DOUBLE)) AS _avg_{idx}')
                agg_exprs.append(f'MIN(CAST("{col}" AS DOUBLE)) AS _min_{idx}')
                agg_exprs.append(f'MAX(CAST("{col}" AS DOUBLE)) AS _max_{idx}')

            grp_sql = (
                f'SELECT CAST("{self.group_col}" AS VARCHAR) AS _g, '
                + ", ".join(agg_exprs)
                + f" FROM {src_sql} GROUP BY 1 ORDER BY 1"
            )
            grp_df = con.execute(grp_sql).fetchdf()

            # Zero-copy Arrow load of required numeric columns only (column pruning!)
            proj_cols = [f'CAST("{self.group_col}" AS VARCHAR) AS _g']
            proj_cols.append(f'CAST("{self.proxy_col}" AS DOUBLE) AS _proxy')
            for idx, col in enumerate(self.covariate_cols):
                proj_cols.append(f'CAST("{col}" AS DOUBLE) AS _cov_{idx}')
            if self.measure_col is not None:
                proj_cols.append(f'CAST("{self.measure_col}" AS DOUBLE) AS _meas')
            if self.oracle_col is not None:
                proj_cols.append(f'CAST("{self.oracle_col}" AS DOUBLE) AS _oracle')

            arrow_tbl = con.sql(f"SELECT {', '.join(proj_cols)} FROM {src_sql}").to_arrow_table()
        finally:
            con.close()

        groups = grp_df["_g"].tolist()
        G = len(groups)
        group_to_idx = {g: i for i, g in enumerate(groups)}
        n_by_group = grp_df["_n_g"].to_numpy(dtype=np.int64)
        n_total = int(n_by_group.sum())

        # Extract Arrow arrays cleanly to contiguous NumPy
        g_col_py = arrow_tbl.column("_g").to_pylist()
        g_ids = np.fromiter((group_to_idx[v] for v in g_col_py), dtype=np.int32, count=n_total)
        # O(N log N) stable grouping via argsort + split.
        order = np.argsort(g_ids, kind="mergesort")
        counts = np.bincount(g_ids, minlength=G)
        split_pts = np.cumsum(counts)[:-1]
        group_row_indices = [arr.copy() for arr in np.split(order, split_pts)]

        proxy_raw = np.nan_to_num(
            arrow_tbl.column("_proxy").to_numpy(zero_copy_only=False).astype(np.float64),
            nan=0.5,
        )
        # Global min-max normalization to [0, 1] using exact DuckDB min/max
        p_min = float(grp_df["_min_0"].min())
        p_max = float(grp_df["_max_0"].max())
        p_span = max(p_max - p_min, 1e-12)
        proxy_norm = np.clip((proxy_raw - p_min) / p_span, 0.0, 1.0)

        cols_list = [np.ones(n_total, dtype=np.float64), proxy_norm]
        cov_norms = []
        for idx in range(len(self.covariate_cols)):
            c_raw = np.nan_to_num(
                arrow_tbl.column(f"_cov_{idx}").to_numpy(zero_copy_only=False).astype(np.float64),
                nan=0.0,
            )
            c_min = float(grp_df[f"_min_{idx + 1}"].min())
            c_max = float(grp_df[f"_max_{idx + 1}"].max())
            c_span = max(c_max - c_min, 1e-12)
            c_norm = np.clip((c_raw - c_min) / c_span, 0.0, 1.0)
            cols_list.append(c_norm)
            cov_norms.append(c_norm)

        if self.include_interactions and cov_norms:
            for c_norm in cov_norms:
                cols_list.append(proxy_norm * c_norm)

        phi_all = np.column_stack(cols_list)
        d = phi_all.shape[1]

        # Exact per-group feature means and exact min/max bounding boxes
        phi_mean_by_group = np.zeros((G, d), dtype=np.float64)
        phi_sum_lo_by_group = np.zeros((G, d), dtype=np.float64)
        phi_sum_hi_by_group = np.zeros((G, d), dtype=np.float64)
        phi_min_by_group = np.zeros((G, d), dtype=np.float64)
        phi_max_by_group = np.ones((G, d), dtype=np.float64)
        for gi in range(G):
            idxs = group_row_indices[gi]
            if len(idxs) > 0:
                sub = phi_all[idxs]
                phi_mean_by_group[gi] = sub.mean(axis=0)
                phi_sum_lo_by_group[gi], phi_sum_hi_by_group[gi] = nonnegative_sum_interval(
                    sub, axis=0
                )
                phi_min_by_group[gi] = sub.min(axis=0)
                phi_max_by_group[gi] = sub.max(axis=0)

        if self.measure_col is not None:
            measure_raw = np.nan_to_num(
                arrow_tbl.column("_meas").to_numpy(zero_copy_only=False).astype(np.float64),
                nan=0.0,
            )
            measure_min = float(min(0.0, np.min(measure_raw)))
            measure_max = float(max(1.0, np.max(measure_raw)))
        else:
            measure_raw = np.ones(n_total, dtype=np.float64)
            measure_min = 0.0
            measure_max = 1.0

        oracle_true = None
        if self.oracle_col is not None:
            oracle_true = arrow_tbl.column("_oracle").to_numpy(zero_copy_only=False).astype(
                np.float64
            )

        duckdb_scan_ms = (time.perf_counter() - t0) * 1000.0
        return _TableProfile(
            groups=groups,
            group_to_idx=group_to_idx,
            group_row_indices=group_row_indices,
            n_by_group=n_by_group,
            n_total=n_total,
            phi_all=phi_all,
            phi_sum_lo_by_group=phi_sum_lo_by_group,
            phi_sum_hi_by_group=phi_sum_hi_by_group,
            phi_mean_by_group=phi_mean_by_group,
            phi_min_by_group=phi_min_by_group,
            phi_max_by_group=phi_max_by_group,
            proxy_raw=proxy_raw,
            measure_raw=measure_raw,
            measure_min=measure_min,
            measure_max=measure_max,
            oracle_true=oracle_true,
            duckdb_scan_ms=duckdb_scan_ms,
        )

    def _validate_grouped_agg_type(self, agg_type: str) -> None:
        if agg_type not in ("count", "sum", "mean"):
            raise ValueError(f"Unsupported aggregate: {agg_type!r}")
        if agg_type == "count" and self.measure_col is not None:
            raise ValueError("COUNT requires an engine constructed without measure_col")

    @staticmethod
    def _aggregate_bounds(cs: WoRControlVariateCS, agg_type: str) -> Tuple[float, float]:
        lo, hi = cs.bounds
        if agg_type == 'mean':
            return lo, hi
        n = cs.n_total
        if agg_type == 'count':
            if cs.exhausted and cs.deterministic_count_bounds is not None:
                return cs.deterministic_count_bounds
            if cs.safe_rounding:
                count_bounds = cs.deterministic_count_bounds
                lower = (count_bounds[0] if count_bounds is not None
                         and lo == count_bounds[0] / n else _down(lo * n))
                upper = (count_bounds[1] if count_bounds is not None
                         and hi == count_bounds[1] / n else _up(hi * n))
                return lower, upper
            seen = int(cs.y_sum_seen)
            feasible_hi = seen + n - cs.n_seen
            # Restoring these endpoints avoids an integer-to-rate-to-count round trip.
            return (seen if lo == seen / n else lo * n,
                    feasible_hi if hi == feasible_hi / n else hi * n)
        if cs.safe_rounding:
            return _down(lo * n), _up(hi * n)
        return lo * n, hi * n

    def _eval_oracle(self, row_indices: np.ndarray, *, binary_count: bool = False) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        if self.oracle_fn is not None:
            o_vals = np.asarray(self.oracle_fn(row_indices), dtype=np.float64)
        else:
            assert self.profile.oracle_true is not None
            o_vals = self.profile.oracle_true[row_indices]
        if binary_count and (o_vals.shape != row_indices.shape or not np.all((o_vals == 0) | (o_vals == 1))):
            raise ValueError('COUNT requires one binary reference label (0 or 1) per sampled row')
        y_vals = o_vals * self.profile.measure_raw[row_indices]
        return y_vals, (time.perf_counter() - t0) * 1000.0

    def _fit_global_theta(
        self,
        all_phi: List[np.ndarray],
        all_y_norm: List[np.ndarray],
        d: int,
        use_covariates: bool,
        ridge_lambda: float = 5.0,
    ) -> np.ndarray:
        """Fit pooled global predictable Ridge predictor strictly on prior samples."""
        theta_prior = np.zeros(d, dtype=np.float64)
        if d > 1:
            theta_prior[1] = 1.0  # start from raw proxy p_i
        if not all_y_norm:
            return theta_prior

        Y = np.concatenate(all_y_norm)
        Phi = np.vstack(all_phi)
        if not use_covariates:
            # Restrict strictly to [1, proxy]
            Phi_sub = Phi[:, :2]
            A = Phi_sub.T @ Phi_sub + ridge_lambda * np.eye(2)
            A[0, 0] -= 0.9 * ridge_lambda
            b = Phi_sub.T @ Y
            try:
                w2 = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                w2 = np.array([0.0, 1.0])
            th = np.zeros(d, dtype=np.float64)
            th[:2] = w2
            return WoRControlVariateCS.project_theta_span(th, max_span=1.0)

        if len(Y) < d + 2:
            return theta_prior

        A = Phi.T @ Phi + ridge_lambda * np.eye(d)
        A[0, 0] -= 0.9 * ridge_lambda
        b = Phi.T @ Y
        try:
            th = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            th = theta_prior
        return WoRControlVariateCS.project_theta_span(th, max_span=1.0)

    def _get_predictable_theta(
        self,
        gi: int,
        phi_seen_by_g: List[List[np.ndarray]],
        y_seen_by_g: List[List[np.ndarray]],
        theta_global: np.ndarray,
        alpha_g: float,
        d: int,
        use_control_variates: bool,
        use_covariates: bool,
        use_hierarchical: bool,
        target_eps: Optional[float] = None,
    ) -> Tuple[np.ndarray, Tuple[float, float]]:
        prof = self.profile
        pmin_g = prof.phi_min_by_group[gi]
        pmax_g = prof.phi_max_by_group[gi]
        idxs = prof.group_row_indices[gi]
        phi_eval = prof.phi_all[idxs] if len(idxs) <= 25000 else None

        if not use_control_variates:
            return np.zeros(d, dtype=np.float64), (0.0, 0.0)

        if not phi_seen_by_g[gi]:
            th0 = theta_global.copy()
            fb0 = linear_prediction_bounds(
                th0, phi_eval, pmin_g, pmax_g, safe_rounding=self.safe_rounding
            )
            return th0, fb0

        phi_g = np.vstack(phi_seen_by_g[gi])
        y_g = np.concatenate(y_seen_by_g[gi])

        if not use_covariates:
            phi_g_sub = phi_g[:, :2]
            phi_eval_sub = phi_eval[:, :2] if phi_eval is not None else None
            tg_sub = theta_global[:2] if use_hierarchical else np.array([0.0, 1.0])
            w2, fb2 = select_optimal_shrinkage_theta(
                phi_g_sub,
                y_g,
                tg_sub,
                alpha=alpha_g,
                phi_group_eval=phi_eval_sub,
                phi_min_g=pmin_g[:2],
                phi_max_g=pmax_g[:2],
                target_eps=target_eps,
                ridge_lambda=8.0,
                gating_mode=self.gating_mode,
            )
            th = np.zeros(d, dtype=np.float64)
            th[:2] = w2
            if self.safe_rounding:
                fb2 = linear_prediction_bounds(th, phi_eval, pmin_g, pmax_g,
                                               safe_rounding=True)
            return th, fb2

        prior_vec = theta_global if use_hierarchical else np.array([0.0, 1.0] + [0.0] * (d - 2))
        result = select_optimal_shrinkage_theta(
            phi_g,
            y_g,
            prior_vec,
            alpha=alpha_g,
            phi_group_eval=phi_eval,
            phi_min_g=pmin_g,
            phi_max_g=pmax_g,
            target_eps=target_eps,
            ridge_lambda=12.0,
            gating_mode=self.gating_mode,
        )
        if self.safe_rounding:
            return result[0], linear_prediction_bounds(
                result[0], phi_eval, pmin_g, pmax_g, safe_rounding=True
            )
        return result

    @staticmethod
    def _predictable_decision_eps(
        cs: WoRControlVariateCS,
        boundary_norm: float,
        fallback_eps: float,
    ) -> float:
        """Clip a decision target using only the currently revealed interval.

        ``gap_lower`` is zero while the interval contains the decision
        boundary.  In that case one quarter of the current half-width supplies
        a positive, history-adaptive floor.  The result never exceeds the
        operator's explicit indifference/one-row fallback.
        """
        lo, hi = cs.lo_norm, cs.hi_norm
        gap_lower = (0.0 if lo <= boundary_norm <= hi
                     else min(abs(lo - boundary_norm), abs(hi - boundary_norm)))
        adaptive_floor = 0.125 * (hi - lo)
        return min(max(float(fallback_eps), 1e-12),
                   max(gap_lower, adaptive_floor, 1e-12))

    # ------------------------------------------------------------------
    # Q1: Grouped Aggregation (GROUP BY + COUNT / SUM / MEAN)
    # ------------------------------------------------------------------
    def execute_grouped_agg(
        self,
        eps: Union[float, Dict[str, float]] = 0.05,
        delta: float = 0.05,
        agg_type: str = "mean",
        batch_size: int = 16,
        pilot_per_group: int = 8,
        max_oracle_budget: Optional[int] = None,
        use_control_variates: bool = True,
        use_covariates: bool = True,
        use_hierarchical: bool = True,
        use_adaptive_alloc: bool = True,
        use_wor: bool = True,
        seed: Optional[int] = None,
        *,
        refit_proxy: bool = True,
    ) -> GroupAggResult:
        """
        Execute Q1: Simultaneous per-group aggregate estimation with half-width <= eps_g
        and joint probability >= 1 - delta across all G groups.
        """
        self._validate_grouped_agg_type(agg_type)
        if not use_wor:
            raise ValueError("Grouped aggregation requires sampling without replacement")
        t_start = time.perf_counter()
        rng = np.random.default_rng(self.seed if seed is None else seed)
        prof = self.profile
        G = len(prof.groups)
        d = prof.phi_all.shape[1]
        alpha_g = delta / max(G, 1)

        # Normalize per-group eps targets onto [0, 1] mean scale
        eps_by_gi = np.zeros(G, dtype=np.float64)
        for gi, g in enumerate(prof.groups):
            e_val = eps[g] if isinstance(eps, dict) else float(eps)
            if agg_type in ("count", "sum"):
                eps_by_gi[gi] = e_val / max(prof.n_by_group[gi], 1)
            else:
                eps_by_gi[gi] = e_val

        # Create shuffled WoR permutations per group
        perms = [rng.permutation(idxs) for idxs in prof.group_row_indices]
        ptrs = np.zeros(G, dtype=np.int64)

        cs_list = [
            WoRControlVariateCS(
                n_total=int(prof.n_by_group[gi]),
                alpha=alpha_g,
                phi_mean_init=prof.phi_mean_by_group[gi],
                phi_sum_interval=(prof.phi_sum_lo_by_group[gi], prof.phi_sum_hi_by_group[gi]),
                phi_min_g=prof.phi_min_by_group[gi],
                phi_max_g=prof.phi_max_by_group[gi],
                y_min=prof.measure_min,
                y_max=prof.measure_max,
                target_eps=float(eps_by_gi[gi]),
                use_wor=use_wor,
                safe_rounding=self.safe_rounding,
            )
            for gi in range(G)
        ]

        phi_seen_by_g: List[List[np.ndarray]] = [[] for _ in range(G)]
        y_seen_by_g: List[List[np.ndarray]] = [[] for _ in range(G)]
        y_span = max(prof.measure_max - prof.measure_min, 1e-12)

        total_oracle_calls = 0
        oracle_ms = 0.0
        budget_cap = max_oracle_budget if max_oracle_budget is not None else prof.n_total

        def _sample_group(gi: int, k_req: int, theta_global: np.ndarray) -> int:
            nonlocal total_oracle_calls, oracle_ms
            if cs_list[gi].exhausted:
                return 0
            rem_budget = budget_cap - total_oracle_calls
            if rem_budget <= 0:
                return 0
            rem_group = int(prof.n_by_group[gi] - ptrs[gi])
            if rem_group <= 0:
                return 0
            k = min(k_req, rem_group, rem_budget)
            start = int(ptrs[gi])
            batch_rows = perms[gi][start : start + k]
            ptrs[gi] += k

            if refit_proxy:
                theta_g, fb_g = self._get_predictable_theta(
                    gi, phi_seen_by_g, y_seen_by_g, theta_global, alpha_g, d,
                    use_control_variates=use_control_variates,
                    use_covariates=use_covariates,
                    use_hierarchical=use_hierarchical,
                    target_eps=float(eps_by_gi[gi]),
                )
            else:
                theta_g, fb_g = theta_init, fixed_bounds[gi]

            phi_b = prof.phi_all[batch_rows]
            y_raw_b, o_ms = self._eval_oracle(batch_rows, binary_count=agg_type == 'count')
            oracle_ms += o_ms
            total_oracle_calls += k

            # Update CS BEFORE storing new samples into history (strict predictability!)
            cs_list[gi].update_batch(y_raw_b, phi_b, theta_g, exact_f_bounds=fb_g)

            y_norm_b = np.clip((y_raw_b - prof.measure_min) / y_span, 0.0, 1.0)
            phi_seen_by_g[gi].append(phi_b)
            y_seen_by_g[gi].append(y_norm_b)
            return k

        # Stage 1: Small uniform pilot across all groups
        theta_init = np.zeros(d, dtype=np.float64)
        if use_control_variates and d > 1:
            theta_init[1] = 1.0
        if not refit_proxy:
            # The raw proxy is feature column 1; its stored extrema are exact.
            fixed_bounds = [(float(pmin[1]), float(pmax[1])) if use_control_variates and d > 1
                            else (0.0, 0.0) for pmin, pmax in zip(prof.phi_min_by_group, prof.phi_max_by_group)]
        for gi in range(G):
            _sample_group(gi, pilot_per_group, theta_init)

        # Stage 2: Adaptive Max-Slack cross-group routing (or round-robin ablation)
        rr_cursor = 0
        steps_since_global_fit = 0
        theta_global = self._fit_global_theta(
            [np.vstack(x) for x in phi_seen_by_g if x],
            [np.concatenate(x) for x in y_seen_by_g if x],
            d,
            use_covariates=use_covariates,
        ) if refit_proxy else theta_init

        while total_oracle_calls < budget_cap:
            slacks = np.array(
                [
                    0.0
                    if (cs_list[gi].exhausted or cs_list[gi].half_width <= eps_by_gi[gi])
                    else cs_list[gi].half_width / max(eps_by_gi[gi], 1e-12)
                    for gi in range(G)
                ],
                dtype=np.float64,
            )
            if float(np.max(slacks)) <= 0.0:
                break

            if use_adaptive_alloc:
                gi_next = int(np.argmax(slacks))
            else:
                # Round-robin among still-unconverged groups
                found = -1
                for offset in range(G):
                    cand = (rr_cursor + offset) % G
                    if slacks[cand] > 0.0:
                        found = cand
                        rr_cursor = (cand + 1) % G
                        break
                if found < 0:
                    break
                gi_next = found

            drawn = _sample_group(gi_next, batch_size, theta_global)
            if drawn == 0:
                break

            steps_since_global_fit += 1
            if refit_proxy and use_control_variates and use_hierarchical and steps_since_global_fit % max(4, G // 2) == 0:
                theta_global = self._fit_global_theta(
                    [np.vstack(x) for x in phi_seen_by_g if x],
                    [np.concatenate(x) for x in y_seen_by_g if x],
                    d,
                    use_covariates=use_covariates,
                )

        # Assemble outputs scaled to requested aggregate type
        estimates, lower, upper, half_widths, calls_by_g, n_by_g_dict = {}, {}, {}, {}, {}, {}
        converged = True
        for gi, g in enumerate(prof.groups):
            lo, hi = self._aggregate_bounds(cs_list[gi], agg_type)
            estimates[g] = (lo + hi) / 2
            lower[g] = lo
            upper[g] = hi
            half_widths[g] = (hi - lo) / 2
            calls_by_g[g] = int(cs_list[gi].n_seen)
            n_by_g_dict[g] = int(prof.n_by_group[gi])
            if not cs_list[gi].exhausted and cs_list[gi].half_width > eps_by_gi[gi] + 1e-12:
                converged = False

        wall_ms = (time.perf_counter() - t_start) * 1000.0
        optimizer_ms = max(0.0, wall_ms - oracle_ms)
        return GroupAggResult(
            groups=prof.groups,
            n_total_by_group=n_by_g_dict,
            estimates=estimates,
            lower=lower,
            upper=upper,
            half_widths=half_widths,
            oracle_calls_by_group=calls_by_g,
            total_oracle_calls=total_oracle_calls,
            total_rows=prof.n_total,
            duckdb_scan_ms=prof.duckdb_scan_ms,
            optimizer_ms=optimizer_ms,
            oracle_ms=oracle_ms,
            wall_ms=wall_ms,
            converged=converged,
            agg_type=agg_type,
        )

    # ------------------------------------------------------------------
    # Q2: Grouped Selective Filtering (Per-Group Recall / Precision Target)
    # ------------------------------------------------------------------
    def execute_grouped_selection(
        self,
        target_gamma: float = 0.90,
        target_mode: str = "recall",
        delta: float = 0.05,
        oracle_budget: Optional[int] = None,
        pilot_frac: float = 0.15,
        n_grid: int = 25,
        batch_size: int = 16,
        use_covariates: bool = True,
        use_adaptive_alloc: bool = True,
        seed: Optional[int] = None,
    ) -> GroupSelectResult:
        """
        Execute Q2: Return tuple set S = union_g S_g guaranteeing simultaneous
        per-group Recall_g >= target_gamma (or Precision_g >= target_gamma)
        with joint probability >= 1 - delta, while maximizing precision (RT) or
        recall (PT) under oracle_budget.

        Mechanism:
        1. Pilot stage (split-sample / cross-fitted so ranking is independent of certification):
           learns covariate-lifted per-group ranking score s_i = clip(theta_g^T phi_i, 0, 1).
        2. Certification stage (strictly fresh WoR oracle samples on remaining tuples):
           - For Recall Target (RT): linearizes Recall_g(tau) >= gamma as
             mean(h_i(tau)) >= 0 where h_i(tau) = (1-gamma)*O(x_i) if s_i>=tau else -gamma*O(x_i)
             in [-gamma, 1-gamma] (span = 1!). Uses anytime WoR Empirical-Bernstein CS
             over a grid of candidate thresholds tau_1 > ... > tau_M (union bound delta/(G*M)),
             picking the highest (most selective / highest precision) certified threshold tau_g*
             for each group g!
           - Adaptive active-group routing concentrates remaining oracle budget on groups
             with the largest potential precision gain / uncertified tail.
        """
        t_start = time.perf_counter()
        rng = np.random.default_rng(self.seed if seed is None else seed)
        prof = self.profile
        G = len(prof.groups)
        d = prof.phi_all.shape[1]
        if oracle_budget is None:
            oracle_budget = max(G * 40, int(0.10 * prof.n_total))
        oracle_budget = min(int(oracle_budget), prof.n_total)

        perms = [rng.permutation(idxs) for idxs in prof.group_row_indices]
        ptrs = np.zeros(G, dtype=np.int64)
        calls_by_gi = np.zeros(G, dtype=np.int64)
        oracle_cache: Dict[int, float] = {}

        # Stage 1: Independent pilot to learn covariate-lifted ranking scores (no certification bias!)
        pilot_total = min(int(oracle_budget * pilot_frac), prof.n_total // 2)
        pilot_per_g = max(4, pilot_total // max(G, 1))

        phi_pilot_list: List[np.ndarray] = []
        y_pilot_list: List[np.ndarray] = []
        phi_pilot_by_g: List[np.ndarray] = []
        y_pilot_by_g: List[np.ndarray] = []

        total_calls = 0
        for gi in range(G):
            k = min(pilot_per_g, int(prof.n_by_group[gi]), oracle_budget - total_calls)
            if k <= 0:
                phi_pilot_by_g.append(np.empty((0, d)))
                y_pilot_by_g.append(np.empty((0,)))
                continue
            rows = perms[gi][:k]
            ptrs[gi] = k
            y_b, _ = self._eval_oracle(rows)
            for r_id, y_v in zip(rows, y_b):
                oracle_cache[int(r_id)] = float(y_v)
            phi_b = prof.phi_all[rows]
            phi_pilot_by_g.append(phi_b)
            y_pilot_by_g.append(y_b)
            phi_pilot_list.append(phi_b)
            y_pilot_list.append(y_b)
            calls_by_gi[gi] += k
            total_calls += k

        theta_glob = self._fit_global_theta(
            phi_pilot_list, y_pilot_list, d, use_covariates=use_covariates, ridge_lambda=5.0
        )

        # Compute independent covariate-lifted scores s_i on remaining unqueried tuples
        scores_all = np.zeros(prof.n_total, dtype=np.float64)
        for gi in range(G):
            if use_covariates and len(y_pilot_by_g[gi]) >= 4:
                th_g, _ = select_optimal_shrinkage_theta(
                    phi_pilot_by_g[gi],
                    y_pilot_by_g[gi],
                    theta_glob,
                    alpha=delta / G,
                    phi_min_g=prof.phi_min_by_group[gi],
                    phi_max_g=prof.phi_max_by_group[gi],
                    ridge_lambda=10.0,
                    gating_mode=self.gating_mode,
                )
            elif use_covariates:
                th_g = theta_glob
            else:
                th_g = np.zeros(d, dtype=np.float64)
                th_g[1] = 1.0
            idxs = prof.group_row_indices[gi]
            scores_all[idxs] = np.clip(prof.phi_all[idxs] @ th_g, 0.0, 1.0)

        # Stage 2: Fresh WoR certification on remaining pool R_g^{\cert} = perms[gi][ptrs[gi]:]
        # Build candidate threshold grid per group from score quantiles (high -> low)
        rem_pools = [perms[gi][int(ptrs[gi]) :] for gi in range(G)]
        rem_ptrs = np.zeros(G, dtype=np.int64)
        # The explicit zero threshold can add one candidate beyond n_grid.
        alpha_cell = delta / max(G * (n_grid + 1), 1)

        grids: List[np.ndarray] = []
        # For each group and each grid threshold m, maintain a WoR CS on h_i(tau_m)
        # Recall mode: h_i(tau) = (1-gamma)*O_i if s_i>=tau else -gamma*O_i in [-gamma, 1-gamma]
        # Precision mode: direct WoR CS on O_i restricted to s_i >= tau_m
        cs_grid: List[List[WoRControlVariateCS]] = []
        prefix_sizes: List[List[int]] = []

        for gi in range(G):
            pool = rem_pools[gi]
            n_rem = len(pool)
            if n_rem == 0:
                grids.append(np.array([0.0]))
                cs_grid.append([])
                prefix_sizes.append([0])
                continue
            s_pool = scores_all[pool]
            # Quantiles from 0.95 down to 0.0 (plus explicit 0.0 so full-set recall=1 is always feasible)
            qs = np.linspace(0.95, 0.0, n_grid)
            taus = np.unique(np.quantile(s_pool, qs))[::-1]
            if taus[-1] > 0.0:
                taus = np.append(taus, 0.0)
            grids.append(taus)
            g_cs_list = []
            g_pref = []
            for tau in taus:
                mask = s_pool >= tau
                g_pref.append(int(mask.sum()))
                if target_mode == "recall":
                    # Linear recall margin h_i(tau) in [-gamma, 1-gamma], span = 1.0
                    # Zero-cost predictable proxy control variate for h_i(tau):
                    # h_hat_i = (1-gamma)*s_i if s_i>=tau else -gamma*s_i
                    h_hat = np.where(mask, (1.0 - target_gamma) * s_pool, -target_gamma * s_pool)
                    # Normalize h_hat to [0, 1] for exact hypercube bounds
                    h_hat_norm = np.clip(h_hat + target_gamma, 0.0, 1.0)
                    phi_m = np.array([1.0, float(np.mean(h_hat_norm))], dtype=np.float64)
                    sum_lo, sum_hi = nonnegative_sum_interval(h_hat_norm)
                    cs_m = WoRControlVariateCS(
                        n_total=n_rem,
                        alpha=alpha_cell,
                        phi_mean_init=phi_m,
                        phi_sum_interval=(np.array([n_rem, sum_lo]), np.array([n_rem, sum_hi])),
                        y_min=-target_gamma,
                        y_max=1.0 - target_gamma,
                        use_wor=True,
                        safe_rounding=self.safe_rounding,
                    )
                    g_cs_list.append(cs_m)
                else:
                    # Precision mode: WoR CS on indicator (O_i - gamma)*1[s_i>=tau] / |prefix|
                    # Equivalently, mean of z_i = 1[s_i>=tau]*(O_i - gamma) >= 0!
                    # z_i in [-gamma, 1-gamma], span = 1.0!
                    z_hat = np.where(mask, s_pool - target_gamma, 0.0)
                    z_hat_norm = np.clip(z_hat + target_gamma, 0.0, 1.0)
                    phi_m = np.array([1.0, float(np.mean(z_hat_norm))], dtype=np.float64)
                    sum_lo, sum_hi = nonnegative_sum_interval(z_hat_norm)
                    cs_m = WoRControlVariateCS(
                        n_total=n_rem,
                        alpha=alpha_cell,
                        phi_mean_init=phi_m,
                        phi_sum_interval=(np.array([n_rem, sum_lo]), np.array([n_rem, sum_hi])),
                        y_min=-target_gamma,
                        y_max=1.0 - target_gamma,
                        use_wor=True,
                        safe_rounding=self.safe_rounding,
                    )
                    g_cs_list.append(cs_m)
            cs_grid.append(g_cs_list)
            prefix_sizes.append(g_pref)

        # Helper to draw a WoR batch from group gi's certification pool and update all M grid CSs
        def _cert_step(gi: int, k_req: int) -> int:
            nonlocal total_calls
            pool = rem_pools[gi]
            rem_left = len(pool) - int(rem_ptrs[gi])
            rem_bud = oracle_budget - total_calls
            k = min(k_req, rem_left, rem_bud)
            if k <= 0:
                return 0
            st = int(rem_ptrs[gi])
            rows = pool[st : st + k]
            rem_ptrs[gi] += k
            y_b, _ = self._eval_oracle(rows)
            for r_id, y_v in zip(rows, y_b):
                oracle_cache[int(r_id)] = float(y_v)
            s_b = scores_all[rows]
            taus = grids[gi]
            # Predictable control-variate weights theta = [0, 0.7] (conservative fixed lift in [0,1])
            th_cv = np.array([0.0, 0.7], dtype=np.float64)
            for m_idx, tau in enumerate(taus):
                mask_b = s_b >= tau
                if target_mode == "recall":
                    h_b = np.where(mask_b, (1.0 - target_gamma) * y_b, -target_gamma * y_b)
                    h_hat_b = np.where(mask_b, (1.0 - target_gamma) * s_b, -target_gamma * s_b)
                    phi_b = np.column_stack([np.ones(k), np.clip(h_hat_b + target_gamma, 0.0, 1.0)])
                else:
                    h_b = np.where(mask_b, y_b - target_gamma, 0.0)
                    h_hat_b = np.where(mask_b, s_b - target_gamma, 0.0)
                    phi_b = np.column_stack([np.ones(k), np.clip(h_hat_b + target_gamma, 0.0, 1.0)])
                cs_grid[gi][m_idx].update_batch(h_b, phi_b, th_cv)
            calls_by_gi[gi] += k
            total_calls += k
            return k

        # Initial certification round across all groups
        init_cert = max(4, min(batch_size, (oracle_budget - total_calls) // max(G * 2, 1)))
        for gi in range(G):
            _cert_step(gi, init_cert)

        # Adaptive active-group budget routing:
        # Prioritize groups that still have large gap between currently certified threshold
        # and empirically optimal threshold (i.e. highest potential precision/recall improvement).
        rr_c = 0
        while total_calls < oracle_budget:
            priorities = np.zeros(G, dtype=np.float64)
            for gi in range(G):
                if int(rem_ptrs[gi]) >= len(rem_pools[gi]) or not cs_grid[gi]:
                    continue
                n_rem = len(rem_pools[gi])
                taus = grids[gi]
                if target_mode == "recall":
                    # Best certified index (smallest prefix = highest tau with lower bound >= 0)
                    cert_idx = len(taus) - 1
                    emp_idx = len(taus) - 1
                    for m_idx, cs_m in enumerate(cs_grid[gi]):
                        lo_m, _ = cs_m.bounds
                        if cs_m.estimate >= 0.0 and emp_idx == len(taus) - 1:
                            emp_idx = m_idx
                        if lo_m >= 0.0:
                            cert_idx = m_idx
                            break
                    # Potential savings = extra tuples currently forced into S_g beyond empirical optimum
                    gap_frac = (prefix_sizes[gi][cert_idx] - prefix_sizes[gi][emp_idx]) / max(n_rem, 1)
                    # Add small exploration bonus for groups with wide CS at emp_idx
                    hw_emp = cs_grid[gi][emp_idx].half_width
                    priorities[gi] = gap_frac + 0.25 * hw_emp
                else:
                    # Precision mode: want LARGEST prefix (highest m_idx) with lower bound >= 0
                    cert_idx = -1
                    emp_idx = -1
                    for m_idx, cs_m in enumerate(cs_grid[gi]):
                        lo_m, _ = cs_m.bounds
                        if cs_m.estimate >= 0.0:
                            emp_idx = m_idx
                        if lo_m >= 0.0:
                            cert_idx = m_idx
                    cert_sz = prefix_sizes[gi][cert_idx] if cert_idx >= 0 else 0
                    emp_sz = prefix_sizes[gi][emp_idx] if emp_idx >= 0 else 0
                    gap_frac = max(0, emp_sz - cert_sz) / max(n_rem, 1)
                    hw_e = cs_grid[gi][max(emp_idx, 0)].half_width
                    priorities[gi] = gap_frac + 0.25 * hw_e

            if float(np.max(priorities)) <= 0.0:
                break

            if use_adaptive_alloc:
                gi_next = int(np.argmax(priorities))
            else:
                gi_next = -1
                for off in range(G):
                    cand = (rr_c + off) % G
                    if int(rem_ptrs[cand]) < len(rem_pools[cand]):
                        gi_next = cand
                        rr_c = (cand + 1) % G
                        break
                if gi_next < 0:
                    break

            if _cert_step(gi_next, batch_size) == 0:
                break

        # Finalize certified thresholds and build selected tuple set S = union_g S_g
        selected_chunks: List[np.ndarray] = []
        thresholds_by_g: Dict[str, float] = {}
        selected_cnt_by_g: Dict[str, int] = {}
        calls_dict: Dict[str, int] = {}

        for gi, g in enumerate(prof.groups):
            pool = rem_pools[gi]
            taus = grids[gi]
            # Select exact oracle-positive tuples among queried ones (always free & exact!)
            queried_rows = np.concatenate(
                [perms[gi][: int(ptrs[gi])], pool[: int(rem_ptrs[gi])]]
            )
            unqueried_rows = pool[int(rem_ptrs[gi]) :]
            q_pos = np.array(
                [r for r in queried_rows if oracle_cache.get(int(r), 0.0) >= 0.5],
                dtype=np.int64,
            )

            if target_mode == "recall":
                tau_star = 0.0  # safe fallback (recall = 1.0)
                for m_idx, cs_m in enumerate(cs_grid[gi]):
                    lo_m, _ = cs_m.bounds
                    if lo_m >= 0.0:
                        tau_star = float(taus[m_idx])
                        break
                unq_sel = unqueried_rows[scores_all[unqueried_rows] >= tau_star]
            else:
                tau_star = 1.01  # safe fallback (select only oracle-verified positives)
                for m_idx in range(len(cs_grid[gi]) - 1, -1, -1):
                    lo_m, _ = cs_grid[gi][m_idx].bounds
                    if lo_m >= 0.0:
                        tau_star = float(taus[m_idx])
                        break
                unq_sel = unqueried_rows[scores_all[unqueried_rows] >= tau_star]

            sel_g = np.concatenate([q_pos, unq_sel])
            selected_chunks.append(sel_g)
            thresholds_by_g[g] = tau_star
            selected_cnt_by_g[g] = int(len(sel_g))
            calls_dict[g] = int(calls_by_gi[gi])

        selected_all = np.concatenate(selected_chunks) if selected_chunks else np.empty((0,), dtype=np.int64)
        wall_ms = (time.perf_counter() - t_start) * 1000.0
        return GroupSelectResult(
            groups=prof.groups,
            selected_indices=selected_all,
            selected_count_by_group=selected_cnt_by_g,
            thresholds_by_group=thresholds_by_g,
            oracle_calls_by_group=calls_dict,
            total_oracle_calls=int(total_calls),
            total_rows=prof.n_total,
            duckdb_scan_ms=prof.duckdb_scan_ms,
            optimizer_ms=wall_ms,
            wall_ms=wall_ms,
            target_mode=target_mode,
            target_gamma=target_gamma,
        )

    # ------------------------------------------------------------------
    # Q3a: Grouped HAVING (GROUP BY g HAVING agg >= tau) with Early Pruning
    # ------------------------------------------------------------------
    def execute_grouped_having(
        self,
        tau: float,
        agg_type: str = "mean",
        delta: float = 0.05,
        indifference_eps: float = 0.005,
        batch_size: int = 16,
        pilot_per_group: int = 8,
        max_oracle_budget: Optional[int] = None,
        use_control_variates: bool = True,
        use_covariates: bool = True,
        use_early_pruning: bool = True,
        seed: Optional[int] = None,
    ) -> GroupHavingResult:
        """
        Execute Q3a: Certify {g : agg_g >= tau} with joint probability >= 1 - delta.
        Prunes any group g immediately once L_g >= tau (IN) or U_g < tau (OUT).
        """
        self._validate_grouped_agg_type(agg_type)
        t_start = time.perf_counter()
        rng = np.random.default_rng(self.seed if seed is None else seed)
        prof = self.profile
        G = len(prof.groups)
        d = prof.phi_all.shape[1]
        alpha_g = delta / max(G, 1)

        eps_indiff_gi = np.array(
            [
                indifference_eps / max(prof.n_by_group[gi], 1)
                if agg_type in ("count", "sum")
                else indifference_eps
                for gi in range(G)
            ],
            dtype=np.float64,
        )

        perms = [rng.permutation(idxs) for idxs in prof.group_row_indices]
        ptrs = np.zeros(G, dtype=np.int64)
        cs_list = [
            WoRControlVariateCS(
                n_total=int(prof.n_by_group[gi]),
                alpha=alpha_g,
                phi_mean_init=prof.phi_mean_by_group[gi],
                phi_sum_interval=(prof.phi_sum_lo_by_group[gi], prof.phi_sum_hi_by_group[gi]),
                y_min=prof.measure_min,
                y_max=prof.measure_max,
                use_wor=True,
                safe_rounding=self.safe_rounding,
            )
            for gi in range(G)
        ]

        phi_seen_by_g: List[List[np.ndarray]] = [[] for _ in range(G)]
        y_seen_by_g: List[List[np.ndarray]] = [[] for _ in range(G)]
        y_span = max(prof.measure_max - prof.measure_min, 1e-12)
        total_calls = 0
        budget_cap = max_oracle_budget if max_oracle_budget is not None else prof.n_total

        def _draw(gi: int, k_req: int, th_glob: np.ndarray) -> int:
            nonlocal total_calls
            if cs_list[gi].exhausted:
                return 0
            k = min(k_req, int(prof.n_by_group[gi] - ptrs[gi]), budget_cap - total_calls)
            if k <= 0:
                return 0
            st = int(ptrs[gi])
            rows = perms[gi][st : st + k]
            ptrs[gi] += k
            th_g, fb_g = self._get_predictable_theta(
                gi,
                phi_seen_by_g,
                y_seen_by_g,
                th_glob,
                alpha_g,
                d,
                use_control_variates=use_control_variates,
                use_covariates=use_covariates,
                use_hierarchical=True,
                target_eps=(
                    self._predictable_decision_eps(
                        cs_list[gi],
                        tau / max(prof.n_by_group[gi], 1)
                        if agg_type in ("count", "sum")
                        else (tau - prof.measure_min) / y_span,
                        eps_indiff_gi[gi] / (1.0 if agg_type in ("count", "sum") else y_span),
                    )
                    if self.gating_mode == "safe_loo" else None
                ),
            )
            phi_b = prof.phi_all[rows]
            y_b, _ = self._eval_oracle(rows, binary_count=agg_type == 'count')
            cs_list[gi].update_batch(y_b, phi_b, th_g, exact_f_bounds=fb_g)
            phi_seen_by_g[gi].append(phi_b)
            y_seen_by_g[gi].append(np.clip((y_b - prof.measure_min) / y_span, 0.0, 1.0))
            total_calls += k
            return k

        th_init = np.zeros(d, dtype=np.float64)
        if use_control_variates and d > 1:
            th_init[1] = 1.0
        for gi in range(G):
            _draw(gi, pilot_per_group, th_init)

        th_glob = self._fit_global_theta(
            [np.vstack(x) for x in phi_seen_by_g if x],
            [np.concatenate(x) for x in y_seen_by_g if x],
            d,
            use_covariates=use_covariates,
        )

        step = 0
        while total_calls < budget_cap:
            active_scores = np.zeros(G, dtype=np.float64)
            for gi in range(G):
                if cs_list[gi].exhausted:
                    continue
                lo, hi = cs_list[gi].bounds
                count_lo, count_hi = self._aggregate_bounds(cs_list[gi], agg_type)
                decided = (count_lo >= tau or count_hi < tau or
                           cs_list[gi].half_width <= eps_indiff_gi[gi])
                if use_early_pruning and decided:
                    continue
                if not use_early_pruning and cs_list[gi].half_width <= eps_indiff_gi[gi]:
                    continue
                # Priority: widest uncertain interval straddling tau
                active_scores[gi] = hi - lo

            if float(np.max(active_scores)) <= 0.0:
                break
            gi_next = int(np.argmax(active_scores))
            if _draw(gi_next, batch_size, th_glob) == 0:
                break
            step += 1
            if use_control_variates and step % max(4, G // 2) == 0:
                th_glob = self._fit_global_theta(
                    [np.vstack(x) for x in phi_seen_by_g if x],
                    [np.concatenate(x) for x in y_seen_by_g if x],
                    d,
                    use_covariates=use_covariates,
                )

        acc, rej, unc = [], [], []
        est_d, lo_d, hi_d, calls_d = {}, {}, {}, {}
        for gi, g in enumerate(prof.groups):
            lo, hi = self._aggregate_bounds(cs_list[gi], agg_type)
            est_d[g], lo_d[g], hi_d[g] = (lo + hi) / 2, lo, hi
            calls_d[g] = int(cs_list[gi].n_seen)
            if lo >= tau:
                acc.append(g)
            elif hi < tau:
                rej.append(g)
            else:
                unc.append(g)

        wall_ms = (time.perf_counter() - t_start) * 1000.0
        return GroupHavingResult(
            groups=prof.groups,
            accepted_groups=acc,
            rejected_groups=rej,
            uncertain_groups=unc,
            estimates=est_d,
            lower=lo_d,
            upper=hi_d,
            oracle_calls_by_group=calls_d,
            total_oracle_calls=total_calls,
            total_rows=prof.n_total,
            duckdb_scan_ms=prof.duckdb_scan_ms,
            optimizer_ms=wall_ms,
            wall_ms=wall_ms,
            tau=tau,
        )

    # ------------------------------------------------------------------
    # Q3b: Grouped TOP-K (GROUP BY g ORDER BY agg DESC LIMIT K) via LUCB-StratQ
    # ------------------------------------------------------------------
    def execute_grouped_topk(
        self,
        k: int = 5,
        agg_type: str = "count",
        delta: float = 0.05,
        indifference_eps: float = 0.0,
        batch_size: int = 16,
        pilot_per_group: int = 8,
        max_oracle_budget: Optional[int] = None,
        use_control_variates: bool = True,
        use_covariates: bool = True,
        use_early_pruning: bool = True,
        seed: Optional[int] = None,
    ) -> GroupTopKResult:
        """Certify top-k membership by strict interval separation.

        At full census, break ties by the profile's group order. A positive
        indifference epsilon can stop sampling early with certified=False.
        """
        self._validate_grouped_agg_type(agg_type)
        t_start = time.perf_counter()
        rng = np.random.default_rng(self.seed if seed is None else seed)
        prof = self.profile
        G = len(prof.groups)
        k_eff = min(max(int(k), 1), G)
        d = prof.phi_all.shape[1]
        alpha_g = delta / max(G, 1)

        perms = [rng.permutation(idxs) for idxs in prof.group_row_indices]
        ptrs = np.zeros(G, dtype=np.int64)
        cs_list = [
            WoRControlVariateCS(
                n_total=int(prof.n_by_group[gi]),
                alpha=alpha_g,
                phi_mean_init=prof.phi_mean_by_group[gi],
                phi_sum_interval=(prof.phi_sum_lo_by_group[gi], prof.phi_sum_hi_by_group[gi]),
                y_min=prof.measure_min,
                y_max=prof.measure_max,
                use_wor=True,
                safe_rounding=self.safe_rounding,
            )
            for gi in range(G)
        ]

        phi_seen_by_g: List[List[np.ndarray]] = [[] for _ in range(G)]
        y_seen_by_g: List[List[np.ndarray]] = [[] for _ in range(G)]
        y_span = max(prof.measure_max - prof.measure_min, 1e-12)
        total_calls = 0
        budget_cap = max_oracle_budget if max_oracle_budget is not None else prof.n_total

        def _draw(gi: int, k_req: int, th_glob: np.ndarray,
                  boundary_norm: Optional[float] = None) -> int:
            nonlocal total_calls
            if cs_list[gi].exhausted:
                return 0
            cnt = min(k_req, int(prof.n_by_group[gi] - ptrs[gi]), budget_cap - total_calls)
            if cnt <= 0:
                return 0
            st = int(ptrs[gi])
            rows = perms[gi][st : st + cnt]
            ptrs[gi] += cnt
            th_g, fb_g = self._get_predictable_theta(
                gi,
                phi_seen_by_g,
                y_seen_by_g,
                th_glob,
                alpha_g,
                d,
                use_control_variates=use_control_variates,
                use_covariates=use_covariates,
                use_hierarchical=True,
                target_eps=(
                    self._predictable_decision_eps(
                        cs_list[gi], boundary_norm,
                        1.0 / max(prof.n_by_group[gi], 1),
                    )
                    if self.gating_mode == "safe_loo" and boundary_norm is not None
                    else None
                ),
            )
            phi_b = prof.phi_all[rows]
            y_b, _ = self._eval_oracle(rows, binary_count=agg_type == 'count')
            cs_list[gi].update_batch(y_b, phi_b, th_g, exact_f_bounds=fb_g)
            phi_seen_by_g[gi].append(phi_b)
            y_seen_by_g[gi].append(np.clip((y_b - prof.measure_min) / y_span, 0.0, 1.0))
            total_calls += cnt
            return cnt

        th_init = np.zeros(d, dtype=np.float64)
        if use_control_variates and d > 1:
            th_init[1] = 1.0
        for gi in range(G):
            _draw(gi, pilot_per_group, th_init)

        th_glob = self._fit_global_theta(
            [np.vstack(x) for x in phi_seen_by_g if x],
            [np.concatenate(x) for x in y_seen_by_g if x],
            d,
            use_covariates=use_covariates,
        )

        step = 0
        while True:
            los, his = np.array([self._aggregate_bounds(cs, agg_type) for cs in cs_list]).reshape(G, 2).T
            ests = (los + his) / 2

            if k_eff == G or all(cs.exhausted for cs in cs_list):
                certified = True
                break

            order_desc = np.argsort(-ests, kind="stable")
            top_set = order_desc[:k_eff]
            bot_set = order_desc[k_eff:]

            h_star = int(top_set[np.argmin(los[top_set])])
            l_star = int(bot_set[np.argmax(his[bot_set])])

            certified = bool(los[h_star] > his[l_star])
            if certified or total_calls >= budget_cap:
                break
            if indifference_eps > 0 and los[h_star] + indifference_eps >= his[l_star]:
                break

            if use_early_pruning:
                boundary_norm = (
                    his[l_star] / max(prof.n_by_group[l_star], 1)
                    if agg_type in ("count", "sum")
                    else (his[l_star] - prof.measure_min) / y_span
                )
                # LUCB: sample boundary pair (h_star, l_star)
                d1 = _draw(h_star, batch_size, th_glob, boundary_norm)
                d2 = _draw(l_star, batch_size, th_glob, boundary_norm)
                if d1 + d2 == 0:
                    # Fallback if both boundary groups exhausted
                    widths = his - los
                    rem_unex = [gi for gi in range(G) if not cs_list[gi].exhausted]
                    if not rem_unex:
                        certified = True
                        break
                    gi_alt = max(rem_unex, key=lambda gi: widths[gi])
                    if _draw(gi_alt, batch_size, th_glob) == 0:
                        break
            else:
                # Non-pruning ablation: sample widest interval across ALL groups
                rem_unex = [gi for gi in range(G) if not cs_list[gi].exhausted]
                if not rem_unex:
                    certified = True
                    break
                gi_w = max(rem_unex, key=lambda gi: his[gi] - los[gi])
                if _draw(gi_w, batch_size, th_glob) == 0:
                    break

            step += 1
            if use_control_variates and step % max(4, G // 2) == 0:
                th_glob = self._fit_global_theta(
                    [np.vstack(x) for x in phi_seen_by_g if x],
                    [np.concatenate(x) for x in y_seen_by_g if x],
                    d,
                    use_covariates=use_covariates,
                )

        order_desc = np.argsort(-ests, kind="stable")
        topk_groups = [prof.groups[gi] for gi in order_desc[:k_eff]]

        est_d, lo_d, hi_d, calls_d = {}, {}, {}, {}
        for gi, g in enumerate(prof.groups):
            est_d[g] = float(ests[gi])
            lo_d[g], hi_d[g] = float(los[gi]), float(his[gi])
            calls_d[g] = int(cs_list[gi].n_seen)

        wall_ms = (time.perf_counter() - t_start) * 1000.0
        return GroupTopKResult(
            groups=prof.groups,
            topk_groups=topk_groups,
            estimates=est_d,
            lower=lo_d,
            upper=hi_d,
            oracle_calls_by_group=calls_d,
            total_oracle_calls=total_calls,
            total_rows=prof.n_total,
            duckdb_scan_ms=prof.duckdb_scan_ms,
            optimizer_ms=wall_ms,
            wall_ms=wall_ms,
            k=k_eff,
            certified=certified,
        )
