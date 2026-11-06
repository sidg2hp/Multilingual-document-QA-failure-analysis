"""Cost-aware OCR-vs-VLM router with leakage guards and grouped evaluation.

Fixes relative to the previous implementation:

- Features are text-free (or declared-language metadata only). Oracle-transcript
  features raise ``OracleFeatureLeakageError``.
- Costs are measured wall-clock milliseconds, not hard-coded relative constants;
  the relative fallback is retained only for unit tests and is labelled.
- Cross-validation is grouped by document when group IDs are supplied, so
  questions from the same page cannot leak across folds.
- Reports the fraction of the per-item oracle gap captured and the gain over the
  best single pipeline, not just a Pareto curve.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold

from src.routing.features import DocumentRoutingFeatures, OracleFeatureLeakageError

PIPELINE_OCR = "cascaded_ocr"
PIPELINE_VLM = "ocr_free_vlm"


@dataclass
class RoutingDecision:
    """Routing prediction for a single document item."""

    sample_id: str
    selected_pipeline: str
    pred_anls_ocr: float
    pred_anls_vlm: float
    confidence_margin: float
    cost_ocr: float
    cost_vlm: float


class AdaptiveDocQARouter:
    """Predicts pipeline-specific ANLS and routes under a measured cost budget."""

    def __init__(
        self,
        ocr_relative_cost: float = 0.25,
        vlm_relative_cost: float = 1.00,
        cost_tradeoff_lambda: float = 0.10,
        random_state: int = 42,
        n_estimators: int = 64,
        max_depth: int = 6,
    ) -> None:
        self.ocr_relative_cost = ocr_relative_cost
        self.vlm_relative_cost = vlm_relative_cost
        self.cost_tradeoff_lambda = cost_tradeoff_lambda
        self.random_state = random_state
        self._reg_ocr = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
        self._reg_vlm = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
        self._is_fitted = False

    # -- validation --------------------------------------------------------
    @staticmethod
    def _validate_features(features: Sequence[DocumentRoutingFeatures]) -> None:
        for feature in features:
            if feature.uses_oracle_text or feature.script_mixing_source == "oracle_text":
                raise OracleFeatureLeakageError(
                    f"Routing feature for {feature.sample_id} uses oracle text; "
                    "this is unavailable at routing time."
                )

    @staticmethod
    def _vectors(features: Sequence[DocumentRoutingFeatures]) -> np.ndarray:
        if not features:
            raise ValueError("At least one feature vector is required")
        return np.asarray([f.to_vector() for f in features], dtype=np.float32)

    # -- training ----------------------------------------------------------
    def fit(
        self,
        features: List[DocumentRoutingFeatures],
        ocr_anls_scores: List[float],
        vlm_anls_scores: List[float],
    ) -> None:
        """Trains dual regressors E[ANLS_ocr | x] and E[ANLS_vlm | x]."""
        self._validate_features(features)
        if not (len(features) == len(ocr_anls_scores) == len(vlm_anls_scores)):
            raise ValueError("features and score lists must have equal length")
        x_mat = self._vectors(features)
        self._reg_ocr.fit(x_mat, np.asarray(ocr_anls_scores, dtype=np.float32))
        self._reg_vlm.fit(x_mat, np.asarray(vlm_anls_scores, dtype=np.float32))
        self._is_fitted = True

    def predict_scores(
        self, features: List[DocumentRoutingFeatures]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts expected ANLS for cascaded OCR and OCR-free VLM."""
        self._validate_features(features)
        x_mat = self._vectors(features)
        if not self._is_fitted:
            # Explicit heuristic fallback, used only before fitting.
            lap = x_mat[:, 0]
            declared_indic = x_mat[:, 6]
            smi = x_mat[:, 7]
            pred_ocr = np.clip(
                0.85 * (lap / (lap + 300.0)) - 0.25 * smi - 0.05 * declared_indic,
                0.0,
                1.0,
            )
            pred_vlm = np.clip(
                0.80 * (lap / (lap + 120.0)) - 0.10 * smi + 0.05 * declared_indic,
                0.0,
                1.0,
            )
            return pred_ocr, pred_vlm
        return self._reg_ocr.predict(x_mat), self._reg_vlm.predict(x_mat)

    def feature_importances(self) -> Dict[str, float]:
        if not self._is_fitted:
            return {}
        names = DocumentRoutingFeatures.feature_names()
        return {
            name: round(float(value), 4)
            for name, value in zip(names, self._reg_vlm.feature_importances_)
        }

    # -- costs -------------------------------------------------------------
    @staticmethod
    def _normalized_costs(
        ocr_cost: Optional[Sequence[float] | float],
        vlm_cost: Optional[Sequence[float] | float],
        n: int,
        ocr_relative_cost: float,
        vlm_relative_cost: float,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Returns per-item normalized costs and a cost-model label."""
        if ocr_cost is None or vlm_cost is None:
            return (
                np.full(n, float(ocr_relative_cost), dtype=np.float64),
                np.full(n, float(vlm_relative_cost), dtype=np.float64),
                "relative_fallback",
            )
        ocr = np.asarray(ocr_cost, dtype=np.float64)
        vlm = np.asarray(vlm_cost, dtype=np.float64)
        if ocr.ndim == 0:
            ocr = np.full(n, float(ocr))
        if vlm.ndim == 0:
            vlm = np.full(n, float(vlm))
        if ocr.size != n or vlm.size != n:
            raise ValueError("Per-item costs must match the number of features")
        scale = max(1e-9, float(np.max([ocr.max(), vlm.max()])))
        return ocr / scale, vlm / scale, "measured_ms"

    # -- routing -----------------------------------------------------------
    def route_batch(
        self,
        features: List[DocumentRoutingFeatures],
        cost_tradeoff_lambda: Optional[float] = None,
        ocr_cost_ms: Optional[Sequence[float] | float] = None,
        vlm_cost_ms: Optional[Sequence[float] | float] = None,
        pred_ocr: Optional[np.ndarray] = None,
        pred_vlm: Optional[np.ndarray] = None,
    ) -> List[RoutingDecision]:
        """Selects the utility-maximizing pipeline per item."""
        lam = (
            self.cost_tradeoff_lambda
            if cost_tradeoff_lambda is None
            else cost_tradeoff_lambda
        )
        s_ocr, s_vlm = (
            (pred_ocr, pred_vlm)
            if pred_ocr is not None and pred_vlm is not None
            else self.predict_scores(features)
        )
        cost_ocr, cost_vlm, _ = self._normalized_costs(
            ocr_cost_ms,
            vlm_cost_ms,
            len(features),
            self.ocr_relative_cost,
            self.vlm_relative_cost,
        )
        decisions: List[RoutingDecision] = []
        for index, feature in enumerate(features):
            u_ocr = float(s_ocr[index]) - lam * float(cost_ocr[index])
            u_vlm = float(s_vlm[index]) - lam * float(cost_vlm[index])
            chosen = PIPELINE_VLM if u_vlm > u_ocr else PIPELINE_OCR
            decisions.append(
                RoutingDecision(
                    sample_id=feature.sample_id,
                    selected_pipeline=chosen,
                    pred_anls_ocr=round(float(s_ocr[index]), 4),
                    pred_anls_vlm=round(float(s_vlm[index]), 4),
                    confidence_margin=round(abs(u_vlm - u_ocr), 4),
                    cost_ocr=round(float(cost_ocr[index]), 4),
                    cost_vlm=round(float(cost_vlm[index]), 4),
                )
            )
        return decisions

    def _curve_rows(
        self,
        decisions_by_lambda: Dict[float, List[RoutingDecision]],
        true_ocr: np.ndarray,
        true_vlm: np.ndarray,
        cost_ocr: np.ndarray,
        cost_vlm: np.ndarray,
        cost_model: str,
    ) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
        always_ocr = float(np.mean(true_ocr))
        always_vlm = float(np.mean(true_vlm))
        oracle = float(np.mean(np.maximum(true_ocr, true_vlm)))
        best_single = max(always_ocr, always_vlm)

        rows: List[Dict[str, object]] = []
        for lam, decisions in decisions_by_lambda.items():
            scores = []
            costs = []
            vlm_count = 0
            for index, decision in enumerate(decisions):
                if decision.selected_pipeline == PIPELINE_VLM:
                    scores.append(float(true_vlm[index]))
                    costs.append(float(cost_vlm[index]))
                    vlm_count += 1
                else:
                    scores.append(float(true_ocr[index]))
                    costs.append(float(cost_ocr[index]))
            routed = float(np.mean(scores))
            gap = oracle - best_single
            rows.append(
                {
                    "lambda": lam,
                    "routed_anls": round(routed, 4),
                    "relative_cost": round(float(np.mean(costs)), 4),
                    "vlm_routing_rate": round(vlm_count / max(1, len(decisions)), 4),
                    "gain_over_best_single": round(routed - best_single, 4),
                    "fraction_of_oracle_gap_captured": (
                        round((routed - best_single) / gap, 4) if gap > 1e-9 else None
                    ),
                }
            )
        baselines = {
            "always_ocr_anls": round(always_ocr, 4),
            "always_vlm_anls": round(always_vlm, 4),
            "best_single_anls": round(best_single, 4),
            "oracle_router_upper_bound_anls": round(oracle, 4),
            "oracle_gap": round(oracle - best_single, 4),
            "cost_model": cost_model,
        }
        return rows, baselines

    def evaluate_pareto_frontier(
        self,
        features: List[DocumentRoutingFeatures],
        true_ocr_anls: List[float],
        true_vlm_anls: List[float],
        lambdas: Optional[List[float]] = None,
        ocr_cost_ms: Optional[Sequence[float] | float] = None,
        vlm_cost_ms: Optional[Sequence[float] | float] = None,
    ) -> Dict[str, object]:
        """In-sample Pareto evaluation (for unit tests, not paper numbers)."""
        if lambdas is None:
            lambdas = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]
        y_ocr = np.asarray(true_ocr_anls, dtype=np.float32)
        y_vlm = np.asarray(true_vlm_anls, dtype=np.float32)
        n = len(features)
        if n != y_ocr.size or n != y_vlm.size:
            raise ValueError("features and score lists must have equal length")
        cost_ocr, cost_vlm, cost_model = self._normalized_costs(
            ocr_cost_ms,
            vlm_cost_ms,
            n,
            self.ocr_relative_cost,
            self.vlm_relative_cost,
        )
        decisions_by_lambda = {
            lam: self.route_batch(
                features,
                cost_tradeoff_lambda=lam,
                ocr_cost_ms=cost_ocr,
                vlm_cost_ms=cost_vlm,
            )
            for lam in lambdas
        }
        rows, baselines = self._curve_rows(
            decisions_by_lambda, y_ocr, y_vlm, cost_ocr, cost_vlm, cost_model
        )
        return {
            "evaluation_protocol": "in_sample",
            **baselines,
            "pareto_frontier": rows,
        }

    def evaluate_pareto_frontier_cv(
        self,
        features: List[DocumentRoutingFeatures],
        true_ocr_anls: List[float],
        true_vlm_anls: List[float],
        lambdas: Optional[List[float]] = None,
        n_splits: int = 5,
        seed: int = 42,
        group_ids: Optional[Sequence[str]] = None,
        ocr_cost_ms: Optional[Sequence[float] | float] = None,
        vlm_cost_ms: Optional[Sequence[float] | float] = None,
    ) -> Dict[str, object]:
        """Out-of-fold, document-grouped router evaluation."""
        if lambdas is None:
            lambdas = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]
        self._validate_features(features)
        n = len(features)
        if n != len(true_ocr_anls) or n != len(true_vlm_anls):
            raise ValueError("features and score lists must have equal length")
        if n < 2:
            return {}

        x_mat = self._vectors(features)
        y_ocr = np.asarray(true_ocr_anls, dtype=np.float32)
        y_vlm = np.asarray(true_vlm_anls, dtype=np.float32)
        effective_splits = min(int(n_splits), n)
        if effective_splits < 2:
            result = self.evaluate_pareto_frontier(
                features,
                true_ocr_anls,
                true_vlm_anls,
                lambdas,
                ocr_cost_ms,
                vlm_cost_ms,
            )
            result["evaluation_protocol"] = "in_sample_fallback_too_few_samples"
            return result

        if group_ids is not None:
            if len(group_ids) != n:
                raise ValueError("group_ids must match the number of features")
            n_groups = len(set(group_ids))
            effective_splits = min(effective_splits, n_groups)
            if effective_splits < 2:
                return {
                    "evaluation_protocol": "grouped_fallback_too_few_documents",
                    "n_splits": effective_splits,
                    "n_clusters": n_groups,
                }
            splitter = GroupKFold(n_splits=effective_splits)
            splits = list(splitter.split(x_mat, groups=list(group_ids)))
            protocol = f"{effective_splits}-fold_grouped_by_document"
        else:
            splitter = KFold(n_splits=effective_splits, shuffle=True, random_state=seed)
            splits = list(splitter.split(x_mat))
            protocol = f"{effective_splits}-fold_cross_validated"

        oof_ocr = np.zeros(n, dtype=np.float32)
        oof_vlm = np.zeros(n, dtype=np.float32)
        for train_idx, test_idx in splits:
            reg_ocr = RandomForestRegressor(
                n_estimators=self._reg_ocr.n_estimators,
                max_depth=self._reg_ocr.max_depth,
                random_state=seed,
            )
            reg_vlm = RandomForestRegressor(
                n_estimators=self._reg_vlm.n_estimators,
                max_depth=self._reg_vlm.max_depth,
                random_state=seed,
            )
            reg_ocr.fit(x_mat[train_idx], y_ocr[train_idx])
            reg_vlm.fit(x_mat[train_idx], y_vlm[train_idx])
            oof_ocr[test_idx] = reg_ocr.predict(x_mat[test_idx])
            oof_vlm[test_idx] = reg_vlm.predict(x_mat[test_idx])

        cost_ocr, cost_vlm, cost_model = self._normalized_costs(
            ocr_cost_ms,
            vlm_cost_ms,
            n,
            self.ocr_relative_cost,
            self.vlm_relative_cost,
        )
        decisions_by_lambda = {
            lam: self.route_batch(
                features,
                cost_tradeoff_lambda=lam,
                ocr_cost_ms=cost_ocr,
                vlm_cost_ms=cost_vlm,
                pred_ocr=oof_ocr,
                pred_vlm=oof_vlm,
            )
            for lam in lambdas
        }
        rows, baselines = self._curve_rows(
            decisions_by_lambda, y_ocr, y_vlm, cost_ocr, cost_vlm, cost_model
        )
        return {
            "evaluation_protocol": protocol,
            "n_splits": effective_splits,
            "n_clusters": len(set(group_ids)) if group_ids is not None else n,
            "oof_prediction_mae_ocr": round(
                float(np.mean(np.abs(oof_ocr - y_ocr))), 4
            ),
            "oof_prediction_mae_vlm": round(
                float(np.mean(np.abs(oof_vlm - y_vlm))), 4
            ),
            **baselines,
            "pareto_frontier": rows,
        }
