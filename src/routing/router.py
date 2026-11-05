"""Adaptive OCR-vs-VLM Routing Classifier and Pareto Frontier Evaluator.

Implements:
1. Utility-based routing:
     pi*(x; lambda) = argmax_{p in {OCR, VLM}} [ E[ANLS_p | x] - lambda * Cost_p ]
2. Supervised training using RandomForest / GradientBoosting over DocumentRoutingFeatures.
3. Pareto curve generation across lambda in [0, 1] comparing:
   - Always Cascaded OCR
   - Always OCR-Free VLM
   - Learned Adaptive Router
   - Oracle Router (Upper Bound)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from src.routing.features import DocumentRoutingFeatures


PIPELINE_OCR = "cascaded_ocr"
PIPELINE_VLM = "ocr_free_vlm"


@dataclass
class RoutingDecision:
    """Routing prediction for a single document sample."""

    sample_id: str
    selected_pipeline: str
    pred_anls_ocr: float
    pred_anls_vlm: float
    confidence_margin: float


class AdaptiveDocQARouter:
    """Predicts pipeline-specific ANLS and routes documents under a compute budget."""

    def __init__(
        self,
        ocr_relative_cost: float = 0.25,
        vlm_relative_cost: float = 1.00,
        cost_tradeoff_lambda: float = 0.10,
        random_state: int = 42,
    ) -> None:
        self.ocr_relative_cost = ocr_relative_cost
        self.vlm_relative_cost = vlm_relative_cost
        self.cost_tradeoff_lambda = cost_tradeoff_lambda
        self._reg_ocr = RandomForestRegressor(
            n_estimators=64, max_depth=6, random_state=random_state
        )
        self._reg_vlm = RandomForestRegressor(
            n_estimators=64, max_depth=6, random_state=random_state
        )
        self._is_fitted = False

    def fit(
        self,
        features: List[DocumentRoutingFeatures],
        ocr_anls_scores: List[float],
        vlm_anls_scores: List[float],
    ) -> None:
        """Trains dual performance regressors E[ANLS_ocr | x] and E[ANLS_vlm | x]."""
        x_mat = np.asarray([f.to_vector() for f in features], dtype=np.float32)
        y_ocr = np.asarray(ocr_anls_scores, dtype=np.float32)
        y_vlm = np.asarray(vlm_anls_scores, dtype=np.float32)
        self._reg_ocr.fit(x_mat, y_ocr)
        self._reg_vlm.fit(x_mat, y_vlm)
        self._is_fitted = True

    def predict_scores(
        self, features: List[DocumentRoutingFeatures]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts expected ANLS for Cascaded OCR and OCR-Free VLM."""
        x_mat = np.asarray([f.to_vector() for f in features], dtype=np.float32)
        if not self._is_fitted:
            # Heuristic fallback before supervised training:
            # High sharpness (laplacian > 400) & low script-mixing -> OCR is strong;
            # Complex layout transitions or blur -> VLM is stronger.
            lap = x_mat[:, 0]
            smi = x_mat[:, 6]
            pred_ocr = np.clip(0.85 * (lap / (lap + 300.0)) - 0.25 * smi, 0.0, 1.0)
            pred_vlm = np.clip(0.80 * (lap / (lap + 120.0)) - 0.10 * smi, 0.0, 1.0)
            return pred_ocr, pred_vlm

        return self._reg_ocr.predict(x_mat), self._reg_vlm.predict(x_mat)

    def route_batch(
        self,
        features: List[DocumentRoutingFeatures],
        cost_tradeoff_lambda: float | None = None,
    ) -> List[RoutingDecision]:
        """Selects the optimal pipeline per document maximizing utility."""
        lam = (
            self.cost_tradeoff_lambda
            if cost_tradeoff_lambda is None
            else cost_tradeoff_lambda
        )
        pred_ocr, pred_vlm = self.predict_scores(features)
        decisions: List[RoutingDecision] = []

        for feat, s_ocr, s_vlm in zip(features, pred_ocr, pred_vlm):
            u_ocr = float(s_ocr) - lam * self.ocr_relative_cost
            u_vlm = float(s_vlm) - lam * self.vlm_relative_cost
            chosen = PIPELINE_VLM if u_vlm > u_ocr else PIPELINE_OCR
            decisions.append(
                RoutingDecision(
                    sample_id=feat.sample_id,
                    selected_pipeline=chosen,
                    pred_anls_ocr=round(float(s_ocr), 4),
                    pred_anls_vlm=round(float(s_vlm), 4),
                    confidence_margin=round(abs(u_vlm - u_ocr), 4),
                )
            )
        return decisions

    def evaluate_pareto_frontier(
        self,
        features: List[DocumentRoutingFeatures],
        true_ocr_anls: List[float],
        true_vlm_anls: List[float],
        lambdas: List[float] | None = None,
    ) -> Dict[str, object]:
        """Evaluates accuracy vs. relative compute cost across lambda values."""
        if lambdas is None:
            lambdas = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]

        y_ocr = np.asarray(true_ocr_anls, dtype=np.float32)
        y_vlm = np.asarray(true_vlm_anls, dtype=np.float32)
        oracle_anls = float(np.mean(np.maximum(y_ocr, y_vlm)))

        curve = []
        for lam in lambdas:
            decisions = self.route_batch(features, cost_tradeoff_lambda=lam)
            routed_scores = []
            routed_costs = []
            vlm_fraction = 0
            for dec, t_ocr, t_vlm in zip(decisions, y_ocr, y_vlm):
                if dec.selected_pipeline == PIPELINE_VLM:
                    routed_scores.append(float(t_vlm))
                    routed_costs.append(self.vlm_relative_cost)
                    vlm_fraction += 1
                else:
                    routed_scores.append(float(t_ocr))
                    routed_costs.append(self.ocr_relative_cost)
            curve.append(
                {
                    "lambda": lam,
                    "routed_anls": round(float(np.mean(routed_scores)), 4),
                    "relative_cost": round(float(np.mean(routed_costs)), 4),
                    "vlm_routing_rate": round(vlm_fraction / max(1, len(decisions)), 4),
                }
            )

        return {
            "evaluation_protocol": "in_sample",
            "always_ocr_anls": round(float(np.mean(y_ocr)), 4),
            "always_ocr_cost": self.ocr_relative_cost,
            "always_vlm_anls": round(float(np.mean(y_vlm)), 4),
            "always_vlm_cost": self.vlm_relative_cost,
            "oracle_router_upper_bound_anls": round(oracle_anls, 4),
            "pareto_frontier": curve,
        }

    def evaluate_pareto_frontier_cv(
        self,
        features: List[DocumentRoutingFeatures],
        true_ocr_anls: List[float],
        true_vlm_anls: List[float],
        lambdas: List[float] | None = None,
        n_splits: int = 5,
        seed: int = 42,
    ) -> Dict[str, object]:
        """Evaluates the router with out-of-fold predictions to prevent leakage.

        The regressors are trained on $k-1$ folds and scored on the held-out
        fold. Pareto decisions are then made from those out-of-fold predictions,
        so the reported accuracy is not inflated by fitting and evaluating on
        the same documents.
        """
        if lambdas is None:
            lambdas = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]

        n = len(features)
        if n != len(true_ocr_anls) or n != len(true_vlm_anls):
            raise ValueError("features and score lists must have equal length")
        if n < 2:
            return {}

        x_mat = np.asarray([f.to_vector() for f in features], dtype=np.float32)
        y_ocr = np.asarray(true_ocr_anls, dtype=np.float32)
        y_vlm = np.asarray(true_vlm_anls, dtype=np.float32)

        effective_splits = min(int(n_splits), n)
        if effective_splits < 2:
            result = self.evaluate_pareto_frontier(features, true_ocr_anls, true_vlm_anls, lambdas)
            result["evaluation_protocol"] = "in_sample_fallback_too_few_samples"
            return result

        kfold = KFold(n_splits=effective_splits, shuffle=True, random_state=seed)
        oof_ocr = np.zeros(n, dtype=np.float32)
        oof_vlm = np.zeros(n, dtype=np.float32)

        for train_idx, test_idx in kfold.split(x_mat):
            reg_ocr = RandomForestRegressor(
                n_estimators=64, max_depth=6, random_state=seed
            )
            reg_vlm = RandomForestRegressor(
                n_estimators=64, max_depth=6, random_state=seed
            )
            reg_ocr.fit(x_mat[train_idx], y_ocr[train_idx])
            reg_vlm.fit(x_mat[train_idx], y_vlm[train_idx])
            oof_ocr[test_idx] = reg_ocr.predict(x_mat[test_idx])
            oof_vlm[test_idx] = reg_vlm.predict(x_mat[test_idx])

        oracle_anls = float(np.mean(np.maximum(y_ocr, y_vlm)))
        curve = []
        for lam in lambdas:
            routed_scores = []
            routed_costs = []
            vlm_fraction = 0
            for index in range(n):
                u_ocr = float(oof_ocr[index]) - lam * self.ocr_relative_cost
                u_vlm = float(oof_vlm[index]) - lam * self.vlm_relative_cost
                if u_vlm > u_ocr:
                    routed_scores.append(float(y_vlm[index]))
                    routed_costs.append(self.vlm_relative_cost)
                    vlm_fraction += 1
                else:
                    routed_scores.append(float(y_ocr[index]))
                    routed_costs.append(self.ocr_relative_cost)
            curve.append(
                {
                    "lambda": lam,
                    "routed_anls": round(float(np.mean(routed_scores)), 4),
                    "relative_cost": round(float(np.mean(routed_costs)), 4),
                    "vlm_routing_rate": round(vlm_fraction / float(n), 4),
                }
            )

        return {
            "evaluation_protocol": f"{effective_splits}-fold_cross_validated",
            "n_splits": effective_splits,
            "oof_prediction_mae_ocr": round(float(np.mean(np.abs(oof_ocr - y_ocr))), 4),
            "oof_prediction_mae_vlm": round(float(np.mean(np.abs(oof_vlm - y_vlm))), 4),
            "always_ocr_anls": round(float(np.mean(y_ocr)), 4),
            "always_ocr_cost": self.ocr_relative_cost,
            "always_vlm_anls": round(float(np.mean(y_vlm)), 4),
            "always_vlm_cost": self.vlm_relative_cost,
            "oracle_router_upper_bound_anls": round(oracle_anls, 4),
            "pareto_frontier": curve,
        }
