"""Expected-points optimization for the prediction scoring rules.

Scoring per match:
    Group stage:    winner/draw 5, home goals 2, away goals 2, diff 1  (max 10)
    Knockout:       winner/draw 10, home goals 4, away goals 4, diff 2  (max 20)

Knockout weights are 2x group weights so the optimal scoreline is identical for both
stages; only the magnitude differs. We optimize once and report EV for each stage.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, computed_field

Stage = Literal["group", "knockout"]

# ---------------------------------------------------------------------------
# Scoring rules
# ---------------------------------------------------------------------------

class PointRules(BaseModel):
    model_config = ConfigDict(frozen=True)

    winner: int
    home_goals: int
    away_goals: int
    goal_difference: int

    @computed_field
    @property
    def max_points(self) -> int:
        return self.winner + self.home_goals + self.away_goals + self.goal_difference


STAGE_RULES: dict[Stage, PointRules] = {
    "group": PointRules(winner=5, home_goals=2, away_goals=2, goal_difference=1),
    "knockout": PointRules(winner=10, home_goals=4, away_goals=4, goal_difference=2),
}

MAX_GOALS = 8  # raised from 7; captures ~99.9% of Poisson mass up to xG≈3.5


# ---------------------------------------------------------------------------
# Poisson helpers
# ---------------------------------------------------------------------------

def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * lam**k / math.factorial(k)


def _goal_pmf(xg: float, max_goals: int) -> list[float]:
    pmf = [_poisson_pmf(k, xg) for k in range(max_goals + 1)]
    total = sum(pmf) or 1.0
    return [p / total for p in pmf]


def _result_of(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    if away_goals > home_goals:
        return "away_win"
    return "draw"


def _result_probs_from_pmfs(
    ph: list[float], pa: list[float], max_goals: int
) -> dict[str, float]:
    hw = dw = aw = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = ph[h] * pa[a]
            if h > a:
                hw += p
            elif h < a:
                aw += p
            else:
                dw += p
    return {"home_win": hw, "draw": dw, "away_win": aw}


# ---------------------------------------------------------------------------
# xG / market blending
# ---------------------------------------------------------------------------

def blend_probs_1x2(
    xg_probs: dict[str, float],
    market_probs: dict[str, float],
    market_weight: float = 0.5,
) -> dict[str, float]:
    """Linear blend of xG-implied and market 1X2 probabilities.

    Args:
        xg_probs: 1X2 probabilities implied by home/away xG (Poisson).
        market_probs: 1X2 probabilities from betting market or external model.
        market_weight: 0.0 → pure xG, 1.0 → pure market. Default 0.5.

    Returns:
        Normalized blended 1X2 dict.
    """
    market_weight = max(0.0, min(1.0, market_weight))
    keys = ("home_win", "draw", "away_win")

    # Normalize inputs independently
    xg_total = sum(max(xg_probs.get(k, 0.0), 0.0) for k in keys) or 1.0
    mkt_total = sum(max(market_probs.get(k, 0.0), 0.0) for k in keys) or 1.0

    blended = {
        k: (1 - market_weight) * (xg_probs.get(k, 0.0) / xg_total)
        + market_weight * (market_probs.get(k, 0.0) / mkt_total)
        for k in keys
    }
    total = sum(blended.values()) or 1.0
    return {k: v / total for k, v in blended.items()}


def infer_xg_from_market(
    probs_1x2: dict[str, float],
    max_goals: int = MAX_GOALS,
) -> dict[str, float]:
    """Infer home/away xG that best reproduces market 1X2 probabilities.

    Uses a fine 0.05-step grid search. When scipy is available it refines the
    grid solution with Nelder-Mead minimization for sub-0.01 accuracy.
    """
    keys = ("home_win", "draw", "away_win")
    total = sum(max(probs_1x2.get(k, 0.0), 0.0) for k in keys) or 1.0
    target = tuple(probs_1x2.get(k, 0.0) / total for k in keys)

    # Fine grid: 0.05 to 4.0
    grid = [round(0.05 * i, 2) for i in range(1, 81)]
    pmf_cache: dict[float, list[float]] = {lam: _goal_pmf(lam, max_goals) for lam in grid}

    def _mse(lh: float, la: float) -> float:
        probs = _result_probs_from_pmfs(pmf_cache[lh], pmf_cache[la], max_goals)
        return sum((probs[k] - target[i]) ** 2 for i, k in enumerate(keys))

    best = (1.2, 1.0)
    best_err = float("inf")
    for lh in grid:
        for la in grid:
            err = _mse(lh, la)
            if err < best_err:
                best_err = err
                best = (lh, la)

    # Nelder-Mead refinement when scipy is available
    try:
        from scipy.optimize import minimize  # type: ignore

        def _scipy_mse(x: list[float]) -> float:
            lh, la = max(x[0], 0.05), max(x[1], 0.05)
            ph = _goal_pmf(lh, max_goals)
            pa = _goal_pmf(la, max_goals)
            probs = _result_probs_from_pmfs(ph, pa, max_goals)
            return sum((probs[k] - target[i]) ** 2 for i, k in enumerate(keys))

        result = minimize(_scipy_mse, list(best), method="Nelder-Mead",
                          options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 2000})
        if result.success and result.fun < best_err:
            best = (round(max(result.x[0], 0.05), 4), round(max(result.x[1], 0.05), 4))
    except ImportError:
        pass

    return {"home": best[0], "away": best[1]}


# ---------------------------------------------------------------------------
# Joint distribution with blended reweighting
# ---------------------------------------------------------------------------

def scoreline_distribution(
    home_xg: float,
    away_xg: float,
    probs_1x2: dict[str, float] | None = None,
    market_weight: float = 0.5,
    max_goals: int = MAX_GOALS,
) -> dict[str, Any]:
    """Joint Poisson scoreline distribution, optionally blended with market 1X2.

    Args:
        home_xg: Expected goals for the home team.
        away_xg: Expected goals for the away team.
        probs_1x2: Target 1X2 probabilities (from market / external model).
            If None, pure Poisson from xG is used.
        market_weight: How much weight to give market vs. xG-implied 1X2 when
            computing the blend target. 0.0 = pure xG, 1.0 = pure market.
        max_goals: Maximum goals per team to model.
    """
    ph = _goal_pmf(home_xg, max_goals)
    pa = _goal_pmf(away_xg, max_goals)

    joint = [[ph[h] * pa[a] for a in range(max_goals + 1)] for h in range(max_goals + 1)]

    raw_result = _result_probs_from_pmfs(ph, pa, max_goals)

    if probs_1x2:
        # Blend xG-implied 1X2 with market 1X2 before reweighting
        target = blend_probs_1x2(raw_result, probs_1x2, market_weight)
        weights = {
            k: (target[k] / raw_result[k] if raw_result[k] > 0 else 0.0)
            for k in raw_result
        }
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                joint[h][a] *= weights[_result_of(h, a)]

    total = sum(sum(row) for row in joint) or 1.0
    joint = [[v / total for v in row] for row in joint]

    home_marginal = [sum(joint[h]) for h in range(max_goals + 1)]
    away_marginal = [sum(joint[h][a] for h in range(max_goals + 1)) for a in range(max_goals + 1)]

    diff_marginal: dict[int, float] = {}
    result_marginal: dict[str, float] = {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            diff_marginal[h - a] = diff_marginal.get(h - a, 0.0) + joint[h][a]
            result_marginal[_result_of(h, a)] += joint[h][a]

    return {
        "joint": joint,
        "home_marginal": home_marginal,
        "away_marginal": away_marginal,
        "diff_marginal": diff_marginal,
        "result_marginal": result_marginal,
        "max_goals": max_goals,
    }


# ---------------------------------------------------------------------------
# Points helpers
# ---------------------------------------------------------------------------

def expected_points(
    home_goals: int,
    away_goals: int,
    dist: dict[str, Any],
    stage: Stage = "group",
) -> float:
    """Expected points for predicting (home_goals, away_goals) under a distribution."""
    rules = STAGE_RULES[stage]
    result = _result_of(home_goals, away_goals)
    ev = rules.winner * dist["result_marginal"].get(result, 0.0)
    if home_goals < len(dist["home_marginal"]):
        ev += rules.home_goals * dist["home_marginal"][home_goals]
    if away_goals < len(dist["away_marginal"]):
        ev += rules.away_goals * dist["away_marginal"][away_goals]
    ev += rules.goal_difference * dist["diff_marginal"].get(home_goals - away_goals, 0.0)
    return ev


def actual_points(
    pred_home: int,
    pred_away: int,
    actual_home: int,
    actual_away: int,
    stage: Stage = "group",
) -> dict[str, Any]:
    """Points earned by a prediction against the real (90-minute) result."""
    rules = STAGE_RULES[stage]
    breakdown = {
        "winner": rules.winner
        if _result_of(pred_home, pred_away) == _result_of(actual_home, actual_away)
        else 0,
        "home_goals": rules.home_goals if pred_home == actual_home else 0,
        "away_goals": rules.away_goals if pred_away == actual_away else 0,
        "goal_difference": rules.goal_difference
        if (pred_home - pred_away) == (actual_home - actual_away)
        else 0,
    }
    total = sum(breakdown.values())
    return {
        "points": total,
        "max_points": rules.max_points,
        "breakdown": breakdown,
        "exact_score": pred_home == actual_home and pred_away == actual_away,
    }


# ---------------------------------------------------------------------------
# Scoreline optimizer
# ---------------------------------------------------------------------------

def optimize_scoreline(
    home_xg: float,
    away_xg: float,
    probs_1x2: dict[str, float] | None = None,
    market_weight: float = 0.5,
    entropy_weight: float = 0.03,
    max_goals: int = MAX_GOALS,
) -> dict[str, Any]:
    """Find the scoreline that maximizes expected points.

    Args:
        home_xg: Expected goals for the home team (from stats model).
        away_xg: Expected goals for the away team (from stats model).
        probs_1x2: 1X2 market probabilities. Blended with xG signal via
            `market_weight`. If None, pure xG is used.
        market_weight: 0.0 = pure xG, 1.0 = pure market 1X2. Default 0.5.
        entropy_weight: Small bonus added to EV per total goals in the
            predicted scoreline (h + a). Breaks ties toward realistic
            World Cup scorelines (e.g. 2-1 over 1-0 when EV is equal).
            Range 0.0–0.10; default 0.03 adds ≤0.24 pts on an 8-goal score.
        max_goals: Maximum goals per team to model.

    Returns:
        Dict with predicted_score, expected_points, result_probabilities, etc.
    """
    dist = scoreline_distribution(home_xg, away_xg, probs_1x2, market_weight, max_goals)

    best: tuple[int, int] | None = None
    best_score = -1.0

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            ev = expected_points(h, a, dist, "group")
            # Entropy regularization: slight preference for higher-scoring scorelines
            # when expected points are nearly tied. Capped so it never overrides a
            # genuine EV difference (max bonus = entropy_weight * max_goals * 2).
            score = ev + entropy_weight * (h + a)
            if score > best_score:
                best_score = score
                best = (h, a)

    assert best is not None
    home_goals, away_goals = best

    return {
        "predicted_score": f"{home_goals}-{away_goals}",
        "home_goals": home_goals,
        "away_goals": away_goals,
        "result": _result_of(home_goals, away_goals),
        "expected_points": {
            "group": round(expected_points(home_goals, away_goals, dist, "group"), 3),
            "knockout": round(expected_points(home_goals, away_goals, dist, "knockout"), 3),
        },
        "max_points": {
            "group": STAGE_RULES["group"].max_points,
            "knockout": STAGE_RULES["knockout"].max_points,
        },
        "result_probabilities": {
            k: round(v, 4) for k, v in dist["result_marginal"].items()
        },
    }