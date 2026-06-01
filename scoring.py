"""Expected-points optimization for the prediction scoring rules.

Scoring per match:
    Group stage (primera ronda):  winner/draw 5, home goals 2, away goals 2, diff 1  (max 10)
    Knockout (fases eliminatorias): winner/draw 10, home goals 4, away goals 4, diff 2 (max 20)

Knockout weights are exactly 2x the group weights, so the expected-points-maximizing
scoreline is identical for both stages; only the magnitude of points differs. We therefore
optimize once and report expected points for each stage.

Strategy: model goals as independent Poisson variables from each side's expected goals (xG),
then reweight the joint scoreline distribution so its 1X2 marginal matches our best estimate
(model blended with market). We then pick the scoreline that maximizes expected points, which
prioritizes the result (worth half the points) and then the most likely exact goals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

Stage = Literal["group", "knockout"]


@dataclass(frozen=True)
class PointRules:
    winner: int
    home_goals: int
    away_goals: int
    goal_difference: int

    @property
    def max_points(self) -> int:
        return self.winner + self.home_goals + self.away_goals + self.goal_difference


STAGE_RULES: dict[Stage, PointRules] = {
    "group": PointRules(winner=5, home_goals=2, away_goals=2, goal_difference=1),
    "knockout": PointRules(winner=10, home_goals=4, away_goals=4, goal_difference=2),
}

MAX_GOALS = 7


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * lam**k / math.factorial(k)


def _result_of(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    if away_goals > home_goals:
        return "away_win"
    return "draw"


def _goal_pmf(xg: float, max_goals: int) -> list[float]:
    pmf = [_poisson_pmf(k, xg) for k in range(max_goals + 1)]
    total = sum(pmf) or 1.0
    return [p / total for p in pmf]


def _result_probs_from_xg(
    ph: list[float], pa: list[float], max_goals: int
) -> tuple[float, float, float]:
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
    return hw, dw, aw


def infer_xg_from_market(
    probs_1x2: dict[str, float], max_goals: int = MAX_GOALS
) -> dict[str, float]:
    """Infer home/away expected goals that best reproduce market 1X2 probabilities.

    Used when a team is missing from the local stats DB, so we can still optimize a
    scoreline from market odds alone. Searches a grid of Poisson means.
    """
    total = sum(max(probs_1x2.get(k, 0.0), 0.0) for k in ("home_win", "draw", "away_win"))
    total = total or 1.0
    target = (
        probs_1x2.get("home_win", 0.0) / total,
        probs_1x2.get("draw", 0.0) / total,
        probs_1x2.get("away_win", 0.0) / total,
    )

    grid = [round(0.2 * i, 2) for i in range(1, 19)]  # 0.2 .. 3.6
    pmfs = {lam: _goal_pmf(lam, max_goals) for lam in grid}

    best = (1.2, 1.0)
    best_err = float("inf")
    for lh in grid:
        ph = pmfs[lh]
        for la in grid:
            hw, dw, aw = _result_probs_from_xg(ph, pmfs[la], max_goals)
            err = (hw - target[0]) ** 2 + (dw - target[1]) ** 2 + (aw - target[2]) ** 2
            if err < best_err:
                best_err = err
                best = (lh, la)
    return {"home": best[0], "away": best[1]}


def scoreline_distribution(
    home_xg: float,
    away_xg: float,
    probs_1x2: dict[str, float] | None = None,
    max_goals: int = MAX_GOALS,
) -> dict[str, Any]:
    """Joint scoreline distribution, optionally reweighted to match a target 1X2."""
    ph = _goal_pmf(home_xg, max_goals)
    pa = _goal_pmf(away_xg, max_goals)

    joint = [[ph[h] * pa[a] for a in range(max_goals + 1)] for h in range(max_goals + 1)]

    raw_result = {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            raw_result[_result_of(h, a)] += joint[h][a]

    if probs_1x2:
        target = {k: max(probs_1x2.get(k, 0.0), 0.0) for k in raw_result}
        t_total = sum(target.values()) or 1.0
        target = {k: v / t_total for k, v in target.items()}
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
    away_marginal = [
        sum(joint[h][a] for h in range(max_goals + 1)) for a in range(max_goals + 1)
    ]
    diff_marginal: dict[int, float] = {}
    result_marginal = {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}
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


def optimize_scoreline(
    home_xg: float,
    away_xg: float,
    probs_1x2: dict[str, float] | None = None,
    max_goals: int = MAX_GOALS,
) -> dict[str, Any]:
    """Find the scoreline that maximizes expected points (identical for both stages)."""
    dist = scoreline_distribution(home_xg, away_xg, probs_1x2, max_goals)

    best: tuple[int, int] | None = None
    best_ev = -1.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            ev = expected_points(h, a, dist, "group")
            if ev > best_ev:
                best_ev = ev
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
            "knockout": round(
                expected_points(home_goals, away_goals, dist, "knockout"), 3
            ),
        },
        "max_points": {
            "group": STAGE_RULES["group"].max_points,
            "knockout": STAGE_RULES["knockout"].max_points,
        },
        "result_probabilities": {
            k: round(v, 4) for k, v in dist["result_marginal"].items()
        },
    }
