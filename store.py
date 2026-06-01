"""Persistent JSON store for predictions and real World Cup results.

The store keeps one record per match (keyed by a stable slug/team-date key) holding both the
agent's prediction and, once played, the real 90-minute result with the points earned. Past
results are used as context to predict later matches (e.g. how Mexico played match 1).

File shape:
    {
      "matches": {
        "<key>": {
          "key", "slug", "stage", "event_date", "home_team", "away_team",
          "prediction": {"home_goals", "away_goals", "winner", "expected_points", "source", "updated_at"},
          "result":     {"home_goals", "away_goals", "winner", "recorded_at"},
          "points_earned": {"points", "max_points", "breakdown", "exact_score"}
        }
      }
    }
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import get_settings
from scoring import Stage, actual_points


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _store_path() -> Path:
    return Path(get_settings().predictions_file)


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")


def match_key(
    home_team: str, away_team: str, event_date: str | None, slug: str | None = None
) -> str:
    """Stable key for a match; prefers the Polymarket slug when available."""
    if slug:
        return slug
    date = event_date or "tbd"
    return f"{_slugify(home_team)}-vs-{_slugify(away_team)}-{date}"


def load() -> dict[str, Any]:
    path = _store_path()
    if not path.exists():
        return {"matches": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"matches": {}}
    data.setdefault("matches", {})
    return data


def save(data: dict[str, Any]) -> None:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _winner(home_team: str, away_team: str, home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return home_team
    if away_goals > home_goals:
        return away_team
    return "Draw"


def _recompute_points(record: dict[str, Any]) -> None:
    pred = record.get("prediction")
    res = record.get("result")
    if not pred or not res:
        record["points_earned"] = None
        return
    record["points_earned"] = actual_points(
        pred["home_goals"],
        pred["away_goals"],
        res["home_goals"],
        res["away_goals"],
        record.get("stage", "group"),
    )


def save_prediction(
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
    *,
    stage: Stage = "group",
    event_date: str | None = None,
    slug: str | None = None,
    expected_points: float | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Insert or update the prediction for a match. Keeps any existing real result."""
    data = load()
    key = match_key(home_team, away_team, event_date, slug)
    record = data["matches"].get(key, {})
    record.update(
        {
            "key": key,
            "slug": slug or record.get("slug"),
            "stage": stage,
            "event_date": event_date or record.get("event_date"),
            "home_team": home_team,
            "away_team": away_team,
        }
    )
    record["prediction"] = {
        "home_goals": home_goals,
        "away_goals": away_goals,
        "winner": _winner(home_team, away_team, home_goals, away_goals),
        "expected_points": expected_points,
        "source": source,
        "updated_at": _now(),
    }
    _recompute_points(record)
    data["matches"][key] = record
    save(data)
    return record


def record_result(
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
    *,
    stage: Stage | None = None,
    event_date: str | None = None,
    slug: str | None = None,
) -> dict[str, Any]:
    """Record the real 90-minute result; computes points if a prediction exists."""
    data = load()
    key = match_key(home_team, away_team, event_date, slug)
    record = data["matches"].get(key, {})
    record.update(
        {
            "key": key,
            "slug": slug or record.get("slug"),
            "stage": stage or record.get("stage", "group"),
            "event_date": event_date or record.get("event_date"),
            "home_team": record.get("home_team", home_team),
            "away_team": record.get("away_team", away_team),
        }
    )
    record["result"] = {
        "home_goals": home_goals,
        "away_goals": away_goals,
        "winner": _winner(
            record["home_team"], record["away_team"], home_goals, away_goals
        ),
        "recorded_at": _now(),
    }
    _recompute_points(record)
    data["matches"][key] = record
    save(data)
    return record


def get_match(
    home_team: str, away_team: str, event_date: str | None = None, slug: str | None = None
) -> dict[str, Any] | None:
    data = load()
    key = match_key(home_team, away_team, event_date, slug)
    if key in data["matches"]:
        return data["matches"][key]
    # Fall back to order-insensitive team match.
    wanted = {home_team.strip().lower(), away_team.strip().lower()}
    for record in data["matches"].values():
        teams = {record.get("home_team", "").lower(), record.get("away_team", "").lower()}
        if teams == wanted and (not event_date or record.get("event_date") == event_date):
            return record
    return None


def get_team_results(team: str) -> list[dict[str, Any]]:
    """Played matches (with real results) for a team, most recent first.

    Each entry is from the team's perspective: opponent, venue, goals_for/against, result (W/D/L).
    """
    data = load()
    key = team.strip().lower()
    out: list[dict[str, Any]] = []
    for record in data["matches"].values():
        res = record.get("result")
        if not res:
            continue
        home = record.get("home_team", "")
        away = record.get("away_team", "")
        if key == home.lower():
            gf, ga, opponent, venue = (
                res["home_goals"],
                res["away_goals"],
                away,
                "home",
            )
        elif key == away.lower():
            gf, ga, opponent, venue = (
                res["away_goals"],
                res["home_goals"],
                home,
                "away",
            )
        else:
            continue
        result = "W" if gf > ga else "L" if gf < ga else "D"
        out.append(
            {
                "opponent": opponent,
                "venue": venue,
                "goals_for": gf,
                "goals_against": ga,
                "result": result,
                "score": f"{gf}-{ga}",
                "event_date": record.get("event_date"),
                "stage": record.get("stage", "group"),
            }
        )
    out.sort(key=lambda r: (r.get("event_date") or ""), reverse=True)
    return out


def summary() -> dict[str, Any]:
    """Totals across all recorded matches: points earned, exacts, accuracy."""
    data = load()
    total_points = 0
    graded = 0
    exacts = 0
    winners_correct = 0
    for record in data["matches"].values():
        pts = record.get("points_earned")
        if not pts:
            continue
        graded += 1
        total_points += pts["points"]
        if pts.get("exact_score"):
            exacts += 1
        if pts["breakdown"].get("winner", 0) > 0:
            winners_correct += 1
    return {
        "graded_matches": graded,
        "total_points": total_points,
        "exact_scores": exacts,
        "winners_correct": winners_correct,
        "predictions_stored": len(data["matches"]),
    }
