import json

from langchain_core.tools import BaseTool, tool

import store
from config import get_settings
from football_data import (
    TeamNotFoundError,
    get_head_to_head,
    get_team_profile,
    list_teams,
    predict_match,
    resolve_team,
)
from polymarket import PolymarketClient, PolymarketError, blend_probabilities
from scoring import infer_xg_from_market, optimize_scoreline


def _canonical(team_name: str) -> str:
    try:
        return resolve_team(team_name)
    except TeamNotFoundError:
        return team_name


def _recent_context(team_name: str) -> list[dict]:
    """Recent in-tournament results for a team (most recent first), for prediction context."""
    results = store.get_team_results(_canonical(team_name))
    return [
        {
            "goals_for": r["goals_for"],
            "goals_against": r["goals_against"],
            "result": r["result"],
        }
        for r in results
    ]


@tool
def lookup_team(team_name: str) -> str:
    """Look up a national team's FIFA ranking, recent form, and goal averages.

    Use this before predicting a match to understand each side's strength.

    Args:
        team_name: The name of the team to lookup.
    Returns:
        A JSON string with the team's profile.
    """
    try:
        return json.dumps(get_team_profile(team_name), indent=2)
    except TeamNotFoundError as exc:
        return json.dumps({"error": str(exc), "known_teams": list_teams()})


@tool
def get_match_head_to_head(team_a: str, team_b: str) -> str:
    """Get head-to-head history between two national teams.

    team_a and team_b are the two sides (order does not matter).
    Args:
        team_a: The name of the first team.
        team_b: The name of the second team.
    Returns:
        A JSON string with the head-to-head history.
    """
    try:
        resolve_team(team_a)
        resolve_team(team_b)
        return json.dumps(get_head_to_head(team_a, team_b), indent=2)
    except TeamNotFoundError as exc:
        return json.dumps({"error": str(exc), "known_teams": list_teams()})


@tool
def get_polymarket_odds(home_team: str, away_team: str, match_date: str = "") -> str:
    """Get live crowd-implied odds for a World Cup match from Polymarket.

    home_team and away_team are the two sides (order does not matter for lookup).
    match_date is an optional ISO date (YYYY-MM-DD) to disambiguate fixtures.
    Returns implied win/draw/away probabilities, raw market prices, volume, and a URL.

    Args:
        home_team: The name of the home team.
        away_team: The name of the away team.
        match_date: The date of the match (optional).
    Returns:
        A JSON string with the odds.
    """
    settings = get_settings()
    if not settings.polymarket_enabled:
        return json.dumps({"error": "Polymarket enrichment is disabled in settings."})
    try:
        client = PolymarketClient(settings)
        odds = client.find_match(home_team, away_team, match_date or None)
        return json.dumps(odds.to_dict(), indent=2)
    except PolymarketError as exc:
        return json.dumps({"error": str(exc)})


@tool
def list_world_cup_fixtures() -> str:
    """List the World Cup match SCHEDULE only (teams, date, slug). Does NOT make predictions.

    Only use this to show the calendar. To predict/pick results, use predict_all_fixtures
    or predict_match_result instead.

    Returns:
        A JSON string with the fixtures.
    """
    settings = get_settings()
    if not settings.polymarket_enabled:
        return json.dumps({"error": "Polymarket enrichment is disabled in settings."})
    try:
        client = PolymarketClient(settings)
        return json.dumps({"fixtures": client.list_fixtures()}, indent=2)
    except PolymarketError as exc:
        return json.dumps({"error": str(exc)})


@tool
def predict_match_result(
    home_team: str, away_team: str, match_date: str = "", stage: str = "group"
) -> str:
    """Predict the points-maximizing scoreline for a football match (read-only).

    home_team is the side playing at home (first name in 'X vs Y' queries).
    away_team is the visiting side. match_date is an optional ISO date (YYYY-MM-DD).
    stage is 'group' (primera ronda) or 'knockout' (fases eliminatorias) for the points scale.

    Picks the scoreline that maximizes expected points under the contest rules, using a
    statistical model (FIFA rank, form, H2H) blended with live Polymarket odds when available.
    The recommended_bet is the score to submit.

    Args:
        home_team: The name of the home team.
        away_team: The name of the away team.
        match_date: The date of the match (optional).
        stage: The stage of the match (optional).
    Returns:
        A JSON string with the prediction.
    """
    settings = get_settings()
    stage = stage if stage in ("group", "knockout") else "group"
    home_recent = _recent_context(home_team)
    away_recent = _recent_context(away_team)
    try:
        model = predict_match(home_team, away_team, home_recent, away_recent)
    except TeamNotFoundError as exc:
        return json.dumps({"error": str(exc), "known_teams": list_teams()})

    xg = model["expected_xg"]
    final_probs = model["probabilities"]
    result: dict = {"stage": stage, "model_prediction": model}
    slug = None
    event_date = match_date or None

    if settings.polymarket_enabled:
        try:
            client = PolymarketClient(settings)
            odds = client.find_match(home_team, away_team, match_date or None)
            market = odds.to_dict()
            slug = odds.slug
            event_date = odds.event_date or event_date
            final_probs = blend_probabilities(
                model["probabilities"],
                market["implied_probabilities"],
                settings.polymarket_market_weight,
            )
            result["polymarket"] = market
            result["blended_probabilities"] = final_probs
        except PolymarketError as exc:
            result["polymarket_error"] = str(exc)

    optimal = optimize_scoreline(xg["home"], xg["away"], final_probs)
    winner = (
        model["home_team"]
        if optimal["result"] == "home_win"
        else model["away_team"]
        if optimal["result"] == "away_win"
        else "Draw"
    )
    result["recommended_bet"] = {
        "home_goals": optimal["home_goals"],
        "away_goals": optimal["away_goals"],
        "score": optimal["predicted_score"],
        "winner": winner,
        "stage": stage,
        "expected_points": optimal["expected_points"][stage],
        "max_points": optimal["max_points"][stage],
        "result_probabilities": optimal["result_probabilities"],
        "source": "model+market" if "polymarket" in result else "model",
    }
    if slug:
        result["slug"] = slug
    if event_date:
        result["event_date"] = event_date
    if home_recent or away_recent:
        result["used_recent_results"] = {
            model["home_team"]: home_recent,
            model["away_team"]: away_recent,
        }

    result["note"] = (
        "recommended_bet maximizes expected points (read-only). Use save_match_prediction "
        "to persist picks. Blends the model with live market odds when available."
    )
    return json.dumps(result, indent=2)


def _recommended_bet(
    home_team: str,
    away_team: str,
    market_probs: dict[str, float] | None,
    settings,
    stage: str,
) -> dict:
    """Compute the points-maximizing bet from the model and/or market odds."""
    probs = market_probs
    source = "market"
    home_recent = _recent_context(home_team)
    away_recent = _recent_context(away_team)
    try:
        model = predict_match(home_team, away_team, home_recent, away_recent)
        xg = model["expected_xg"]
        if market_probs:
            probs = blend_probabilities(
                model["probabilities"], market_probs, settings.polymarket_market_weight
            )
            source = "model+market"
        else:
            probs = model["probabilities"]
            source = "model"
    except TeamNotFoundError:
        if not market_probs:
            raise
        xg = infer_xg_from_market(market_probs)

    optimal = optimize_scoreline(xg["home"], xg["away"], probs)
    winner = (
        home_team
        if optimal["result"] == "home_win"
        else away_team
        if optimal["result"] == "away_win"
        else "Draw"
    )
    return {
        "match": f"{home_team} vs {away_team}",
        "home_goals": optimal["home_goals"],
        "away_goals": optimal["away_goals"],
        "score": optimal["predicted_score"],
        "winner": winner,
        "expected_points": optimal["expected_points"][stage],
        "max_points": optimal["max_points"][stage],
        "source": source,
        "used_recent": bool(home_recent or away_recent),
    }


@tool
def predict_all_fixtures(stage: str = "group", match_date: str = "", limit: int = 30) -> str:
    """Make PICKS/PREDICTIONS for many World Cup matches at once (e.g. 'all matches', 'today's games').

    Read-only: returns the points-maximizing score per fixture. Use save_match_prediction to persist.
    stage is 'group' or 'knockout'. match_date filters to one ISO date (YYYY-MM-DD) — pass it
    whenever the user mentions a specific day. limit caps results (default 30). Uses the
    90-minute result only (draws allowed in knockout).

    Args:
        stage: The stage of the match (optional).
        match_date: The date of the match (optional).
        limit: The limit of the matches (optional).
    Returns:
        A JSON string with the predictions.
    """
    settings = get_settings()
    stage = stage if stage in ("group", "knockout") else "group"
    if not settings.polymarket_enabled:
        return json.dumps({"error": "Polymarket enrichment is disabled in settings."})
    try:
        client = PolymarketClient(settings)
        odds_list = client.all_match_odds(match_date=match_date or None)
    except PolymarketError as exc:
        return json.dumps({"error": str(exc)})

    picks = []
    for odds in odds_list[: max(1, limit)]:
        market = odds.to_dict()["implied_probabilities"]
        try:
            bet = _recommended_bet(odds.home_team, odds.away_team, market, settings, stage)
        except TeamNotFoundError:
            continue
        picks.append(
            {
                "match": odds.title,
                "home_team": _canonical(odds.home_team),
                "away_team": _canonical(odds.away_team),
                "home_goals": bet["home_goals"],
                "away_goals": bet["away_goals"],
                "event_date": odds.event_date,
                "slug": odds.slug,
                "score": bet["score"],
                "winner": bet["winner"],
                "expected_points": bet["expected_points"],
                "source": bet["source"],
            }
        )

    total_ep = round(sum(p["expected_points"] for p in picks), 2)
    return json.dumps(
        {
            "stage": stage,
            "fixtures_returned": len(picks),
            "projected_total_points": total_ep,
            "picks": picks,
        },
        indent=2,
    )


@tool
def save_match_prediction(
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
    match_date: str = "",
    stage: str = "group",
    slug: str = "",
    expected_points: float = 0.0,
    source: str = "",
) -> str:
    """Save a prediction pick for the polla (does not compute scores).

    Call after predict_match_result or predict_all_fixtures when the user wants picks stored.
    Pass home_goals, away_goals, and metadata from the prediction output (slug, event_date,
    expected_points, source help match and score the entry later).

    Args:
        home_team: The name of the home team.
        away_team: The name of the away team.
        home_goals: Predicted home goals.
        away_goals: Predicted away goals.
        match_date: ISO date YYYY-MM-DD (optional).
        stage: 'group' or 'knockout'.
        slug: Polymarket fixture slug from prediction output (optional).
        expected_points: Expected points from prediction output (optional).
        source: 'model', 'market', or 'model+market' from prediction output (optional).
    Returns:
        A JSON string with the saved record.
    """
    stage = stage if stage in ("group", "knockout") else "group"
    record = store.save_prediction(
        _canonical(home_team),
        _canonical(away_team),
        int(home_goals),
        int(away_goals),
        stage=stage,
        event_date=match_date or None,
        slug=slug or None,
        expected_points=expected_points or None,
        source=source or None,
    )
    return json.dumps(record, indent=2, ensure_ascii=False)


@tool
def record_match_result(
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
    match_date: str = "",
    stage: str = "group",
) -> str:
    """Save the REAL 90-minute result of a played World Cup match.

    Stores the actual score so it becomes context for future predictions and scores the
    points earned vs the saved prediction. home_team is the side that played at home.

    Args:
        home_team: The name of the home team.
        away_team: The name of the away team.
        home_goals: The number of goals scored by the home team.
        away_goals: The number of goals scored by the away team.
        match_date: The date of the match (optional).
        stage: The stage of the match (optional).
    Returns:
        A JSON string with the result.
    """
    stage = stage if stage in ("group", "knockout") else "group"
    settings = get_settings()
    slug = None
    event_date = match_date or None
    if settings.polymarket_enabled:
        try:
            odds = PolymarketClient(settings).find_match(
                home_team, away_team, match_date or None
            )
            slug = odds.slug
            event_date = odds.event_date or event_date
        except PolymarketError:
            pass
    record = store.record_result(
        _canonical(home_team),
        _canonical(away_team),
        int(home_goals),
        int(away_goals),
        stage=stage,
        event_date=event_date,
        slug=slug,
    )
    return json.dumps(record, indent=2, ensure_ascii=False)


@tool
def get_team_recent_results(team_name: str) -> str:
    """Show a team's actual results so far in this World Cup (most recent first).

    Use this for context before predicting a team's next match.

    Args:
        team_name: The name of the team to get the recent results.
    Returns:
        A JSON string with the recent results.
    """
    results = store.get_team_results(_canonical(team_name))
    return json.dumps(
        {"team": _canonical(team_name), "matches_played": len(results), "results": results},
        indent=2,
        ensure_ascii=False,
    )


@tool
def get_points_summary() -> str:
    """Show total points earned so far and prediction accuracy across recorded matches."""
    return json.dumps(store.summary(), indent=2)


@tool
def list_supported_teams() -> str:
    """List national teams available in the prediction database."""
    return json.dumps({"teams": list_teams()}, indent=2)


TOOLS: list[BaseTool] = [
    lookup_team,
    get_match_head_to_head,
    get_polymarket_odds,
    list_world_cup_fixtures,
    predict_match_result,
    predict_all_fixtures,
    save_match_prediction,
    record_match_result,
    get_team_recent_results,
    get_points_summary,
    list_supported_teams,
]
