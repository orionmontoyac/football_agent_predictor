import json

from langchain_core.tools import BaseTool, tool

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


@tool
def lookup_team(team_name: str) -> str:
    """Look up a national team's FIFA ranking, recent form, and goal averages.

    Use this before predicting a match to understand each side's strength.
    """
    try:
        return json.dumps(get_team_profile(team_name), indent=2)
    except TeamNotFoundError as exc:
        return json.dumps({"error": str(exc), "known_teams": list_teams()})


@tool
def get_match_head_to_head(team_a: str, team_b: str) -> str:
    """Get head-to-head history between two national teams.

    team_a and team_b are the two sides (order does not matter).
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
    """List upcoming World Cup fixtures tracked on Polymarket (teams, date, slug)."""
    settings = get_settings()
    if not settings.polymarket_enabled:
        return json.dumps({"error": "Polymarket enrichment is disabled in settings."})
    try:
        client = PolymarketClient(settings)
        return json.dumps({"fixtures": client.list_fixtures()}, indent=2)
    except PolymarketError as exc:
        return json.dumps({"error": str(exc)})


@tool
def predict_match_result(home_team: str, away_team: str, match_date: str = "") -> str:
    """Predict the likely result of a football match.

    home_team is the side playing at home (first name in 'X vs Y' queries).
    away_team is the visiting side. match_date is an optional ISO date (YYYY-MM-DD).

    Combines a statistical model (FIFA rank, recent form, head-to-head) with live
    Polymarket odds when available, and returns a blended forecast plus both sources.
    """
    settings = get_settings()
    try:
        model = predict_match(home_team, away_team)
    except TeamNotFoundError as exc:
        return json.dumps({"error": str(exc), "known_teams": list_teams()})

    result: dict = {"model_prediction": model}

    if settings.polymarket_enabled:
        try:
            client = PolymarketClient(settings)
            odds = client.find_match(home_team, away_team, match_date or None)
            market = odds.to_dict()
            result["polymarket"] = market
            result["blended_probabilities"] = blend_probabilities(
                model["probabilities"],
                market["implied_probabilities"],
                settings.polymarket_market_weight,
            )
            result["note"] = (
                "blended_probabilities mixes the statistical model with live market "
                f"odds (market weight {settings.polymarket_market_weight})."
            )
        except PolymarketError as exc:
            result["polymarket_error"] = str(exc)
            result["note"] = "Live market odds unavailable; using model prediction only."

    return json.dumps(result, indent=2)


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
    list_supported_teams,
]
