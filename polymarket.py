"""Polymarket Gamma API client for live World Cup match odds.

The public Gamma API (https://gamma-api.polymarket.com) requires no auth. World Cup
fixtures live under a single series; each event exposes three Yes/No markets
(home win, draw, away win) whose prices are crowd-implied probabilities.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

from config import Settings, get_settings

POLYMARKET_WORLD_CUP_URL = "https://polymarket.com/sports/world-cup"


class PolymarketError(RuntimeError):
    """Raised when Polymarket data cannot be fetched or parsed."""


@dataclass(frozen=True)
class MatchOdds:
    slug: str
    title: str
    home_team: str
    away_team: str
    event_date: str | None
    kickoff_utc: str | None
    # Normalized implied probabilities (sum to 1.0), overround removed.
    home_win: float
    draw: float
    away_win: float
    # Raw last-trade prices straight from the markets (still include overround).
    raw_prices: dict[str, float]
    volume_usd: float | None
    url: str
    fetched_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def implied_winner(self) -> str:
        best = max(
            (("home", self.home_win), ("draw", self.draw), ("away", self.away_win)),
            key=lambda kv: kv[1],
        )[0]
        if best == "home":
            return self.home_team
        if best == "away":
            return self.away_team
        return "Draw"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": "polymarket",
            "match": self.title,
            "slug": self.slug,
            "event_date": self.event_date,
            "kickoff_utc": self.kickoff_utc,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "implied_probabilities": {
                "home_win": round(self.home_win, 4),
                "draw": round(self.draw, 4),
                "away_win": round(self.away_win, 4),
            },
            "implied_winner": self.implied_winner,
            "raw_market_prices": self.raw_prices,
            "volume_usd": self.volume_usd,
            "url": self.url,
            "fetched_at": self.fetched_at,
            "note": "Prices are crowd-implied probabilities from a prediction market, normalized to remove the bookmaker overround.",
        }


# Module-level cache for the fixtures list: (timestamp, raw_events).
_fixtures_cache: tuple[float, list[dict[str, Any]]] | None = None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _market_yes_price(market: dict[str, Any]) -> float | None:
    """Best available 'Yes' probability for a market."""
    last = _as_float(market.get("lastTradePrice"))
    if last is not None:
        return last
    raw = market.get("outcomePrices")
    outcomes = market.get("outcomes")
    try:
        prices = json.loads(raw) if isinstance(raw, str) else raw
        labels = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
        if prices and labels:
            for label, price in zip(labels, prices):
                if str(label).strip().lower() == "yes":
                    return _as_float(price)
            return _as_float(prices[0])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return None


def _is_fulltime_match(event: dict[str, Any]) -> bool:
    """True only for the full-time 90-minute result (moneyline) event, not props."""
    return any(
        m.get("sportsMarketType") == "moneyline" for m in event.get("markets", [])
    )


def _parse_event(event: dict[str, Any]) -> MatchOdds:
    teams = event.get("teams") or []
    home_name = away_name = None
    for team in teams:
        if team.get("ordering") == "home":
            home_name = team.get("name")
        elif team.get("ordering") == "away":
            away_name = team.get("name")
    if not home_name or not away_name:
        # Fallback: derive from "A vs. B" title.
        title = event.get("title", "")
        parts = title.replace(" vs. ", " vs ").split(" vs ")
        if len(parts) == 2:
            home_name = home_name or parts[0].strip()
            away_name = away_name or parts[1].strip()

    markets = event.get("markets", [])
    moneyline = [m for m in markets if m.get("sportsMarketType") == "moneyline"]
    markets = moneyline or markets

    home_p = draw_p = away_p = None
    for market in markets:
        group = (market.get("groupItemTitle") or "").strip()
        price = _market_yes_price(market)
        if price is None:
            continue
        group_lower = group.lower()
        if group_lower.startswith("draw"):
            draw_p = price
        elif home_name and group_lower == home_name.lower():
            home_p = price
        elif away_name and group_lower == away_name.lower():
            away_p = price

    if home_p is None or draw_p is None or away_p is None:
        raise PolymarketError(
            f"Could not parse 1X2 markets for event {event.get('slug')!r}."
        )

    total = home_p + draw_p + away_p
    if total <= 0:
        raise PolymarketError(f"Invalid market prices for {event.get('slug')!r}.")

    return MatchOdds(
        slug=event.get("slug", ""),
        title=event.get("title", f"{home_name} vs. {away_name}"),
        home_team=home_name or "",
        away_team=away_name or "",
        event_date=event.get("eventDate"),
        kickoff_utc=event.get("startTime"),
        home_win=home_p / total,
        draw=draw_p / total,
        away_win=away_p / total,
        raw_prices={
            "home_win": round(home_p, 4),
            "draw": round(draw_p, 4),
            "away_win": round(away_p, 4),
        },
        volume_usd=_as_float(event.get("volume")),
        url=f"{POLYMARKET_WORLD_CUP_URL}/{event.get('slug', '')}",
    )


class PolymarketClient:
    """Fetches World Cup match odds from the Polymarket Gamma API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @property
    def settings(self) -> Settings:
        return self._settings

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        url = f"{self._settings.polymarket_base_url.rstrip('/')}/{path.lstrip('/')}"
        try:
            response = httpx.get(
                url, params=params, timeout=self._settings.polymarket_timeout_seconds
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            raise PolymarketError(f"Polymarket request failed: {exc}") from exc

    def fetch_event_by_slug(self, slug: str) -> MatchOdds:
        data = self._get("events", {"slug": slug})
        if not data:
            raise PolymarketError(f"No Polymarket event found for slug {slug!r}.")
        event = data[0] if isinstance(data, list) else data
        return _parse_event(event)

    def fetch_world_cup_fixtures(self, *, force_refresh: bool = False) -> list[dict[str, Any]]:
        """Raw event dicts for every World Cup fixture in the series (cached)."""
        global _fixtures_cache
        ttl = self._settings.polymarket_cache_ttl_seconds
        now = time.monotonic()
        if (
            not force_refresh
            and _fixtures_cache is not None
            and now - _fixtures_cache[0] < ttl
        ):
            return _fixtures_cache[1]

        events: list[dict[str, Any]] = []
        offset = 0
        limit = 100
        while True:
            batch = self._get(
                "events",
                {
                    "series_id": self._settings.polymarket_world_cup_series_id,
                    "closed": "false",
                    "limit": str(limit),
                    "offset": str(offset),
                },
            )
            if not isinstance(batch, list) or not batch:
                break
            events.extend(batch)
            if len(batch) < limit:
                break
            offset += limit

        _fixtures_cache = (now, events)
        return events

    def all_match_odds(
        self, *, match_date: str | None = None
    ) -> list[MatchOdds]:
        """Parsed odds for every fixture (optionally filtered by ISO date)."""
        results: list[MatchOdds] = []
        for event in self.fetch_world_cup_fixtures():
            if match_date and event.get("eventDate") != match_date:
                continue
            if not _is_fulltime_match(event):
                continue
            try:
                results.append(_parse_event(event))
            except PolymarketError:
                continue
        return results

    def list_fixtures(self) -> list[dict[str, Any]]:
        """Lightweight summary of upcoming fixtures (date, teams, slug)."""
        fixtures = []
        for event in self.fetch_world_cup_fixtures():
            if not _is_fulltime_match(event):
                continue
            fixtures.append(
                {
                    "match": event.get("title"),
                    "event_date": event.get("eventDate"),
                    "kickoff_utc": event.get("startTime"),
                    "slug": event.get("slug"),
                    "url": f"{POLYMARKET_WORLD_CUP_URL}/{event.get('slug', '')}",
                }
            )
        return fixtures

    def find_match(
        self,
        home_team: str,
        away_team: str,
        match_date: str | None = None,
    ) -> MatchOdds:
        """Find odds for a fixture by team names (order-insensitive), optional date."""
        wanted = {home_team.strip().lower(), away_team.strip().lower()}
        best: dict[str, Any] | None = None
        for event in self.fetch_world_cup_fixtures():
            if not _is_fulltime_match(event):
                continue
            teams = {
                str(t.get("name", "")).strip().lower()
                for t in (event.get("teams") or [])
            }
            if not teams:
                title = (event.get("title") or "").lower()
                if not all(w in title for w in wanted):
                    continue
            elif teams != wanted:
                continue
            if match_date and event.get("eventDate") != match_date:
                continue
            best = event
            break

        if best is None:
            raise PolymarketError(
                f"No Polymarket World Cup fixture found for "
                f"{home_team} vs {away_team}"
                + (f" on {match_date}" if match_date else "")
                + "."
            )
        return _parse_event(best)


def blend_probabilities(
    model: dict[str, float],
    market: dict[str, float],
    market_weight: float = 0.6,
) -> dict[str, float]:
    """Weighted blend of heuristic model and market-implied probabilities."""
    mw = min(max(market_weight, 0.0), 1.0)
    blended = {
        key: (1 - mw) * model.get(key, 0.0) + mw * market.get(key, 0.0)
        for key in ("home_win", "draw", "away_win")
    }
    total = sum(blended.values()) or 1.0
    return {key: round(value / total, 4) for key, value in blended.items()}
