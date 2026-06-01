"""Static football knowledge used by agent tools (expand or replace with live APIs later)."""


from __future__ import annotations


from dataclasses import dataclass
from typing import Any



@dataclass(frozen=True)
class TeamProfile:
    name: str
    confederation: str
    fifa_rank: int
    form: tuple[str, ...]  # last 5: W, D, or L
    avg_goals_scored: float
    avg_goals_conceded: float
    notes: str = ""



@dataclass(frozen=True)
class HeadToHeadRecord:
    matches_played: int
    home_team_wins: int
    away_team_wins: int
    draws: int
    last_meeting: str
    summary: str


TEAM_ALIASES: dict[str, str] = {
    "mexico": "Mexico",
    "mex": "Mexico",
    "el tri": "Mexico",
    "south africa": "South Africa",
    "rsa": "South Africa",
    "bafana bafana": "South Africa",
    "argentina": "Argentina",
    "brazil": "Brazil",
    "brasil": "Brazil",
    "england": "England",
    "france": "France",
    "germany": "Germany",
    "spain": "Spain",
    "usa": "United States",
    "us": "United States",
    "united states": "United States",
    "portugal": "Portugal",
    "netherlands": "Netherlands",
    "holland": "Netherlands",
    "italy": "Italy",
    "japan": "Japan",
    "morocco": "Morocco",
    "croatia": "Croatia",
    "canada": "Canada",
    "south korea": "South Korea",
    "korea": "South Korea",
    "kor": "South Korea",
    "czech republic": "Czech Republic",
    "czechia": "Czech Republic",
    "bosnia": "Bosnia and Herzegovina",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
    "bih": "Bosnia and Herzegovina",
    "qatar": "Qatar",
    "switzerland": "Switzerland",
    "sui": "Switzerland",
    "paraguay": "Paraguay",
    "par": "Paraguay",
    "australia": "Australia",
    "aus": "Australia",
    "turkey": "Türkiye",
    "tur": "Türkiye",
    "curaçao": "Curaçao",
    "cur": "Curaçao",
    "ivory coast": "Ivory Coast",
    "côte d'ivoire": "Ivory Coast",
    "civ": "Ivory Coast",
    "ecuador": "Ecuador",
    "ecu": "Ecuador",
    "sweden": "Sweden",
    "swe": "Sweden",
    "tunisia": "Tunisia",
    "tun": "Tunisia",
    "belgium": "Belgium",
    "bel": "Belgium",
    "egypt": "Egypt",
    "egy": "Egypt",
    "iran": "Iran",
    "irn": "Iran",
    "new zealand": "New Zealand",
    "nzl": "New Zealand",
    "cape verde": "Cape Verde",
    "cpv": "Cape Verde",
    "saudi arabia": "Saudi Arabia",
    "ksa": "Saudi Arabia",
    "uruguay": "Uruguay",
    "uru": "Uruguay",
    "senegal": "Senegal",
    "sen": "Senegal",
    "iraq": "Iraq",
    "irq": "Iraq",
    "norway": "Norway",
    "nor": "Norway",
    "algeria": "Algeria",
    "alg": "Algeria",
    "austria": "Austria",
    "aut": "Austria",
    "jordan": "Jordan",
    "JOR": "Jordan",
    "dr congo": "DR Congo",
    "congo dr": "DR Congo",
    "cod": "DR Congo",
    "uzbekistan": "Uzbekistan",
    "uzb": "Uzbekistan",
    "colombia": "Colombia",
    "col": "Colombia",
    "seleccion cafetera": "Colombia",
    "ghana": "Ghana",
    "gha": "Ghana",
    "panama": "Panama",
    "pan": "Panama",
    "haiti": "Haiti",
    "hai": "Haiti",
    "scotland": "Scotland",
    "sco": "Scotland",
}


TEAMS: dict[str, TeamProfile] = {
    # Group A
    "Mexico": TeamProfile(
        name="Mexico",
        confederation="CONCACAF",
        fifa_rank=15,
        form=("W", "D", "W", "W", "L"),
        avg_goals_scored=1.8,
        avg_goals_conceded=1.1,
        notes="Strong CONCACAF side with solid midfield control and set-piece threat. Co-host.",
    ),
    "South Africa": TeamProfile(
        name="South Africa",
        confederation="CAF",
        fifa_rank=60,
        form=("D", "L", "W", "D", "L"),
        avg_goals_scored=1.2,
        avg_goals_conceded=1.4,
        notes="Physical side; results can swing with defensive organization.",
    ),
    "South Korea": TeamProfile(
        name="South Korea",
        confederation="AFC",
        fifa_rank=25,
        form=("W", "W", "D", "L", "W"),
        avg_goals_scored=1.7,
        avg_goals_conceded=1.0,
        notes="Fast-paced attacking team with strong wing play.",
    ),
    "Czech Republic": TeamProfile(
        name="Czech Republic",
        confederation="UEFA",
        fifa_rank=41,
        form=("D", "W", "L", "D", "W"),
        avg_goals_scored=1.4,
        avg_goals_conceded=1.2,
        notes="UEFA playoff winner; disciplined defensive structure.",
    ),
    # Group B
    "Canada": TeamProfile(
        name="Canada",
        confederation="CONCACAF",
        fifa_rank=30,
        form=("W", "L", "W", "D", "W"),
        avg_goals_scored=1.6,
        avg_goals_conceded=1.1,
        notes="Co-host with emerging talent; physical and direct style.",
    ),
    "Bosnia and Herzegovina": TeamProfile(
        name="Bosnia and Herzegovina",
        confederation="UEFA",
        fifa_rank=65,
        form=("L", "D", "W", "L", "D"),
        avg_goals_scored=1.3,
        avg_goals_conceded=1.5,
        notes="UEFA playoff winner; technical midfield but defensive vulnerabilities.",
    ),
    "Qatar": TeamProfile(
        name="Qatar",
        confederation="AFC",
        fifa_rank=55,
        form=("W", "D", "L", "W", "D"),
        avg_goals_scored=1.4,
        avg_goals_conceded=1.3,
        notes="Defending Asian champions; organized and compact.",
    ),
    "Switzerland": TeamProfile(
        name="Switzerland",
        confederation="UEFA",
        fifa_rank=19,
        form=("W", "W", "D", "W", "L"),
        avg_goals_scored=1.8,
        avg_goals_conceded=0.9,
        notes="Tactically disciplined European side with strong team chemistry.",
    ),
    # Group C
    "Brazil": TeamProfile(
        name="Brazil",
        confederation="CONMEBOL",
        fifa_rank=6,
        form=("W", "L", "W", "W", "D"),
        avg_goals_scored=2.0,
        avg_goals_conceded=0.9,
        notes="Five-time champions with world-class attacking talent.",
    ),
    "Morocco": TeamProfile(
        name="Morocco",
        confederation="CAF",
        fifa_rank=8,
        form=("W", "D", "W", "W", "D"),
        avg_goals_scored=1.5,
        avg_goals_conceded=0.8,
        notes="2022 semi-finalists; strong defense and quick counter-attacks.",
    ),
    "Haiti": TeamProfile(
        name="Haiti",
        confederation="CONCACAF",
        fifa_rank=83,
        form=("L", "W", "D", "L", "W"),
        avg_goals_scored=1.1,
        avg_goals_conceded=1.7,
        notes="CONCACAF playoff winner; debutant with resilient spirit.",
    ),
    "Scotland": TeamProfile(
        name="Scotland",
        confederation="UEFA",
        fifa_rank=43,
        form=("W", "D", "L", "W", "D"),
        avg_goals_scored=1.5,
        avg_goals_conceded=1.2,
        notes="UEFA playoff winner; physical and competitive.",
    ),
    # Group D
    "United States": TeamProfile(
        name="United States",
        confederation="CONCACAF",
        fifa_rank=16,
        form=("W", "D", "L", "W", "W"),
        avg_goals_scored=1.7,
        avg_goals_conceded=1.2,
        notes="Co-host with young, athletic squad; improving rapidly.",
    ),
    "Paraguay": TeamProfile(
        name="Paraguay",
        confederation="CONMEBOL",
        fifa_rank=40,
        form=("D", "W", "L", "D", "W"),
        avg_goals_scored=1.3,
        avg_goals_conceded=1.1,
        notes="Defensive-minded CONMEBOL side; tough to break down.",
    ),
    "Australia": TeamProfile(
        name="Australia",
        confederation="AFC",
        fifa_rank=27,
        form=("W", "W", "L", "D", "W"),
        avg_goals_scored=1.6,
        avg_goals_conceded=1.0,
        notes="Physical AFC side with strong aerial presence.",
    ),
    "Türkiye": TeamProfile(
        name="Türkiye",
        confederation="UEFA",
        fifa_rank=22,
        form=("W", "L", "W", "W", "D"),
        avg_goals_scored=1.8,
        avg_goals_conceded=1.1,
        notes="Passionate team with talented individuals; unpredictable.",
    ),
    # Group E
    "Germany": TeamProfile(
        name="Germany",
        confederation="UEFA",
        fifa_rank=10,
        form=("D", "W", "L", "W", "D"),
        avg_goals_scored=1.9,
        avg_goals_conceded=1.0,
        notes="Four-time champions; rebuilding with young talent.",
    ),
    "Curaçao": TeamProfile(
        name="Curaçao",
        confederation="CONCACAF",
        fifa_rank=82,
        form=("D", "L", "W", "D", "L"),
        avg_goals_scored=1.0,
        avg_goals_conceded=1.6,
        notes="CONCACAF playoff winner; debutant with Dutch-connected players.",
    ),
    "Ivory Coast": TeamProfile(
        name="Ivory Coast",
        confederation="CAF",
        fifa_rank=34,
        form=("W", "W", "D", "L", "W"),
        avg_goals_scored=1.5,
        avg_goals_conceded=1.1,
        notes="2023 Africa Cup of Nations champions; explosive attackers.",
    ),
    "Ecuador": TeamProfile(
        name="Ecuador",
        confederation="CONMEBOL",
        fifa_rank=23,
        form=("W", "D", "W", "L", "W"),
        avg_goals_scored=1.6,
        avg_goals_conceded=0.7,
        notes="Strong defense; only 5 goals conceded in 18 qualifiers.",
    ),
    # Group F
    "Netherlands": TeamProfile(
        name="Netherlands",
        confederation="UEFA",
        fifa_rank=7,
        form=("W", "D", "W", "L", "W"),
        avg_goals_scored=2.1,
        avg_goals_conceded=1.0,
        notes="Total football heritage; tactically flexible and strong.",
    ),
    "Japan": TeamProfile(
        name="Japan",
        confederation="AFC",
        fifa_rank=18,
        form=("W", "W", "D", "W", "L"),
        avg_goals_scored=1.8,
        avg_goals_conceded=1.0,
        notes="Asia's top team; technical and disciplined.",
    ),
    "Sweden": TeamProfile(
        name="Sweden",
        confederation="UEFA",
        fifa_rank=38,
        form=("W", "D", "L", "W", "W"),
        avg_goals_scored=1.5,
        avg_goals_conceded=1.0,
        notes="UEFA playoff winner; physical and organized.",
    ),
    "Tunisia": TeamProfile(
        name="Tunisia",
        confederation="CAF",
        fifa_rank=44,
        form=("D", "W", "L", "D", "W"),
        avg_goals_scored=1.3,
        avg_goals_conceded=1.1,
        notes="Consistent African side; strong defensive organization.",
    ),
    # Group G
    "Belgium": TeamProfile(
        name="Belgium",
        confederation="UEFA",
        fifa_rank=9,
        form=("W", "W", "D", "L", "W"),
        avg_goals_scored=1.9,
        avg_goals_conceded=0.9,
        notes="Golden generation winding down; still full of talent.",
    ),
    "Egypt": TeamProfile(
        name="Egypt",
        confederation="CAF",
        fifa_rank=29,
        form=("W", "D", "W", "L", "W"),
        avg_goals_scored=1.6,
        avg_goals_conceded=1.0,
        notes="Led by Mohamed Salah; dangerous on counter-attacks.",
    ),
    "Iran": TeamProfile(
        name="Iran",
        confederation="AFC",
        fifa_rank=21,
        form=("W", "W", "D", "W", "L"),
        avg_goals_scored=1.5,
        avg_goals_conceded=0.8,
        notes="Defensively solid AFC side; physically strong.",
    ),
    "New Zealand": TeamProfile(
        name="New Zealand",
        confederation="OFC",
        fifa_rank=85,
        form=("L", "D", "W", "L", "D"),
        avg_goals_scored=1.0,
        avg_goals_conceded=1.5,
        notes="OFC champions; debutant in expanded World Cup.",
    ),
    # Group H
    "Spain": TeamProfile(
        name="Spain",
        confederation="UEFA",
        fifa_rank=2,
        form=("W", "W", "D", "W", "D"),
        avg_goals_scored=2.0,
        avg_goals_conceded=0.8,
        notes="Euro 2024 champions; possession-based attacking football.",
    ),
    "Cape Verde": TeamProfile(
        name="Cape Verde",
        confederation="CAF",
        fifa_rank=69,
        form=("W", "L", "D", "W", "L"),
        avg_goals_scored=1.2,
        avg_goals_conceded=1.3,
        notes="Debutant; surprising African qualifier with strong defense.",
    ),
    "Saudi Arabia": TeamProfile(
        name="Saudi Arabia",
        confederation="AFC",
        fifa_rank=61,
        form=("D", "W", "L", "D", "W"),
        avg_goals_scored=1.3,
        avg_goals_conceded=1.2,
        notes="2022 upset winners over Argentina; technical Asian side.",
    ),
    "Uruguay": TeamProfile(
        name="Uruguay",
        confederation="CONMEBOL",
        fifa_rank=17,
        form=("W", "W", "L", "W", "D"),
        avg_goals_scored=1.7,
        avg_goals_conceded=0.9,
        notes="Two-time champions; gritty and competitive.",
    ),
    # Group I
    "France": TeamProfile(
        name="France",
        confederation="UEFA",
        fifa_rank=1,
        form=("W", "D", "W", "W", "W"),
        avg_goals_scored=2.3,
        avg_goals_conceded=0.9,
        notes="Defending runners-up; deepest squad in the tournament.",
    ),
    "Senegal": TeamProfile(
        name="Senegal",
        confederation="CAF",
        fifa_rank=14,
        form=("W", "W", "D", "W", "L"),
        avg_goals_scored=1.7,
        avg_goals_conceded=0.9,
        notes="2022 Africa Cup of Nations runners-up; athletic and skilled.",
    ),
    "Iraq": TeamProfile(
        name="Iraq",
        confederation="AFC",
        fifa_rank=57,
        form=("W", "D", "L", "W", "D"),
        avg_goals_scored=1.2,
        avg_goals_conceded=1.1,
        notes="UEFA playoff winner; passionate and resilient.",
    ),
    "Norway": TeamProfile(
        name="Norway",
        confederation="UEFA",
        fifa_rank=31,
        form=("W", "W", "L", "D", "W"),
        avg_goals_scored=1.8,
        avg_goals_conceded=1.1,
        notes="UEFA playoff winner; Haaland-led attacking threat.",
    ),
    # Group J
    "Argentina": TeamProfile(
        name="Argentina",
        confederation="CONMEBOL",
        fifa_rank=3,
        form=("W", "W", "D", "W", "W"),
        avg_goals_scored=2.1,
        avg_goals_conceded=0.7,
        notes="Defending World Cup champions; Messi-led且tactically brilliant.",
    ),
    "Algeria": TeamProfile(
        name="Algeria",
        confederation="CAF",
        fifa_rank=28,
        form=("W", "L", "D", "W", "W"),
        avg_goals_scored=1.5,
        avg_goals_conceded=1.0,
        notes="2019 Africa Cup of Nations runners-up; physical and technical.",
    ),
    "Austria": TeamProfile(
        name="Austria",
        confederation="UEFA",
        fifa_rank=24,
        form=("W", "W", "L", "D", "W"),
        avg_goals_scored=1.7,
        avg_goals_conceded=1.0,
        notes="High-pressing German-coached side; well-organized.",
    ),
    "Jordan": TeamProfile(
        name="Jordan",
        confederation="AFC",
        fifa_rank=63,
        form=("D", "W", "W", "L", "D"),
        avg_goals_scored=1.2,
        avg_goals_conceded=1.0,
        notes="2023 Asian Cup runners-up; debutant with strong defense.",
    ),
    # Group K
    "Portugal": TeamProfile(
        name="Portugal",
        confederation="UEFA",
        fifa_rank=5,
        form=("W", "W", "L", "W", "D"),
        avg_goals_scored=1.9,
        avg_goals_conceded=0.9,
        notes="Euro 2016 champions; deep squad with world-class talent.",
    ),
    "DR Congo": TeamProfile(
        name="DR Congo",
        confederation="CAF",
        fifa_rank=46,
        form=("W", "D", "L", "W", "W"),
        avg_goals_scored=1.4,
        avg_goals_conceded=1.2,
        notes="Inter-confederation playoff winner; athletic and improving.",
    ),
    "Uzbekistan": TeamProfile(
        name="Uzbekistan",
        confederation="AFC",
        fifa_rank=50,
        form=("W", "W", "D", "L", "W"),
        avg_goals_scored=1.4,
        avg_goals_conceded=0.9,
        notes="Inter-confederation playoff winner; debutant with strong youth.",
    ),
    "Colombia": TeamProfile(
        name="Colombia",
        confederation="CONMEBOL",
        fifa_rank=13,
        form=("W", "W", "W", "D", "L"),
        avg_goals_scored=1.8,
        avg_goals_conceded=0.8,
        notes="2024 Copa América runners-up; technical and attacking.",
    ),
    # Group L
    "England": TeamProfile(
        name="England",
        confederation="UEFA",
        fifa_rank=4,
        form=("W", "W", "D", "W", "L"),
        avg_goals_scored=2.2,
        avg_goals_conceded=0.8,
        notes="Euro 2020 runners-up; 8 wins 22 goals 0 conceded in qualifiers.",
    ),
    "Croatia": TeamProfile(
        name="Croatia",
        confederation="UEFA",
        fifa_rank=11,
        form=("W", "D", "L", "W", "W"),
        avg_goals_scored=1.7,
        avg_goals_conceded=1.1,
        notes="2018 runners-up; experienced midfield with Modrić.",
    ),
    "Ghana": TeamProfile(
        name="Ghana",
        confederation="CAF",
        fifa_rank=74,
        form=("L", "W", "D", "W", "L"),
        avg_goals_scored=1.3,
        avg_goals_conceded=1.3,
        notes="Four-time African nations standout; energetic and attacking.",
    ),
    "Panama": TeamProfile(
        name="Panama",
        confederation="CONCACAF",
        fifa_rank=33,
        form=("W", "D", "W", "L", "D"),
        avg_goals_scored=1.4,
        avg_goals_conceded=1.2,
        notes="CONCACAF stalwart; improving and physically strong.",
    ),
}


# Key is sorted tuple of canonical team names.
H2H: dict[tuple[str, str], HeadToHeadRecord] = {
    ("Mexico", "South Africa"): HeadToHeadRecord(
        matches_played=2,
        home_team_wins=1,
        away_team_wins=0,
        draws=1,
        last_meeting="June 2010 friendly — Mexico 1-1 South Africa",
        summary="Limited history; meetings have been tight and low-scoring.",
    ),
    ("Argentina", "Brazil"): HeadToHeadRecord(
        matches_played=100,
        home_team_wins=40,
        away_team_wins=35,
        draws=25,
        last_meeting="November 2024 World Cup qualifier",
        summary="Classic rivalry; historically balanced with frequent draws.",
    ),
}


HOME_ADVANTAGE = 0.12
FORM_POINTS = {"W": 3, "D": 1, "L": 0}



class TeamNotFoundError(ValueError):
    pass



def resolve_team(name: str) -> str:
    """Map user input to a canonical team name."""
    key = name.strip().lower()
    if key in TEAM_ALIASES:
        return TEAM_ALIASES[key]
    for canonical in TEAMS:
        if canonical.lower() == key:
            return canonical
    raise TeamNotFoundError(
        f"Unknown team: {name!r}. Known teams: {', '.join(sorted(TEAMS))}"
    )



def list_teams() -> list[str]:
    return sorted(TEAMS)



def get_team_profile(team_name: str) -> dict[str, Any]:
    canonical = resolve_team(team_name)
    team = TEAMS[canonical]
    form_points = sum(FORM_POINTS[r] for r in team.form)
    return {
        "name": team.name,
        "confederation": team.confederation,
        "fifa_rank": team.fifa_rank,
        "recent_form": list(team.form),
        "form_points_last_5": form_points,
        "avg_goals_scored": team.avg_goals_scored,
        "avg_goals_conceded": team.avg_goals_conceded,
        "notes": team.notes,
    }



def get_head_to_head(team_a: str, team_b: str) -> dict[str, Any]:
    home = resolve_team(team_a)
    away = resolve_team(team_b)
    key = tuple(sorted([home, away]))
    record = H2H.get(key)
    if record is None:
        return {
            "team_a": home,
            "team_b": away,
            "matches_played": 0,
            "message": "No recorded head-to-head data; rely on rankings and recent form.",
        }
    return {
        "team_a": home,
        "team_b": away,
        "matches_played": record.matches_played,
        "wins_for_team_a": record.home_team_wins if key[0] == home else record.away_team_wins,
        "wins_for_team_b": record.away_team_wins if key[0] == home else record.home_team_wins,
        "draws": record.draws,
        "last_meeting": record.last_meeting,
        "summary": record.summary,
    }



def _form_strength(team: TeamProfile) -> float:
    points = sum(FORM_POINTS[r] for r in team.form)
    return points / 15.0



def _rank_strength(rank: int) -> float:
    return max(0.2, 1.0 - (rank - 1) / 200.0)



def predict_match(home_team: str, away_team: str) -> dict[str, Any]:
    """Heuristic match prediction from rankings, form, and optional H2H."""
    home_name = resolve_team(home_team)
    away_name = resolve_team(away_team)
    home = TEAMS[home_name]
    away = TEAMS[away_name]


    home_strength = _rank_strength(home.fifa_rank) * 0.55 + _form_strength(home) * 0.45
    away_strength = _rank_strength(away.fifa_rank) * 0.55 + _form_strength(away) * 0.45
    home_strength += HOME_ADVANTAGE


    h2h = get_head_to_head(home_name, away_name)
    if h2h.get("matches_played", 0) > 0 and h2h.get("draws", 0) >= h2h.get("matches_played", 1) // 2:
        home_strength *= 0.98
        away_strength *= 0.98


    total = home_strength + away_strength
    home_win_p = home_strength / total * 0.72
    away_win_p = away_strength / total * 0.72
    draw_p = max(0.12, 1.0 - home_win_p - away_win_p)


    scale = 1.0 / (home_win_p + draw_p + away_win_p)
    home_win_p *= scale
    draw_p *= scale
    away_win_p *= scale


    home_xg = home.avg_goals_scored * (away.avg_goals_conceded / 1.2) * (1 + HOME_ADVANTAGE)
    away_xg = away.avg_goals_scored * (home.avg_goals_conceded / 1.2)
    home_goals = max(0, round(home_xg + (home_win_p - away_win_p) * 0.8))
    away_goals = max(0, round(away_xg + (away_win_p - home_win_p) * 0.8))


    if home_goals == away_goals and home_win_p > away_win_p + 0.08:
        home_goals += 1
    elif home_goals == away_goals and away_win_p > home_win_p + 0.08:
        away_goals += 1


    if home_goals > away_goals:
        winner = home_name
    elif away_goals > home_goals:
        winner = away_name
    else:
        winner = "Draw"


    spread = max(home_win_p, draw_p, away_win_p) - min(home_win_p, draw_p, away_win_p)
    confidence = "high" if spread > 0.25 else "medium" if spread > 0.12 else "low"


    factors = [
        f"{home_name} FIFA rank #{home.fifa_rank} vs {away_name} #{away.fifa_rank}",
        f"Recent form: {home_name} {''.join(home.form)} | {away_name} {''.join(away.form)}",
        f"Home advantage applied for {home_name}",
    ]
    if h2h.get("matches_played"):
        factors.append(h2h["summary"])


    return {
        "home_team": home_name,
        "away_team": away_name,
        "predicted_score": f"{home_goals}-{away_goals}",
        "likely_winner": winner,
        "probabilities": {
            "home_win": round(home_win_p, 3),
            "draw": round(draw_p, 3),
            "away_win": round(away_win_p, 3),
        },
        "confidence": confidence,
        "key_factors": factors,
        "disclaimer": "Statistical estimate only — not a guarantee of the actual result.",
    }
