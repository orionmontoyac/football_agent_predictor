# ⚽ Football Agent Predictor

A local, agentic CLI that predicts FIFA World Cup match results to **maximize your score in a "Polla Mundialista"** (prediction pool). It combines a statistical model with live [Polymarket](https://polymarket.com/sports/world-cup/games) prediction-market odds, then chooses the scoreline that yields the **highest expected points** under the pool's scoring rules.

Built with **LangGraph** + a local **Ollama** model (default `granite4.1:3b`), so everything runs on your machine — no API keys required.

---

## What it does

- **Predicts the points-maximizing scoreline** for any World Cup fixture, not just the most likely score.
- **Blends two signals**: a heuristic model (FIFA ranking, recent form, head-to-head) and live Polymarket crowd-implied odds.
- **Fills the whole pool at once** (`predict_all_fixtures`) or one match at a time.
- **Learns during the tournament**: you record real results, and they become context for later predictions (e.g. how Mexico played match 1 informs match 2).
- **Tracks your score**: stores predictions + real results in JSON and computes points earned.
- **Styled terminal output**: clean ANSI boxes and colored stats (auto-disabled when piped).

---

## Scoring rules (Polla Mundialista)

The optimizer targets these exact rules. **Only the 90-minute result counts** (no extra time or penalties), so a draw is a valid pick even in knockouts.

| Outcome predicted correctly | Group stage (primera ronda) | Knockout (fases eliminatorias) |
|-----------------------------|:---------------------------:|:------------------------------:|
| Winner or draw              | 5                           | 10                             |
| Home team goals             | 2                           | 4                              |
| Away team goals             | 2                           | 4                              |
| Goal difference             | 1                           | 2                              |
| **Max per match**           | **10**                      | **20**                         |

Because the winner/draw is worth half the points, the agent prioritizes getting the result right, then optimizes the exact goals.

---

## How it works

```
                    ┌──────────────────────────────────────────────┐
   user query  ───► │  LangGraph agent (Ollama, tool-calling loop)   │
                    │                                                │
                    │   agent ──► tools ──► agent ──► … ──► answer    │
                    └───────────────┬────────────────────────────────┘
                                    │ calls tools
            ┌───────────────────────┼─────────────────────────────┐
            ▼                       ▼                             ▼
   football_data.py          polymarket.py                    store.py
   (model: rank, form,     (live Gamma API odds          (JSON: predictions
    H2H, expected goals)     for each fixture)             + real results)
            └───────────────┬───────┘                             │
                            ▼                                      │
                       scoring.py  ◄──────────────────────────────┘
              (Poisson scoreline distribution +
               expected-points optimizer)
```

### The prediction pipeline

1. **Model** (`football_data.py`): each team has a `TeamProfile` (Pydantic model) with FIFA rank, last-5 form, and goal averages. `predict_match()` derives win/draw/loss probabilities and expected goals (xG) for both sides. If a team has played in the tournament already, its stored results adjust its form and goal averages.
2. **Market** (`polymarket.py`): fetches the fixture's full-time "moneyline" market from the public Polymarket Gamma API and normalizes the cent prices into implied home/draw/away probabilities (overround removed).
3. **Blend**: model and market probabilities are mixed (`POLYMARKET_MARKET_WEIGHT`, default `0.6` toward the market). If a team isn't in the local DB, xG is inferred from the market odds alone.
4. **Optimize** (`scoring.py`): goals are modeled as Poisson variables, the joint scoreline distribution is reweighted to match the blended 1X2 probabilities, and the scoreline with the **maximum expected points** is selected. (Knockout weights are exactly 2× group weights, so the optimal score is the same for both stages — only the magnitude differs.)
5. **Persist** (`save_match_prediction` → `store.py`): when the agent should lock in picks, they are saved to `data/world_cup.json`.

---

## Project structure

| File | Responsibility |
|------|----------------|
| `main.py` | CLI entry point; styled terminal rendering of predictions |
| `graph.py` | LangGraph agent definition (agent ↔ tools loop) |
| `config.py` | Pydantic `Settings` (env-driven) + system prompt |
| `model_builder.py` | Builds the `ChatOllama` model from settings |
| `tools.py` | The agent's tools (predict, record, lookup, summary) |
| `football_data.py` | Team profiles, head-to-head, heuristic `predict_match()` |
| `polymarket.py` | Polymarket Gamma API client + odds parsing |
| `scoring.py` | Point rules, Poisson model, expected-points optimizer |
| `store.py` | JSON persistence of predictions and real results |
| `logging_config.py` | Loguru bootstrap + key=value event helper |
| `terminal.py` | Dependency-free ANSI colors and score banners |

---

## Requirements

- **Python 3.11+**
- **[Ollama](https://ollama.com)** running locally with the model pulled:

```bash
ollama pull granite4.1:3b
```

- Python dependencies (see `requirements.txt`): `langchain-core`, `langchain-ollama`, `langgraph`, `pydantic-settings`, `python-dotenv`, `httpx`, `loguru`.

---

## Setup

```bash
# 1. Create a virtual environment and install deps
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. (Optional) configure via .env
cp .env.example .env

# 3. Make sure Ollama is running and the model is available
ollama pull granite4.1:3b
```

---

## Usage

Run a single-match query:

```bash
python main.py "What will be the result of Mexico vs South Africa?"
```

Example output:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  Mexico  2-1  South Africa   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
▸ Winner: Mexico
▸ Win prob: H 58% / D 24% / A 18%
▸ Expected points: 4.383/10 (group)

▸ Why: higher ranking, better recent form, and home advantage.
```

Other things you can ask:

```bash
# Fill all picks for a given date
python main.py "Give me picks for all matches on 2026-06-13"

# Knockout-stage prediction (draws still allowed — 90 min only)
python main.py "Predict France vs Brazil in the knockout stage"

# Record a real result (becomes context + scores your prediction)
python main.py "Record the result: Mexico 2, South Africa 0"

# See how a team has played so far
python main.py "How did Mexico play so far?"

# Check your running score
python main.py "How many points do I have?"
```

### Flags

- `--no-stream` — print only the final answer (hides intermediate node updates).
- `--verbose` — enable `DEBUG` logging for this run (overrides `LOG_LEVEL`).

### Color control

Colors auto-enable on a TTY. Override with environment variables:

- `FORCE_COLOR=1` — force colors even when piped.
- `NO_COLOR=1` — disable colors.

---

## The agent's tools

| Tool | Purpose |
|------|---------|
| `predict_match_result` | Points-maximizing score for one fixture (read-only) |
| `predict_all_fixtures` | Picks for many fixtures at once (read-only) |
| `save_match_prediction` | Persist a pick to the polla store |
| `record_match_result` | Save a real 90-minute result; auto-scores your prediction |
| `get_team_recent_results` | A team's actual results so far this tournament |
| `get_points_summary` | Total points, exact scores, winners correct |
| `get_polymarket_odds` | Raw live market odds for a fixture |
| `list_world_cup_fixtures` | The match schedule (not predictions) |
| `lookup_team` / `get_match_head_to_head` / `list_supported_teams` | Model data lookups |

---

## Configuration

All settings are environment-driven via `config.py` (`Settings`) and an optional `.env` file. Key options:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `granite4.1:3b` | Ollama model tag |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_TEMPERATURE` | _(model default)_ | Sampling temperature |
| `POLYMARKET_ENABLED` | `true` | Toggle live market enrichment |
| `POLYMARKET_MARKET_WEIGHT` | `0.6` | Blend weight toward market vs model |
| `POLYMARKET_CACHE_TTL_SECONDS` | `300` | Fixtures cache lifetime |
| `POLYMARKET_WORLD_CUP_SERIES_ID` | `11433` | Gamma series id for WC fixtures |
| `LANGGRAPH_RECURSION_LIMIT` | `25` | Max graph steps per run |
| `LANGGRAPH_STREAM_MODE` | `updates` | LangGraph stream mode |
| `PREDICTIONS_FILE` | `data/world_cup.json` | Where predictions/results are stored |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |

See `.env.example` for a starting point.

---

## Observability

Structured `event=... key=value` logs trace what the agent does: which tool ran with what
arguments, whether Polymarket hit or fell back to the model, and how long each step took.
Logging is powered by [loguru](https://github.com/Delgan/loguru), which prints colorized,
level-tagged lines to **stderr** so they never mix with the styled CLI output on **stdout** —
you can redirect them independently (for example `python main.py "..." 2> agent.log`).

Set the level via `LOG_LEVEL` (or `--verbose` for a single `DEBUG` run):

```bash
LOG_LEVEL=DEBUG python main.py "What will be the result of Mexico vs South Africa?"
```

Sample output at `INFO` (colors stripped):

```
2026-06-04 12:00:01 INFO     football_agent.main event=run_start stream=True query_len=42
2026-06-04 12:00:02 INFO     football_agent.graph event=agent_done duration_ms=820
2026-06-04 12:00:02 INFO     football_agent.graph event=agent_tool_plan tools=predict_match_result
2026-06-04 12:00:02 INFO     football_agent.graph event=tool_start tool=predict_match_result
2026-06-04 12:00:03 INFO     football_agent.polymarket event=polymarket_http_done path=events duration_ms=210 status=ok
2026-06-04 12:00:03 INFO     football_agent.polymarket event=polymarket_match_found home=Mexico away="South Africa" slug=...
2026-06-04 12:00:03 INFO     football_agent.tools event=market_blend home=Mexico away="South Africa" source=model+market
2026-06-04 12:00:03 INFO     football_agent.graph event=tool_done tool=predict_match_result duration_ms=1200 status=ok
2026-06-04 12:00:05 INFO     football_agent.graph event=agent_final content_len=96
2026-06-04 12:00:05 INFO     football_agent.main event=run_end duration_ms=4500
```

`DEBUG` adds per-call tool arguments, routing decisions, cache hits, and raw HTTP requests.

---

## Data persistence

Predictions and results are stored in `data/world_cup.json`, keyed by the Polymarket fixture slug. Each record holds the prediction, the real result (once recorded), and the points earned:

```json
{
  "matches": {
    "fifwc-mex-rsa-2026-06-11": {
      "home_team": "Mexico", "away_team": "South Africa",
      "stage": "group", "event_date": "2026-06-11",
      "prediction": { "home_goals": 2, "away_goals": 1, "winner": "Mexico", "expected_points": 4.38 },
      "result":     { "home_goals": 1, "away_goals": 0, "winner": "Mexico" },
      "points_earned": { "points": 6, "max_points": 10, "exact_score": false }
    }
  }
}
```

This file is git-ignored and serves as the agent's memory across runs.

---

## Notes & limitations

- Predictions are **statistical/market estimates, not guarantees**.
- Goals are modeled as independent Poisson variables (no Dixon-Coles low-score correlation adjustment).
- Polymarket odds reflect crowd belief and update in real time; the fixtures list is cached briefly to avoid excessive requests.
- The default model (`granite4.1:3b`) is small; responses may vary in phrasing, but the score, probabilities, and points shown in the banner are computed deterministically from tool output (not the LLM).
