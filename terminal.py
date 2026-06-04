"""Lightweight ANSI terminal styling for the prediction CLI (no dependencies).

Colors are disabled automatically when output is not a TTY or when NO_COLOR is set.
"""

from __future__ import annotations

import os
import re
import sys

_ENABLED = (
    os.environ.get("FORCE_COLOR") not in (None, "", "0")
    or (sys.stdout.isatty() and os.environ.get("NO_COLOR") is None)
)

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

_CODES = {
    "green": "\033[32m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "grey": "\033[90m",
}

# Match a scoreline like "Mexico vs South Africa (2-1)" with hyphen or unicode dashes.
_SCORE_RE = re.compile(
    r"^(?P<home>.+?)\s+vs\.?\s+(?P<away>.+?)\s*[\(\[]\s*(?P<hg>\d+)\s*[-\u2010-\u2015]\s*(?P<ag>\d+)\s*[\)\]]",
    re.IGNORECASE,
)
_LABEL_RE = re.compile(r"^\s*[-•*]?\s*(?P<label>[A-Za-z][A-Za-z /]+?):\s*(?P<value>.*)$")


def style(text: str, *colors: str) -> str:
    if not _ENABLED or not colors:
        return text
    prefix = "".join(_CODES.get(c, "") if c in _CODES else {"bold": BOLD, "dim": DIM}.get(c, "") for c in colors)
    return f"{prefix}{text}{RESET}"


def _visible_len(text: str) -> int:
    return len(re.sub(r"\033\[[0-9;]*m", "", text))


def stat(label: str, value: str, color: str = "cyan") -> str:
    """A colored '▸ Label: value' stat line."""
    bullet = style("▸", "blue")
    return f"{bullet} {style(f'{label}:', 'bold', color)} {value}"


def step(text: str, kind: str = "run") -> str:
    """A dim progress line shown while the agent streams (e.g. tool calls)."""
    icon = style("→", "cyan") if kind == "run" else style("✓", "green")
    return f"{icon} {style(text, 'dim')}"


def banner(home: str, away: str, score: str) -> list[str]:
    home, away, score = home.strip(), away.strip(), str(score).strip()
    plain = f"{home}  {score}  {away}"
    inner = max(_visible_len(plain) + 4, 30)  # 2 spaces padding each side, min width
    pad = inner - _visible_len(plain)
    left = pad // 2
    right = pad - left
    colored = (
        " " * left
        + style(home, "bold", "cyan")
        + "  "
        + style(score, "bold", "yellow")
        + "  "
        + style(away, "bold", "cyan")
        + " " * right
    )
    top = style("┏" + "━" * inner + "┓", "blue")
    mid = style("┃", "blue") + colored + style("┃", "blue")
    bottom = style("┗" + "━" * inner + "┛", "blue")
    return [top, mid, bottom]


def _color_for_label(label: str) -> str:
    low = label.lower()
    if low.startswith("winner"):
        return "green"
    if low.startswith("win prob") or low.startswith("probab"):
        return "magenta"
    if low.startswith("expected points") or low.startswith("points"):
        return "yellow"
    if low.startswith("why") or low.startswith("key"):
        return "grey"
    return "cyan"


def render(text: str, skip_banner: bool = False) -> str:
    """Render plain (markdown-stripped) agent text with ANSI colors and a score banner.

    When skip_banner is True, the score line is colored inline instead of boxed (used when
    banners are already rendered from structured tool output).
    """
    lines = text.splitlines()
    out: list[str] = []
    banner_done = skip_banner

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            out.append("")
            continue

        match = _SCORE_RE.match(line.strip())
        if match and not banner_done:
            g = match.groupdict()
            out.extend(banner(g["home"], g["away"], f"{g['hg']}-{g['ag']}"))
            banner_done = True
            trailing = line.strip()[match.end():].strip(" -–—")
            if trailing:
                out.append(style(trailing, "dim"))
            continue

        label = _LABEL_RE.match(line)
        if label:
            name = label.group("label").strip()
            value = label.group("value").strip()
            bullet = style("▸", "blue")
            colored_label = style(f"{name}:", "bold", _color_for_label(name))
            out.append(f"{bullet} {colored_label} {value}")
            continue

        out.append(line)

    return "\n".join(out)
