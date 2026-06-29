"""Render a SymbolForecast / RunResult into Telegram-ready text.

Plain text with light Markdown, safe for Telegram. Kept separate from the models
so the JSON result and the human message can evolve independently.
"""

from __future__ import annotations

from market_forecaster.models import RunResult, SymbolForecast

_ARROW = {"up": "▲", "down": "▼", "flat": "→"}


def _direction(current: float | None, target: float | None) -> str:
    if current is None or target is None:
        return "flat"
    if target > current * 1.0005:
        return "up"
    if target < current * 0.9995:
        return "down"
    return "flat"


def format_symbol(sf: SymbolForecast) -> str:
    if sf.error:
        return f"*{sf.display}*\n⚠️ Could not forecast: {sf.error}"

    lines: list[str] = []
    lines.append(f"*{sf.display}*")
    if sf.is_proxy and sf.proxy_note:
        lines.append(f"_proxy: {sf.proxy_note}_")

    target = sf.point_forecast[-1] if sf.point_forecast else None
    direction = _direction(sf.current_price, target)
    arrow = _ARROW[direction]

    if sf.current_price is not None:
        lines.append(f"Current: {sf.current_price:,.2f}")
    if target is not None:
        pct = ""
        if sf.current_price:
            pct = f" ({(target / sf.current_price - 1) * 100:+.2f}%)"
        lines.append(f"Forecast (+{sf.horizon}d): {arrow} {target:,.2f}{pct}")

    band = sf.quantiles.band()
    if band and band[0] and band[1]:
        lines.append(f"Range (10-90%): {band[0][-1]:,.2f} – {band[1][-1]:,.2f}")

    # Baseline comparison is ALWAYS shown.
    if sf.baseline:
        b = sf.baseline
        if b.beats_naive is True:
            verdict = f"✅ beats naive (MASE {b.mase})"
        elif b.beats_naive is False:
            verdict = f"❌ worse than naive (MASE {b.mase})"
        else:
            verdict = f"baseline: {b.note or 'no backtest'}"
        lines.append(f"Vs naive: {verdict}")

    if sf.news:
        sent = sf.news.sentiment.capitalize()
        conf = f" ({sf.news.confidence:.0%})" if sf.news.confidence is not None else ""
        lines.append(f"News: {sent}{conf}")
        if sf.news.narrative:
            lines.append(sf.news.narrative)

    lines.append(f"_source: {sf.source or 'n/a'}_")
    return "\n".join(lines)


def format_run(run: RunResult) -> str:
    header = f"\U0001f4c8 *Market Forecast* — {run.generated_at:%Y-%m-%d %H:%M UTC}"
    blocks = [header]
    for sf in run.results:
        blocks.append(format_symbol(sf))
    if run.errors:
        blocks.append("⚠️ " + "; ".join(run.errors))
    blocks.append(f"_run {run.correlation_id}_")
    return "\n\n".join(blocks)
