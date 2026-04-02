# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo>=0.20.2"]
# ///
"""Minimal repro for marimo button + mo.state (see scalable_postselection.py).

Run:  marimo edit test.py   or   marimo run test.py

What to check:
1. Section A: raw button.value — should increment on each click if on_click runs.
2. Section B: same pattern as the game (compare total clicks vs state _clicks).
3. Section C: options_pricing-style — on_click calls set_* directly (no .value relay).

If A/B stay at 0 but C works, the issue is relying on button.value + a separate handler cell.
If nothing works, check marimo version and browser console for errors.
"""

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    get_game, set_game = mo.state(
        {
            "clicks_recorded": 0,
            "_clicks": 0,
            "_accept_n": 0,
            "direct_clicks": 0,
        }
    )
    return get_game, set_game


@app.cell
def _(mo):
    _tick = lambda v: (v or 0) + 1
    btn_value_only = mo.ui.button(
        label="A: value + on_click tick",
        value=0,
        on_click=_tick,
    )
    return (btn_value_only,)


@app.cell
def _(btn_value_only, get_game, mo):
    mo.md(f"**A — button.value:** `{btn_value_only.value}`  (expect +1 per click)")
    mo.hstack([btn_value_only, get_game()], justify="start")
    return


@app.cell
def _(get_game, mo, set_game):
    def _direct(_):
        s = get_game()
        set_game({**s, "direct_clicks": s["direct_clicks"] + 1})

    btn_direct = mo.ui.button(label="C: on_click -> set_game()", on_click=_direct)
    _d = get_game()["direct_clicks"]
    mo.md(f"**C — direct state:** `{_d}`  (expect +1 per click)")
    mo.hstack([btn_direct], justify="start")
    return


@app.cell
def _(btn_value_only, get_game, set_game):
    """Mirrors scalable_postselection handler: separate cell reads .value vs mo.state."""
    _s = get_game()
    _a = btn_value_only.value or 0
    _total = _a
    _prev = _s.get("_clicks", 0)

    if _total > _prev:
        _new = dict(_s)
        _new["_clicks"] = _total
        _new["_accept_n"] = _a
        _new["clicks_recorded"] = _s.get("clicks_recorded", 0) + 1
        set_game(_new)
    return


@app.cell
def _(get_game, mo):
    _s = get_game()
    mo.md(
        f"**B — game-style counter** (handler updates when `_total > _prev`):\n\n"
        f"- `clicks_recorded`: **{_s['clicks_recorded']}**\n"
        f"- `_clicks` (stored total): **{_s['_clicks']}**\n"
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    "
        <!-- "**Debug:** open browser devtools → Console when clicking. " -->
        "If A stays 0, `on_click` is not updating `.value`. "
        "If A moves but B stays 0, the handler cell is not re-running after the button updates.
    """)
    return


if __name__ == "__main__":
    app.run()
