"""Generative AI layer: recommendations, sustainability report, Q&A.

Uses Claude if ANTHROPIC_API_KEY is set. Otherwise falls back to a template so the
demo never breaks (say so honestly if you present the fallback).
To use Gemini/OpenAI instead, only change `_call()`.
"""
import json
import os

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")

SYSTEM = (
    "You are a campus sustainability analyst for an Indian college. Be specific, concise and "
    "practical. Use INR for money. Do not use markdown tables. Use '## ' headings and '- ' bullets. "
    "Only use numbers present in the data given; label any extra estimate as an estimate."
)


def llm_available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def _call(user: str, max_tokens: int = 1400) -> str:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=MODEL, max_tokens=max_tokens, system=SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def recommendations(issues: list, kpis: dict) -> str:
    """Actionable recommendations + a short sustainability report."""
    if not llm_available():
        return _fallback(issues, kpis)
    prompt = (
        "Detected campus issues (JSON):\n" + json.dumps(issues, indent=1)
        + "\n\nCampus KPIs (JSON):\n" + json.dumps(kpis)
        + "\n\nWrite:\n## Executive summary (2-3 lines)\n"
          "## Prioritised actions (for each issue: what to do, who should do it, expected monthly "
          "saving in INR, and a prevention measure)\n"
          "## Sustainability snapshot (energy, water, waste, CO2 impact)\n"
          "## Next steps this week"
    )
    try:
        return _call(prompt)
    except Exception as e:  # network / key problems -> don't break the demo
        return _fallback(issues, kpis) + f"\n\n(LLM call failed: {e}. Showing template output.)"


def ask(question: str, issues: list, kpis: dict) -> str:
    if not llm_available():
        return ("Set ANTHROPIC_API_KEY to enable the chat assistant. Current open issues: "
                + (", ".join(f"{i['type']} in {i['building']}" for i in issues) or "none") + ".")
    prompt = (f"Campus data:\nIssues: {json.dumps(issues)}\nKPIs: {json.dumps(kpis)}\n\n"
              f"Facility manager question: {question}\nAnswer briefly using only this data.")
    try:
        return _call(prompt, 500)
    except Exception as e:
        return f"LLM call failed: {e}"


ACTIONS = {
    "Water leakage": ("Dispatch plumbing to inspect pipes/valves, isolate the supply line, and "
                      "install a flow sensor with a night-time zero-flow alarm."),
    "Energy wastage": ("Switch off lights/AC in the empty area now; install occupancy-based "
                       "switches and an auto-shutoff schedule."),
    "Waste accumulation": ("Send housekeeping to empty the bin; add a mid-day pickup and a "
                           "segregated second bin at peak hours."),
    "Inefficient resource usage": "Review recent usage with the building manager and audit equipment schedules.",
}


def _fallback(issues, kpis) -> str:
    lines = ["## Executive summary",
             f"{len(issues)} active issue(s) detected. Estimated recurring loss: "
             f"INR {kpis['monthly_loss_inr']:,}/month and {kpis['monthly_co2_kg']:,} kg CO2/month.",
             "", "## Prioritised actions"]
    for i in issues:
        lines.append(f"- **{i['type']} - {i['building']}** ({i['severity']}): {ACTIONS.get(i['type'], '')} "
                     f"Potential saving: INR {i['monthly_cost_inr']:,}/month.")
    if not issues:
        lines.append("- No action needed. Keep monitoring.")
    lines += ["", "## Sustainability snapshot",
              f"- Energy (24h): {kpis['energy_kwh_24h']:,} kWh ({kpis['co2_kg_24h']:,} kg CO2)",
              f"- Water (24h): {kpis['water_kl_24h']} kL",
              "", "(Template output - set ANTHROPIC_API_KEY for AI-written analysis.)"]
    return "\n".join(lines)
