"""Detection layer: statistical rules + Isolation Forest on sensor streams."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

STEP_H = 5 / 60
TARIFF_INR_PER_KWH = 8.0
WATER_INR_PER_LITRE = 0.03   # ~Rs 30 per kilolitre
CO2_KG_PER_KWH = 0.82        # approx. Indian grid factor
SUSTAIN_STEPS = 6            # condition must hold 30 min to raise an alert
EMPTY_ROOM_HOURS_PER_DAY = 10  # assumption used for projections


def _features(d):
    h = d["time"].dt.hour + d["time"].dt.minute / 60
    return np.column_stack([
        d["energy_kw"], d["water_lpm"], d["occupancy"],
        np.sin(h / 24 * 2 * np.pi), np.cos(h / 24 * 2 * np.pi),
    ])


def build_baseline(hist: pd.DataFrame) -> dict:
    """Learn 'normal' from healthy history: hourly profile + Isolation Forest per building."""
    hist = hist.assign(hour=hist["time"].dt.hour)
    table = hist.groupby(["building", "hour"]).agg(
        w_mu=("water_lpm", "mean"), w_sd=("water_lpm", "std"),
        e_mu=("energy_kw", "mean"),
    ).reset_index()
    idle, models = {}, {}
    for b, d in hist.groupby("building"):
        empty = d[d["occupancy"] == 0]["energy_kw"]
        idle[b] = (float(empty.median()) if len(empty)
                   else float(table[table["building"] == b]["e_mu"].min()))
        models[b] = IsolationForest(n_estimators=100, contamination=0.005,
                                    random_state=0).fit(_features(d))
    return {"table": table, "idle": idle, "models": models}


def _trailing_run(mask: np.ndarray) -> int:
    n = 0
    for v in mask[::-1]:
        if not v:
            break
        n += 1
    return n


def detect_issues(live: pd.DataFrame, base: dict) -> list[dict]:
    issues = []
    for b, d in live.groupby("building"):
        d = d.sort_values("time").assign(hour=lambda x: x["time"].dt.hour)
        d = d.merge(base["table"][base["table"]["building"] == b].drop(columns="building"),
                    on="hour", how="left")
        found = False

        # 1) Water leakage: flow far above the hourly norm, sustained
        sd = d["w_sd"].fillna(0.3).clip(lower=0.3)
        excess = d["water_lpm"] - d["w_mu"]
        mask = ((excess / sd) > 4) & (excess > 2)
        run = _trailing_run(mask.values)
        if run >= SUSTAIN_STEPS:
            r = d.tail(run)
            rate = float(excess.tail(run).mean())
            per_day = rate * 60 * 24
            issues.append(dict(
                building=b, type="Water leakage",
                severity="high" if rate >= 5 else "medium",
                since=str(r["time"].iloc[0]),
                detail=(f"Flow {r['water_lpm'].mean():.1f} L/min vs normal {r['w_mu'].mean():.1f} L/min "
                        f"for this time, sustained {run * 5} min. ~{per_day:,.0f} L/day lost."),
                loss_per_day=round(per_day), loss_unit="L",
                monthly_cost_inr=round(per_day * 30 * WATER_INR_PER_LITRE),
                co2_kg_month=0.0,
            ))
            found = True

        # 2) Energy wastage: room empty (from CV / occupancy) but lights/AC still on
        mask = ((d["occupancy"] == 0) & d["lights_on"]).values
        run = _trailing_run(mask)
        if run >= SUSTAIN_STEPS:
            r = d.tail(run)
            kw = max(float(r["energy_kw"].mean() - base["idle"][b]), 0.0)
            kwh_month = kw * EMPTY_ROOM_HOURS_PER_DAY * 30
            issues.append(dict(
                building=b, type="Energy wastage",
                severity="high" if kw >= 10 else "medium",
                since=str(r["time"].iloc[0]),
                detail=(f"0 occupants but lights/AC ON for {run * 5} min, drawing "
                        f"{r['energy_kw'].mean():.1f} kW ({kw:.1f} kW above idle)."),
                loss_per_day=round(kw * EMPTY_ROOM_HOURS_PER_DAY), loss_unit="kWh",
                monthly_cost_inr=round(kwh_month * TARIFF_INR_PER_KWH),
                co2_kg_month=round(kwh_month * CO2_KG_PER_KWH),
            ))
            found = True

        # 3) Waste accumulation: bin nearly full
        level = float(d["bin_pct"].iloc[-1])
        if level >= 85:
            slope = float(d["bin_pct"].iloc[-1] - d["bin_pct"].iloc[-7]) / 6  # % per step
            since = d.loc[d["bin_pct"] >= 85, "time"].iloc[0]
            issues.append(dict(
                building=b, type="Waste accumulation",
                severity="high" if level >= 95 else "medium",
                since=str(since),
                detail=(f"Bin {level:.0f}% full" + (" - overflow risk, litter likely nearby."
                        if level >= 95 else f", rising {slope * 12:.1f}%/hour. Schedule pickup.")),
                loss_per_day=0, loss_unit="",
                monthly_cost_inr=0, co2_kg_month=0.0,
            ))
            found = True

        # 4) ML: unusual overall pattern not explained by rules above
        if not found:
            pred = base["models"][b].predict(_features(d.tail(12))) == -1
            if pred.sum() >= 10:
                issues.append(dict(
                    building=b, type="Inefficient resource usage", severity="low",
                    since=str(d["time"].iloc[-12]),
                    detail="Isolation Forest flags the last hour as unusual vs. learned behaviour "
                           "(energy/water/occupancy combination).",
                    loss_per_day=0, loss_unit="", monthly_cost_inr=0, co2_kg_month=0.0,
                ))
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(issues, key=lambda i: order[i["severity"]])


def building_status(issues, b) -> str:
    sev = [i["severity"] for i in issues if i["building"] == b]
    if "high" in sev:
        return "red"
    return "yellow" if sev else "green"


def campus_kpis(live: pd.DataFrame, issues: list[dict]) -> dict:
    kwh = float(live["energy_kw"].sum() * STEP_H)
    return dict(
        energy_kwh_24h=round(kwh),
        water_kl_24h=round(float(live["water_lpm"].sum() * 5) / 1000, 1),
        co2_kg_24h=round(kwh * CO2_KG_PER_KWH),
        active_issues=len(issues),
        monthly_loss_inr=int(sum(i["monthly_cost_inr"] for i in issues)),
        monthly_co2_kg=int(sum(i["co2_kg_month"] for i in issues)),
    )
