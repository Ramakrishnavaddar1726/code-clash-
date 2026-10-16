"""Synthetic campus sensor data + fault injection.

Replace this module with MQTT / HTTP feeds from real IoT sensors later.
Every reading is 5 minutes apart and has the same columns a real feed would:
time, building, energy_kw, water_lpm, occupancy, lights_on, bin_pct
"""
import zlib

import numpy as np
import pandas as pd

STEP_MIN = 5
FAULT_TYPES = {
    "water_leak": "Water leakage",
    "energy_waste": "Energy wastage (empty room, lights/AC on)",
    "bin_overflow": "Waste accumulation (bin overflow)",
}

# pos = (x, y) on the 2D twin map
BUILDINGS = {
    "Library":     dict(kind="academic",    pos=(1, 3), base_kw=12, kw_pp=0.06, base_lpm=0.2, lpm_pp=0.015, peak=150, bin_rate=0.55),
    "Block A":     dict(kind="academic",    pos=(3, 3), base_kw=20, kw_pp=0.05, base_lpm=0.3, lpm_pp=0.012, peak=400, bin_rate=0.40),
    "Hostel":      dict(kind="residential", pos=(5, 3), base_kw=15, kw_pp=0.07, base_lpm=0.5, lpm_pp=0.020, peak=300, bin_rate=0.30),
    "Canteen":     dict(kind="canteen",     pos=(2, 1), base_kw=10, kw_pp=0.04, base_lpm=0.5, lpm_pp=0.030, peak=200, bin_rate=0.70),
    "Admin Block": dict(kind="academic",    pos=(4, 1), base_kw=9,  kw_pp=0.05, base_lpm=0.2, lpm_pp=0.010, peak=80,  bin_rate=0.50),
}


def _profile(kind, h, weekend):
    """Occupancy profile 0..1 by hour of day."""
    if kind == "academic":
        p = np.where((h >= 7) & (h < 18), np.sin((h - 7) / 11 * np.pi), 0.0).clip(0, 1)
        return p * np.where(weekend, 0.25, 1.0)
    if kind == "residential":
        return 0.65 + 0.3 * np.cos((h - 3) / 24 * 2 * np.pi)
    # canteen: breakfast, lunch, dinner peaks
    p = sum(np.exp(-((h - m) ** 2) / (2 * 0.7 ** 2)) for m in (8.0, 13.0, 19.5))
    return np.minimum(p, 1.0)


def generate_building(name, end, days=8, faults=(), seed=42):
    cfg = BUILDINGS[name]
    rng = np.random.default_rng(seed + zlib.crc32(name.encode()) % 10_000)
    n = days * 24 * 60 // STEP_MIN
    idx = pd.date_range(end=end, periods=n, freq=f"{STEP_MIN}min")
    h = idx.hour.values + idx.minute.values / 60
    weekend = idx.dayofweek.values >= 5

    prof = _profile(cfg["kind"], h, weekend)
    occ = np.rint(cfg["peak"] * prof * (1 + 0.06 * rng.standard_normal(n))).clip(0).astype(int)
    if cfg["kind"] != "residential":
        occ[prof < 0.02] = 0
    lights = occ > 0

    energy = (cfg["base_kw"] * np.where(lights, 1.0, 0.45) + cfg["kw_pp"] * occ
              + 0.3 * rng.standard_normal(n))
    water = (cfg["base_lpm"] * np.where(lights, 1.0, 0.2) + cfg["lpm_pp"] * occ
             + 0.05 * np.abs(rng.standard_normal(n)))

    # bin fill level (%), collected every day at 07:00
    bins = np.empty(n)
    level = 10.0
    for i in range(n):
        if idx[i].hour == 7 and idx[i].minute < STEP_MIN:
            level = 3.0
        level = min(level + cfg["bin_rate"] * prof[i] + 0.02 * abs(rng.standard_normal()), 100.0)
        bins[i] = level

    # ---- fault injection (last 3 hours) ----
    start = idx[-1] - pd.Timedelta(hours=3)
    m = idx >= start
    if "water_leak" in faults:
        water[m] += 7 + 0.3 * rng.standard_normal(m.sum())
    if "energy_waste" in faults:
        occ[m] = 0
        lights[m] = True
        energy[m] = (cfg["base_kw"] + 8 + 0.5 * cfg["kw_pp"] * cfg["peak"]
                     + 0.3 * rng.standard_normal(m.sum()))
    if "bin_overflow" in faults:
        elapsed = ((idx[m] - start) / pd.Timedelta(minutes=90)).values
        bins[m] = np.maximum(bins[m], 55 + 45 * np.clip(elapsed, 0, 1))

    return pd.DataFrame({
        "time": idx, "building": name,
        "energy_kw": energy.round(2), "water_lpm": water.round(2),
        "occupancy": occ, "lights_on": lights, "bin_pct": bins.round(1),
    })


def generate_campus(end=None, days=8, faults=None):
    """faults: {building_name: [fault_type, ...]}"""
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.now().floor(f"{STEP_MIN}min")
    faults = faults or {}
    return pd.concat(
        [generate_building(b, end, days, faults.get(b, ())) for b in BUILDINGS],
        ignore_index=True,
    )


if __name__ == "__main__":
    print(generate_campus().tail())
