"""Indian retail calendar: festivals, national holidays, and chain-wide sale events.

Festival dates follow the lunar calendar, so they move every year. That matters for
forecasting: a plain "same day last year" lag (lag_364) misses Diwali when it moves
from Oct 24 to Nov 12. The `days_to_*` features built from this table let the model
learn the festival effect regardless of the calendar date.
"""
import pandas as pd

FESTIVALS = {
    "diwali":   ["2022-10-24", "2023-11-12", "2024-10-31", "2025-10-20", "2026-11-08"],
    "dussehra": ["2022-10-05", "2023-10-24", "2024-10-12", "2025-10-02", "2026-10-20"],
    "holi":     ["2022-03-18", "2023-03-08", "2024-03-25", "2025-03-14", "2026-03-04"],
    "eid":      ["2022-05-03", "2023-04-22", "2024-04-11", "2025-03-31", "2026-03-20"],
    "rakhi":    ["2022-08-11", "2023-08-30", "2024-08-19", "2025-08-09", "2026-08-28"],
}

# Fixed-date national holidays / occasions (month, day)
FIXED_HOLIDAYS = {
    "republic_day": (1, 26),
    "independence_day": (8, 15),
    "gandhi_jayanti": (10, 2),
    "christmas": (12, 25),
    "new_year": (1, 1),
}

YEARS = range(2022, 2027)


def festival_dates(name: str) -> pd.DatetimeIndex:
    if name in FESTIVALS:
        return pd.to_datetime(FESTIVALS[name])
    m, d = FIXED_HOLIDAYS[name]
    return pd.to_datetime([f"{y}-{m:02d}-{d:02d}" for y in YEARS])


def holiday_table() -> pd.DataFrame:
    """One row per holiday date: date, holiday_name."""
    rows = []
    for name in list(FESTIVALS) + list(FIXED_HOLIDAYS):
        for dt in festival_dates(name):
            rows.append((dt, name))
    df = pd.DataFrame(rows, columns=["date", "holiday_name"])
    # If two holidays fall on the same day (Dussehra 2025 = Gandhi Jayanti), keep both names
    return (df.groupby("date")["holiday_name"].agg(" + ".join).reset_index())


def mega_sale_windows() -> list[tuple[pd.Timestamp, pd.Timestamp, str, str | None]]:
    """Chain-wide sale events: (start, end, name, category_or_None_for_all)."""
    out = []
    for y in YEARS:
        out.append((pd.Timestamp(f"{y}-01-20"), pd.Timestamp(f"{y}-01-26"), "Republic Day Sale", None))
        out.append((pd.Timestamp(f"{y}-08-10"), pd.Timestamp(f"{y}-08-15"), "Independence Day Sale", None))
        out.append((pd.Timestamp(f"{y}-06-25"), pd.Timestamp(f"{y}-07-05"), "End of Season Sale", "Apparel"))
    for d in festival_dates("diwali"):
        out.append((d - pd.Timedelta(days=10), d - pd.Timedelta(days=3), "Big Festive Sale", None))
    return out


def days_to_event(dates: pd.Series, event_dates: pd.DatetimeIndex, clip: int = 30) -> pd.Series:
    """Signed days to the nearest occurrence of an event: negative = before, positive = after.
    Clipped to [-clip, clip] so days far from any event all look the same to the model."""
    d = pd.to_datetime(dates).values.astype("datetime64[D]").astype("int64")
    e = event_dates.values.astype("datetime64[D]").astype("int64")
    diff = d[:, None] - e[None, :]
    nearest = diff[range(len(d)), abs(diff).argmin(axis=1)]
    return pd.Series(nearest, index=dates.index).clip(-clip, clip)
