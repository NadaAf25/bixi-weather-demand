"""
build_architecture.py
Draw the architecture diagram used in the README -> docs/architecture.png
Run: py docs/build_architecture.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / "architecture.png"

INK, MUTED = "#264653", "#5c6b73"
COL = {"source": "#f4a261", "python": "#4682b4", "data": "#2a9d8f",
       "db": "#264653", "bi": "#e76f51"}

fig, ax = plt.subplots(figsize=(16.5, 7.2))
ax.set_xlim(0, 16.5)
ax.set_ylim(0, 7.2)
ax.axis("off")


def box(x, y, w, h, title, sub="", kind="python", dark=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=COL[kind], ec="none", alpha=0.95))
    colour = "white"
    ax.text(x + w / 2, y + h * (0.64 if sub else 0.5), title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=colour)
    if sub:
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
                fontsize=8.6, color=colour, linespacing=1.25)


def arrow(x1, y1, x2, y2, rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6,
                                shrinkA=2, shrinkB=2,
                                connectionstyle=f"arc3,rad={rad}"))


def header(x, w, text):
    ax.text(x + w / 2, 6.85, text, ha="center", va="center", fontsize=11,
            fontweight="bold", color=INK)
    ax.plot([x, x + w], [6.6, 6.6], color=INK, lw=1.2)


# column x positions
SX, EX, FX, MX, DX, BX = 0.2, 2.9, 5.6, 8.3, 11.1, 14.2
W = 2.2

header(SX, W, "1. Sources")
header(EX, W + 0.2, "2. Extract")
header(FX, W, "3. Features")
header(MX, W + 0.3, "4. Analyse & model")
header(DX, W + 0.6, "5. Load (PostgreSQL)")
header(BX, W, "6. Serve")

# 1 sources
box(SX, 4.9, W, 1.2, "ECCC GeoMet API", "hourly weather\nMontreal-Trudeau (YUL)", "source")
box(SX, 2.0, W, 1.2, "BIXI open data", "trips 2026\nZIP, ~2 GB", "source")

# 2 extract
box(EX, 4.9, W + 0.2, 1.2, "01_extract_weather.py", "paginated API calls\nretries, UTC to local time", "python")
box(EX, 2.0, W + 0.2, 1.2, "02_aggregate_bixi.py", "chunked read (1M rows)\ncleaning, trips per hour", "python")

# 3 features
box(FX, 3.45, W, 1.2, "03_build_features.py", "merge weather + trips\ncalendar + holiday flags", "python")

# 4 analyse & model
box(MX, 5.0, W + 0.3, 1.1, "04_explore.py", "charts 1 to 4", "python")
box(MX, 3.45, W + 0.3, 1.2, "05_train_model.py\n06_weather_ablation.py", "", "python")
ax.text(MX + (W + 0.3) / 2, 3.18, "4 models, time-based split\nablation study  |  charts 5 to 7",
        ha="center", va="top", fontsize=8.6, color=INK)
box(MX, 1.4, W + 0.3, 1.2, "07_build_scenarios.py", "16,632 what-if scenarios\nchart 8", "python")

# 5 PostgreSQL
box(DX, 3.45, W + 0.6, 1.2, "08_load_postgres.py", "COPY bulk load\nquality checks, 1 transaction", "python")
box(DX, 0.55, W + 0.6, 2.35, "PostgreSQL  (schema bixi)",
    "staging: weather_hourly, bixi_hourly\nmodelling: features_hourly\nresults: model_results, ablation,\n        scenario_predictions\nviews: vw_hourly, vw_daily,\n        vw_scenarios ...", "db")

# 6 Power BI
box(BX, 3.45, W, 1.2, "Power BI", "Overview | Simulator | Model", "bi")
ax.text(BX + W / 2, 4.85, '"At 22 °C with rain,\nhow many trips?"', ha="center", va="bottom",
        fontsize=9, style="italic", color=INK)

# arrows
arrow(SX + W, 5.5, EX, 5.5)
arrow(SX + W, 2.6, EX, 2.6)
arrow(EX + W + 0.2, 5.5, FX + 0.4, 4.65, rad=0.1)
arrow(EX + W + 0.2, 2.6, FX + 0.4, 3.45, rad=-0.1)
arrow(FX + W, 4.4, MX, 5.4, rad=-0.15)
arrow(FX + W, 4.05, MX, 4.05)
arrow(FX + W, 3.7, MX, 2.3, rad=0.15)
arrow(MX + W + 0.3, 4.05, DX, 4.05)             # models -> loader
arrow(MX + W + 0.3, 2.0, DX, 1.9)               # scenarios -> db (via loader files)
arrow(DX + (W + 0.6) / 2, 3.45, DX + (W + 0.6) / 2, 2.9)   # loader -> db
arrow(DX + W + 0.6, 1.9, BX + W / 2, 3.45, rad=0.2)       # db -> Power BI

ax.text(8.25, 0.12, "All intermediate results are CSV files in data/processed/  |  "
        "only the loader talks to the database", ha="center", fontsize=9, color=MUTED)

fig.suptitle("BIXI x weather: data architecture", x=0.02, ha="left", fontsize=15,
             fontweight="bold", color=INK, y=0.985)
plt.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT, dpi=150, facecolor="white")
print(f"Saved {OUT}")
