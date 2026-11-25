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
       "db": "#264653", "bi": "#e76f51", "eval": "#dce6f2"}
W_FIG, H_FIG = 18.0, 9.8

fig, ax = plt.subplots(figsize=(W_FIG, H_FIG))
ax.set_xlim(0, W_FIG)
ax.set_ylim(0, 9.75)
ax.axis("off")


def box(x, y, w, h, title, sub="", kind="python", dashed=False):
    light = kind == "eval"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=COL[kind], alpha=0.95,
                                ec=("#4682b4" if light else "none"),
                                lw=(1.8 if light else 0),
                                ls=("--" if dashed else "-")))
    colour = INK if light else "white"
    n_title = title.count("\n") + 1
    ty = y + h * (0.66 if sub else 0.5)
    ax.text(x + w / 2, ty, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=colour, linespacing=1.15)
    if sub:
        ax.text(x + w / 2, y + h * (0.28 if n_title == 1 else 0.22), sub, ha="center",
                va="center", fontsize=8.6, color=colour, linespacing=1.25)


def arrow(x1, y1, x2, y2, rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6,
                                shrinkA=2, shrinkB=2,
                                connectionstyle=f"arc3,rad={rad}"))


def header(x, w, text):
    ax.text(x + w / 2, 9.45, text, ha="center", va="center", fontsize=11,
            fontweight="bold", color=INK)
    ax.plot([x, x + w], [9.2, 9.2], color=INK, lw=1.2)


# column x positions and widths
SX, SW = 0.2, 2.5
EX, EW = 3.2, 2.7
FX, FW = 6.4, 2.5
MX, MW = 9.4, 2.9
DX, DW = 12.8, 2.8
BX, BW = 16.0, 1.8

header(SX, SW, "1. Sources")
header(EX, EW, "2. Extract")
header(FX, FW, "3. Features")
header(MX, MW, "4. Analyse & model")
header(DX, DW, "5. Load (PostgreSQL)")
header(BX, BW, "6. Serve")

# ---- 1 sources
box(SX, 7.55, SW, 1.25, "ECCC GeoMet API", "hourly weather: Trudeau (YUL)\n+ McTavish (comparison)", "source")
box(SX, 5.25, SW, 1.25, "BIXI open data", "trips 2026\nZIP, ~2 GB", "source")

# ---- 2 extract
box(EX, 7.55, EW, 1.25, "01_extract_weather.py", "paginated API calls, retries\nUTC to local time", "python")
box(EX, 5.25, EW, 1.25, "02_aggregate_bixi.py", "chunked read (1M rows)\ncleaning, trips per hour", "python")

# ---- 3 features
box(FX, 6.15, FW, 1.25, "03_build_features.py", "merge weather + trips\ncalendar + holiday flags\n--year 2025 for the test year", "python")
ax.text(FX + FW / 2, 6.0, "features_hourly.csv", ha="center", va="top",
        fontsize=8.6, style="italic", color=MUTED)

# ---- 4 analyse & model (main chain)
box(MX, 8.0, MW, 0.85, "04_explore.py", "charts 1 to 4", "python")
box(MX, 6.75, MW, 1.0, "05_train_model.py\n06_weather_ablation.py", "", "python")
ax.text(MX + MW / 2, 6.68, "4 models, time-based split\nablation study  |  charts 5 to 7",
        ha="center", va="top", fontsize=8.4, color=INK)
box(MX, 5.1, MW, 0.95, "07_build_scenarios.py", "16,632 what-if scenarios  |  chart 8", "python")
box(MX, 3.95, MW, 0.95, "11_scenario_intervals.py", "90% interval per scenario", "python")

# ---- 5 PostgreSQL
box(DX, 6.35, DW, 1.2, "08_load_postgres.py", "COPY bulk load of 6 CSVs\nquality checks, 1 transaction", "python")
box(DX, 3.95, DW, 2.1, "PostgreSQL  (schema bixi)",
    "staging: weather_hourly, bixi_hourly\nmodelling: features_hourly\nresults: model_results, ablation,\nscenario_predictions (with intervals)\nviews: vw_hourly, vw_daily,\nvw_scenarios ...", "db")

# ---- 6 Power BI
box(BX, 6.35, BW, 1.2, "Power BI", "Overview\nSimulator | Model", "bi")
ax.text(BX + BW / 2, 7.7, '"At 22 °C with rain,\nhow many trips?"', ha="center", va="bottom",
        fontsize=9, style="italic", color=INK)

# ---- evaluation & experiments band (CSV only)
BAND_X, BAND_Y, BAND_W, BAND_H = 3.2, 0.95, 12.4, 2.75
ax.add_patch(FancyBboxPatch((BAND_X, BAND_Y), BAND_W, BAND_H,
                            boxstyle="round,pad=0.02,rounding_size=0.15", fc="none",
                            ec="#4682b4", lw=1.8, ls="--"))
ax.text(BAND_X + BAND_W / 2, BAND_Y + BAND_H - 0.2,
        "evaluation and experiments: read the CSVs, write a result CSV, not loaded in the database",
        ha="center", va="center", fontsize=9, style="italic", color="#4682b4")

EW_BOX, EH_BOX, GAP = 2.85, 0.95, 0.2
ex = [BAND_X + 0.2 + i * (EW_BOX + GAP) for i in range(4)]
row1, row2 = BAND_Y + 1.35, BAND_Y + 0.2
box(ex[0], row1, EW_BOX, EH_BOX, "09_validate_models.py", "rolling validation, 3 test months\nchart 9", "eval", True)
box(ex[1], row1, EW_BOX, EH_BOX, "10_prediction_intervals.py", "real coverage of the 90% intervals\nchart 10", "eval", True)
box(ex[2], row1, EW_BOX, EH_BOX, "12_station_comparison.py", "airport vs McTavish\nsame model, same folds", "eval", True)
box(ex[3], row1, EW_BOX, EH_BOX, "13_error_analysis.py", "MAE, RMSE, bias by hour,\nrain and temperature", "eval", True)
box(ex[0], row2, EW_BOX, EH_BOX, "14_lag_experiment.py", "lagged demand: day ahead\nand one hour ahead", "eval", True)
box(ex[1], row2, EW_BOX, EH_BOX, "15_rain_experiment.py", "monotone rain constraint\nrecent-rain features", "eval", True)
box(ex[2], row2, EW_BOX, EH_BOX, "16_year_transfer.py", "trained on 2026, scored on 2025\nautumn never seen", "eval", True)
ax.text(ex[3] + EW_BOX / 2, row2 + EH_BOX / 2,
        "inputs: features_hourly.csv (2026)\nand features_hourly_2025.csv\nhelpers: lags.py, rain_features.py,\nconformal.py",
        ha="center", va="center", fontsize=8.4, color=MUTED, linespacing=1.3)

# ---- 2025 test year (left, feeds the band)
box(SX, 1.75, SW, 1.45, "2025 data (test year)", "BIXI trips + weather\nrun through 02 and 03\nnever used to train", "source")

# ---- foundations strip
box(0.2, 0.05, 8.7, 0.72, "src/config.py + conformal.py + lags.py + rain_features.py",
    "", "data")
ax.text(4.55, 0.2, "every path, setting and the what-if grid in one place, imported by scripts 01 to 16",
        ha="center", va="center", fontsize=8.4, color="white")
box(9.1, 0.05, 8.7, 0.72, "tests/ + GitHub Actions", "", "data")
ax.text(13.45, 0.2, "44 pytest checks (3 need PostgreSQL, run in CI), on every push, Python 3.14",
        ha="center", va="center", fontsize=8.4, color="white")

# ---- arrows
arrow(SX + SW, 8.17, EX, 8.17)
arrow(SX + SW, 5.87, EX, 5.87)
arrow(EX + EW, 8.17, FX + 0.5, 7.4, rad=0.12)
arrow(EX + EW, 5.87, FX + 0.5, 6.15, rad=-0.12)
arrow(FX + FW, 7.2, MX, 8.4, rad=-0.12)
arrow(FX + FW, 6.85, MX, 7.25)
arrow(FX + FW, 6.5, MX, 5.6, rad=0.12)
arrow(MX + MW / 2, 5.1, MX + MW / 2, 4.9)                  # 07 -> 11
arrow(MX + MW, 7.25, DX, 6.95)                             # models -> loader
arrow(MX + MW, 4.4, DX, 6.5, rad=-0.3)                     # scenarios -> loader
arrow(DX + DW / 2, 6.35, DX + DW / 2, 6.05)                 # loader -> db
arrow(DX + DW, 5.0, BX + BW / 2, 6.35, rad=0.2)            # db -> Power BI
arrow(FX + FW / 2, 5.55, FX + FW / 2, BAND_Y + BAND_H)     # features -> band
ax.text(FX + FW / 2 - 0.12, 4.6, "2026 + 2025\nfeatures", ha="right", va="center",
        fontsize=8.4, style="italic", color=MUTED)
arrow(SX + SW, 2.45, BAND_X, 2.45)                         # 2025 data -> band

fig.suptitle("BIXI x weather: data architecture", x=0.02, ha="left", fontsize=15,
             fontweight="bold", color=INK, y=0.985)
fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.935)
fig.savefig(OUT, dpi=150, facecolor="white")
print(f"Saved {OUT}")
