# Power BI dashboard: build guide

The dashboard reads the `bixi.vw_*` views of the PostgreSQL database. It has three pages:
**Overview**, **Simulator** ("22 °C and rain: how many trips?") and **Model**.

> The `.pbix` file is a binary made in Power BI Desktop (Windows). This guide gives
> every step and every DAX measure, so it takes about 30 minutes. Save the result as
> `powerbi/bixi_weather.pbix` and add screenshots to `docs/`.

## 1. Connect

1. Start the database and load it: `docker compose up -d` then `py src/08_load_postgres.py`
   (with your own PostgreSQL, pass the connection string:
   `py src/08_load_postgres.py --dsn postgresql://bixi:bixi@localhost:5433/bixi`)
2. Power BI Desktop → **Get data → PostgreSQL database**
   - Server `localhost:5432` (use your own port if it differs, e.g. `localhost:5433`),
     Database `bixi`, mode **Import**
   - Credentials: user `bixi`, password `bixi` (Database tab)
   - If Power BI reports a missing driver, install the Npgsql provider and restart.
3. In the Navigator tick: `vw_hourly`, `vw_daily`, `vw_trips_by_hour`,
   `vw_daytime_by_temp_rain`, `vw_scenarios`, `model_results`, `ablation_results`
   → **Load**.
4. **Model view**: delete any relationship Power BI auto-created. Every table is a
   self-contained report table. The simulator must not be filtered by anything but its slicers.
   (If a visual mixes fields from two tables, Power BI shows "Can't determine relationships":
   do not click *Fix this*, keep one table per visual.)
5. **View → Themes → Browse for themes** → `powerbi/theme.json`.

Table settings to apply (Data view → select the column → *Column tools* → *Sort by column*):

| Table | Column | Setting |
|---|---|---|
| `vw_scenarios` | `day_name` | Sort by column → `day_of_week` |
| `vw_scenarios` | `rain_label` | Sort by column → `rain_order` |
| `vw_hourly` | `day_name` | Sort by column → `day_of_week` |
| `vw_hourly` | `month_name` | Sort by column → `month` |
| `vw_daytime_by_temp_rain` | `temp_bin` | Sort by column → `temp_bin_low` |

*Note on `rain_label`:* the `vw_scenarios` view already provides a `rain_order` column
(1 = No rain, 2 = Light rain, 3 = Heavy rain), so no calculated column is needed:
just sort `rain_label` by `rain_order`. If `rain_order` is missing, refresh the data
(Home → Refresh) after reloading the database with `08_load_postgres.py`.

## 2. DAX measures

Create a table for them: **Home → Enter data**, name it `_Measures`, then add these.

```DAX
-- ===== Simulator page =====

Predicted trips =
-- With hour on the axis: trips for that hour. Without it: trips for the whole day.
-- If several days are selected it is the average day.
IF (
    HASONEVALUE ( vw_scenarios[temp_c] ) && HASONEVALUE ( vw_scenarios[rain_label] ),
    DIVIDE (
        SUM ( vw_scenarios[predicted_trips] ),
        DISTINCTCOUNT ( vw_scenarios[day_of_week] )
    )
)

Predicted trips if dry =
CALCULATE ( [Predicted trips], vw_scenarios[rain_label] = "No rain" )

Rain impact % =
DIVIDE ( [Predicted trips] - [Predicted trips if dry], [Predicted trips if dry] )

Average observed day =
AVERAGE ( vw_daily[trips] )

Reliability message =
VAR similar = MIN ( vw_scenarios[n_similar_hours] )
RETURN
    IF (
        MIN ( vw_scenarios[is_reliable] ) = 0,
        "Warning: only " & similar & " similar hours were observed. Treat this estimate with caution.",
        "Based on " & similar & " similar observed hours (temperature within 2 °C)."
    )

Scenario headline =
IF (
    HASONEVALUE ( vw_scenarios[temp_c] ) && HASONEVALUE ( vw_scenarios[rain_label] ),
    "At " & SELECTEDVALUE ( vw_scenarios[temp_c] ) & " °C with "
        & LOWER ( SELECTEDVALUE ( vw_scenarios[rain_label] ) ) & ": about "
        & FORMAT ( ROUND ( [Predicted trips], -2 ), "#,##0" ) & " trips per day"
)

-- ===== Overview page =====

Total trips = SUM ( vw_daily[trips] )
Days analysed = COUNTROWS ( vw_daily )
Rainy days share = DIVIDE ( COUNTROWS ( FILTER ( vw_daily, vw_daily[had_rain] ) ), [Days analysed] )
Avg trips/hour (daytime, rain) =
    CALCULATE ( AVERAGE ( vw_hourly[trips] ), vw_hourly[is_daytime] = TRUE (), vw_hourly[is_raining] = TRUE () )
Avg trips/hour (daytime, dry) =
    CALCULATE ( AVERAGE ( vw_hourly[trips] ), vw_hourly[is_daytime] = TRUE (), vw_hourly[is_raining] = FALSE () )
```

Format `Predicted trips`, `Predicted trips if dry`, `Average observed day` as whole numbers
with thousands separator, `Rain impact %` and `Rainy days share` as percentages.

## 3. Page 2: Simulator (the headline page)

Layout, from top left:

1. **Slicers** (left column, **single select** on; in the French interface the setting is
   called *Sélection simple*):
   - `vw_scenarios[temp_c]`: Style *Dropdown*; set it to 22
   - `vw_scenarios[rain_label]`: Style *Tile*; set it to *Light rain*
   - `vw_scenarios[day_name]`: Style *Dropdown* (or *Tile*); set it to *Wednesday*
   - Add a title "Build a scenario" above them
2. **Big card**: `Scenario headline` (this is the answer to the question)
3. **Three small cards**: `Predicted trips`, `Predicted trips if dry`, `Rain impact %`
4. **Line chart "A day hour by hour"**: X = `vw_scenarios[hour]`, Y = `Predicted trips`
   and `Predicted trips if dry` (the gap between the two lines is the rain effect)
5. **Card** with `Reliability message`, conditional font colour:
   Format → Font colour → *fx* → Rules based on field `MIN of is_reliable`:
   `= 0` → `#e76f51`, otherwise `#264653`
6. **Small card**: `Average observed day` with the caption "average real day, Apr to Aug 2026", as a reference point

Why the "dry" line: rainy hours are rare (8% of the data) and unevenly spread across
temperatures and months, so the *gap* between the two lines is easier to interpret than either absolute number.

Tip: tile slicers fill column by column. In a two-column layout the three rain levels can
look out of order (No rain, Heavy, Light) although the sort is correct. Make the slicer
narrower so the tiles stack in one column, or use the *List* style.

> Why a table of scenarios rather than a What-If parameter? DAX cannot run a
> scikit-learn model. `src/07_build_scenarios.py` scores the model on a grid
> (33 temperatures × 3 rain levels × 24 hours × 7 days) and Power BI only filters it.

## 4. Page 1: Overview

| Visual | Fields |
|---|---|
| Cards (4) | `Total trips`, `Days analysed`, `Rainy days share`, `Avg trips/hour (daytime, rain)` next to `Avg trips/hour (daytime, dry)` |
| Line chart "Trips per day" | X `vw_daily[date]`, Y `vw_daily[trips]` |
| Line chart "A typical day" | X `vw_trips_by_hour[hour]`, Y `avg_trips`, Legend `day_type` |
| Clustered column "Trips by temperature" | X `vw_daytime_by_temp_rain[temp_bin]`, Y `avg_trips`, Legend `rain_label` |
| Slicer | `vw_hourly[month_name]`, linked to the cards only |

Tip: in the temperature chart, add `hours_observed` to *Tooltips* so the reader sees how
few hours some rain bins contain.

## 5. Page 3: Model

| Visual | Fields |
|---|---|
| Table "Model comparison" | `model_results[model]`, `mae`, `rmse`, `r2`, `mae_pct_of_mean` |
| Bar chart "What does weather add?" | Y `ablation_results[feature_set]`, X `mae_mean` (sort ascending) |
| Image | `figures/5_actual_vs_predicted.png` and `figures/6_feature_importance.png` (Insert → Image) |

## 6. Publish

- Save as `powerbi/bixi_weather.pbix` (data is imported, so the file opens without the database).
- Export screenshots of the three pages to `docs/powerbi_overview.png`,
  `docs/powerbi_simulator.png`, `docs/powerbi_model.png` (File → Export → PDF, or Win + Shift + S).
- Power BI Service (free) cannot refresh from a local PostgreSQL without a gateway:
  for a portfolio, share the `.pbix` and the screenshots rather than a live link.
