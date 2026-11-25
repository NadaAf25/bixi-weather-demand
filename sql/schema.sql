-- schema.sql
-- PostgreSQL schema for the BIXI x weather project.
-- Applied by src/08_load_postgres.py (safe to re-run: everything is dropped first).

DROP SCHEMA IF EXISTS bixi CASCADE;
CREATE SCHEMA bixi;
SET search_path TO bixi;

-- ---------------------------------------------------------------------------
-- Staging tables: one per pipeline output, loaded as-is from data/processed/
-- ---------------------------------------------------------------------------

-- Hourly observations, Montreal-Trudeau airport (ECCC). ~3,670 rows.
CREATE TABLE weather_hourly (
    hour_local         TIMESTAMP PRIMARY KEY,      -- Montreal wall-clock time
    ts_utc             TIMESTAMPTZ NOT NULL UNIQUE,
    source_id          TEXT NOT NULL,              -- the API's own observation ID
    temp               NUMERIC(4,1),               -- deg C
    dew_point_temp     NUMERIC(4,1),
    humidex            NUMERIC(4,1),
    windchill          NUMERIC(4,1),
    precip_amount      NUMERIC(5,1),               -- mm in the hour
    relative_humidity  SMALLINT CHECK (relative_humidity BETWEEN 0 AND 100),
    station_pressure   NUMERIC(5,2),               -- kPa
    visibility         NUMERIC(5,1),               -- km
    wind_speed         SMALLINT CHECK (wind_speed >= 0),   -- km/h
    wind_direction     SMALLINT CHECK (wind_direction BETWEEN 0 AND 360),
    weather_desc       TEXT
);

-- BIXI trips counted per hour (hours without any trip are absent). ~5,830 rows.
CREATE TABLE bixi_hourly (
    hour_local  TIMESTAMP PRIMARY KEY,
    trips       INTEGER NOT NULL CHECK (trips >= 0)
);

-- ---------------------------------------------------------------------------
-- Modelling table: weather + trips + calendar features (one row per hour)
-- ---------------------------------------------------------------------------
CREATE TABLE features_hourly (
    hour_local         TIMESTAMP PRIMARY KEY,
    temp               NUMERIC(4,1) NOT NULL,
    dew_point_temp     NUMERIC(4,1) NOT NULL,
    precip_amount      NUMERIC(5,1) NOT NULL CHECK (precip_amount >= 0),
    relative_humidity  SMALLINT     NOT NULL,
    station_pressure   NUMERIC(5,2) NOT NULL,
    wind_speed         SMALLINT     NOT NULL,
    wind_direction     SMALLINT     NOT NULL,
    trips              INTEGER      NOT NULL CHECK (trips >= 0),
    hour               SMALLINT     NOT NULL CHECK (hour BETWEEN 0 AND 23),
    day_of_week        SMALLINT     NOT NULL CHECK (day_of_week BETWEEN 0 AND 6),  -- 0 = Monday
    month              SMALLINT     NOT NULL CHECK (month BETWEEN 1 AND 12),
    is_weekend         SMALLINT     NOT NULL CHECK (is_weekend IN (0, 1)),
    is_holiday         SMALLINT     NOT NULL CHECK (is_holiday IN (0, 1))
);
CREATE INDEX ix_features_month ON features_hourly (month);

-- ---------------------------------------------------------------------------
-- Model outputs
-- ---------------------------------------------------------------------------
CREATE TABLE model_results (            -- 05_train_model.py (test = August)
    model          TEXT PRIMARY KEY,
    mae            NUMERIC(8,1) NOT NULL,
    rmse           NUMERIC(8,1) NOT NULL,
    r2             NUMERIC(6,3) NOT NULL,
    mae_pct_of_mean NUMERIC(5,1) NOT NULL
);

CREATE TABLE ablation_results (         -- 06_weather_ablation.py
    feature_set                    TEXT PRIMARY KEY,
    mae_july                       NUMERIC(8,1) NOT NULL,
    mae_august                     NUMERIC(8,1) NOT NULL,
    mae_mean                       NUMERIC(8,1) NOT NULL,
    error_reduction_vs_calendar_pct NUMERIC(5,1) NOT NULL
);

-- What-if grid scored by the model (07_build_scenarios.py), with a 90%
-- prediction interval added by 11_scenario_intervals.py. 16,632 rows.
CREATE TABLE scenario_predictions (
    temp_c           SMALLINT NOT NULL,
    rain_label       TEXT     NOT NULL,
    precip_mm        NUMERIC(3,1) NOT NULL,
    hour             SMALLINT NOT NULL CHECK (hour BETWEEN 0 AND 23),
    day_of_week      SMALLINT NOT NULL CHECK (day_of_week BETWEEN 0 AND 6),
    day_name         TEXT     NOT NULL,
    is_weekend       SMALLINT NOT NULL,
    predicted_trips  INTEGER  NOT NULL CHECK (predicted_trips >= 0),
    n_similar_hours  INTEGER  NOT NULL,     -- real hours that resemble this scenario
    is_reliable      SMALLINT NOT NULL CHECK (is_reliable IN (0, 1)),
    lower_trips      INTEGER  NOT NULL CHECK (lower_trips >= 0),   -- 90% interval, low bound
    upper_trips      INTEGER  NOT NULL CHECK (upper_trips >= 0),   -- 90% interval, high bound
    PRIMARY KEY (temp_c, rain_label, hour, day_of_week)
);

-- ---------------------------------------------------------------------------
-- Reporting views: what Power BI connects to (friendly names, no raw codes)
-- ---------------------------------------------------------------------------

-- One row per hour with readable labels and a rain flag
CREATE VIEW vw_hourly AS
SELECT
    f.hour_local,
    f.hour_local::date                                   AS date,
    f.hour,
    f.day_of_week,
    (ARRAY['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
        [f.day_of_week + 1]                              AS day_name,
    TO_CHAR(f.hour_local, 'Mon')                         AS month_name,
    f.month,
    CASE WHEN f.is_weekend = 1 THEN 'Weekend' ELSE 'Weekday' END AS day_type,
    f.is_holiday = 1                                     AS is_holiday,
    f.temp,
    f.precip_amount,
    f.precip_amount > 0                                  AS is_raining,
    CASE WHEN f.precip_amount > 0 THEN 'Rain' ELSE 'No rain' END AS rain_label,
    f.relative_humidity,
    f.wind_speed,
    f.trips,
    (f.hour BETWEEN 7 AND 21)                            AS is_daytime,
    -- 5 degree bins, sorted by lower bound: (-5,0], (0,5], ...
    (CEIL(f.temp / 5.0) * 5 - 5)::int                    AS temp_bin_low
FROM features_hourly f;

-- Daily totals with the day's weather
CREATE VIEW vw_daily AS
SELECT
    hour_local::date                  AS date,
    SUM(trips)                        AS trips,
    ROUND(AVG(temp), 1)               AS avg_temp,
    ROUND(SUM(precip_amount), 1)      AS total_precip_mm,
    BOOL_OR(precip_amount > 0)        AS had_rain
FROM features_hourly
GROUP BY hour_local::date;

-- Average trips per hour for each hour of day x day type (chart 1)
CREATE VIEW vw_trips_by_hour AS
SELECT hour, day_type, ROUND(AVG(trips)) AS avg_trips
FROM vw_hourly
GROUP BY hour, day_type;

-- Daytime trips by temperature bin and rain (charts 2 and 3)
CREATE VIEW vw_daytime_by_temp_rain AS
SELECT
    temp_bin_low,
    temp_bin_low || ' to ' || (temp_bin_low + 5) || ' °C' AS temp_bin,
    rain_label,
    COUNT(*)                AS hours_observed,
    ROUND(AVG(trips))       AS avg_trips
FROM vw_hourly
WHERE is_daytime
GROUP BY temp_bin_low, rain_label;

-- Simulator table with the same naming as the rest of the model
CREATE VIEW vw_scenarios AS
SELECT
    temp_c,
    rain_label,
    CASE rain_label
        WHEN 'No rain' THEN 1
        WHEN 'Light rain (1 mm/h)' THEN 2
        ELSE 3
    END                                                         AS rain_order,
    hour,
    day_of_week,
    day_name,
    CASE WHEN is_weekend = 1 THEN 'Weekend' ELSE 'Weekday' END AS day_type,
    predicted_trips,
    lower_trips,
    upper_trips,
    n_similar_hours,
    is_reliable                                                 -- 1 / 0 (easy to use in DAX)
FROM scenario_predictions;