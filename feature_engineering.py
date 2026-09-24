import argparse
import csv
import json

import numpy as np
import pandas as pd

from chronos2_utils import (
    CSV_ENCODINGS,
    DEFAULT_WEATHER_SOURCE,
    POWER_PATH,
    WEATHER_PATH,
    read_csv_compatible,
    update_forecast,
)


OUTPUT_PATH = POWER_PATH.parent / "chronos2_features.csv"
ROLES_PATH = POWER_PATH.parent / "feature_roles.json"
START = pd.Timestamp("2025-06-28")

POWER_NAMES = [
    "source_no", "source_factor", "source_temperature", "source_apparent_temperature",
    "source_humidity", "source_rain", "source_rain_probability", "source_low_cloud",
    "source_wind", "source_central_radiation", "source_distributed_radiation",
    "source_weather_text", "source_wind_direction", "source_uv", "source_weekday",
    "source_day_type", "source_date", "source_time", "da_price", "rt_price", "spread",
    "source_strategy", "source_strategy_result", "load_forecast", "a_generation_forecast",
    "b_generation_forecast", "local_generation_forecast", "west_import_forecast",
    "gdhkmo_forecast", "total_generation_forecast", "market_renewable_forecast",
    "solar_forecast", "wind_forecast", "hydro_pumped_forecast", "pumped_storage_forecast",
    "positive_reserve_forecast", "negative_reserve_forecast", "primary_reserve_forecast",
    "maintenance_forecast", "congestion_count_forecast", "must_run_capacity",
    "must_stop_capacity", "load_actual", "a_generation_actual", "b_generation_actual",
    "local_generation_actual", "west_import_actual", "gdhk_link_actual", "renewable_actual",
    "hydro_pumped_actual", "source_load_deviation", "source_b_deviation",
    "source_renewable_deviation", "source_combined_deviation", "source_strategy_revenue",
]

WEATHER_FEATURES = {
    "气温2m(°C)": "wx_gd_mean_temperature_c",
    "体感温度(°C)": "wx_gd_mean_apparent_temperature_c",
    "湿度2m(%)": "wx_gd_mean_humidity_pct",
    "降雨量(mm)": "wx_gd_mean_rain_mm",
    "风速10m(m/s)": "wx_gd_mean_wind_speed_raw",
    "地表气压(hPa)": "wx_gd_mean_pressure_hpa",
    "露点温度(°C)": "wx_gd_mean_dewpoint_c",
    "短波辐射(W/m²)": "wx_gd_mean_shortwave_wm2",
    "低云量(%)": "wx_gd_mean_low_cloud_pct",
    "湿球温度(°C)": "wx_gd_mean_wetbulb_c",
}


def load_power():
    failures = []
    rows = None
    for encoding in CSV_ENCODINGS:
        try:
            with POWER_PATH.open(encoding=encoding, newline="") as source:
                candidate = list(csv.reader(source))
        except UnicodeError as error:
            failures.append(f"{encoding}: {error}")
            continue
        if len(candidate) < 4:
            failures.append(f"{encoding}: 文件不足4行")
            continue
        if len(candidate[3]) != len(POWER_NAMES):
            failures.append(
                f"{encoding}: 24点数据列数={len(candidate[3])}, expected={len(POWER_NAMES)}"
            )
            continue
        rows = candidate
        break
    if rows is None:
        raise ValueError(
            f"无法识别24点数据的编码或列结构: {POWER_PATH}; tried={'; '.join(failures)}"
        )
    power = pd.DataFrame(rows[4:], columns=POWER_NAMES)
    power = power.map(
        lambda value: (
            np.nan if isinstance(value, str) and not value.strip()
            else value.strip() if isinstance(value, str)
            else value
        )
    ).infer_objects(copy=False)
    power = power.loc[power.source_date.notna()].copy()
    power["timestamp"] = pd.to_datetime(
        power.source_date + " " + power.source_time,
        format="%Y/%m/%d %H:%M",
        errors="raise",
    )
    numeric = ["source_factor", *POWER_NAMES[18:21], *POWER_NAMES[23:]]
    for column in numeric:
        power[column] = pd.to_numeric(power[column], errors="raise")
    power = power.sort_values("timestamp").set_index("timestamp")
    if power.index.duplicated().any():
        raise ValueError("24点数据存在重复时间")
    return power


def load_weather():
    weather = read_csv_compatible(
        WEATHER_PATH,
        required_columns=["城市", "时间", *WEATHER_FEATURES],
    )
    weather["时间"] = pd.to_datetime(weather["时间"], errors="raise")
    if weather.duplicated(["城市", "时间"]).any():
        raise ValueError("天气表存在重复城市/时间")
    city_counts = weather.groupby("时间")["城市"].nunique()
    if not city_counts.eq(21).all():
        bad = city_counts[city_counts.ne(21)].head().to_dict()
        raise ValueError(f"天气表存在非21城完整时点: {bad}")
    for column in WEATHER_FEATURES:
        weather[column] = pd.to_numeric(weather[column], errors="raise")
    return weather


def build_features():
    power_full = load_power()
    weather = load_weather()
    forecast_columns = [
        column for column in POWER_NAMES[23:42]
        if column not in {"pumped_storage_forecast", "congestion_count_forecast"}
    ]
    required_power = ["source_day_type", *forecast_columns]
    valid_power = power_full[required_power].notna().all(axis=1)
    power_end = power_full.index[valid_power].max()
    weather_end = weather["时间"].max()
    end = min(power_end, weather_end)
    index = pd.date_range(START, end, freq="h")
    power = power_full.reindex(index)
    if power[required_power].isna().any().any():
        missing = power[required_power].isna().sum()
        raise ValueError(f"电力已知特征存在中间缺口: {missing[missing.gt(0)].to_dict()}")

    base = pd.DataFrame(index=index)
    base["hour"] = index.hour
    base["weekday"] = index.dayofweek
    base["month"] = index.month
    base["hour_sin"] = np.sin(2 * np.pi * index.hour / 24)
    base["hour_cos"] = np.cos(2 * np.pi * index.hour / 24)
    base["weekday_sin"] = np.sin(2 * np.pi * index.dayofweek / 7)
    base["weekday_cos"] = np.cos(2 * np.pi * index.dayofweek / 7)
    day_types = {
        "工作日": "workday", "周六": "saturday", "周日": "sunday",
        "调休节假日": "adjusted_holiday", "法定节假日": "statutory_holiday",
    }
    unknown_types = set(power.source_day_type.dropna()) - set(day_types)
    if unknown_types:
        raise ValueError(f"未知日期类型: {sorted(unknown_types)}")
    for source_name, target_name in day_types.items():
        base["day_type_" + target_name] = power.source_day_type.eq(source_name).astype(int)
    for column in forecast_columns:
        base[column] = power[column]
    base["net_load_forecast"] = base.load_forecast - base.solar_forecast - base.wind_forecast
    base["load_ramp_next_hour"] = (
        base.load_forecast.groupby(base.index.normalize()).shift(-1) - base.load_forecast
    )
    base["load_ramp_next_hour_valid"] = (base.index.hour != 23).astype(int)
    base["net_load_ramp_next_hour"] = (
        base.net_load_forecast.groupby(base.index.normalize()).shift(-1) - base.net_load_forecast
    )
    base.loc[base.index.hour == 23, ["load_ramp_next_hour", "net_load_ramp_next_hour"]] = 0

    weather_mean = (
        weather.groupby("时间", as_index=True)[list(WEATHER_FEATURES)]
        .mean()
        .rename(columns=WEATHER_FEATURES)
        .reindex(index)
    )
    if weather_mean.isna().any().any():
        missing = weather_mean.isna().sum()
        raise ValueError(f"天气均值存在缺口: {missing[missing.gt(0)].to_dict()}")
    base = base.join(weather_mean)
    # 原项目2的基础表以8位小数落盘；保留该舍入才能逐值复现。
    base = base.round(8)

    strict = base.copy()
    actual_columns = [
        "da_price", "rt_price", "load_actual", "a_generation_actual",
        "b_generation_actual", "local_generation_actual", "west_import_actual",
        "gdhk_link_actual",
    ]
    past_only = []
    for column in actual_columns:
        name = "history_" + column
        strict[name] = power[column]
        past_only.append(name)

    base_known = list(base.columns)
    known = list(base_known)
    for column in [
        "da_price", "rt_price", "spread", "load_actual",
        "a_generation_actual", "b_generation_actual", "west_import_actual",
    ]:
        name = column + "_d3_same_hour"
        strict[name] = power_full[column].shift(72).reindex(index)
        known.append(name)
    for column in ["da_price", "rt_price", "spread", "load_actual"]:
        name = column + "_d3_daily_mean"
        daily = power_full[column].resample("D").mean().shift(3)
        strict[name] = strict.index.normalize().map(daily)
        known.append(name)
    shifted = power_full.spread.shift(72)
    by_hour = shifted.groupby(shifted.index.hour)
    for statistic in ["mean", "std"]:
        name = "spread_same_hour_7days_ending_d3_" + statistic
        strict[name] = by_hour.transform(
            lambda series: getattr(series.rolling(7, min_periods=7), statistic)()
        )
        known.append(name)
    daily_std = power_full.spread.resample("D").std().shift(3)
    strict["spread_d3_daily_std"] = strict.index.normalize().map(daily_std)
    strict["load_forecast_minus_d3_actual"] = (
        strict.load_forecast - power_full.load_actual.shift(72).reindex(index)
    )
    strict["load_error_actual_minus_forecast_d3"] = (
        power_full.load_actual - power_full.load_forecast
    ).shift(72).reindex(index)
    known += [
        "spread_d3_daily_std",
        "load_forecast_minus_d3_actual",
        "load_error_actual_minus_forecast_d3",
    ]

    strict.insert(0, "spread", power.spread)
    strict.insert(0, "item_id", "GD")
    strict.index.name = "timestamp"
    strict = strict.reset_index()
    derived_known = known[len(base_known):]
    # 保留项目2原CSV列顺序；训练时仍按feature_roles中的known/past角色取列。
    strict = strict[
        ["timestamp", "item_id", "spread", *base_known, *past_only, *derived_known]
    ]
    if len(known) != 59 or len(past_only) != 8:
        raise AssertionError(f"feature roles changed: known={len(known)}, past={len(past_only)}")
    if strict.timestamp.diff().dropna().ne(pd.Timedelta(hours=1)).any():
        raise AssertionError("总表时间不连续")
    if strict[known].isna().any().any():
        missing = strict[known].isna().sum()
        raise ValueError(f"未来可用特征存在缺失: {missing[missing.gt(0)].to_dict()}")

    strict.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    ROLES_PATH.write_text(json.dumps({
        "decision_time": "D-1 12:00",
        "history_end": "D-3 23:00",
        "forecast_horizon_hours": 72,
        "evaluated_horizon": "last 24 hours (D day)",
        "target": "spread = day-ahead price - real-time price",
        "known_future_covariates": known,
        "past_only_covariates": past_only,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return strict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="更新每日源表并生成项目2严格D-3总特征表")
    parser.add_argument("--update", action="store_true", help="先同步24点与天气，并补天气预报")
    parser.add_argument(
        "--power-source",
        default=None,
        help="可选；指定时复制到项目data目录。默认直接使用你手工放入的data/24点数据.csv",
    )
    parser.add_argument("--weather-source", default=str(DEFAULT_WEATHER_SOURCE))
    parser.add_argument("--target-end", default=None, help="天气补齐日期，YYYY-MM-DD；默认明天")
    parser.add_argument("--no-weather-api", action="store_true", help="只同步源文件，不请求天气API")
    args = parser.parse_args()
    if args.update:
        print(json.dumps(update_forecast(
            power_source=args.power_source,
            weather_source=args.weather_source,
            target_end=args.target_end,
            refresh_weather=not args.no_weather_api,
        ), ensure_ascii=False, indent=2))
    result = build_features()
    print(
        f"saved rows={len(result)}, columns={len(result.columns)}, "
        f"range={result.timestamp.min()}..{result.timestamp.max()} to {OUTPUT_PATH}"
    )
