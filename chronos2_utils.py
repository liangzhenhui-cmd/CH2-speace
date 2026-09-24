from datetime import date, timedelta
import json
import os
from pathlib import Path
import shutil
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


CSV_ENCODINGS = (
    "utf-8-sig",
    "utf-8",
    "gb18030",
    "gbk",
    "cp936",
    "big5",
    "utf-16",
)


def read_csv_compatible(path, *, required_columns=None, **kwargs):
    """Read a CSV using common Chinese/Unicode encodings and validate its schema."""
    path = Path(path)
    required_columns = set(required_columns or [])
    failures = []
    for encoding in CSV_ENCODINGS:
        try:
            frame = pd.read_csv(path, encoding=encoding, **kwargs)
        except (UnicodeError, pd.errors.ParserError) as error:
            failures.append(f"{encoding}: {error}")
            continue
        missing = required_columns - set(frame.columns)
        if missing:
            failures.append(f"{encoding}: missing columns {sorted(missing)}")
            continue
        return frame
    details = "; ".join(failures)
    raise ValueError(f"无法识别CSV编码或列结构: {path}; tried={details}")


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "chronos2_features.csv"
ROLES_PATH = ROOT / "data" / "feature_roles.json"
POWER_PATH = ROOT / "data" / "24点数据.csv"
WEATHER_PATH = ROOT / "data" / "weather_21cities_history.csv"
MODEL_PATH = ROOT / "models" / "chronos2_spread" / "lora-adapter"
OUTPUT_DIR = ROOT / "outputs"

DEFAULT_WEATHER_SOURCE = Path(
    r"D:\heyuan_predict\weather\weather\weather_21cities_history.csv"
)

CONTEXT_HOURS = int(os.environ.get("CONTEXT_HOURS", 28 * 24))
PREDICTION_HOURS = int(os.environ.get("PREDICTION_HOURS", 72))
TARGET_DAY_HOURS = 24
if CONTEXT_HOURS <= 0 or CONTEXT_HOURS % TARGET_DAY_HOURS:
    raise ValueError("CONTEXT_HOURS必须是24的正整数倍")
if PREDICTION_HOURS < TARGET_DAY_HOURS or PREDICTION_HOURS % TARGET_DAY_HOURS:
    raise ValueError("PREDICTION_HOURS必须是不小于24的24小时整数倍")
BASE_MODEL = "amazon/chronos-2"
BASE_MODEL_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
PROBABILITY_QUANTILES = [
    0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
    0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
    0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99,
]

WEATHER_CITIES = [
    ("广州", 23.1291, 113.2644), ("深圳", 22.5431, 114.0579),
    ("珠海", 22.2710, 113.5767), ("汕头", 23.3541, 116.6822),
    ("佛山", 23.0219, 113.1214), ("韶关", 24.8104, 113.5975),
    ("湛江", 21.2713, 110.3589), ("肇庆", 23.0470, 112.4653),
    ("江门", 22.5787, 113.0816), ("茂名", 21.6630, 110.9256),
    ("惠州", 23.1110, 114.4168), ("梅州", 24.2886, 116.1224),
    ("汕尾", 22.7872, 115.3739), ("河源", 23.7437, 114.6979),
    ("阳江", 21.8590, 111.9822), ("清远", 23.6819, 113.0562),
    ("东莞", 23.0208, 113.7518), ("中山", 22.5210, 113.3929),
    ("潮州", 23.6618, 116.6226), ("揭阳", 23.5496, 116.3727),
    ("云浮", 22.9158, 112.0445),
]

WEATHER_COLUMNS = {
    "temperature_2m": "气温2m(°C)",
    "apparent_temperature": "体感温度(°C)",
    "relative_humidity_2m": "湿度2m(%)",
    "precipitation": "降雨量(mm)",
    "wind_speed_10m": "风速10m(m/s)",
    "surface_pressure": "地表气压(hPa)",
    "dew_point_2m": "露点温度(°C)",
    "shortwave_radiation": "短波辐射(W/m²)",
    "cloud_cover_low": "低云量(%)",
    "weather_code": "天气现象代码",
    "cape": "CAPE(J/kg)",
}


def _weather_description(code):
    if pd.isna(code):
        return "未知"
    code = int(code)
    names = {
        0: "晴朗", **{value: "少云/多云" for value in range(1, 4)},
        **{value: "雾/霾" for value in range(45, 50)},
        **{value: "毛毛雨" for value in range(51, 56)}, 56: "冻雨", 57: "冻雨",
        **{value: "雨" for value in range(61, 66)}, 66: "冻雨(较强)", 67: "冻雨(较强)",
        **{value: "雪" for value in range(71, 78)},
        **{value: "阵雨" for value in range(80, 83)}, 85: "阵雪", 86: "阵雪",
        **{value: "雷暴/冰雹" for value in range(95, 100)},
    }
    return names.get(code, "其他")


def _enrich_weather(frame):
    temperature = frame["气温2m(°C)"].astype(float)
    humidity = frame["湿度2m(%)"].astype(float)
    pressure = frame["地表气压(hPa)"].astype(float)
    frame["湿球温度(°C)"] = np.round(
        temperature * np.arctan(0.151977 * np.sqrt(humidity + 8.313659))
        + np.arctan(temperature + humidity)
        - np.arctan(humidity - 1.676331)
        + 0.00391838 * humidity**1.5 * np.arctan(0.023101 * humidity)
        - 4.686035
        - (1013 - pressure) * 0.001,
        1,
    )
    codes = frame["天气现象代码"].fillna(-1).astype(int)
    frame["天气现象描述"] = codes.map(_weather_description)
    frame["强对流标记"] = np.where(
        codes.between(95, 99),
        "雷暴/冰雹",
        np.where(frame["CAPE(J/kg)"].fillna(0) > 1000, "强对流不稳定", "无"),
    )
    frame["数据源"] = "Forecast"
    return frame


def _sync_weather_source(source, destination):
    source = Path(source)
    destination = Path(destination)
    if not source.exists():
        raise FileNotFoundError(f"weather source not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
        return
    required = ["城市", "时间"]
    external = read_csv_compatible(source, required_columns=required)
    local = read_csv_compatible(destination, required_columns=required)
    merged = pd.concat([local, external], ignore_index=True)
    merged["时间"] = pd.to_datetime(merged["时间"])
    merged = merged.drop_duplicates(["城市", "时间"], keep="last")
    merged = merged.sort_values(["时间", "城市"]).reset_index(drop=True)
    merged.to_csv(destination, index=False, encoding="utf-8-sig")


def _fetch_weather(start_date, end_date):
    forecasts = []
    for city, latitude, longitude in WEATHER_CITIES:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(WEATHER_COLUMNS),
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "Asia/Shanghai",
        }
        request = Request(
            "https://api.open-meteo.com/v1/forecast?" + urlencode(params),
            headers={"User-Agent": "Mozilla/5.0 (compatible; Chronos2Updater/1.0)"},
        )
        last_error = None
        for attempt in range(3):
            try:
                with urlopen(request, timeout=120) as response:
                    hourly = json.load(response)["hourly"]
                break
            except Exception as error:
                last_error = error
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))
        else:
            raise RuntimeError(f"weather request failed for {city}: {last_error}")
        forecast = pd.DataFrame({"城市": city, "时间": pd.to_datetime(hourly["time"])})
        for source_name, target_name in WEATHER_COLUMNS.items():
            forecast[target_name] = hourly[source_name]
        forecasts.append(_enrich_weather(forecast))
    return pd.concat(forecasts, ignore_index=True)


def update_forecast(
    power_source=None,
    weather_source=DEFAULT_WEATHER_SOURCE,
    target_end=None,
    refresh_weather=True,
):
    """同步每日24点数据与21城天气，并补齐天气预报到target_end。"""
    weather_source = Path(weather_source)
    POWER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if power_source is not None:
        power_source = Path(power_source)
        if not power_source.exists():
            raise FileNotFoundError(f"power source not found: {power_source}")
        if power_source.resolve() != POWER_PATH.resolve():
            shutil.copy2(power_source, POWER_PATH)
    elif not POWER_PATH.exists():
        raise FileNotFoundError(
            f"请先把最新24点表复制到项目目录: {POWER_PATH}"
        )
    if weather_source.exists():
        _sync_weather_source(weather_source, WEATHER_PATH)
    elif not WEATHER_PATH.exists():
        raise FileNotFoundError(
            f"weather source not found: {weather_source}; "
            f"local weather file also missing: {WEATHER_PATH}"
        )

    weather = read_csv_compatible(
        WEATHER_PATH,
        required_columns=["城市", "时间", "数据源"],
    )
    weather["时间"] = pd.to_datetime(weather["时间"])
    if refresh_weather:
        target_end = pd.Timestamp(target_end or date.today() + timedelta(days=1)).date()
        non_forecast = weather.loc[weather["数据源"].ne("Forecast"), "时间"]
        start_base = non_forecast.max().date() + timedelta(days=1)
        counts = weather.drop_duplicates(["城市", "时间"]).groupby(weather["时间"].dt.date).size()
        dates = pd.date_range(start_base, target_end).date
        missing_dates = [day for day in dates if counts.get(day, 0) < len(WEATHER_CITIES) * 24]
        if missing_dates:
            start_date = missing_dates[0]
            forecast = _fetch_weather(start_date, target_end)
            successful_cities = set(forecast["城市"])
            replace = (
                (weather["时间"].dt.date >= start_date)
                & (weather["时间"].dt.date <= target_end)
                & weather["数据源"].eq("Forecast")
                & weather["城市"].isin(successful_cities)
            )
            weather = pd.concat([weather.loc[~replace], forecast], ignore_index=True)
            weather = weather.sort_values(["时间", "城市"]).reset_index(drop=True)
            weather.to_csv(WEATHER_PATH, index=False, encoding="utf-8-sig")

    return {
        "power_path": str(POWER_PATH),
        "weather_path": str(WEATHER_PATH),
        "power_bytes": POWER_PATH.stat().st_size,
        "weather_rows": len(weather),
        "weather_end": str(weather["时间"].max()),
    }


def load_roles():
    roles = json.loads(ROLES_PATH.read_text(encoding="utf-8"))
    return roles["known_future_covariates"], roles["past_only_covariates"]


def load_data():
    data = read_csv_compatible(
        DATA_PATH,
        required_columns=["timestamp"],
        parse_dates=["timestamp"],
    )
    data = data.sort_values("timestamp").reset_index(drop=True)
    assert not data["timestamp"].duplicated().any()
    assert data["timestamp"].diff().dropna().eq(pd.Timedelta(hours=1)).all()
    return data


def split_daily_window(data, target_date, known, past_only):
    target_date = pd.Timestamp(target_date).normalize()
    forecast_start = target_date - pd.Timedelta(
        hours=PREDICTION_HOURS - TARGET_DAY_HOURS
    )
    history_start = forecast_start - pd.Timedelta(hours=CONTEXT_HOURS)
    history = data.loc[
        (data.timestamp >= history_start)
        & (data.timestamp < forecast_start),
        ["timestamp", "spread", *known, *past_only],
    ].copy()
    future = data.loc[
        (data.timestamp >= forecast_start)
        & (data.timestamp < target_date + pd.Timedelta(hours=TARGET_DAY_HOURS)),
        ["timestamp", *known],
    ].copy()
    if len(history) != CONTEXT_HOURS:
        raise ValueError(f"{target_date.date()} history rows={len(history)}, expected {CONTEXT_HOURS}")
    if len(future) != PREDICTION_HOURS:
        raise ValueError(f"{target_date.date()} future rows={len(future)}, expected {PREDICTION_HOURS}")
    required_history = ["spread", *known, *past_only]
    if history[required_history].isna().any().any():
        missing = history[required_history].isna().sum()
        raise ValueError(f"{target_date.date()} missing history: {missing[missing.gt(0)].to_dict()}")
    if future[known].isna().any().any():
        missing = future[known].isna().sum()
        raise ValueError(f"{target_date.date()} missing future: {missing[missing.gt(0)].to_dict()}")
    item_id = target_date.strftime("%Y%m%d")
    history["item_id"] = item_id
    future["item_id"] = item_id
    history = history[["timestamp", "item_id", "spread", *known, *past_only]]
    future = future[["timestamp", "item_id", *known]]
    return history, future


def build_prediction_frames(data, target_dates, known, past_only):
    histories, futures = [], []
    for target_date in pd.to_datetime(target_dates):
        history, future = split_daily_window(data, target_date, known, past_only)
        histories.append(history)
        futures.append(future)
    return pd.concat(histories, ignore_index=True), pd.concat(futures, ignore_index=True)


def target_day_only(predictions):
    target_dates = pd.to_datetime(predictions["item_id"], format="%Y%m%d")
    return predictions.loc[predictions.timestamp.dt.normalize().eq(target_dates)].copy()


def add_probability_columns(predictions):
    """从密集分位数插值得到负价差概率；只作展示，不改变q50策略。"""
    columns = [str(level) for level in PROBABILITY_QUANTILES]
    values = predictions[columns].to_numpy(dtype=float)
    values = np.maximum.accumulate(values, axis=1)
    probabilities = np.asarray([
        np.interp(0.0, row, PROBABILITY_QUANTILES, left=0.0, right=1.0)
        for row in values
    ])
    predictions["negative_probability"] = probabilities
    predictions["direction_confidence"] = np.maximum(probabilities, 1 - probabilities)
    return predictions
