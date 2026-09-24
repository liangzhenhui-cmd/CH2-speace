from pathlib import Path
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline

from chronos2_utils import (
    BASE_MODEL,
    CONTEXT_HOURS,
    MODEL_PATH,
    OUTPUT_DIR,
    PREDICTION_HOURS,
    PROBABILITY_QUANTILES,
    add_probability_columns,
    build_prediction_frames,
    load_data,
    load_roles,
    target_day_only,
)

BEST_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "chronos2_spread"
    / "lora-recent180-w3-lr12e6-r16-a16-b136-s450"
    / "lora-adapter"
)


def resolve_model(model):
    if model == "zero-shot":
        return BASE_MODEL, "zero_shot"
    if model == "lora":
        return MODEL_PATH, "lora_step30"
    if model == "best":
        if not BEST_MODEL_PATH.exists():
            raise FileNotFoundError(f"Best model not found: {BEST_MODEL_PATH}")
        return BEST_MODEL_PATH, "best"
    return Path(model), Path(model).name


def build_simple_output(result):
    return pd.DataFrame({
        "时间": result["forecast_time"],
        "策略": result["strategy"].map(lambda value: f"{value:.0%}"),
        "负价差概率": result["negative_probability"].map(lambda value: f"{value:.2%}"),
    })


def save_csv(frame, output):
    try:
        frame.to_csv(output, index=False, encoding="utf-8-sig")
        return output
    except PermissionError:
        fallback = output.with_name(
            f"{output.stem}_{datetime.now():%H%M%S}{output.suffix}"
        )
        frame.to_csv(fallback, index=False, encoding="utf-8-sig")
        return fallback


def predict(target_date, model="zero-shot", save=True, device="cpu"):
    torch.set_num_threads(8)
    data = load_data()
    known, past_only = load_roles()
    history, future = build_prediction_frames(data, [target_date], known, past_only)
    checkpoint, model_name = resolve_model(model)
    pipeline = Chronos2Pipeline.from_pretrained(str(checkpoint), device_map=device)
    prediction = pipeline.predict_df(
        history,
        future_df=future,
        target="spread",
        prediction_length=PREDICTION_HOURS,
        context_length=CONTEXT_HOURS,
        quantile_levels=PROBABILITY_QUANTILES,
        batch_size=1 + len(known) + len(past_only),
        cross_learning=False,
    )
    result = add_probability_columns(target_day_only(prediction))
    result = result.rename(columns={
        "predictions": "predicted_spread",
        "0.1": "predicted_q10",
        "0.5": "predicted_q50",
        "0.9": "predicted_q90",
    })
    actual = data[["timestamp", "spread"]].rename(columns={"spread": "actual_spread"})
    result = result.merge(actual, on="timestamp", how="left", validate="one_to_one")
    result["forecast_date"] = result.timestamp.dt.strftime("%Y-%m-%d")
    # Text identifier for Excel: keeps every hourly timestamp visible instead
    # of displaying #### when a date-formatted column is too narrow.
    result["forecast_time"] = result["timestamp"].map(
        lambda value: (
            f"{value.year}/{value.month}/{value.day} "
            f"{value.hour}:{value.minute:02d}:{value.second:02d}"
        )
    )
    result["decision_time"] = result.timestamp.dt.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=12)
    result["predicted_direction"] = np.where(
        result.predicted_spread < 0, "实时高于日前", "实时不高于日前"
    )
    result["strategy"] = np.where(result.predicted_spread < 0, 1.2, 0.8)
    result["model"] = model_name
    columns = [
        "forecast_time", "strategy", "negative_probability", "direction_confidence",
        "predicted_direction", "predicted_spread", "predicted_q10", "predicted_q50",
        "predicted_q90", "forecast_date", "decision_time", "timestamp",
        "actual_spread", "model",
    ]
    result = result[columns]
    if len(result) != 24:
        raise AssertionError(f"target-day rows={len(result)}, expected 24")
    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_prefix = f"{pd.Timestamp(target_date):%Y-%m-%d}-ch2策略"
        output = OUTPUT_DIR / f"{output_prefix}.csv"
        simple_output = OUTPUT_DIR / f"{output_prefix}_simple.csv"
        saved_simple_output = save_csv(build_simple_output(result), simple_output)
        saved_output = save_csv(result, output)
        print(f"saved {saved_output}")
        print(f"saved {saved_simple_output}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_date")
    parser.add_argument(
        "--model",
        default="zero-shot",
        help="zero-shot, lora, best, or checkpoint path",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    args = parser.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )
    result = predict(args.target_date, args.model, device=device)
    print(build_simple_output(result).to_string(index=False))
