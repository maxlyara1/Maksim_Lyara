import argparse
from pathlib import Path

import pandas as pd

from .constants import RANDOM_SEED
from .inference import _apply_ce_rerank
from .preprocessing import preprocess_dataframe
from .utils import detect_torch_device, set_seed


def add_rerank_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Добавляет аргументы командной строки для reranking."""
    parser.add_argument(
        "--catboost-preds",
        type=Path,
        default=Path("submission.csv"),
        help="Path to CatBoost predictions with columns: id,prediction",
    )
    return parser


def run_rerank(args: argparse.Namespace) -> None:
    """
    Применяет cross-encoder reranking к предсказаниям CatBoost.

    Args:
        args: Аргументы командной строки.
    """
    set_seed(RANDOM_SEED)

    test_df = pd.read_csv(args.test_path)
    test_df = preprocess_dataframe(test_df)

    cb_preds = pd.read_csv(args.catboost_preds)
    required_cols = {"id", "prediction"}
    if not required_cols.issubset(cb_preds.columns):
        raise ValueError(f"CatBoost predictions file must contain columns: {required_cols}")

    pred_map = cb_preds.set_index("id")["prediction"]
    prediction = test_df["id"].map(pred_map)
    if prediction.isna().any():
        missing = int(prediction.isna().sum())
        raise ValueError(f"Missing predictions for {missing} ids in {args.catboost_preds}")

    prediction_arr = prediction.to_numpy(dtype=float)
    ce_device = args.ce_device or detect_torch_device()
    prediction_arr = _apply_ce_rerank(
        test_df,
        prediction_arr,
        ce_model_dir=str(args.ce_model_dir),
        top_k=args.ce_rerank_top_k,
        weight=args.ce_rerank_weight,
        max_length=args.ce_rerank_max_length,
        batch_size=args.ce_rerank_batch_size,
        device=ce_device,
    )

    submission = pd.DataFrame({"id": test_df["id"], "prediction": prediction_arr})
    output_path = args.submission_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Saved reranked submission to {output_path}")
