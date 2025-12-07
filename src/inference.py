import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from sentence_transformers import CrossEncoder

from .constants import CATEGORICAL_COLUMNS, EMBEDDING_BATCH_SIZE, MODEL_NAME, RANDOM_SEED
from .features import (
    BM25FeatureGenerator,
    CrossEncoderFeatureGenerator,
    EmbeddingFeatureGenerator,
    build_brand_features,
    build_color_features,
    build_coverage_features,
    build_simple_features,
    apply_target_encoding,
    load_metadata,
    prepare_categorical,
)
from .preprocessing import preprocess_dataframe
from .utils import detect_catboost_task_type, detect_torch_device, set_seed


def _build_ce_text(row: pd.Series, max_chars: int = 512) -> str:
    """Собирает текст продукта для cross-encoder из полей продукта."""
    def as_str(val: object) -> str:
        return val if isinstance(val, str) else "" if val is None else str(val)

    parts = []
    for key in ["product_title", "product_description", "product_bullet_point", "product_brand", "product_color"]:
        val = as_str(row.get(key, ""))
        if val:
            parts.append(val)
    combined = " [SEP] ".join(parts)
    if len(combined) > max_chars:
        combined = combined[:max_chars]
    return combined


def _apply_ce_rerank(
    df: pd.DataFrame,
    prediction: np.ndarray,
    ce_model_dir: str,
    top_k: int,
    weight: float,
    max_length: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """
    Применяет cross-encoder reranking к топ-k предсказаниям.

    Args:
        df: DataFrame с данными.
        prediction: Исходные предсказания.
        ce_model_dir: Путь к директории с cross-encoder моделью.
        top_k: Количество топ элементов для reranking.
        weight: Вес для смешивания скоров (1.0 заменяет полностью, 0.0 не применяет).
        max_length: Максимальная длина последовательности для CE.
        batch_size: Размер батча для CE.
        device: Устройство для CE модели.

    Returns:
        Обновленные предсказания после reranking.
    """
    if top_k <= 0 or weight == 0.0:
        return prediction
    ce_path = Path(ce_model_dir)
    if not ce_path.exists():
        print(f"CE rerank skipped: model not found at {ce_path}")
        return prediction
    model = CrossEncoder(ce_model_dir, device=device, max_length=max_length)
    model.tokenizer.model_max_length = 100000
    model.max_seq_length = max_length

    pairs: List[Tuple[str, str]] = []
    indices: List[int] = []
    for _, group in df.groupby("query_id"):
        group_idx = group.index.to_numpy()
        group_pred = prediction[group_idx]
        order = np.argsort(-group_pred)
        top_idx = group_idx[order[:top_k]]
        top_rows = df.loc[top_idx]
        pairs.extend([(row["query"], _build_ce_text(row, max_chars=max_length * 3)) for _, row in top_rows.iterrows()])
        indices.extend(top_idx.tolist())

    ce_scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    ce_scores = np.asarray(ce_scores, dtype=float)
    indices_arr = np.asarray(indices, dtype=int)
    if weight >= 1.0:
        prediction[indices_arr] = ce_scores
    elif weight <= 0.0:
        return prediction
    else:
        prediction[indices_arr] = (1.0 - weight) * prediction[indices_arr] + weight * ce_scores
    return prediction


def build_feature_matrix(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """
    Строит матрицу признаков для инференса.

    Args:
        df: DataFrame с данными.
        args: Аргументы командной строки.

    Returns:
        DataFrame с признаками.
    """
    bm25 = BM25FeatureGenerator(k1=args.bm25_k1, b=args.bm25_b)
    bm25.fit(df)
    bm25_features = bm25.transform(df)

    embedder = EmbeddingFeatureGenerator(
        model_name=args.embed_model,
        query_instruction=args.embed_query_instruction,
        batch_size=args.embed_batch_size,
        show_progress_bar=not args.no_embed_progress,
        cache_dir=getattr(args, "feature_cache_dir", None),
    )
    embedding_features = embedder.transform(df)

    cross_encoder = CrossEncoderFeatureGenerator(
        batch_size=16,
        show_progress_bar=not args.no_embed_progress,
        cache_dir=getattr(args, "feature_cache_dir", None),
    )
    cross_encoder_features = cross_encoder.transform(df)

    simple_features = build_simple_features(df)
    brand_features = build_brand_features(df)
    color_features = build_color_features(df)
    coverage_features = build_coverage_features(df)

    cat_ready = prepare_categorical(df)
    cat_blocks = [cat_ready[col] for col in CATEGORICAL_COLUMNS if col in cat_ready.columns]

    feature_blocks = [
        bm25_features,
        embedding_features,
        cross_encoder_features,
        simple_features,
        brand_features,
        color_features,
        coverage_features,
    ]
    if cat_blocks:
        feature_blocks.append(pd.concat(cat_blocks, axis=1))

    feature_matrix = pd.concat(feature_blocks, axis=1)
    return feature_matrix


def parse_args() -> argparse.Namespace:
    """Парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Inference pipeline for product ranking.")
    add_inference_arguments(parser)
    return parser.parse_args()


def add_inference_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Добавляет аргументы командной строки для инференса."""
    parser.add_argument("--test-path", type=Path, default=Path("data/test.csv"), help="Path to test.csv")
    parser.add_argument("--models-dir", type=Path, default=Path("models"), help="Directory with trained models")
    parser.add_argument(
        "--submission-path",
        type=Path,
        default=Path("results/submission.csv"),
        help="Output submission path (defaults to results/submission.csv)",
    )
    parser.add_argument("--bm25-k1", type=float, default=1.5, help="BM25 k1 parameter")
    parser.add_argument("--bm25-b", type=float, default=0.75, help="BM25 b parameter")
    parser.add_argument("--embed-model", type=str, default=MODEL_NAME, help="Embedding model name")
    parser.add_argument(
        "--embed-query-instruction",
        type=str,
        default=None,
        help="Instruction template for query embeddings (e.g. 'Instruct: ... {query}')",
    )
    parser.add_argument("--embed-batch-size", type=int, default=EMBEDDING_BATCH_SIZE, help="Embedding batch size")
    parser.add_argument("--no-embed-progress", action="store_true", help="Disable embedding progress bars")
    parser.add_argument(
        "--ce-blend-weight",
        type=float,
        default=0.2,
        help="Weight to blend CatBoost predictions with mean cross-encoder score (set to 0 to disable)",
    )
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=Path("cache"),
        help="Optional cache directory for embeddings and cross-encoder scores",
    )
    parser.add_argument("--ce-model-dir", type=Path, default=Path("models_ce_minilm"), help="Path to CE model dir (for rerank).")
    parser.add_argument("--ce-rerank-top-k", type=int, default=250, help="Top-k per query for CE rerank.")
    parser.add_argument(
        "--ce-rerank-weight",
        type=float,
        default=0.9,
        help="Blend weight for CE rerank (1.0 replaces CatBoost scores in top-k).",
    )
    parser.add_argument("--ce-rerank-max-length", type=int, default=160, help="Max sequence length for CE rerank model.")
    parser.add_argument("--ce-rerank-batch-size", type=int, default=16, help="Batch size for CE rerank.")
    parser.add_argument("--ce-device", type=str, default=None, help="Device for CE rerank (cpu/cuda/mps); auto if unset.")
    return parser


def run_inference(args: argparse.Namespace) -> None:
    """
    Запускает инференс на тестовых данных.

    Args:
        args: Аргументы командной строки.
    """
    set_seed(RANDOM_SEED)

    metadata_path = args.models_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")
    metadata = load_metadata(metadata_path)
    feature_columns: List[str] = metadata["feature_columns"]
    cat_columns: List[str] = metadata["categorical_columns"]
    te_meta = metadata.get("target_encoding")

    test_df = pd.read_csv(args.test_path)
    test_df = preprocess_dataframe(test_df)

    feature_matrix = build_feature_matrix(test_df, args)
    if te_meta:
        te_columns = te_meta.get("columns", [])
        te_mappings = te_meta.get("mappings", {})
        te_prior = te_meta.get("prior", 0.0)
        te_features = apply_target_encoding(test_df, te_columns, te_mappings, te_prior)
        feature_matrix = pd.concat([feature_matrix, te_features], axis=1)
    feature_matrix = feature_matrix.reindex(columns=feature_columns)
    cat_features = [feature_matrix.columns.get_loc(col) for col in cat_columns if col in feature_matrix.columns]

    model_paths = sorted(args.models_dir.glob("catboost_fold*.cbm"))
    if not model_paths:
        raise FileNotFoundError(f"No CatBoost models found in {args.models_dir}")

    torch_device = detect_torch_device()
    task_type = detect_catboost_task_type()
    print(f"Inference embeddings on {torch_device}; CatBoost task_type={task_type}; models={len(model_paths)}")

    preds = []
    pool = Pool(feature_matrix, cat_features=cat_features)
    for model_path in model_paths:
        model = CatBoostRanker(task_type=task_type)
        model.load_model(model_path)
        preds.append(model.predict(pool))

    prediction = np.mean(preds, axis=0)
    if args.ce_blend_weight != 0.0:
        ce_cols = [c for c in feature_matrix.columns if c.startswith("cross_encoder_score_")]
        if ce_cols:
            ce_mean = feature_matrix[ce_cols].mean(axis=1).to_numpy()
            prediction = prediction + args.ce_blend_weight * ce_mean
        else:
            print("Warning: ce-blend-weight set but cross_encoder_score_* columns not found; skipping blend.")

    ce_model_dir = args.ce_model_dir or args.models_dir / "models_ce_minilm"
    prediction = _apply_ce_rerank(
        test_df,
        prediction,
        ce_model_dir=str(ce_model_dir),
        top_k=args.ce_rerank_top_k,
        weight=args.ce_rerank_weight,
        max_length=args.ce_rerank_max_length,
        batch_size=args.ce_rerank_batch_size,
        device=args.ce_device or detect_torch_device(),
    )

    submission = pd.DataFrame({"id": test_df["id"], "prediction": prediction})
    if args.submission_path.parent != Path("."):
        args.submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.submission_path, index=False)
    print(f"Saved submission to {args.submission_path}")


def main() -> None:
    """Точка входа для инференса."""
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
