import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupKFold

from .constants import CATEGORICAL_COLUMNS, EMBEDDING_BATCH_SIZE, MODEL_NAME, RANDOM_SEED
from .features import (
    BM25FeatureGenerator,
    CrossEncoderFeatureGenerator,
    EmbeddingFeatureGenerator,
    build_brand_features,
    build_color_features,
    build_coverage_features,
    build_simple_features,
    compute_target_encoding,
    prepare_categorical,
    save_metadata,
)
from .preprocessing import preprocess_dataframe
from .utils import detect_catboost_task_type, detect_torch_device, set_seed


def compute_group_ndcg(df: pd.DataFrame, preds: np.ndarray, k: int = 10) -> float:
    """
    Вычисляет средний nDCG@k по группам запросов.

    Args:
        df: DataFrame с колонками query_id и relevance.
        preds: Предсказания модели.
        k: Параметр k для nDCG.

    Returns:
        Средний nDCG@k.
    """
    scores: List[float] = []
    for _, group in df.groupby("query_id"):
        true_labels = group["relevance"].to_numpy().reshape(1, -1)
        pred_scores = preds[group.index].reshape(1, -1)
        scores.append(ndcg_score(true_labels, pred_scores, k=k))
    return float(np.mean(scores)) if scores else 0.0


def build_feature_matrix(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """
    Строит матрицу признаков для обучения.

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
    parser = argparse.ArgumentParser(description="Training pipeline for product ranking.")
    add_train_arguments(parser)
    return parser.parse_args()


def add_train_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Добавляет аргументы командной строки для обучения."""
    parser.add_argument("--train-path", type=Path, default=Path("data/train.csv"), help="Path to train.csv")
    parser.add_argument(
        "--models-dir", type=Path, default=Path("models"), help="Directory to store trained models and metadata"
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of folds for GroupKFold")
    parser.add_argument("--iterations", type=int, default=3500, help="CatBoost iterations")
    parser.add_argument("--learning-rate", type=float, default=0.025, help="CatBoost learning rate")
    parser.add_argument("--depth", type=int, default=7, help="CatBoost tree depth")
    parser.add_argument("--loss-function", type=str, default="YetiRank", help="CatBoost loss function")
    parser.add_argument("--early-stopping-rounds", type=int, default=300, help="Early stopping rounds")
    parser.add_argument("--verbose-every", type=int, default=100, help="Logging frequency for CatBoost")
    parser.add_argument("--embed-model", type=str, default=MODEL_NAME, help="Embedding model name")
    parser.add_argument(
        "--embed-query-instruction",
        type=str,
        default=None,
        help="Instruction template for query embeddings (e.g. 'Instruct: ... {query}')",
    )
    parser.add_argument("--bm25-k1", type=float, default=1.5, help="BM25 k1 parameter")
    parser.add_argument("--bm25-b", type=float, default=0.75, help="BM25 b parameter")
    parser.add_argument("--embed-batch-size", type=int, default=EMBEDDING_BATCH_SIZE, help="Embedding batch size")
    parser.add_argument("--no-embed-progress", action="store_true", help="Disable embedding progress bars")
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory for embeddings and cross-encoder scores",
    )
    return parser


def run_training(args: argparse.Namespace) -> None:
    """
    Запускает обучение CatBoost модели с кросс-валидацией.

    Args:
        args: Аргументы командной строки.
    """
    set_seed(RANDOM_SEED)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_path)
    train_df = preprocess_dataframe(train_df)

    target = train_df["relevance"].to_numpy()
    groups = train_df["query_id"].to_numpy()

    te_columns = ["product_id", "product_title", "query"]
    te_df, te_mappings, te_prior = compute_target_encoding(
        train_df, train_df["relevance"], te_columns, folds=args.folds, groups=train_df["query_id"]
    )

    feature_matrix = pd.concat([build_feature_matrix(train_df, args), te_df], axis=1)

    cat_features = [feature_matrix.columns.get_loc(col) for col in CATEGORICAL_COLUMNS if col in feature_matrix.columns]

    gkf = GroupKFold(n_splits=args.folds)
    oof_preds = np.zeros(len(train_df))
    model_paths: List[str] = []

    torch_device = detect_torch_device()
    task_type = detect_catboost_task_type()
    print(f"Embedding device: {torch_device}; CatBoost task_type: {task_type}")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(feature_matrix, target, groups=groups)):
        train_pool = Pool(
            data=feature_matrix.iloc[train_idx],
            label=target[train_idx],
            group_id=groups[train_idx],
            cat_features=cat_features,
        )
        val_pool = Pool(
            data=feature_matrix.iloc[val_idx],
            label=target[val_idx],
            group_id=groups[val_idx],
            cat_features=cat_features,
        )

        model = CatBoostRanker(
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            loss_function=args.loss_function,
            eval_metric="NDCG:top=10",
            random_seed=RANDOM_SEED,
            task_type=task_type,
            od_type="Iter",
            od_wait=args.early_stopping_rounds,
            bootstrap_type="Bernoulli",
        )

        model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=args.verbose_every)
        oof_preds[val_idx] = model.predict(val_pool)

        model_path = args.models_dir / f"catboost_fold{fold}.cbm"
        model.save_model(model_path)
        model_paths.append(str(model_path))

        fold_ndcg = compute_group_ndcg(train_df.iloc[val_idx], oof_preds)
        print(f"[Fold {fold}] nDCG@10={fold_ndcg:.4f}")

    overall_ndcg = compute_group_ndcg(train_df, oof_preds)
    print(f"OOF nDCG@10: {overall_ndcg:.4f}")

    metadata_path = args.models_dir / "metadata.json"
    save_metadata(
        metadata_path,
        feature_columns=list(feature_matrix.columns),
        cat_columns=CATEGORICAL_COLUMNS,
        te_mappings=te_mappings,
        te_prior=te_prior,
        te_columns=te_columns,
    )
    print(f"Saved metadata to {metadata_path}")


def main() -> None:
    """Точка входа для обучения."""
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
