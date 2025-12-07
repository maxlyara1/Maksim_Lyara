import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from .constants import RANDOM_SEED
from .preprocessing import preprocess_dataframe
from .utils import detect_torch_device, set_seed


def _build_text(row: pd.Series, max_chars: int = 512) -> str:
    """Собирает текст продукта из полей для cross-encoder."""
    def as_str(val: object) -> str:
        return val if isinstance(val, str) else "" if val is None else str(val)

    parts: List[str] = []
    for key in ["product_title", "product_description", "product_bullet_point", "product_brand", "product_color"]:
        val = as_str(row.get(key))
        if val:
            parts.append(val)
    combined = " [SEP] ".join(parts)
    if len(combined) > max_chars:
        combined = combined[:max_chars]
    return combined


def _prepare_examples(df: pd.DataFrame, max_chars: int) -> List[InputExample]:
    """
    Подготавливает примеры для обучения cross-encoder.

    Args:
        df: DataFrame с колонками query, relevance и полями продукта.
        max_chars: Максимальная длина текста продукта.

    Returns:
        Список InputExample для обучения.
    """
    examples: List[InputExample] = []
    for _, row in df.iterrows():
        text_a = row["query"]
        text_b = _build_text(row, max_chars=max_chars)
        label = float(row["relevance"]) / 3.0
        examples.append(InputExample(texts=[text_a, text_b], label=label))
    return examples


def add_ce_train_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Добавляет аргументы командной строки для обучения cross-encoder."""
    parser.add_argument("--ce-train-path", type=Path, default=Path("data/train.csv"), help="Path to train.csv for CE fine-tuning")
    parser.add_argument(
        "--ce-output-dir", type=Path, default=Path("models_ce_minilm"), help="Where to save fine-tuned CE model"
    )
    parser.add_argument(
        "--ce-model-name",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L6-v2",
        help="Base cross-encoder model to fine-tune",
    )
    parser.add_argument("--ce-batch-size", type=int, default=16, help="Batch size for CE training (per device)")
    parser.add_argument("--ce-epochs", type=int, default=2, help="Number of CE training epochs")
    parser.add_argument("--ce-learning-rate", type=float, default=2e-5, help="Learning rate for CE training")
    parser.add_argument("--ce-max-length", type=int, default=160, help="Max sequence length for CE")
    parser.add_argument("--ce-warmup-ratio", type=float, default=0.1, help="Warmup ratio for CE scheduler")
    parser.add_argument("--ce-val-size", type=float, default=0.1, help="Validation fraction grouped by query_id")
    parser.add_argument(
        "--ce-subset-fraction",
        type=float,
        default=1.0,
        help="Optional fraction of query_id groups to sample for quick CE training (0 < frac <= 1)",
    )
    parser.add_argument("--ce-seed", type=int, default=RANDOM_SEED, help="Random seed for CE training")
    parser.add_argument("--ce-device", type=str, default=None, help="Device for CE model (cpu/cuda/mps). Auto if not set")
    parser.add_argument("--ce-fp16", action="store_true", help="Run CE model in fp16 (if supported by device)")
    parser.add_argument(
        "--force-ce-train",
        action="store_true",
        help="Always train CE even if ce-output-dir already contains a model (used in full mode)",
    )
    return parser


def run_ce_training(args: argparse.Namespace) -> None:
    """
    Запускает обучение cross-encoder модели.

    Args:
        args: Аргументы командной строки.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    set_seed(args.ce_seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = args.ce_device or detect_torch_device()
    print(f"CE training device: {device}")

    train_df = pd.read_csv(args.ce_train_path)
    train_df = preprocess_dataframe(train_df)

    if args.ce_subset_fraction < 1.0:
        unique_queries = train_df["query_id"].unique()
        rng = np.random.default_rng(args.ce_seed)
        keep_queries = rng.choice(
            unique_queries,
            size=max(1, int(len(unique_queries) * args.ce_subset_fraction)),
            replace=False,
        )
        train_df = train_df[train_df["query_id"].isin(keep_queries)].reset_index(drop=True)
        print(f"CE uses subset of queries: {len(keep_queries)} / {len(unique_queries)}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.ce_val_size, random_state=args.ce_seed)
    (train_idx, val_idx) = next(splitter.split(train_df, groups=train_df["query_id"]))
    train_split = train_df.iloc[train_idx].reset_index(drop=True)
    val_split = train_df.iloc[val_idx].reset_index(drop=True)

    train_examples = _prepare_examples(train_split, max_chars=args.ce_max_length * 3)
    val_examples = _prepare_examples(val_split, max_chars=args.ce_max_length * 3)

    model = CrossEncoder(args.ce_model_name, num_labels=1, max_length=args.ce_max_length, device=device)
    model.tokenizer.model_max_length = 100000
    model.max_seq_length = args.ce_max_length
    if args.ce_fp16 and device != "cpu":
        model.model.half()

    generator = None
    if torch is not None:
        generator = torch.Generator()
        generator.manual_seed(args.ce_seed)

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.ce_batch_size,
        collate_fn=model.smart_batching_collate,
        pin_memory=False,
        generator=generator,
    )
    warmup_steps = int(len(train_dataloader) * args.ce_epochs * args.ce_warmup_ratio)

    evaluator = CECorrelationEvaluator.from_input_examples(val_examples, name="val")

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=args.ce_epochs,
        evaluation_steps=len(train_dataloader) // 2 or 1,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.ce_learning_rate},
        output_path=str(args.ce_output_dir),
        save_best_model=True,
        loss_fct=torch.nn.MSELoss(),
    )

    print(f"Saved fine-tuned cross-encoder to {args.ce_output_dir}")
