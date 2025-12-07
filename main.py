import argparse
from pathlib import Path
import shutil
import os

from src.ce_train import add_ce_train_arguments, run_ce_training
from src.inference import add_inference_arguments, run_inference
from src.rerank import add_rerank_arguments, run_rerank
from src.train import add_train_arguments, run_training


def create_submission(submission_path: Path) -> Path:
    """
    Копирует (или оставляет на месте) итоговый файл сабмита в папку results/submission.csv.
    """
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    target = results_dir / "submission.csv"
    submission_path = Path(submission_path)
    if submission_path.resolve() != target.resolve():
        shutil.copy2(submission_path, target)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Полный цикл: препроцессинг, обучение CatBoostRanker и инференс.",
        conflict_handler="resolve",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "infer", "full", "rerank", "ce-train"],
        default="full",
        help="Режим работы: обучение CatBoost, инференс, полный цикл, CE-rerank или обучение CE.",
    )
    add_train_arguments(parser)
    add_inference_arguments(parser)
    add_rerank_arguments(parser)
    add_ce_train_arguments(parser)
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args = parse_args()
    if args.mode == "rerank":
        run_rerank(args)
        create_submission(args.submission_path)
        return
    if args.mode == "ce-train":
        run_ce_training(args)
        return
    if args.mode in {"train", "full"}:
        run_training(args)
    if args.mode == "full":
        ce_dir = Path(args.ce_output_dir)
        if args.force_ce_train or not ce_dir.exists() or not any(ce_dir.iterdir()):
            # Обучаем cross-encoder перед инференсом, чтобы rerank не пропускался.
            run_ce_training(args)
        else:
            print(f"CE model found at {ce_dir}, skip training. Use --force-ce-train to retrain.")
        # Прокидываем путь к уже обученной/найденной CE модели в инференс.
        args.ce_model_dir = ce_dir
    if args.mode in {"infer", "full"}:
        run_inference(args)
        create_submission(args.submission_path)


if __name__ == "__main__":
    main()
