import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import Levenshtein
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.model_selection import GroupKFold

from .constants import (
    CATEGORICAL_COLUMNS,
    CROSS_ENCODER_MODEL,
    DEFAULT_BM25_B,
    DEFAULT_BM25_K1,
    EMBEDDING_BATCH_SIZE,
    MODEL_NAME,
    RANDOM_SEED,
)
from .utils import detect_torch_device


def _tokenize(text: str) -> List[str]:
    """Разбивает текст на токены по пробелам."""
    return text.split()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Вычисляет косинусное сходство между векторами."""
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-9, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-9, None)
    return np.sum(a_norm * b_norm, axis=1)


COLOR_MAP = {
    "grey": "gray",
    "light grey": "gray",
    "dark grey": "gray",
    "space gray": "gray",
    "space grey": "gray",
    "graphite": "gray",
    "lt gray": "gray",
    "lt. gray": "gray",
    "charcoal": "gray",
    "gunmetal": "gray",
    "gun metal": "gray",
    "navy": "blue",
    "navy blue": "blue",
    "royal": "blue",
    "royal blue": "blue",
    "teal": "blue",
    "stainless steel": "silver",
    "steel": "silver",
    "metal": "silver",
    "chrome": "silver",
    "brushed nickel": "silver",
    "nickel": "silver",
    "satin nickel": "silver",
    "rose gold": "gold",
    "warm white": "white",
    "natural": "beige",
    "tan": "beige",
    "ivory": "beige",
    "espresso": "brown",
    "chocolate": "brown",
    "bronze": "brown",
    "burgundy": "red",
    "wine red": "red",
}
COLOR_UNKNOWN = {"", "unknown", "unknown_brand", "none", "[null]", "null", "brand_masked"}
BASE_COLORS = [
    "black",
    "white",
    "gray",
    "silver",
    "blue",
    "red",
    "green",
    "pink",
    "purple",
    "yellow",
    "orange",
    "brown",
    "beige",
    "gold",
    "clear",
    "multicolor",
    "navy",
    "ivory",
    "tan",
    "turquoise",
    "teal",
    "burgundy",
    "bronze",
    "copper",
    "cream",
    "khaki",
    "nude",
    "maroon",
    "violet",
    "lavender",
    "coral",
    "mint",
    "peach",
    "charcoal",
    "rose",
]
COMMON_COLOR_TOKENS = set(BASE_COLORS)


def normalize_color(raw: str) -> str:
    """
    Нормализует название цвета к базовому цвету.

    Args:
        raw: Исходное название цвета.

    Returns:
        Нормализованное название цвета или "unknown" если не удалось определить.
    """
    if not isinstance(raw, str):
        return ""
    norm = raw.strip().lower()
    if not norm:
        return ""
    if norm in COLOR_MAP:
        return COLOR_MAP[norm]
    tokens = [tok for tok in re.split(r"[^a-z]+", norm) if tok and not tok.isdigit()]
    if not tokens:
        return "unknown"
    for tok in tokens:
        if tok in BASE_COLORS:
            return tok
        if tok in COLOR_MAP:
            return COLOR_MAP[tok]
    if len(norm) <= 15:
        return norm
    return "unknown"


class EmbeddingFeatureGenerator:
    """Генератор признаков на основе эмбеддингов запросов и продуктов."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        query_instruction: Optional[str] = None,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        show_progress_bar: bool = True,
        cache_dir: Optional[Path] = None,
    ) -> None:
        """
        Args:
            model_name: Название модели для генерации эмбеддингов.
            device: Устройство для вычислений (cpu/cuda/mps). Автоматически определяется если None.
            query_instruction: Инструкция для форматирования запросов. Для E5 моделей устанавливается автоматически.
            batch_size: Размер батча для обработки.
            show_progress_bar: Показывать ли прогресс-бар.
            cache_dir: Директория для кэширования эмбеддингов.
        """
        self.model_name = model_name
        self.device = device or detect_torch_device()
        if query_instruction is None and "e5" in model_name.lower():
            query_instruction = "Instruct: Given a product search query, retrieve relevant product information.\\nQuery: {query}"
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model.to(self.device)

    def _model_tag(self) -> str:
        """Возвращает тег модели для использования в именах файлов."""
        return self.model_name.replace("/", "__").replace("-", "_")

    def _cache_path(self, kind: str) -> Optional[Path]:
        """Возвращает путь к файлу кэша для указанного типа."""
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{self._model_tag()}_{kind}_embeddings.pkl"

    def _load_embedding_cache(self, kind: str) -> Dict[str, np.ndarray]:
        """Загружает кэш эмбеддингов из файла."""
        path = self._cache_path(kind)
        if path and path.exists():
            try:
                cached = pd.read_pickle(path)
                if isinstance(cached, dict):
                    return cached
            except Exception:
                pass
        return {}

    def _save_embedding_cache(self, kind: str, cache: Dict[str, np.ndarray]) -> None:
        """Сохраняет кэш эмбеддингов в файл."""
        path = self._cache_path(kind)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            pd.to_pickle(cache, path)
        except Exception:
            pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Генерирует признаки на основе косинусного сходства эмбеддингов.

        Args:
            df: DataFrame с колонками query, product_title, product_description, product_bullet_point,
                product_brand, product_color.

        Returns:
            DataFrame с признаками косинусного сходства между запросами и различными частями продуктов.
        """
        queries_raw = df["query"].tolist()
        if self.query_instruction:
            queries = [
                self.query_instruction.format(query=q) if "{query}" in self.query_instruction else f"{self.query_instruction} {q}"
                for q in queries_raw
            ]
        else:
            queries = queries_raw
        titles = df["product_title"].tolist()
        descs = df["product_description"].tolist()
        bullets = df.get("product_bullet_point", pd.Series([""] * len(df))).tolist()
        brands = df.get("product_brand", pd.Series([""] * len(df))).tolist()
        colors = df.get("product_color", pd.Series([""] * len(df))).tolist()

        combined_all = []
        combined_title_brand_color = []
        brand_markers = {"unknown", "unknown_brand", "generic", "none", "brand_masked", ""}
        for t, d, b, brand, color in zip(titles, descs, bullets, brands, colors):
            combined_all.append(" [SEP] ".join(part for part in [t, b, d] if part))
            norm_brand = brand.strip().lower() if isinstance(brand, str) else ""
            if norm_brand in brand_markers:
                norm_brand = ""
            norm_color = normalize_color(color)
            if norm_color in COLOR_UNKNOWN:
                norm_color = ""
            combined_title_brand_color.append(" [SEP] ".join(part for part in [t, norm_brand, norm_color] if part))

        unique_queries = list(dict.fromkeys([q for q in queries if q]))
        query_map: Dict[str, np.ndarray] = self._load_embedding_cache("query")
        missing_queries = [q for q in unique_queries if q not in query_map]
        if unique_queries:
            if missing_queries:
                encode_kwargs = {
                    "batch_size": self.batch_size,
                    "convert_to_numpy": True,
                    "show_progress_bar": self.show_progress_bar,
                    "device": self.device,
                }
                if not self.query_instruction:
                    encode_kwargs["prompt_name"] = "query"
                query_embeds = self.model.encode(
                    missing_queries,
                    **encode_kwargs,
                )
                query_map.update(dict(zip(missing_queries, query_embeds)))
                self._save_embedding_cache("query", query_map)

        doc_texts = list(
            dict.fromkeys([t for t in titles + descs + bullets + combined_all + combined_title_brand_color if t])
        )
        doc_map: Dict[str, np.ndarray] = self._load_embedding_cache("doc")
        missing_docs = [t for t in doc_texts if t not in doc_map]
        if doc_texts:
            if missing_docs:
                doc_embeds = self.model.encode(
                    missing_docs,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=self.show_progress_bar,
                    device=self.device,
                )
                doc_map.update(dict(zip(missing_docs, doc_embeds)))
                self._save_embedding_cache("doc", doc_map)

        dim = self.model.get_sentence_embedding_dimension()
        zero_vec = np.zeros(dim, dtype=np.float32)

        query_embeddings = np.vstack([query_map.get(q, zero_vec) for q in queries])
        title_embeddings = np.vstack([doc_map.get(t, zero_vec) for t in titles])
        desc_embeddings = np.vstack([doc_map.get(d, zero_vec) for d in descs])
        bullet_embeddings = np.vstack([doc_map.get(b, zero_vec) for b in bullets])
        all_embeddings = np.vstack([doc_map.get(c, zero_vec) for c in combined_all])
        brand_color_embeddings = np.vstack([doc_map.get(c, zero_vec) for c in combined_title_brand_color])

        cos_sim_query_title = _cosine_similarity(query_embeddings, title_embeddings)
        cos_sim_query_desc = _cosine_similarity(query_embeddings, desc_embeddings)
        cos_sim_query_bullet = _cosine_similarity(query_embeddings, bullet_embeddings)
        cos_sim_query_all = _cosine_similarity(query_embeddings, all_embeddings)
        cos_sim_query_brand_color = _cosine_similarity(query_embeddings, brand_color_embeddings)

        return pd.DataFrame(
            {
                "cos_sim_query_title": cos_sim_query_title,
                "cos_sim_query_desc": cos_sim_query_desc,
                "cos_sim_query_bullet": cos_sim_query_bullet,
                "cos_sim_query_all": cos_sim_query_all,
                "cos_sim_query_brand_color": cos_sim_query_brand_color,
            }
        )


class CrossEncoderFeatureGenerator:
    """Генератор признаков на основе cross-encoder модели."""

    def __init__(
        self,
        model_name: str = CROSS_ENCODER_MODEL,
        device: Optional[str] = None,
        batch_size: int = 16,
        show_progress_bar: bool = True,
        cache_dir: Optional[Path] = None,
    ) -> None:
        """
        Args:
            model_name: Название cross-encoder модели.
            device: Устройство для вычислений (cpu/cuda/mps). Автоматически определяется если None.
            batch_size: Размер батча для обработки.
            show_progress_bar: Показывать ли прогресс-бар.
            cache_dir: Директория для кэширования скоров.
        """
        self.model_name = model_name
        self.device = device or detect_torch_device()
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model = CrossEncoder(model_name, device=self.device, num_labels=1)

    def _model_tag(self) -> str:
        """Возвращает тег модели для использования в именах файлов."""
        return self.model_name.replace("/", "__").replace("-", "_")

    def _cache_path(self, kind: str) -> Optional[Path]:
        """Возвращает путь к файлу кэша для указанного типа."""
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{self._model_tag()}_{kind}_scores.pkl"

    def _load_cache(self, kind: str) -> Dict[str, float]:
        """Загружает кэш скоров из файла."""
        path = self._cache_path(kind)
        if path and path.exists():
            try:
                cached = pd.read_pickle(path)
                if isinstance(cached, dict):
                    return cached
            except Exception:
                pass
        return {}

    def _save_cache(self, kind: str, cache: Dict[str, float]) -> None:
        """Сохраняет кэш скоров в файл."""
        path = self._cache_path(kind)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            pd.to_pickle(cache, path)
        except Exception:
            pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Генерирует признаки на основе скоров cross-encoder модели.

        Args:
            df: DataFrame с колонками query, product_title, product_description, product_bullet_point,
                product_brand, product_color.

        Returns:
            DataFrame с признаками скоров cross-encoder для различных комбинаций запросов и продуктов.
        """
        queries = df["query"].tolist()
        titles = df["product_title"].tolist()
        descs = df.get("product_description", pd.Series([""] * len(df))).tolist()
        bullets = df.get("product_bullet_point", pd.Series([""] * len(df))).tolist()
        brands = df.get("product_brand", pd.Series([""] * len(df))).tolist()
        colors = df.get("product_color", pd.Series([""] * len(df))).tolist()

        brand_markers = {"unknown", "unknown_brand", "generic", "none", "brand_masked", ""}
        combined_all = []
        combined_title_brand_color = []
        for t, d, b, brand, color in zip(titles, descs, bullets, brands, colors):
            combined_all.append(" [SEP] ".join(part for part in [t, b, d] if part))
            norm_brand = brand.strip().lower() if isinstance(brand, str) else ""
            if norm_brand in brand_markers:
                norm_brand = ""
            norm_color = normalize_color(color)
            if norm_color in COLOR_UNKNOWN:
                norm_color = ""
            combined_title_brand_color.append(" [SEP] ".join(part for part in [t, norm_brand, norm_color] if part))

        def predict_with_cache(pairs: List[tuple[str, str]], kind: str) -> np.ndarray:
            cache = self._load_cache(kind)
            keys = [f"{a}|||{b}" for a, b in pairs]
            unique_pairs = list(dict.fromkeys(zip(keys, pairs)))
            to_compute = [(k, p) for k, p in unique_pairs if k not in cache]
            if to_compute:
                new_scores = self.model.predict(
                    [p for _, p in to_compute],
                    batch_size=self.batch_size,
                    show_progress_bar=self.show_progress_bar,
                    convert_to_numpy=True,
                )
                for (k, _), s in zip(to_compute, new_scores):
                    cache[k] = float(s)
                self._save_cache(kind, cache)
            return np.asarray([cache.get(k, 0.0) for k in keys], dtype=np.float32)

        scores_title = predict_with_cache(list(zip(queries, titles)), "title")
        scores_desc = predict_with_cache(list(zip(queries, descs)), "desc")
        scores_bullet = predict_with_cache(list(zip(queries, bullets)), "bullet")
        scores_all = predict_with_cache(list(zip(queries, combined_all)), "all")
        scores_brand_color = predict_with_cache(list(zip(queries, combined_title_brand_color)), "brand_color")
        return pd.DataFrame(
            {
                "cross_encoder_score_title": scores_title,
                "cross_encoder_score_desc": scores_desc,
                "cross_encoder_score_bullet": scores_bullet,
                "cross_encoder_score_all": scores_all,
                "cross_encoder_score_brand_color": scores_brand_color,
            }
        )


@dataclass
class _BM25Corpus:
    """Внутренняя структура данных для BM25 корпуса."""

    doc_term_freqs: List[Counter]
    doc_lengths: List[int]
    idf: Dict[str, float]
    avgdl: float
    vocab: Dict[str, int]


class BM25FeatureGenerator:
    """Генератор признаков на основе BM25 алгоритма."""

    def __init__(self, k1: float = DEFAULT_BM25_K1, b: float = DEFAULT_BM25_B) -> None:
        """
        Args:
            k1: Параметр k1 для BM25 (контролирует насыщение частоты термина).
            b: Параметр b для BM25 (контролирует нормализацию по длине документа).
        """
        self.k1 = k1
        self.b = b
        self._title_corpus: Optional[_BM25Corpus] = None
        self._desc_corpus: Optional[_BM25Corpus] = None
        self._bullet_corpus: Optional[_BM25Corpus] = None
        self._all_corpus: Optional[_BM25Corpus] = None

    @staticmethod
    def _build_corpus(documents: Sequence[str], k1: float, b: float) -> _BM25Corpus:
        """Строит BM25 корпус из документов."""
        tokenized_docs = [_tokenize(doc) for doc in documents]
        doc_term_freqs = [Counter(doc) for doc in tokenized_docs]
        doc_lengths = [len(doc) for doc in tokenized_docs]
        avgdl = float(np.mean(doc_lengths)) if doc_lengths else 0.0

        df_counts: Counter = Counter()
        for freq in doc_term_freqs:
            df_counts.update(freq.keys())

        vocab = {term: idx for idx, term in enumerate(df_counts.keys())}
        N = len(tokenized_docs)
        idf: Dict[str, float] = {}
        for term, df in df_counts.items():
            idf[term] = math.log(1 + (N - df + 0.5) / (df + 0.5))

        return _BM25Corpus(doc_term_freqs=doc_term_freqs, doc_lengths=doc_lengths, idf=idf, avgdl=avgdl, vocab=vocab)

    def fit(self, df: pd.DataFrame) -> None:
        """
        Обучает генератор на данных.

        Args:
            df: DataFrame с колонками product_title, product_description, product_bullet_point.
        """
        titles = df["product_title"].tolist()
        descriptions = df["product_description"].tolist()
        bullets = df.get("product_bullet_point", pd.Series([""] * len(df))).tolist()
        self._title_corpus = self._build_corpus(titles, self.k1, self.b)
        self._desc_corpus = self._build_corpus(descriptions, self.k1, self.b)
        self._bullet_corpus = self._build_corpus(bullets, self.k1, self.b)
        combined = [(t or "") + " " + (d or "") + " " + (b or "") for t, d, b in zip(titles, descriptions, bullets)]
        self._all_corpus = self._build_corpus(combined, self.k1, self.b)

    def _score(self, corpus: _BM25Corpus, query_tokens: List[str], doc_idx: int) -> float:
        """Вычисляет BM25 скор для запроса и документа."""
        if doc_idx >= len(corpus.doc_term_freqs):
            return 0.0
        doc_tf = corpus.doc_term_freqs[doc_idx]
        doc_len = corpus.doc_lengths[doc_idx] if corpus.doc_lengths else 0
        if doc_len == 0 or corpus.avgdl == 0:
            return 0.0

        score = 0.0
        for term in query_tokens:
            if term not in doc_tf:
                continue
            freq = doc_tf[term]
            idf = corpus.idf.get(term, 0.0)
            denom = freq + self.k1 * (1 - self.b + self.b * doc_len / corpus.avgdl)
            score += idf * freq * (self.k1 + 1) / denom
        return score

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Генерирует BM25 признаки для запросов.

        Args:
            df: DataFrame с колонкой query.

        Returns:
            DataFrame с BM25 скорами для различных частей продуктов.
        """
        if (
            self._title_corpus is None
            or self._desc_corpus is None
            or self._bullet_corpus is None
            or self._all_corpus is None
        ):
            raise RuntimeError("BM25FeatureGenerator must be fitted before calling transform.")

        queries = df["query"].tolist()
        q_tokens = [_tokenize(q) for q in queries]

        title_scores = [self._score(self._title_corpus, tokens, idx) for idx, tokens in enumerate(q_tokens)]
        desc_scores = [self._score(self._desc_corpus, tokens, idx) for idx, tokens in enumerate(q_tokens)]
        bullet_scores = [self._score(self._bullet_corpus, tokens, idx) for idx, tokens in enumerate(q_tokens)]
        all_scores = [self._score(self._all_corpus, tokens, idx) for idx, tokens in enumerate(q_tokens)]

        return pd.DataFrame(
            {
                "bm25_score_title": title_scores,
                "bm25_score_desc": desc_scores,
                "bm25_score_bullet": bullet_scores,
                "bm25_score_all": all_scores,
            }
        )


def build_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Генерирует простые текстовые и числовые признаки.

    Args:
        df: DataFrame с колонками query, product_title, product_description, product_bullet_point.

    Returns:
        DataFrame с признаками: длины, Jaccard, Levenshtein, числовые совпадения и т.д.
    """
    query_tokens = df["query"].str.split()
    title_tokens = df["product_title"].str.split()

    def separate_num_units(text: str) -> str:
        text = re.sub(r"(?<=\d)(?=[a-zA-Z])", " ", text)
        text = re.sub(r"(?<=[a-zA-Z])(?=\d)", " ", text)
        return text

    num_re = re.compile(r"\d+\.?\d*")
    unit_re = re.compile(r"(?P<num>\d+\.?\d*)\s*(?P<unit>[a-z]{1,6})")

    def extract_nums_units(text: str) -> Tuple[set[str], set[str]]:
        nums = set(num_re.findall(text))
        units = set()
        for match in unit_re.finditer(text):
            units.add(match.group("unit"))
        return nums, units

    def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
        set_a, set_b = set(a), set(b)
        if not set_a and not set_b:
            return 0.0
        return len(set_a & set_b) / max(1e-9, len(set_a | set_b))

    def digit_overlap(q_str: str, t_str: str) -> float:
        q_nums = set(filter(str.isdigit, q_str.split()))
        t_nums = set(filter(str.isdigit, t_str.split()))
        if not q_nums:
            return 0.5
        if not t_nums:
            return 0.0
        return len(q_nums & t_nums) / len(q_nums)

    def levenshtein_ratio(q_str: str, t_str: str) -> float:
        return Levenshtein.ratio(q_str, t_str)

    queries = df["query"].fillna("").apply(separate_num_units)
    titles = df["product_title"].fillna("").apply(separate_num_units)
    bullets = df.get("product_bullet_point", pd.Series([""] * len(df))).fillna("").apply(separate_num_units)
    descs = df["product_description"].fillna("").apply(separate_num_units)

    query_has_num = []
    title_has_num = []
    num_overlap = []
    unit_overlap = []
    num_abs_diff = []
    query_num_count = []
    title_num_count = []
    query_phrase_in_title = []
    query_phrase_in_bullet = []
    query_phrase_in_desc = []
    query_token_count = []
    title_token_count = []
    desc_present = []
    bullet_present = []
    query_starts_digit = []
    query_starts_punct = []
    title_token_coverage = []
    query_has_currency = []
    num_min_abs_diff = []
    num_min_rel_diff = []

    for q_str, t_str, b_str, d_str in zip(queries.str.lower(), titles.str.lower(), bullets.str.lower(), descs.str.lower()):
        q_nums, q_units = extract_nums_units(q_str)
        t_nums, t_units = extract_nums_units(t_str)

        query_has_num.append(1 if q_nums else 0)
        title_has_num.append(1 if t_nums else 0)
        num_overlap.append(1 if q_nums and t_nums and (q_nums & t_nums) else 0)
        unit_overlap.append(1 if q_units and t_units and (q_units & t_units) else 0)
        query_num_count.append(len(q_nums))
        title_num_count.append(len(t_nums))
        if q_nums and t_nums:
            try:
                q_vals = [float(x) for x in q_nums]
                t_vals = [float(x) for x in t_nums]
                min_abs = min(abs(qv - tv) for qv in q_vals for tv in t_vals)
                min_rel = min(abs(qv - tv) / max(abs(qv), abs(tv), 1e-9) for qv in q_vals for tv in t_vals)
            except Exception:
                min_abs = np.nan
                min_rel = np.nan
            num_min_abs_diff.append(min_abs)
            num_min_rel_diff.append(min_rel)
        else:
            num_min_abs_diff.append(np.nan)
            num_min_rel_diff.append(np.nan)

        if len(q_nums) == 1 and len(t_nums) == 1:
            try:
                num_abs_diff.append(abs(float(next(iter(q_nums))) - float(next(iter(t_nums)))))
            except Exception:
                num_abs_diff.append(np.nan)
        else:
            num_abs_diff.append(np.nan)

        query_phrase_in_title.append(1 if q_str and q_str in t_str else 0)
        query_phrase_in_bullet.append(1 if b_str and q_str and q_str in b_str else 0)
        query_phrase_in_desc.append(1 if d_str and q_str and q_str in d_str else 0)
        query_token_count.append(len(q_str.split()))
        title_token_count.append(len(t_str.split()))
        desc_present.append(1 if d_str else 0)
        bullet_present.append(1 if b_str else 0)
        query_starts_digit.append(1 if q_str[:1].isdigit() else 0)
        query_starts_punct.append(1 if q_str[:1] and not q_str[:1].isalnum() else 0)
        query_has_currency.append(1 if any(sym in q_str for sym in ["$", "€", "£", "¥"]) else 0)
        t_tokens = t_str.split()
        title_token_coverage.append(len(set(q_str.split()) & set(t_tokens)) / (len(t_tokens) + 1e-9))

    features = pd.DataFrame(
        {
            "query_len": queries.str.len(),
            "title_len": titles.str.len(),
            "len_ratio": queries.str.len() / (titles.str.len() + 1),
            "desc_len": df["product_description"].fillna("").str.len(),
            "bullet_len": df.get("product_bullet_point", pd.Series([0] * len(df))).fillna("").str.len(),
            "jaccard_query_title": [jaccard(q, t) for q, t in zip(query_tokens, title_tokens)],
            "digit_overlap": [digit_overlap(q, t) for q, t in zip(queries, titles)],
            "levenshtein_query_title": [levenshtein_ratio(q, t) for q, t in zip(queries, titles)],
            "query_has_num": query_has_num,
            "title_has_num": title_has_num,
            "num_overlap": num_overlap,
            "unit_overlap": unit_overlap,
            "num_abs_diff": num_abs_diff,
            "num_min_abs_diff": num_min_abs_diff,
            "num_min_rel_diff": num_min_rel_diff,
            "query_num_count": query_num_count,
            "title_num_count": title_num_count,
            "query_phrase_in_title": query_phrase_in_title,
            "query_phrase_in_bullet": query_phrase_in_bullet,
            "query_phrase_in_desc": query_phrase_in_desc,
            "query_token_count": query_token_count,
            "title_token_count": title_token_count,
            "desc_present": desc_present,
            "bullet_present": bullet_present,
            "query_starts_digit": query_starts_digit,
            "query_starts_punct": query_starts_punct,
            "query_has_currency": query_has_currency,
            "title_token_coverage": title_token_coverage,
        }
    )
    return features


def build_brand_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Генерирует признаки на основе бренда продукта.

    Args:
        df: DataFrame с колонками query, product_brand.

    Returns:
        DataFrame с признаками совпадения бренда с запросом.
    """
    queries = df["query"].fillna("").str.lower()
    brands = df["product_brand"].fillna("").str.lower()

    markers = {"unknown", "unknown_brand", "generic", "none", "brand_masked", ""}
    top_brands = [
        b for b, _ in brands.value_counts().head(50).items() if b not in markers and len(b) >= 2
    ]

    brand_in_query = []
    brand_is_unknown = []
    brand_partial_ratio = []
    brand_mismatch = []
    query_has_brand = []
    for q, b in zip(queries, brands):
        has_brand_token = any(tb in q for tb in top_brands)
        query_has_brand.append(1 if has_brand_token else 0)
        if b in markers:
            brand_in_query.append(0)
            brand_is_unknown.append(1)
            brand_partial_ratio.append(0.0)
            brand_mismatch.append(1 if has_brand_token else 0)
        else:
            in_query = 1 if b in q else 0
            brand_in_query.append(in_query)
            brand_is_unknown.append(0)
            brand_partial_ratio.append(fuzz.partial_ratio(b, q) / 100.0 if q else 0.0)
            brand_mismatch.append(1 if has_brand_token and not in_query else 0)

    return pd.DataFrame(
        {
            "brand_in_query": brand_in_query,
            "brand_is_unknown": brand_is_unknown,
            "brand_partial_ratio": brand_partial_ratio,
            "brand_mismatch": brand_mismatch,
            "query_has_brand": query_has_brand,
        }
    )


def build_color_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Генерирует признаки на основе цвета продукта.

    Args:
        df: DataFrame с колонками query, product_color.

    Returns:
        DataFrame с признаками совпадения цвета с запросом.
    """
    queries = df["query"].fillna("").str.lower()
    colors = df["product_color"].fillna("").str.lower().apply(normalize_color)

    color_in_query = []
    color_any_in_query = []
    color_is_unknown = []
    color_mismatch = []
    color_partial_ratio = []
    for q, c in zip(queries, colors):
        has_color_token = any(token in q for token in COMMON_COLOR_TOKENS)
        color_any_in_query.append(1 if has_color_token else 0)
        if not c or c in COLOR_UNKNOWN:
            color_in_query.append(0)
            color_is_unknown.append(1)
            color_mismatch.append(1 if has_color_token else 0)
            color_partial_ratio.append(0.0)
            continue
        color_is_unknown.append(0)
        match = 1 if c in q else 0
        color_in_query.append(match)
        color_mismatch.append(1 if has_color_token and not match else 0)
        color_partial_ratio.append(fuzz.partial_ratio(c, q) / 100.0 if q else 0.0)

    return pd.DataFrame(
        {
            "color_in_query": color_in_query,
            "color_any_in_query": color_any_in_query,
            "color_is_unknown": color_is_unknown,
            "color_mismatch": color_mismatch,
            "color_partial_ratio": color_partial_ratio,
        }
    )


def _build_idf(texts: Iterable[str]) -> Dict[str, float]:
    """Вычисляет IDF для токенов в коллекции текстов."""
    df_counts: Counter = Counter()
    texts_list = list(texts)
    for tokens in (t.split() for t in texts_list):
        df_counts.update(set(tokens))
    N = len(texts_list)
    return {term: math.log(1 + (N - df + 0.5) / (df + 0.5)) for term, df in df_counts.items()}


def _idf_weighted_overlap(query_tokens: List[str], doc_tokens: List[str], idf: Dict[str, float]) -> float:
    """Вычисляет IDF-взвешенное пересечение токенов запроса и документа."""
    if not query_tokens:
        return 0.0
    q_set = set(query_tokens)
    overlap = q_set & set(doc_tokens)
    denom = sum(idf.get(tok, 0.0) for tok in q_set)
    if denom == 0.0:
        return 0.0
    num = sum(idf.get(tok, 0.0) for tok in overlap)
    return num / denom


def build_coverage_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Генерирует признаки покрытия запроса текстом продукта на основе IDF.

    Args:
        df: DataFrame с колонками query, product_title, product_description, product_bullet_point.

    Returns:
        DataFrame с признаками IDF-взвешенного покрытия запроса различными частями продукта.
    """
    queries = df["query"].fillna("").str.lower()
    titles = df["product_title"].fillna("").str.lower()
    descs = df["product_description"].fillna("").str.lower()
    bullets = df.get("product_bullet_point", pd.Series([""] * len(df))).fillna("").str.lower()
    combined = titles + " " + descs + " " + bullets

    idf_title = _build_idf(titles)
    idf_desc = _build_idf(descs)
    idf_bullet = _build_idf(bullets)
    idf_all = _build_idf(combined)

    coverage_title = []
    coverage_desc = []
    coverage_bullet = []
    coverage_all = []
    for q, t, d, b, a in zip(queries, titles, descs, bullets, combined):
        q_tokens = q.split()
        coverage_title.append(_idf_weighted_overlap(q_tokens, t.split(), idf_title))
        coverage_desc.append(_idf_weighted_overlap(q_tokens, d.split(), idf_desc))
        coverage_bullet.append(_idf_weighted_overlap(q_tokens, b.split(), idf_bullet))
        coverage_all.append(_idf_weighted_overlap(q_tokens, a.split(), idf_all))

    return pd.DataFrame(
        {
            "coverage_title_idf": coverage_title,
            "coverage_desc_idf": coverage_desc,
            "coverage_bullet_idf": coverage_bullet,
            "coverage_all_idf": coverage_all,
        }
    )


def prepare_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Подготавливает категориальные признаки для CatBoost.

    Args:
        df: DataFrame с категориальными колонками.

    Returns:
        DataFrame с нормализованными категориальными признаками.
    """
    prepared = df.copy()
    for col in CATEGORICAL_COLUMNS:
        if col in prepared.columns:
            prepared[col] = prepared[col].fillna("").str.strip()
            if col == "product_color":
                prepared[col] = prepared[col].apply(normalize_color).replace("", "unknown")
            elif col == "product_brand":
                prepared[col] = (
                    prepared[col]
                    .str.lower()
                    .replace(
                        {
                            "unknown_brand": "unknown",
                            "brand_masked": "unknown",
                            "generic": "unknown",
                            "none": "unknown",
                            "": "unknown",
                        }
                    )
                )
            else:
                prepared[col] = prepared[col].replace("", "unknown")
    return prepared


def save_metadata(
    path: Path,
    feature_columns: List[str],
    cat_columns: List[str],
    te_mappings: Optional[Dict[str, Dict[str, float]]] = None,
    te_prior: Optional[float] = None,
    te_columns: Optional[List[str]] = None,
) -> None:
    """
    Сохраняет метаданные модели в JSON файл.

    Args:
        path: Путь к файлу для сохранения.
        feature_columns: Список названий признаков.
        cat_columns: Список категориальных колонок.
        te_mappings: Маппинги для target encoding.
        te_prior: Приор для target encoding.
        te_columns: Колонки для target encoding.
    """
    payload: Dict[str, object] = {
        "feature_columns": feature_columns,
        "categorical_columns": cat_columns,
    }
    if te_mappings is not None:
        payload["target_encoding"] = {
            "mappings": te_mappings,
            "prior": te_prior,
            "columns": te_columns,
        }
    path.write_text(json.dumps(payload, indent=2))


def load_metadata(path: Path) -> Dict[str, object]:
    """
    Загружает метаданные модели из JSON файла.

    Args:
        path: Путь к файлу с метаданными.

    Returns:
        Словарь с метаданными.
    """
    return json.loads(path.read_text())


def compute_target_encoding(
    df: pd.DataFrame,
    target: pd.Series,
    columns: List[str],
    folds: int,
    groups: pd.Series,
    smoothing: float = 10.0,
) -> tuple[pd.DataFrame, Dict[str, Dict[str, float]], float]:
    """
    Вычисляет target encoding без утечки данных через GroupKFold.

    Args:
        df: DataFrame с данными.
        target: Целевая переменная.
        columns: Колонки для кодирования.
        folds: Количество фолдов для GroupKFold.
        groups: Группы для группировки (query_id).
        smoothing: Параметр сглаживания.

    Returns:
        Tuple из DataFrame с закодированными признаками, маппингов для инференса и приора.
    """
    prior = float(target.mean())
    enc_features: Dict[str, List[float]] = {f"te_{col}": [prior] * len(df) for col in columns}
    mappings: Dict[str, Dict[str, float]] = {}

    gkf = GroupKFold(n_splits=folds)
    for train_idx, val_idx in gkf.split(df, target, groups=groups):
        fold_df = df.iloc[train_idx]
        fold_target = target.iloc[train_idx]
        for col in columns:
            stats = fold_df.groupby(col)[fold_target.name].agg(["sum", "count"])
            stats["te"] = (stats["sum"] + prior * smoothing) / (stats["count"] + smoothing)
            mapping = stats["te"].to_dict()
            enc_series = df[col].iloc[val_idx].map(mapping).fillna(prior)
            enc_array = np.asarray(enc_features[f"te_{col}"], dtype=float)
            enc_array[val_idx] = enc_series.to_numpy()
            enc_features[f"te_{col}"] = enc_array.tolist()

    for col in columns:
        stats_full = df.groupby(col)[target.name].agg(["sum", "count"])
        stats_full["te"] = (stats_full["sum"] + prior * smoothing) / (stats_full["count"] + smoothing)
        mappings[col] = stats_full["te"].to_dict()

    te_df = pd.DataFrame(enc_features)
    return te_df, mappings, prior


def apply_target_encoding(df: pd.DataFrame, columns: List[str], mappings: Dict[str, Dict[str, float]], prior: float) -> pd.DataFrame:
    """
    Применяет target encoding к данным на основе предвычисленных маппингов.

    Args:
        df: DataFrame с данными.
        columns: Колонки для кодирования.
        mappings: Маппинги значений к закодированным значениям.
        prior: Приор для неизвестных значений.

    Returns:
        DataFrame с закодированными признаками.
    """
    te_data: Dict[str, List[float]] = {}
    for col in columns:
        mapping = mappings.get(col, {})
        te_data[f"te_{col}"] = df[col].map(mapping).fillna(prior)
    return pd.DataFrame(te_data)
