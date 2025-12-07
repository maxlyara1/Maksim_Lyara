import re
from typing import Iterable, Optional

import ftfy
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from .constants import DROP_COLUMNS, TEXT_COLUMNS

NULL_MARKERS = {"none", "null", "[null]", "nan", "na", "n/a", "undefined", "0", "1"}
LEADING_NOISE_RE = re.compile(r"^[\s\W_]+", flags=re.UNICODE)
MAX_LONG_TEXT_CHARS = 256
RELEVANT_WINDOW = 160
PLACEHOLDER_MIN_COUNT = 8
PLACEHOLDER_MAX_LEN = 320
PLACEHOLDER_MIN_LEN = 3
PLACEHOLDER_MAX_QUERY_OVERLAP = 0.1


def _normalize_null(text: Optional[str]) -> str:
    """Нормализует null-значения к пустой строке."""
    if text is None:
        return ""
    if isinstance(text, float) and np.isnan(text):
        return ""
    raw = str(text).strip()
    if raw in {"None"}:
        return ""
    if raw.lower() in NULL_MARKERS:
        return ""
    return raw


def clean_text(text: Optional[str]) -> str:
    """
    Очищает текст от HTML, исправляет кодировку и нормализует.

    Args:
        text: Исходный текст.

    Returns:
        Очищенный и нормализованный текст.
    """
    normalized = _normalize_null(text)
    if not normalized:
        return ""
    try:
        soup = BeautifulSoup(normalized, "lxml")
        cleaned = soup.get_text(separator=" ")
    except Exception:
        cleaned = normalized
    fixed = ftfy.fix_text(cleaned)
    stripped = LEADING_NOISE_RE.sub("", fixed)
    lowered = stripped.lower()
    return " ".join(lowered.split())


def _drop_placeholders(text: str) -> str:
    """Устаревшая функция для обратной совместимости."""
    return text


def _find_common_placeholders(texts: pd.Series, queries: pd.Series) -> set[str]:
    """
    Находит шаблонные строки, которые часто повторяются и не пересекаются с запросами.

    Args:
        texts: Серия текстов для анализа.
        queries: Серия запросов для проверки пересечения.

    Returns:
        Множество шаблонных строк.
    """
    vc = texts.value_counts()
    candidates = vc[
        (vc >= PLACEHOLDER_MIN_COUNT)
        & (vc.index.str.len() >= PLACEHOLDER_MIN_LEN)
        & (vc.index.str.len() <= PLACEHOLDER_MAX_LEN)
    ]
    placeholders: set[str] = set()
    if candidates.empty:
        return placeholders
    query_tokens = queries.str.split()
    for text in candidates.index:
        idx = texts.index[texts == text]
        tokens_text = set(text.split())
        if not tokens_text:
            placeholders.add(text)
            continue
        overlap_frac = query_tokens.loc[idx].apply(lambda q: bool(tokens_text & set(q))).mean()
        if overlap_frac <= PLACEHOLDER_MAX_QUERY_OVERLAP:
            placeholders.add(text)
    return placeholders


def _strip_placeholders(texts: pd.Series, queries: pd.Series) -> pd.Series:
    """Удаляет шаблонные строки из текстов."""
    placeholders = _find_common_placeholders(texts, queries)
    if not placeholders:
        return texts
    return texts.where(~texts.isin(placeholders), "")


def _truncate_long(text: str, max_chars: int = MAX_LONG_TEXT_CHARS) -> str:
    """Обрезает текст до указанной длины."""
    if not text:
        return ""
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _extract_relevant_span(text: str, query: str, max_chars: int = MAX_LONG_TEXT_CHARS, window: int = RELEVANT_WINDOW) -> str:
    """
    Извлекает релевантный фрагмент текста вокруг первого совпавшего токена запроса.

    Args:
        text: Исходный текст.
        query: Запрос для поиска.
        max_chars: Максимальная длина результата.
        window: Размер окна вокруг найденного токена.

    Returns:
        Релевантный фрагмент текста.
    """
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    q_tokens = [t for t in query.split() if len(t) >= 3]
    if not q_tokens:
        return text[:max_chars]
    lower_text = text.lower()
    hit_pos = None
    for tok in q_tokens:
        pos = lower_text.find(tok)
        if pos != -1:
            hit_pos = pos
            break
    if hit_pos is None:
        return text[:max_chars]
    start = max(0, hit_pos - window)
    end = min(len(text), hit_pos + window)
    snippet = text[start:end]
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars]
    return snippet


def preprocess_dataframe(df: pd.DataFrame, text_columns: Iterable[str] = TEXT_COLUMNS) -> pd.DataFrame:
    """
    Предобрабатывает DataFrame: очищает текстовые колонки и извлекает релевантные фрагменты.

    Args:
        df: Исходный DataFrame.
        text_columns: Список текстовых колонок для обработки.

    Returns:
        Предобработанный DataFrame.
    """
    prepared = df.copy()
    for col in DROP_COLUMNS:
        if col in prepared.columns:
            prepared = prepared.drop(columns=[col])
    for col in text_columns:
        if col in prepared.columns:
            prepared[col] = prepared[col].apply(clean_text)
            if col in ["product_description", "product_bullet_point"]:
                prepared[col] = [
                    _extract_relevant_span(text, query) for text, query in zip(prepared[col], prepared["query"])
                ]
                prepared[col] = prepared[col].apply(_truncate_long)
                too_short = prepared[col].str.len() < 3
                prepared.loc[too_short, col] = ""
    return prepared
