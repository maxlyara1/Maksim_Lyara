from __future__ import annotations

import os
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from tqdm import tqdm

try:
    from catboost import CatBoostRegressor
except ImportError:
    raise SystemExit("Ошибка: библиотека catboost не найдена. Установите её через pip install catboost")

# --- КОНФИГУРАЦИЯ ---
@dataclass
class Config:
    train_path: str = "data/train.csv"
    test_path: str = "data/test.csv"
    submission_path: str = "results/submission.csv"

    # Сиды для ансамбля и системный сид
    system_seed: int = 322
    ensemble_seeds: Tuple[int, ...] = (322, 322, 322, 322, 322, 322, 322, 322, 322)

    train_window_days: int = 90

    # Параметры Uncertainty (CatBoost)
    unc_iterations: int = 3200
    unc_lr: float = 0.012
    unc_depth: int = 7
    unc_l2: float = 15.0

    # Параметры Quantile (CatBoost)
    q_iterations: int = 3800
    q_lr: float = 0.015
    q_depth: int = 6
    q_l2: float = 5.0

    # Параметры разнообразия ансамбля
    ensemble_depth_offsets: Tuple[int, ...] = (-2, -1, 0, 1, 2, -2, -1, 0, 1)
    ensemble_rsm: Tuple[float, ...] = (0.9, 0.95, 1.0, 0.9, 0.85, 0.8, 0.88, 0.75, 0.92)
    ensemble_bootstrap: Tuple[str, ...] = (
        "Bayesian", "Bayesian", "Bernoulli", "Bernoulli", "Bayesian",
        "Bernoulli", "Bayesian", "Bernoulli", "Bayesian"
    )
    ensemble_bagging_temperature: Tuple[Optional[float], ...] = (0.2, 0.7, None, None, 1.5, None, 1.2, None, 0.4)
    ensemble_subsample: Tuple[Optional[float], ...] = (None, None, 0.7, 0.85, None, 0.6, None, 0.8, None)

    # Общие параметры моделей
    min_data_in_leaf: int = 20
    
    # Генерация признаков
    smoothing: float = 50.0
    n_clusters: int = 5
    
    # Калибровка и Блендинг
    min_gamma_samples: int = 10
    blend_mu: float = 0.5
    blend_sigma: float = 0.5
    new_gamma_boost: float = 1.04
    
    # Сетки поиска
    tune_blend: bool = True
    blend_mu_grid: Tuple[float, ...] = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8)
    blend_sigma_grid: Tuple[float, ...] = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8)
    
    tune_new_gamma_boost: bool = False
    new_gamma_boost_grid: Tuple[float, ...] = (0.9, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.1)
    
    # Параметры оптимизации Gamma
    gamma_min: float = 0.05
    gamma_max: float = 4.0
    gamma_steps: int = 160
    
    # Валидация
    proxy_days: int = 30
    proxy_new_products: int = 240
    proxy_stratify_ratio: float = 1.0
    
    # Пост-процессинг (сглаживание)
    use_smoothing: bool = True
    smoothing_alpha_old: float = 0.5
    smoothing_alpha_new: float = 0.6
    tune_smoothing_split: bool = True
    smoothing_alpha_old_grid: Tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
    smoothing_alpha_new_grid: Tuple[float, ...] = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    
    print_proxy_iou: bool = True


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_device_type() -> str:
    """Определяет доступное устройство: GPU -> CPU"""
    try:
        from catboost.utils import get_gpu_device_count
        if get_gpu_device_count() > 0:
            return "GPU"
    except Exception:
        pass
        
    return "CPU"

def create_submission(predictions: pd.DataFrame | np.ndarray) -> str:
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """
    cfg = Config()
    if not isinstance(predictions, pd.DataFrame):
        predictions = pd.DataFrame(predictions, columns=["row_id", "price_p05", "price_p95"])
    
    # Валидация колонок
    required = {"row_id", "price_p05", "price_p95"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Отсутствуют колонки: {required - set(predictions.columns)}")

    os.makedirs(os.path.dirname(cfg.submission_path), exist_ok=True)
    
    out_df = predictions[["row_id", "price_p05", "price_p95"]]
    out_df.to_csv(cfg.submission_path, index=False)
    print(f"Submission файл сохранен: {cfg.submission_path}")
    return cfg.submission_path

def iou_1d_metric(y_l: np.ndarray, y_h: np.ndarray, p_l: np.ndarray, p_h: np.ndarray) -> float:
    """Метрика IoU для 1D интервалов."""
    epsilon = 1e-6
    # Пересечение
    inter_l = np.maximum(y_l, p_l)
    inter_h = np.minimum(y_h, p_h)
    intersection = np.maximum(0, inter_h - inter_l)
    
    # Объединение
    w_true = (y_h - y_l) + epsilon
    w_pred = (p_h - p_l) + epsilon
    union = w_true + w_pred - intersection
    
    return np.mean(intersection / union)


# --- ОСНОВНОЙ КЛАСС МОДЕЛИ ---
class PricingModel:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.task_type = get_device_type()
        print(f">>> Device detected for CatBoost: {self.task_type}")
        
    def _get_te(self, train_df: pd.DataFrame, test_df: pd.DataFrame, col: str, target: str) -> Tuple[np.ndarray, pd.Series]:
        """Target Encoding с KFold регуляризацией."""
        train_res = np.zeros(len(train_df))
        kf = KFold(n_splits=5, shuffle=True, random_state=self.cfg.system_seed)
        global_mean = train_df[target].mean()
        
        # Для трейна считаем фолдами, чтобы избежать утечки данных
        for tr_idx, val_idx in kf.split(train_df):
            X_tr, X_val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
            stats = X_tr.groupby(col)[target].agg(["mean", "count"])
            
            m = X_val[col].map(stats["mean"])
            c = X_val[col].map(stats["count"]).fillna(0)
            # Байесовское сглаживание: взвешенное среднее между групповым средним и глобальным
            train_res[val_idx] = (m * c + self.cfg.smoothing * global_mean) / (c + self.cfg.smoothing)
            
        # Для теста берем статистику со всего трейна
        stats_all = train_df.groupby(col)[target].agg(["mean", "count"])
        counts = test_df[col].map(stats_all["count"]).fillna(0)
        means = test_df[col].map(stats_all["mean"]).fillna(global_mean)
        test_res = (means * counts + self.cfg.smoothing * global_mean) / (counts + self.cfg.smoothing)
        
        return train_res, test_res.fillna(global_mean)

    def feature_engineering(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Генерация признаков."""
        tr, ts = train_df.copy(), test_df.copy()
        
        # 1. Кластеризация поведения товаров
        aggs = tr.groupby("product_id").agg({
            "price_mid": ["mean", "std"], 
            "price_spread": ["mean"], 
            "activity_flag": ["mean"]
        })
        aggs.columns = ["_".join(c) for c in aggs.columns]
        aggs.fillna(0, inplace=True)
        
        # Масштабирование и кластеризация
        scaler = RobustScaler()
        X_cluster = scaler.fit_transform(aggs)
        kmeans = KMeans(n_clusters=self.cfg.n_clusters, random_state=self.cfg.system_seed, n_init=10)
        aggs["cluster_id"] = kmeans.fit_predict(X_cluster)
        
        # Для новых товаров в тесте используем самый частый кластер как fallback
        common_cluster = aggs["cluster_id"].mode()[0]
        
        tr = tr.merge(aggs[["cluster_id"]], on="product_id", how="left")
        tr["cluster_id"] = tr["cluster_id"].fillna(common_cluster).astype(int)
        
        ts = ts.merge(aggs[["cluster_id"]], on="product_id", how="left")
        ts["cluster_id"] = ts["cluster_id"].fillna(common_cluster).astype(int)

        # 2. Target Encoding
        hier = ["management_group_id", "first_category_id", "second_category_id", "third_category_id", "product_id"]
        for col in hier:
            tr[f"te_lvl_{col}"], ts[f"te_lvl_{col}"] = self._get_te(tr, ts, col, "log_mid")
            tr[f"te_spr_{col}"], ts[f"te_spr_{col}"] = self._get_te(tr, ts, col, "log_spread")

        # 3. Контекстные и календарные признаки
        for df in [tr, ts]:
            df["dow"] = df["dt"].dt.dayofweek
            df["day"] = df["dt"].dt.day
            # Относительный объем: как товар соотносится со своей категорией
            df["rel_vol"] = df["te_lvl_product_id"] / (df["te_lvl_third_category_id"] + 1e-9)
            # Взаимодействие активности и кластера: разные кластеры по-разному реагируют на акции
            df["act_x_clust"] = df["activity_flag"] * df["cluster_id"]

        # 4. Детекция аномалий
        anom_feats = ["te_lvl_product_id", "te_spr_product_id", "activity_flag"]
        iso = IsolationForest(contamination=0.03, random_state=self.cfg.system_seed)
        iso.fit(tr[anom_feats])
        tr["anom"] = iso.decision_function(tr[anom_feats])
        ts["anom"] = iso.decision_function(ts[anom_feats])
        
        return tr, ts

    def _get_asymmetric_preds(self, mu, sigma, g_low, g_high):
        """Расчет интервалов на основе mu, sigma и множителей gamma."""
        p_l = np.expm1(mu - g_low * sigma)
        p_h = np.expm1(mu + g_high * sigma)
        # Гарантируем, что lower >= 0 и lower < upper
        return np.maximum(0, p_l), np.maximum(p_l + 1e-6, p_h)

    def _optimize_gammas(self, mu, sigma, true_l, true_h):
        """Жадный поиск оптимальных множителей Gamma для асимметричных интервалов."""
        grid = np.linspace(self.cfg.gamma_min, self.cfg.gamma_max, self.cfg.gamma_steps)
        
        # Сначала оптимизируем верхнюю границу (при нижней = 1.0)
        best_h, best_s = 1.0, -1.0
        for h in grid:
            pl, ph = self._get_asymmetric_preds(mu, sigma, 1.0, h)
            score = iou_1d_metric(true_l, true_h, pl, ph)
            if score > best_s:
                best_s, best_h = score, h
                
        # Затем оптимизируем нижнюю границу при найденной верхней
        best_l, best_s = 1.0, -1.0
        for l in grid:
            pl, ph = self._get_asymmetric_preds(mu, sigma, l, best_h)
            score = iou_1d_metric(true_l, true_h, pl, ph)
            if score > best_s:
                best_s, best_l = score, l
                
        return best_l, best_h

    def _apply_smoothing(self, df: pd.DataFrame, p05: np.ndarray, p95: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        """Экспоненциальное сглаживание предсказаний."""
        if alpha >= 1.0:
            return p05, p95
            
        temp = pd.DataFrame({
            "product_id": df["product_id"],
            "dt": df["dt"],
            "p05": p05,
            "p95": p95,
            "idx": np.arange(len(df))  # Сохраняем исходный порядок для восстановления
        })
        # Сортируем по времени для корректного экспоненциального сглаживания
        temp = temp.sort_values(by=["product_id", "dt"])
        
        # Сглаживаем по каждому товару отдельно
        grouped = temp.groupby("product_id", sort=False)
        temp["p05"] = grouped["p05"].transform(lambda x: x.ewm(alpha=alpha, adjust=False).mean())
        temp["p95"] = grouped["p95"].transform(lambda x: x.ewm(alpha=alpha, adjust=False).mean())
        
        # Возвращаем в исходном порядке (как в test_df)
        temp = temp.sort_values(by="idx")
        return temp["p05"].values, temp["p95"].values

    def run(self):
        """Основной пайплайн выполнения."""
        print(">>> Loading Data...")
        train_full = pd.read_csv(self.cfg.train_path, parse_dates=["dt"])
        test_df = pd.read_csv(self.cfg.test_path, parse_dates=["dt"])
        
        # Подготовка таргетов в логарифмической шкале
        train_full["price_mid"] = (train_full["price_p05"] + train_full["price_p95"]) / 2.0
        train_full["price_spread"] = train_full["price_p95"] - train_full["price_p05"]
        train_full["log_mid"] = np.log1p(train_full["price_mid"])
        train_full["log_spread"] = np.log1p(train_full["price_spread"])
        
        # Отбираем последние N дней для трейна
        last_dates = sorted(train_full["dt"].unique())[-self.cfg.train_window_days:]
        train_df = train_full[train_full["dt"].isin(last_dates)].copy()
        
        print(">>> Feature Engineering...")
        train_df, test_df = self.feature_engineering(train_df, test_df)
        
        features = [c for c in train_df.columns if c.startswith(("te_", "rel_", "anom", "is_", "act_")) 
                    or c in ["activity_flag", "holiday_flag", "dow", "day",
                             "product_id", "cluster_id", "management_group_id",
                             "first_category_id", "second_category_id", "third_category_id"]]
        cat_feats = [
            "product_id", "cluster_id", "management_group_id",
            "first_category_id", "second_category_id", "third_category_id"
        ]
        
        dates = sorted(train_df["dt"].unique())
        v_dates = dates[-self.cfg.proxy_days:]
        t_dates = dates[:-self.cfg.proxy_days]
        
        # Выделяем прокси новых товаров для симуляции холодного старта
        # Стратифицируем по management_group_id, чтобы распределение было похоже на тест
        rng = np.random.default_rng(self.cfg.system_seed)
        train_products = train_df["product_id"].unique()
        train_prod_mode_mg = (
            train_df.groupby("product_id")["management_group_id"]
            .agg(lambda s: s.mode().iat[0])
        )
        test_new_products = np.setdiff1d(test_df["product_id"].unique(), train_products, assume_unique=False)
        test_new_mode_mg = (
            test_df[test_df["product_id"].isin(test_new_products)]
            .groupby("product_id")["management_group_id"]
            .agg(lambda s: s.mode().iat[0])
        )
        # Распределение management_group_id среди новых товаров в тесте
        mg_probs = test_new_mode_mg.value_counts(normalize=True)

        # Вычисляем количество товаров для каждого management_group_id
        target_n = int(round(self.cfg.proxy_new_products * self.cfg.proxy_stratify_ratio))
        raw_counts = mg_probs * target_n
        base_counts = np.floor(raw_counts).astype(int)
        remainder = int(target_n - base_counts.sum())
        # Распределяем остаток по группам с наибольшими дробными частями
        if remainder > 0:
            frac = (raw_counts - base_counts).sort_values(ascending=False)
            for mg in frac.index[:remainder]:
                base_counts.loc[mg] += 1

        # Выбираем товары из трейна, сохраняя распределение по management_group_id
        selected = []
        selected_set = set()
        for mg, cnt in base_counts.items():
            if cnt <= 0:
                continue
            candidates = train_prod_mode_mg[train_prod_mode_mg == mg].index.values
            if len(candidates) == 0:
                continue
            if cnt > len(candidates):
                cnt = len(candidates)
            picks = rng.choice(candidates, size=cnt, replace=False).tolist()
            selected.extend(picks)
            selected_set.update(picks)

        # Если не хватило товаров, добираем случайные из оставшихся
        if len(selected) < self.cfg.proxy_new_products:
            remaining = np.array([p for p in train_products if p not in selected_set])
            extra = rng.choice(remaining, size=self.cfg.proxy_new_products - len(selected), replace=False)
            selected.extend(extra.tolist())

        proxy_prods = np.array(selected)
        
        p_train = train_df[train_df["dt"].isin(t_dates) & (~train_df["product_id"].isin(proxy_prods))]
        p_val = train_df[train_df["dt"].isin(v_dates)]
        
        # Аккумуляторы предсказаний
        preds = {
            "p_unc_mu": np.zeros(len(p_val)), "p_unc_var": np.zeros(len(p_val)),
            "p_q_p05": np.zeros(len(p_val)), "p_q_p50": np.zeros(len(p_val)), "p_q_p95": np.zeros(len(p_val)),
            "t_unc_mu": np.zeros(len(test_df)), "t_unc_var": np.zeros(len(test_df)),
            "t_q_p05": np.zeros(len(test_df)), "t_q_p50": np.zeros(len(test_df)), "t_q_p95": np.zeros(len(test_df))
        }

        n_ens = len(self.cfg.ensemble_seeds)
        if not all(len(x) == n_ens for x in (
            self.cfg.ensemble_depth_offsets,
            self.cfg.ensemble_rsm,
            self.cfg.ensemble_bootstrap,
            self.cfg.ensemble_bagging_temperature,
            self.cfg.ensemble_subsample,
        )):
            raise ValueError("Ensemble config lengths must match")
        ensemble_plan = list(zip(
            self.cfg.ensemble_seeds,
            self.cfg.ensemble_depth_offsets,
            self.cfg.ensemble_rsm,
            self.cfg.ensemble_bootstrap,
            self.cfg.ensemble_bagging_temperature,
            self.cfg.ensemble_subsample,
        ))

        print(f">>> Training Ensemble ({len(ensemble_plan) * 2} models)...")

        for seed, depth_offset, rsm, bootstrap_type, bagging_temperature, subsample in tqdm(ensemble_plan, desc="Ensemble"):
            set_seed(seed)
            unc_depth = max(2, self.cfg.unc_depth + depth_offset)
            q_depth = max(2, self.cfg.q_depth + depth_offset)
            
            # --- 1. Модель неопределенности ---
            params_unc = {
                "loss_function": "RMSEWithUncertainty", "iterations": self.cfg.unc_iterations, 
                "learning_rate": self.cfg.unc_lr, "depth": unc_depth, 
                "l2_leaf_reg": self.cfg.unc_l2, "min_data_in_leaf": self.cfg.min_data_in_leaf,
                "verbose": 0, "random_seed": seed, "task_type": self.task_type, "allow_writing_files": False
            }
            params_unc["rsm"] = rsm
            params_unc["bootstrap_type"] = bootstrap_type
            if bagging_temperature is not None:
                params_unc["bagging_temperature"] = bagging_temperature
            if subsample is not None:
                params_unc["subsample"] = subsample
            
            m_unc = CatBoostRegressor(**params_unc)
            m_unc.fit(p_train[features], p_train["log_mid"], cat_features=cat_feats, 
                      eval_set=(p_val[features], p_val["log_mid"]), use_best_model=True)
            pp = m_unc.predict(p_val[features])
            preds["p_unc_mu"] += pp[:, 0]
            preds["p_unc_var"] += pp[:, 1]
            
            # Переобучаем на полных данных с +15% итераций от лучшей итерации на валидации
            m_unc_full = CatBoostRegressor(**params_unc)
            m_unc_full.set_params(iterations=int(m_unc.get_best_iteration() * 1.15))
            m_unc_full.fit(train_df[features], train_df["log_mid"], cat_features=cat_feats)
            tp = m_unc_full.predict(test_df[features])
            preds["t_unc_mu"] += tp[:, 0]
            preds["t_unc_var"] += tp[:, 1]
            
            # --- 2. Quantile Model ---
            params_q = {
                "loss_function": "MultiQuantile:alpha=0.05,0.5,0.95", "iterations": self.cfg.q_iterations,
                "learning_rate": self.cfg.q_lr, "depth": q_depth, 
                "l2_leaf_reg": self.cfg.q_l2, "min_data_in_leaf": self.cfg.min_data_in_leaf,
                "verbose": 0, "random_seed": seed, "task_type": self.task_type, "allow_writing_files": False
            }
            params_q["rsm"] = rsm
            params_q["bootstrap_type"] = bootstrap_type
            if bagging_temperature is not None:
                params_q["bagging_temperature"] = bagging_temperature
            if subsample is not None:
                params_q["subsample"] = subsample
            
            m_q = CatBoostRegressor(**params_q)
            m_q.fit(p_train[features], p_train["log_mid"], cat_features=cat_feats, 
                    eval_set=(p_val[features], p_val["log_mid"]), use_best_model=True)
            pp_q = m_q.predict(p_val[features])
            preds["p_q_p05"] += pp_q[:, 0]; preds["p_q_p50"] += pp_q[:, 1]; preds["p_q_p95"] += pp_q[:, 2]
            
            # Переобучаем на полных данных с +15% итераций от лучшей итерации на валидации
            m_q_full = CatBoostRegressor(**params_q)
            m_q_full.set_params(iterations=int(m_q.get_best_iteration() * 1.15))
            m_q_full.fit(train_df[features], train_df["log_mid"], cat_features=cat_feats)
            tp_q = m_q_full.predict(test_df[features])
            preds["t_q_p05"] += tp_q[:, 0]; preds["t_q_p50"] += tp_q[:, 1]; preds["t_q_p95"] += tp_q[:, 2]

        # Усреднение ансамбля
        n = len(ensemble_plan)
        for k in preds: preds[k] /= n
        
        # --- КАЛИБРОВКА ---
        print("\n>>> Calibrating & Blending...")
        
        # Вспомогательная функция для блендинга двух подходов
        def blend_dist(unc_mu, unc_var, q_p05, q_p50, q_p95, w_mu, w_sigma):
            unc_sigma = np.sqrt(np.maximum(unc_var, 1e-9))
            # Преобразуем квантильный интервал в sigma: (p95 - p05) / (2 * 1.645) ≈ sigma для нормального распределения
            q_sigma = np.maximum((q_p95 - q_p05) / 3.29, 1e-9)
            # Взвешенное среднее для mu и sigma
            mu = w_mu * unc_mu + (1.0 - w_mu) * q_p50
            sigma = w_sigma * unc_sigma + (1.0 - w_sigma) * q_sigma
            return mu, np.maximum(sigma, 1e-9)

        # Маски для валидации
        is_new_p = p_val["product_id"].isin(proxy_prods).values
        clusters = p_val["cluster_id"].unique()
        
        # 1. Подбор весов блендинга
        if self.cfg.tune_blend:
            best_sc, best_w = -1.0, (self.cfg.blend_mu, self.cfg.blend_sigma)
            for w_mu in self.cfg.blend_mu_grid:
                for w_sigma in self.cfg.blend_sigma_grid:
                    mu, sig = blend_dist(preds["p_unc_mu"], preds["p_unc_var"], 
                                         preds["p_q_p05"], preds["p_q_p50"], preds["p_q_p95"], w_mu, w_sigma)
                    gl, gh = self._optimize_gammas(mu, sig, p_val["price_p05"].values, p_val["price_p95"].values)
                    p05, p95 = self._get_asymmetric_preds(mu, sig, gl, gh)
                    sc = iou_1d_metric(p_val["price_p05"].values, p_val["price_p95"].values, p05, p95)
                    if sc > best_sc: best_sc, best_w = sc, (w_mu, w_sigma)
            self.cfg.blend_mu, self.cfg.blend_sigma = best_w
            print(f"   Best Blend: mu={best_w[0]}, sigma={best_w[1]}")

        # Финальные распределения
        mu_p, sigma_p = blend_dist(preds["p_unc_mu"], preds["p_unc_var"], 
                                   preds["p_q_p05"], preds["p_q_p50"], preds["p_q_p95"], 
                                   self.cfg.blend_mu, self.cfg.blend_sigma)
        mu_t, sigma_t = blend_dist(preds["t_unc_mu"], preds["t_unc_var"], 
                                   preds["t_q_p05"], preds["t_q_p50"], preds["t_q_p95"], 
                                   self.cfg.blend_mu, self.cfg.blend_sigma)

        # 2. Детальная калибровка гамм с иерархией: глобальная -> группа -> кластер -> кластер+активность
        calib_state = {}
        
        # Глобальные гаммы для новых и старых товаров отдельно
        calib_state["glob_new"] = self._optimize_gammas(mu_p[is_new_p], sigma_p[is_new_p], 
                                                        p_val.loc[is_new_p, "price_p05"], p_val.loc[is_new_p, "price_p95"])
        calib_state["glob_old"] = self._optimize_gammas(mu_p[~is_new_p], sigma_p[~is_new_p], 
                                                        p_val.loc[~is_new_p, "price_p05"], p_val.loc[~is_new_p, "price_p95"])
        
        # По группам для новых товаров (если достаточно данных)
        calib_state["groups"] = {}
        for grp in p_val["management_group_id"].unique():
            mask = is_new_p & (p_val["management_group_id"] == grp)
            if mask.sum() >= self.cfg.min_gamma_samples:
                calib_state["groups"][grp] = self._optimize_gammas(mu_p[mask], sigma_p[mask], 
                                                                   p_val.loc[mask, "price_p05"], p_val.loc[mask, "price_p95"])
        
        # По кластерам для старых товаров
        calib_state["clusters"] = {}
        calib_state["clusters_act"] = {}
        for c in clusters:
            mask = (~is_new_p) & (p_val["cluster_id"] == c)
            if mask.sum() >= self.cfg.min_gamma_samples:
                calib_state["clusters"][c] = self._optimize_gammas(mu_p[mask], sigma_p[mask], 
                                                                   p_val.loc[mask, "price_p05"], p_val.loc[mask, "price_p95"])
            # Дополнительно по активности внутри кластера (акции меняют поведение цен)
            for act in [0, 1]:
                mask_a = mask & (p_val["activity_flag"] == act)
                if mask_a.sum() >= self.cfg.min_gamma_samples:
                    calib_state["clusters_act"][(c, act)] = self._optimize_gammas(mu_p[mask_a], sigma_p[mask_a], 
                                                                                  p_val.loc[mask_a, "price_p05"], p_val.loc[mask_a, "price_p95"])

        # Функция применения гамм с иерархией: специфичные -> групповые -> глобальные
        def apply_gammas(df, is_new_mask, gamma_boost=1.0):
            gl_vec, gh_vec = np.zeros(len(df)), np.zeros(len(df))
            
            # Для старых товаров: кластер -> кластер+активность -> глобальные
            for c in df["cluster_id"].unique():
                base_gl, base_gh = calib_state["clusters"].get(c, calib_state["glob_old"])
                for act in [0, 1]:
                    mask = (df["cluster_id"] == c) & (df["activity_flag"] == act) & (~is_new_mask)
                    spec_gl, spec_gh = calib_state["clusters_act"].get((c, act), (base_gl, base_gh))
                    gl_vec[mask], gh_vec[mask] = spec_gl, spec_gh
            
            # Для новых товаров: группа -> глобальные (с бустом для учета неопределенности)
            for grp in df.loc[is_new_mask, "management_group_id"].unique():
                mask = is_new_mask & (df["management_group_id"] == grp)
                spec_gl, spec_gh = calib_state["groups"].get(grp, calib_state["glob_new"])
                gl_vec[mask], gh_vec[mask] = spec_gl * gamma_boost, spec_gh * gamma_boost
                
            return gl_vec, gh_vec

        # 3. Тюнинг буста для новых товаров
        if self.cfg.tune_new_gamma_boost:
            best_boost, best_sc = self.cfg.new_gamma_boost, -1.0
            for b in self.cfg.new_gamma_boost_grid:
                gl, gh = apply_gammas(p_val, is_new_p, gamma_boost=b)
                p05, p95 = self._get_asymmetric_preds(mu_p, sigma_p, gl, gh)
                sc = iou_1d_metric(p_val["price_p05"].values, p_val["price_p95"].values, p05, p95)
                if sc > best_sc: best_sc, best_boost = sc, b
            self.cfg.new_gamma_boost = best_boost
            print(f"   Best Gamma Boost: {self.cfg.new_gamma_boost}")

        # 4. Тюнинг сглаживания (разные параметры для старых и новых товаров)
        gl_p, gh_p = apply_gammas(p_val, is_new_p, self.cfg.new_gamma_boost)
        p05_p, p95_p = self._get_asymmetric_preds(mu_p, sigma_p, gl_p, gh_p)
        
        if self.cfg.tune_smoothing_split:
            best_sc, best_a_old, best_a_new = -1.0, self.cfg.smoothing_alpha_old, self.cfg.smoothing_alpha_new
            for ao in self.cfg.smoothing_alpha_old_grid:
                for an in self.cfg.smoothing_alpha_new_grid:
                    s05_old, s95_old = self._apply_smoothing(p_val[~is_new_p], p05_p[~is_new_p], p95_p[~is_new_p], ao)
                    s05_new, s95_new = self._apply_smoothing(p_val[is_new_p], p05_p[is_new_p], p95_p[is_new_p], an)
                    
                    # Собираем обратно в исходном порядке
                    full_05, full_95 = p05_p.copy(), p95_p.copy()
                    full_05[~is_new_p], full_95[~is_new_p] = s05_old, s95_old
                    full_05[is_new_p], full_95[is_new_p] = s05_new, s95_new
                    
                    sc = iou_1d_metric(p_val["price_p05"].values, p_val["price_p95"].values, full_05, full_95)
                    if sc > best_sc: best_sc, best_a_old, best_a_new = sc, ao, an
            self.cfg.smoothing_alpha_old, self.cfg.smoothing_alpha_new = best_a_old, best_a_new
            print(f"   Best Smoothing: old={best_a_old}, new={best_a_new}, IoU={best_sc:.6f}")

        # --- ФИНАЛЬНОЕ ПРЕДСКАЗАНИЕ ---
        print(">>> Generating Submission...")
        # Определяем новые товары в тесте (не встречались в трейне)
        is_new_t = ~test_df["product_id"].isin(train_df["product_id"].unique())
        
        # Применяем калиброванные гаммы и преобразуем в исходную шкалу
        gl_t, gh_t = apply_gammas(test_df, is_new_t, self.cfg.new_gamma_boost)
        p05_t, p95_t = self._get_asymmetric_preds(mu_t, sigma_t, gl_t, gh_t)
        
        if self.cfg.use_smoothing:
            # Применяем сглаживание раздельно для старых и новых товаров
            if (~is_new_t).sum() > 0:
                s05, s95 = self._apply_smoothing(test_df[~is_new_t], p05_t[~is_new_t], p95_t[~is_new_t], self.cfg.smoothing_alpha_old)
                p05_t[~is_new_t], p95_t[~is_new_t] = s05, s95
            if is_new_t.sum() > 0:
                s05, s95 = self._apply_smoothing(test_df[is_new_t], p05_t[is_new_t], p95_t[is_new_t], self.cfg.smoothing_alpha_new)
                p05_t[is_new_t], p95_t[is_new_t] = s05, s95

        submission = pd.DataFrame({
            "row_id": test_df["row_id"],
            "price_p05": p05_t,
            "price_p95": p95_t
        })
        create_submission(submission)


def main():
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)
    
    Config.system_seed = 322
    set_seed(Config.system_seed)
    
    model = PricingModel(Config())
    model.run()
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)

if __name__ == "__main__":
    main()
