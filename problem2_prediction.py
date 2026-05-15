"""
problem2_prediction.py — 问题2: 就餐人数、营养需求与销售总额预测
============================================================
题目要求:
  根据餐厅销售记录，对该餐厅每天就餐人数、各类营养素需求量以及销售总额
  进行预测研究，并讨论预测模型的合理性和结果的可靠性。
  请给出2025年5月份工作日的就餐人数、各类营养素需求量以及销售总额预测结果。

解题思路:
  1. 数据准备: 按日汇总交易数据，构建连续时间序列
     - 补全缺失日期 (停业日填充 0)
     - 标记 is_closed 指示变量
  2. 特征工程 (3 类特征):
     (a) 时间特征: 星期几 (dow_0..dow_6), 是否周末, 月份, 日期, 周次
     (b) 滞后特征: lag_1, lag_2, lag_3, lag_7, lag_14
     (c) 滑动窗口统计: ma_3/7/14 (均值), std_3/7/14 (标准差)
  3. 预测模型 (4 种, 进行模型比较):
     - 基准模型 (Baseline): 历史同星期均值
     - SARIMA(1,1,1)(1,1,1,7): 季节性差分整合移动平均自回归
     - XGBoost: 梯度提升树集成学习
     - Ensemble (组合预测): 按 1/MAPE 加权融合
  4. 预测目标变量 (6 个):
     - total_orders: 每日就餐人数
     - total_sales: 每日销售总额
     - total_calories/protein/fat/carbohydrates: 每日营养素总量
  5. 模型评估: MAE, RMSE, MAPE
  6. 外推预测: 基于历史同星期×同月份均值外推 2025年5月

模型评估指标:
  - MAE (Mean Absolute Error): 预测误差的绝对平均值
  - RMSE (Root Mean Square Error): 对大误差更敏感
  - MAPE (Mean Absolute Percentage Error): 相对误差百分比

参考文献:
  [1]  Rodrigues M. et al. "Machine learning models for short-term demand
       forecasting in food catering services: A solution to reduce food waste."
       Journal of Cleaner Production, 2024.
       https://www.sciencedirect.com/science/article/pii/S0959652623044232
  [2]  Posch K. et al. "A Bayesian Approach for Predicting Food and Beverage
       Sales in Staff Canteens and Restaurants."
       International Journal of Forecasting, 2022.
       https://www.sciencedirect.com/science/article/pii/S0169207021001011
  [3]  Thomassey S. et al. "Machine Learning Based Restaurant Sales
       Forecasting." Machine Learning and Knowledge Extraction, 2022.
       https://www.mdpi.com/2504-4990/4/1/6
  [4]  Hyndman R.J., Athanasopoulos G. "Forecasting: Principles and Practice."
       3rd Edition. OTexts.
       https://otexts.com/fpp3/
  [12] Chen T., Guestrin C. "XGBoost: A Scalable Tree Boosting System."
       KDD 2016. https://arxiv.org/abs/1603.02754
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from data_loader import load_all_data
from config import (OUTPUT_DIR, COLORS, COLOR_CYCLE, RANDOM_SEED,
                     PREDICTION_YEAR, PREDICTION_MONTH)
from utils import (mape_score, smape_score, create_lag_features,
                   create_rolling_features, print_metrics_table)


class Problem2Prediction:
    """
    问题2: 多模型需求预测

    预测 6 个目标变量:
    - total_orders: 每日就餐人数
    - total_sales: 每日销售总额 (元)
    - total_calories: 每日热量需求 (kcal)
    - total_protein: 每日蛋白质需求 (g)
    - total_fat: 每日脂肪需求 (g)
    - total_carbohydrates: 每日碳水需求 (g)

    使用 4 种模型进行对比:
    1. Baseline (历史同星期均值)
    2. SARIMA(1,1,1)(1,1,1,7)
    3. XGBoost (梯度提升树)
    4. Ensemble (加权组合)
    """

    # 目标变量列表及中文标签
    TARGET_COLS = ['total_orders', 'total_sales', 'total_calories',
                   'total_protein', 'total_fat', 'total_carbohydrates']
    TARGET_NAMES = {
        'total_orders': 'Daily Diners',
        'total_sales': 'Daily Sales (Yuan)',
        'total_calories': 'Daily Calories (kcal)',
        'total_protein': 'Daily Protein (g)',
        'total_fat': 'Daily Fat (g)',
        'total_carbohydrates': 'Daily Carbs (g)',
    }

    def __init__(self, loader=None):
        """
        初始化预测模型

        Args:
            loader: DataLoader 实例或 None
        """
        print('\n' + '=' * 60)
        print('问题2: 需求预测模型')
        print('=' * 60)

        if loader is None:
            self.loader = load_all_data()
        else:
            self.loader = loader

        # 获取日级数据并设置日期索引
        self.df_daily = self.loader.get_daily_data().copy()
        self.df_daily = self.df_daily.set_index('date').sort_index()

        # 补全日期范围 (包含停业日, 如周末/节假日)
        full_idx = pd.date_range(
            start=self.df_daily.index.min(),
            end=self.df_daily.index.max(),
            freq='D'
        )
        self.df_daily = self.df_daily.reindex(full_idx)

        # 填充缺失的日期特征
        self.df_daily['day_of_week'] = self.df_daily.index.dayofweek
        self.df_daily['is_weekend'] = (
            self.df_daily['day_of_week'].isin([5, 6]).astype(int)
        )
        self.df_daily['month'] = self.df_daily.index.month
        self.df_daily['day'] = self.df_daily.index.day

        # 停业日检测: total_orders 为 NaN → 填 0, is_closed = 1
        for col in self.TARGET_COLS:
            self.df_daily[col] = self.df_daily[col].fillna(0)
        self.df_daily['is_closed'] = (
            (self.df_daily['total_orders'] == 0).astype(int)
        )

        self.results = {}

    def run(self):
        """
        运行完整预测流程

        步骤:
        1. 时间序列特征分析 (ADF, ACF/PACF)
        2. 构建并评估多模型
        3. 模型比较可视化
        4. 预测 2025年5月工作日
        """
        print('\n>>> 2.1 时间序列特征分析')
        self._time_series_analysis()

        print('\n>>> 2.2 构建预测模型')
        predictions = self._build_and_evaluate_models()

        print('\n>>> 2.3 模型比较与选择')
        self._model_comparison()

        print('\n>>> 2.4 预测2025年5月工作日')
        may_predictions = self._predict_may_2025()

        print('\n问题2预测完成!')
        return self.results

    def _time_series_analysis(self):
        """
        时间序列特征分析 — 输出 p2_time_series_overview.png 和 p2_acf_pacf.png

        分析方法:
        1. ADF (Augmented Dickey-Fuller) 平稳性检验
           H0: 序列存在单位根 (非平稳)
           若 p < 0.05 → 拒绝 H0 → 序列平稳
        2. ACF (Autocorrelation Function) 自相关图
           反映时间序列在不同滞后期的自相关系数
        3. PACF (Partial Autocorrelation Function) 偏自相关图
           反映排除中间滞后影响后的直接自相关
        """
        df = self.df_daily[self.df_daily['is_closed'] == 0].copy()

        # ==== 图1: 6个目标变量的时间序列总览 ====
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))

        for i, (col, name) in enumerate(self.TARGET_NAMES.items()):
            ax = axes[i // 2, i % 2]
            series = df[col].dropna()

            if len(series) > 20:
                # ADF 平稳性检验
                adf_result = adfuller(series.dropna(), autolag='AIC')
                is_stationary = adf_result[1] < 0.05

                ax.plot(series.index, series.values,
                       color=COLORS['primary'], linewidth=1, alpha=0.8)
                ax.set_title(f'{name}\n(ADF p={adf_result[1]:.4f}, '
                            f'{"Stationary" if is_stationary else "Non-stationary"})',
                            fontsize=10, fontweight='bold')
                ax.set_ylabel(name)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.tick_params(axis='x', rotation=45, labelsize=8)
            else:
                ax.plot(series.index, series.values,
                       color=COLORS['primary'])
                ax.set_title(name, fontsize=10)

        plt.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/p2_time_series_overview.png', dpi=300)
        plt.close()
        print('  已保存: p2_time_series_overview.png')

        # ==== 图2: ACF / PACF 分析 (以 total_orders 为例) ====
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        plot_acf(df['total_orders'].dropna(), lags=30, ax=axes[0])
        axes[0].set_title('ACF - Daily Orders', fontweight='bold')
        plot_pacf(df['total_orders'].dropna(), lags=30, ax=axes[1])
        axes[1].set_title('PACF - Daily Orders', fontweight='bold')
        plt.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/p2_acf_pacf.png', dpi=300)
        plt.close()
        print('  已保存: p2_acf_pacf.png')

    def _build_features(self, df, target_col):
        """
        构建预测特征矩阵

        特征类别 (共约 30+ 个特征):
        1. 时间特征 (10个):
           - day_of_week (0-6), is_weekend (0/1)
           - month (1-12), day (1-31), week_of_year (1-53)
           - dow_0 到 dow_6: 星期几的 one-hot 编码 (7个)

        2. 滞后特征 (5个):
           - lag_1, lag_2, lag_3, lag_7, lag_14
           - 原理: y_t 可能依赖 y_{t-1}, y_{t-7} 等历史观测值
           - lag_7 特别重要: 捕捉星期周期性

        3. 滑动窗口统计 (6个):
           - ma_3, ma_7, ma_14: 移动平均 (反映短期/中期趋势)
           - std_3, std_7, std_14: 移动标准差 (反映波动性)

        Args:
            df: 原始日级 DataFrame (需设置日期索引)
            target_col: 目标变量列名

        Returns:
            DataFrame: 包含所有特征的 DataFrame
        """
        data = df.copy()

        # ---- 时间特征 ----
        data['day_of_week'] = data.index.dayofweek
        data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
        data['month'] = data.index.month
        data['day'] = data.index.day
        data['week_of_year'] = data.index.isocalendar().week.astype(int)

        # 星期几 one-hot 编码
        for d in range(7):
            data[f'dow_{d}'] = (data['day_of_week'] == d).astype(int)

        # ---- 滞后特征 ----
        data = create_lag_features(
            data, target_col, lags=[1, 2, 3, 7, 14]
        )

        # ---- 滑动窗口统计 ----
        data = create_rolling_features(data, target_col, windows=[3, 7, 14])

        return data

    def _build_and_evaluate_models(self):
        """
        构建并比较 4 种预测模型

        模型列表:
        1. Baseline: 历史同星期均值
           - 方法: 对每个预测日期，取历史上所有相同星期几的观测均值
           - 优点: 简单直观，捕捉星期周期性
           - 缺点: 忽略趋势和季节性变化

        2. SARIMA(1,1,1)(1,1,1,7):
           - 非季节性阶数 (p,d,q) = (1,1,1): 处理趋势
           - 季节性阶数 (P,D,Q,s) = (1,1,1,7): 以7天为周期
           - 优点: 理论基础扎实，同时处理趋势和季节性
           - 缺点: 可能收敛失败，对大样本慢

        3. XGBoost:
           - 参数: n_estimators=100, max_depth=4, lr=0.1
           - 特征: 时间 + 滞后 + 滑动窗口 (~30个)
           - 使用 TimeSeriesSplit (3-fold) 进行交叉验证
           - 优点: 可捕捉复杂的非线性关系
           - 缺点: 需要足够的特征工程

        4. Ensemble (组合预测):
           - 方法: 按 1/MAPE 加权融合前3种模型
           - weight_model = (1/MAPE_model) / Σ(1/MAPE)
           - 优点: 降低单一模型风险，通常优于任一单模型
        """
        # 排除停业日 (is_closed=1) 进行建模
        df = self.df_daily[self.df_daily['is_closed'] == 0].copy()

        all_predictions = {}
        all_metrics = {}

        for target_col in self.TARGET_COLS:
            print(f'\n  --- 预测目标: {self.TARGET_NAMES[target_col]} ---')

            # 模型1: Baseline (历史同星期均值)
            baseline_preds = self._baseline_forecast(df, target_col)

            # 模型2: SARIMA
            try:
                sarima_preds = self._sarima_forecast(df, target_col)
            except Exception as e:
                print(f'    SARIMA 失败: {e}，使用 NaN 填充')
                sarima_preds = pd.Series(np.nan, index=df.index)

            # 模型3: XGBoost
            xgb_preds, xgb_feat_importance = self._xgboost_forecast(
                df, target_col
            )

            # 模型4: Ensemble (加权组合)
            ensemble_preds = self._ensemble_forecast(
                df, target_col, baseline_preds, sarima_preds, xgb_preds
            )

            # 存储预测结果
            pred_df = pd.DataFrame({
                'actual': df[target_col],
                'baseline': baseline_preds,
                'sarima': sarima_preds,
                'xgboost': xgb_preds,
                'ensemble': ensemble_preds,
            }, index=df.index)

            all_predictions[target_col] = pred_df

            # 计算各模型的评估指标
            metrics = {}
            actual = df[target_col].values

            for model_name in ['baseline', 'sarima', 'xgboost', 'ensemble']:
                pred = pred_df[model_name].values
                valid = ~np.isnan(pred) & ~np.isnan(actual)
                if valid.sum() > 0:
                    metrics[model_name] = {
                        'MAE': mean_absolute_error(actual[valid],
                                                    pred[valid]),
                        'RMSE': np.sqrt(mean_squared_error(actual[valid],
                                                            pred[valid])),
                        'MAPE': mape_score(actual[valid], pred[valid]),
                    }

            all_metrics[target_col] = metrics
            print_metrics_table(metrics, self.TARGET_NAMES[target_col])

            if xgb_feat_importance is not None:
                self.results[f'feat_imp_{target_col}'] = xgb_feat_importance

        self.results['predictions'] = all_predictions
        self.results['metrics'] = all_metrics

        return all_predictions

    def _baseline_forecast(self, df, target_col):
        """
        基准模型: 历史同星期均值预测

        算法:
        For each date d:
          1. 确定 d 是星期几
          2. 找到历史上所有相同星期几且早于 d 的日期
          3. 计算这些日期的目标变量均值作为预测值
          4. 若历史上无相同星期数据 (冷启动), 用全局均值

        特点:
        - 无训练过程，直接计算
        - 捕捉稳定的星期周期性模式
        - 无法捕捉趋势变化和新模式

        Args:
            df: 营业日 DataFrame
            target_col: 目标列名

        Returns:
            pd.Series: 与 df.index 对齐的预测值
        """
        preds = pd.Series(index=df.index, dtype=float)
        dow_values = df.index.dayofweek

        for i, date in enumerate(df.index):
            target_dow = date.dayofweek  # 属性而非方法
            # 历史同星期且早于当前日期的记录
            past_same_dow = (dow_values == target_dow) & (df.index < date)
            if past_same_dow.sum() > 0:
                preds.iloc[i] = df.loc[past_same_dow, target_col].mean()
            else:
                preds.iloc[i] = df[target_col].mean()

        return preds

    def _sarima_forecast(self, df, target_col):
        """
        SARIMA 模型预测

        模型规格: SARIMA(1,1,1)(1,1,1,7)
        - 非季节性 AR(1): y_t 依赖 y_{t-1} (自回归)
        - 非季节性 I(1): 1阶差分 (去除趋势)
        - 非季节性 MA(1): 误差项依赖上一时刻误差 (移动平均)
        - 季节性 P=1, D=1, Q=1, s=7: 以7天为周期

        这等价于:
          (1 - φ₁B)(1 - Φ₁B⁷)(1 - B)(1 - B⁷)y_t
        = (1 + θ₁B)(1 + Θ₁B⁷)ε_t

        其中 B 是滞后算子: B^k y_t = y_{t-k}

        Args:
            df: 营业日 DataFrame
            target_col: 目标列名

        Returns:
            pd.Series: 拟合值序列
        """
        series = df[target_col].dropna().values

        try:
            model = SARIMAX(
                series,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False, maxiter=100)
            fitted_values = fitted.fittedvalues

            # 将拟合值映射回日期索引
            preds = pd.Series(index=df.index, dtype=float)
            n_fitted = len(fitted_values)
            if n_fitted > 0 and n_fitted <= len(df):
                # SARIMAX 因差分导致拟合值序列可能短于原始序列
                start_idx = len(df) - n_fitted
                for j in range(n_fitted):
                    preds.iloc[start_idx + j] = fitted_values[j]

            return preds
        except Exception as e:
            print(f'    SARIMA 拟合失败: {e}')
            return pd.Series(np.nan, index=df.index)

    def _xgboost_forecast(self, df, target_col):
        """
        XGBoost 梯度提升树预测

        特征 (~30个):
        - 时间特征: day_of_week, is_weekend, month, day, week_of_year
          + dow_0 至 dow_6 (one-hot)
        - 滞后特征: lag_1, lag_2, lag_3, lag_7, lag_14
        - 滑动窗口特征: ma_3/7/14, std_3/7/14

        训练策略:
        - 使用 TimeSeriesSplit (3折) 进行交叉验证
        - 保持时间顺序，避免未来信息泄露
        - 每折训练独立模型，对对应测试集预测

        超参数:
        - n_estimators=100: 100棵树
        - max_depth=4: 限制树深度防止过拟合
        - learning_rate=0.1: 学习率
        - subsample=0.8, colsample_bytree=0.8: 行/列采样

        Args:
            df: 营业日 DataFrame
            target_col: 目标列名

        Returns:
            tuple: (pd.Series 预测值, dict 特征重要性 Top10)
        """
        # 构建特征
        data = self._build_features(df, target_col)

        # 特征列选择 (排除目标列和标识列)
        feat_cols = [
            c for c in data.columns
            if c not in self.TARGET_COLS + ['is_closed', 'weekday_name']
            and data[c].dtype in ['float64', 'int64', 'int32', 'float32', 'bool']
            and not c.startswith('total_fiber')
        ]

        # 清洗数据 (去除无穷值和 NaN)
        data_clean = data.replace([np.inf, -np.inf], np.nan).dropna(
            subset=feat_cols + [target_col]
        )

        if len(data_clean) < 30:
            return pd.Series(np.nan, index=df.index), None

        X = data_clean[feat_cols].values
        y = data_clean[target_col].values

        # 时间序列交叉验证 (3折)
        tscv = TimeSeriesSplit(n_splits=3)
        all_preds = np.zeros(len(data_clean))

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]

            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_SEED,
                verbosity=0,
            )
            model.fit(X_train, y_train)
            all_preds[test_idx] = model.predict(X_test)

        # 提取特征重要性 (Top 10)
        importance = dict(zip(feat_cols, model.feature_importances_))
        importance = {
            k: v for k, v in sorted(
                importance.items(), key=lambda x: x[1], reverse=True
            )[:10]
        }

        # 将预测值映射回完整日期索引
        preds = pd.Series(np.nan, index=df.index)
        preds.loc[data_clean.index] = all_preds

        return preds, importance

    def _ensemble_forecast(self, df, target_col, baseline, sarima, xgb):
        """
        组合预测: 基于 MAPE 的加权平均

        权重计算:
        weight_model = (1/MAPE_model) / Σ(1/MAPE)
        即 MAPE 越小的模型权重越大。

        对于无法计算 MAPE 的模型 (如 SARIMA 失败), 权重 = 0。
        若所有模型均失败，则回退到全局均值。

        Args:
            df: 营业日 DataFrame
            target_col: 目标列名
            baseline, sarima, xgb: 三个模型的预测序列

        Returns:
            pd.Series: 组合预测值
        """
        actual = df[target_col].values
        weights = {}

        for name, pred in [('baseline', baseline), ('sarima', sarima),
                           ('xgboost', xgb)]:
            valid = ~np.isnan(pred.values) & ~np.isnan(actual)
            if valid.sum() > 20:
                m = mape_score(actual[valid], pred.values[valid])
                weights[name] = 1.0 / max(m, 0.001) if m > 0 else 1.0
            else:
                weights[name] = 0

        total_weight = sum(weights.values())
        if total_weight == 0:
            return pd.Series(actual.mean(), index=df.index)

        # 归一化权重
        for k in weights:
            weights[k] /= total_weight

        # 加权组合
        ensemble = pd.Series(0.0, index=df.index)
        for name, pred in [('baseline', baseline), ('sarima', sarima),
                           ('xgboost', xgb)]:
            if weights[name] > 0:
                valid_idx = ~pred.isna()
                ensemble[valid_idx] += weights[name] * pred[valid_idx]

        return ensemble

    def _model_comparison(self):
        """
        模型比较可视化 — 输出 p2_model_comparison.png

        展示 4 种模型在 6 个目标变量上的 MAPE 对比。
        柱状图使用 Nature NPG 配色。
        """
        metrics = self.results.get('metrics', {})
        if not metrics:
            return

        models = ['baseline', 'sarima', 'xgboost', 'ensemble']
        model_labels = ['Baseline', 'SARIMA', 'XGBoost', 'Ensemble']
        model_colors = [COLORS['purple'], COLORS['primary'],
                        COLORS['accent'], COLORS['success']]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        for i, (target_col, name) in enumerate(self.TARGET_NAMES.items()):
            ax = axes[i // 3, i % 3]

            target_metrics = metrics.get(target_col, {})
            mape_values = []
            for m in models:
                mape_values.append(
                    target_metrics.get(m, {}).get('MAPE', np.nan)
                )

            bars = ax.bar(model_labels, mape_values,
                         color=model_colors,
                         edgecolor='white', linewidth=1)
            ax.set_title(name, fontweight='bold', fontsize=10)
            ax.set_ylabel('MAPE (%)')
            ax.tick_params(axis='x', rotation=30)

            # 标注数值
            for bar, val in zip(bars, mape_values):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width()/2,
                           bar.get_height(),
                           f'{val:.1f}%', ha='center', va='bottom',
                           fontsize=8)

        plt.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/p2_model_comparison.png', dpi=300)
        plt.close()
        print('  已保存: p2_model_comparison.png')

    def _predict_may_2025(self):
        """
        预测 2025 年 5 月工作日 — 输出 p2_may2025_predictions.png 和 CSV

        外推策略:
        1. 识别 2025年5月的所有工作日 (周一至周五)
        2. 对每个目标变量，使用历史相同月份+相同星期几的均值作为基准预测
        3. 应用偏差修正因子 (基于 XGBoost 近期预测的平均偏差)
        4. 限制偏差修正范围在 [0.8, 1.2] 内 (避免极端修正)

        局限性说明 (诚实标注):
        - 外推预测假设历史模式在未来延续
        - 2025 年距离训练数据末尾 (2023-11) 有约 18 个月的间隔
        - 未考虑节假日 (五一劳动节) 的影响
        - 未考虑餐厅在此期间可能发生的运营变化
        - 结果应被视为"如果历史模式持续"的情景估计

        Returns:
            pd.DataFrame: 预测结果表
        """
        print('\n  预测2025年5月 (工作日)...')

        # 生成 2025年5月所有日期
        may_dates = pd.date_range(
            start=f'{PREDICTION_YEAR}-{PREDICTION_MONTH}-01',
            end=f'{PREDICTION_YEAR}-{PREDICTION_MONTH}-31',
            freq='D'
        )

        # 筛选工作日 (周一至周五)
        may_workdays = may_dates[may_dates.dayofweek < 5]
        print(f'  2025年5月工作日: {len(may_workdays)} 天')

        # 准备预测 DataFrame
        pred_df = pd.DataFrame(index=may_workdays)
        pred_df['day_of_week'] = pred_df.index.dayofweek
        pred_df['is_weekend'] = 0
        pred_df['month'] = pred_df.index.month
        pred_df['day'] = pred_df.index.day
        pred_df['week_of_year'] = pred_df.index.isocalendar().week.astype(int)

        for d in range(7):
            pred_df[f'dow_{d}'] = (pred_df['day_of_week'] == d).astype(int)

        # 历史营业数据 (排除停业日)
        df = self.df_daily[self.df_daily['is_closed'] == 0]

        # 逐目标变量预测
        for target_col in self.TARGET_COLS:
            # 策略: 历史相同月份 + 相同星期几的均值
            for i, date in enumerate(may_workdays):
                dow = date.dayofweek

                # 同月 + 同星期 + 非周末 的历史记录
                same_condition = (
                    (df.index.month == 5) &
                    (df.index.dayofweek == dow) &
                    (df['is_weekend'] == 0)
                )
                historical_same = df.loc[same_condition, target_col]

                if len(historical_same) > 0:
                    pred_df.loc[date, target_col] = historical_same.mean()
                else:
                    # 回退: 同星期所有历史均值 (不限月份)
                    same_dow = (
                        (df.index.dayofweek == dow) &
                        (df['is_weekend'] == 0)
                    )
                    pred_df.loc[date, target_col] = (
                        df.loc[same_dow, target_col].mean()
                    )

            # 偏差修正: 基于 XGBoost 近期预测的系统性偏差
            try:
                recent_data = self._build_features(
                    df.iloc[-60:].copy(), target_col
                )
                feat_cols = [
                    c for c in recent_data.columns
                    if c not in self.TARGET_COLS + ['is_closed', 'weekday_name']
                    and recent_data[c].dtype in ['float64', 'int64', 'int32']
                    and not c.startswith('total_fiber')
                ]
                recent_clean = recent_data.replace(
                    [np.inf, -np.inf], np.nan
                ).dropna()

                if len(recent_clean) > 30:
                    X_recent = recent_clean[feat_cols].values
                    y_recent = recent_clean[target_col].values

                    model = xgb.XGBRegressor(
                        n_estimators=100, max_depth=4, learning_rate=0.1,
                        random_state=RANDOM_SEED, verbosity=0
                    )
                    model.fit(X_recent, y_recent)

                    # 预测值与实际值的偏差修正因子
                    pred_train = model.predict(X_recent)
                    bias = (
                        np.mean(y_recent) / max(np.mean(pred_train), 0.001)
                    )
                    bias = np.clip(bias, 0.8, 1.2)  # 限制在 ±20% 内

                    pred_df[target_col] = pred_df[target_col] * bias

                    print(f'    {target_col}: bias correction = {bias:.3f}')
            except Exception as e:
                print(f'    {target_col}: bias correction failed ({e})')

        # 就餐人数需要整数化
        pred_df['total_orders'] = (
            pred_df['total_orders'].round().astype(int)
        )

        # 保存预测结果
        self.results['may_2025_predictions'] = pred_df

        # ==== 预测结果可视化: p2_may2025_predictions.png ====
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        for i, (col, name) in enumerate(self.TARGET_NAMES.items()):
            ax = axes[i // 3, i % 3]

            bars = ax.bar(range(len(may_workdays)), pred_df[col].values,
                         color=COLORS['primary'], alpha=0.7,
                         edgecolor='white')
            ax.set_xticks(range(0, len(may_workdays), 5))
            ax.set_xticklabels(
                [may_workdays[d].strftime('%m/%d')
                 for d in range(0, len(may_workdays), 5)],
                rotation=45, fontsize=8
            )
            ax.set_title(f'{name} - May 2025 Workdays',
                        fontweight='bold', fontsize=10)
            ax.set_ylabel(name)
            ax.grid(axis='y', alpha=0.3)

            # 均值参考线
            mean_val = pred_df[col].mean()
            ax.axhline(y=mean_val, color=COLORS['accent'], linestyle='--',
                      linewidth=1, alpha=0.7,
                      label=f'Mean: {mean_val:.0f}')
            ax.legend(fontsize=8)

        plt.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/p2_may2025_predictions.png', dpi=300)
        plt.close()
        print('  已保存: p2_may2025_predictions.png')

        # 输出预测表格
        print('\n  === 2025年5月工作日预测结果 ===')
        print(pred_df[self.TARGET_COLS].round(0).to_string())

        # 保存 CSV (UTF-8 BOM 用于 Excel 兼容性)
        pred_df[self.TARGET_COLS].round(0).to_csv(
            f'{OUTPUT_DIR}/p2_may2025_predictions.csv',
            encoding='utf-8-sig'
        )
        print(f'\n  预测结果已保存: p2_may2025_predictions.csv')

        return pred_df


if __name__ == '__main__':
    # 模块自检
    pred = Problem2Prediction()
    results = pred.run()
