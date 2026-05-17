"""problem2_prediction.py — 问题2: 就餐人数、营养需求与销售总额预测"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from data_loader import load_all_data
from config import (OUTPUT_DIR, COLORS, COLOR_CYCLE, RANDOM_SEED,
                     PREDICTION_YEAR, PREDICTION_MONTH)
from utils import (mape_score, smape_score, create_lag_features,
                   create_rolling_features, print_metrics_table)


class Problem2Prediction:

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
        print('\n>>> 2.1 时间序列特征分析')
        self._time_series_analysis()

        print('\n>>> 2.2 构建预测模型')
        predictions = self._build_and_evaluate_models()

        print('\n>>> 2.3 模型比较与选择')
        self._model_comparison()

        print('\n>>> 2.4 残差诊断与分组误差分析')
        self._residual_diagnostics()

        print('\n>>> 2.5 Walk-forward 验证')
        self._walk_forward_validation()

        print('\n>>> 2.6 预测2025年5月工作日')
        may_predictions = self._predict_may_2025()

        print('\n问题2预测完成!')
        return self.results

    def _time_series_analysis(self):
        df = self.df_daily[self.df_daily['is_closed'] == 0].copy()

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
        data = df.copy()

        data['day_of_week'] = data.index.dayofweek
        data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
        data['month'] = data.index.month
        data['day'] = data.index.day
        data['week_of_year'] = data.index.isocalendar().week.astype(int)

        # 星期几 one-hot 编码
        for d in range(7):
            data[f'dow_{d}'] = (data['day_of_week'] == d).astype(int)

        data = create_lag_features(
            data, target_col, lags=[1, 2, 3, 7, 14]
        )

        data = create_rolling_features(data, target_col, windows=[3, 7, 14])

        return data

    def _build_and_evaluate_models(self):
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

    def _residual_diagnostics(self):
        df = self.df_daily[self.df_daily['is_closed'] == 0]

        # Use baseline predictions for demonstration
        preds = {}
        actuals = {}
        for col in self.TARGET_COLS:
            bl = self._baseline_forecast(df, col)
            valid = ~bl.isna()
            preds[col] = bl[valid].values
            actuals[col] = df.loc[valid.index[valid], col].values

        fig, axes = plt.subplots(3, 3, figsize=(20, 16))

        # Row 1: Residual distributions for 3 key targets
        for i, col in enumerate(['total_orders', 'total_sales', 'total_calories']):
            ax = axes[0, i]
            resid = actuals[col] - preds[col]
            ax.hist(resid, bins=40, color=COLORS['primary'], alpha=0.7, edgecolor='white')
            ax.axvline(0, color=COLORS['accent'], linestyle='--', linewidth=2)
            ax.set_title(f'Residual Dist: {self.TARGET_NAMES[col]}\n'
                        f'(mean={np.mean(resid):.1f}, std={np.std(resid):.1f})')
            ax.set_xlabel('Residual')
            ax.set_ylabel('Frequency')
            ax.grid(alpha=0.3)

        # Row 2: Residual ACF for 3 key targets
        for i, col in enumerate(['total_orders', 'total_sales', 'total_calories']):
            ax = axes[1, i]
            resid = actuals[col] - preds[col]
            valid_resid = resid[~np.isnan(resid)]
            if len(valid_resid) > 20:
                plot_acf(valid_resid, lags=min(30, len(valid_resid)//4), ax=ax)
                # Ljung-Box test
                lb_result = acorr_ljungbox(valid_resid, lags=[7, 14, 21], return_df=True)
                lb_pvals = lb_result['lb_pvalue'].values
                ax.set_title(f'Residual ACF: {self.TARGET_NAMES[col]}\n'
                            f'LB p(lag=7)={lb_pvals[0]:.3f}, '
                            f'p(14)={lb_pvals[1]:.3f}')
            else:
                ax.set_title(f'Residual ACF: {self.TARGET_NAMES[col]} (insufficient data)')

        # Row 3: MAPE by weekday for 3 key targets
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for i, col in enumerate(['total_orders', 'total_sales', 'total_calories']):
            ax = axes[2, i]
            dow_mapes = []
            df_valid = df.dropna(subset=[col])
            for d in range(7):
                mask = df_valid['day_of_week'] == d
                if mask.sum() > 3:
                    bl = self._baseline_forecast(df_valid[mask], col)
                    valid = ~bl.isna()
                    if valid.sum() > 0:
                        m = mape_score(df_valid.loc[mask].loc[valid.index[valid], col].values,
                                      bl[valid].values)
                        dow_mapes.append(m if not np.isnan(m) else 0)
                    else:
                        dow_mapes.append(0)
                else:
                    dow_mapes.append(0)
            
            ax.bar(dow_names, dow_mapes, color=COLORS['primary'], alpha=0.8, edgecolor='white')
            ax.set_title(f'MAPE by Weekday: {self.TARGET_NAMES[col]}')
            ax.set_ylabel('MAPE (%)')
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/p2_residual_diagnostics.png', dpi=300)
        plt.close()
        print('  已保存: p2_residual_diagnostics.png')

    def _walk_forward_validation(self):
        df = self.df_daily[self.df_daily['is_closed'] == 0].copy()
        target_col = 'total_orders'  # Focus on primary target

        # Build features once
        data = self._build_features(df, target_col)
        feat_cols = [c for c in data.columns
                    if c not in self.TARGET_COLS + ['is_closed', 'weekday_name']
                    and data[c].dtype in ['float64', 'int64', 'int32']
                    and not c.startswith('total_fiber')]
        data_clean = data.replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols + [target_col])

        if len(data_clean) < 50:
            print('  Walk-forward validation skipped (insufficient data)')
            return

        # Walk-forward: initial train on first 80%, then expand by 7 days
        n_total = len(data_clean)
        init_size = int(n_total * 0.8)
        step = 7  # predict 1 week at a time

        wf_predictions = []
        wf_actuals = []
        wf_dates = []

        X_all = data_clean[feat_cols].values
        y_all = data_clean[target_col].values
        dates_all = data_clean.index

        for start in range(init_size, n_total, step):
            end = min(start + step, n_total)
            X_train = X_all[:start]
            y_train = y_all[:start]
            X_test = X_all[start:end]
            y_test = y_all[start:end]

            model = xgb.XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                random_state=RANDOM_SEED, verbosity=0
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            wf_predictions.extend(pred)
            wf_actuals.extend(y_test)
            wf_dates.extend(dates_all[start:end])

        wf_mae = mean_absolute_error(wf_actuals, wf_predictions)
        wf_rmse = np.sqrt(mean_squared_error(wf_actuals, wf_predictions))
        wf_mape = mape_score(np.array(wf_actuals), np.array(wf_predictions))

        print(f'  Walk-forward (XGBoost, {target_col}): '
              f'MAE={wf_mae:.1f}, RMSE={wf_rmse:.1f}, MAPE={wf_mape:.1f}%')

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        ax = axes[0]
        ax.plot(wf_dates, wf_actuals, 'o-', color=COLORS['primary'], 
                markersize=3, linewidth=1, alpha=0.7, label='Actual')
        ax.plot(wf_dates, wf_predictions, 's-', color=COLORS['accent'],
                markersize=3, linewidth=1, alpha=0.7, label='Predicted')
        ax.set_title(f'Walk-Forward Validation: {self.TARGET_NAMES[target_col]}\n'
                    f'(MAE={wf_mae:.1f}, MAPE={wf_mape:.1f}%)')
        ax.set_ylabel(self.TARGET_NAMES[target_col])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1]
        resid = np.array(wf_actuals) - np.array(wf_predictions)
        ax.scatter(wf_predictions, resid, alpha=0.5, s=15, c=COLORS['primary'])
        ax.axhline(0, color=COLORS['accent'], linestyle='--', linewidth=2)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Residual')
        ax.set_title(f'Residual vs Predicted\n(bias={np.mean(resid):.1f})')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/p2_walk_forward.png', dpi=300)
        plt.close()
        print('  已保存: p2_walk_forward.png')

        self.results['walk_forward'] = {
            'mae': wf_mae, 'rmse': wf_rmse, 'mape': wf_mape,
            'n_predictions': len(wf_predictions),
        }

    def _predict_may_2025(self):
        from chinese_calendar import is_workday, is_holiday

        print('\n  预测2025年5月 (工作日, 排除法定假日)...')

        # 生成 2025年5月所有日期
        all_may = pd.date_range(
            start=f'{PREDICTION_YEAR}-{PREDICTION_MONTH}-01',
            end=f'{PREDICTION_YEAR}-{PREDICTION_MONTH}-31',
            freq='D'
        )

        # 筛选真正的工作日: Monday-Friday AND not Chinese holiday
        may_workdays = all_may[
            (all_may.dayofweek < 5) &
            [not is_holiday(d.date()) for d in all_may]
        ]
        n_days = len(may_workdays)
        print(f'  2025年5月: {len(all_may)}天, 排除周末+假期后 {n_days} 个工作日')

        # Mark holidays for reporting
        holidays_may = [d for d in all_may if is_holiday(d.date()) and d.dayofweek < 5]
        if holidays_may:
            print(f'  排除的假日(工作日): {[d.strftime("%m/%d") for d in holidays_may]}')

        # 历史营业数据
        df = self.df_daily[self.df_daily['is_closed'] == 0]

        # 准备预测 DataFrame
        pred_df = pd.DataFrame(index=may_workdays)
        pred_df['day_of_week'] = pred_df.index.dayofweek
        pred_df['is_weekend'] = 0
        pred_df['month'] = pred_df.index.month
        pred_df['day'] = pred_df.index.day

        for d in range(7):
            pred_df[f'dow_{d}'] = (pred_df['day_of_week'] == d).astype(int)

        for target_col in self.TARGET_COLS:
            series = df[target_col].dropna().values

            sarima_forecast = None
            sarima_ci_lower = None
            sarima_ci_upper = None

            try:
                model = SARIMAX(
                    series,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 7),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fitted = model.fit(disp=False, maxiter=100)

                # True out-of-sample forecast with 95% CI
                forecast_result = fitted.get_forecast(steps=n_days)
                forecast_frame = forecast_result.summary_frame(alpha=0.05)

                sarima_forecast = forecast_frame['mean'].values
                sarima_ci_lower = forecast_frame['mean_ci_lower'].values
                sarima_ci_upper = forecast_frame['mean_ci_upper'].values

                print(f'    {target_col}: SARIMA forecast successful, '
                      f'mean={sarima_forecast.mean():.0f}, '
                      f'CI width={np.mean(sarima_ci_upper - sarima_ci_lower):.0f}')
            except Exception as e:
                print(f'    {target_col}: SARIMA forecast FAILED ({e}), '
                      f'falling back to ensemble')
                sarima_forecast = None

            if sarima_forecast is None:
                # Use same-month same-weekday historical mean (Baseline)
                for i, date in enumerate(may_workdays):
                    dow = date.dayofweek
                    same_condition = (
                        (df.index.month == 5) &
                        (df.index.dayofweek == dow) &
                        (df['is_weekend'] == 0)
                    )
                    historical_same = df.loc[same_condition, target_col]
                    if len(historical_same) > 0:
                        pred_df.loc[date, target_col] = historical_same.mean()
                    else:
                        same_dow = (df.index.dayofweek == dow) & (df['is_weekend'] == 0)
                        pred_df.loc[date, target_col] = df.loc[same_dow, target_col].mean()

                # Add XGBoost bias correction
                try:
                    recent_data = self._build_features(df.iloc[-60:].copy(), target_col)
                    feat_cols = [c for c in recent_data.columns
                                if c not in self.TARGET_COLS + ['is_closed', 'weekday_name']
                                and recent_data[c].dtype in ['float64', 'int64', 'int32']
                                and not c.startswith('total_fiber')]
                    recent_clean = recent_data.replace([np.inf, -np.inf], np.nan).dropna()
                    if len(recent_clean) > 30:
                        X_recent = recent_clean[feat_cols].values
                        y_recent = recent_clean[target_col].values
                        xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=4,
                                                     learning_rate=0.1,
                                                     random_state=RANDOM_SEED, verbosity=0)
                        xgb_model.fit(X_recent, y_recent)
                        pred_train = xgb_model.predict(X_recent)
                        bias = np.mean(y_recent) / max(np.mean(pred_train), 0.001)
                        bias = np.clip(bias, 0.8, 1.2)
                        pred_df[target_col] = pred_df[target_col] * bias
                        print(f'    {target_col}: ensemble (baseline+xgb), bias={bias:.3f}')
                except Exception:
                    print(f'    {target_col}: using pure baseline (no correction)')
            else:
                # Use SARIMA forecast with CI
                pred_df[target_col] = sarima_forecast
                pred_df[f'{target_col}_lower'] = sarima_ci_lower
                pred_df[f'{target_col}_upper'] = sarima_ci_upper

        # Integer rounding for orders
        pred_df['total_orders'] = pred_df['total_orders'].round().astype(int)

        # Ensure non-negative
        for target_col in self.TARGET_COLS:
            pred_df[target_col] = pred_df[target_col].clip(lower=0)

        self.results['may_2025_predictions'] = pred_df

        has_ci = any(f'{c}_lower' in pred_df.columns for c in self.TARGET_COLS)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        for i, (col, name) in enumerate(self.TARGET_NAMES.items()):
            ax = axes[i // 3, i % 3]
            x_idx = range(len(may_workdays))

            # Bars for point forecast
            ax.bar(x_idx, pred_df[col].values, color=COLORS['primary'],
                   alpha=0.7, edgecolor='white', label='Forecast')

            # CI ribbon if available
            if f'{col}_lower' in pred_df.columns and f'{col}_upper' in pred_df.columns:
                ax.fill_between(x_idx,
                                pred_df[f'{col}_lower'].values,
                                pred_df[f'{col}_upper'].values,
                                alpha=0.2, color=COLORS['accent'],
                                label='95% CI')

            ax.set_xticks(range(0, len(may_workdays), 5))
            ax.set_xticklabels(
                [may_workdays[d].strftime('%m/%d')
                 for d in range(0, len(may_workdays), 5)],
                rotation=45, fontsize=8
            )
            ax.set_title(f'{name} - May 2025 Workdays', fontweight='bold', fontsize=10)
            ax.set_ylabel(name)
            ax.grid(axis='y', alpha=0.3)
            ax.legend(fontsize=8)

        plt.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/p2_may2025_predictions.png', dpi=300)
        plt.close()
        print('  已保存: p2_may2025_predictions.png')

        # Print table
        print('\n  === 2025年5月工作日预测结果 (SARIMA + 95%CI) ===')
        display_cols = self.TARGET_COLS.copy()
        if has_ci:
            for c in self.TARGET_COLS:
                if f'{c}_lower' in pred_df.columns:
                    print(f'    {c}: 95% CI width = '
                          f'{np.mean(pred_df[f"{c}_upper"] - pred_df[f"{c}_lower"]):.0f}')
        print(pred_df[display_cols].round(0).to_string())

        # Save CSV
        pred_df[display_cols].round(0).to_csv(
            f'{OUTPUT_DIR}/p2_may2025_predictions.csv',
            encoding='utf-8-sig'
        )
        print(f'\n  预测结果已保存: p2_may2025_predictions.csv')

        return pred_df


if __name__ == '__main__':
    # 模块自检
    pred = Problem2Prediction()
    results = pred.run()
