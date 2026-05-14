"""
problem2_prediction.py — 问题2：就餐人数、营养需求与销售总额预测
============================================================
题目要求：
  根据餐厅销售记录，对该餐厅每天就餐人数、各类营养素需求量以及销售总额
  进行预测研究，并讨论预测模型的合理性和结果的可靠性。请给出2025年5月份
  工作日的就餐人数、各类营养素需求量以及销售总额预测结果。

解题思路：
  1. 数据准备：按日汇总交易数据，构建时间序列
  2. 特征工程：
     - 时间特征：星期、月份、是否工作日
     - 滞后特征：前1天、前7天的观察值
     - 滑动窗口特征：3日/7日移动平均
  3. 预测模型（多模型比较）：
     - 基准模型：历史同星期均值法
     - SARIMA：捕捉季节性和趋势
     - XGBoost：机器学习集成方法
     - 组合预测：加权平均融合
  4. 预测目标变量：
     - daily_diners：每日就餐人数
     - daily_sales：每日销售总额
     - daily_calories/protein/fat/carbs/fiber：每日营养素总量
  5. 模型评估：MAE、RMSE、MAPE
  6. 预测2025年5月工作日

参考文献：
  [1] Rodrigues M. et al. "Machine learning models for short-term demand
      forecasting in food catering services" J. Cleaner Production, 2024.
  [2] Posch K. et al. "A Bayesian Approach for Predicting Food and Beverage
      Sales" Intl. J. Forecasting, 2022.
  [3] Thomassey S. et al. "Machine Learning Based Restaurant Sales Forecasting"
      Machine Learning and Knowledge Extraction, 2022.
  [4] Hyndman R.J. "Forecasting: Principles and Practice" OTexts.
      https://otexts.com/fpp3/
  [12] Chen T., Guestrin C. "XGBoost: A Scalable Tree Boosting System"
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
from config import (OUTPUT_DIR, COLORS, RANDOM_SEED,
                    PREDICTION_YEAR, PREDICTION_MONTH)
from utils import mape_score, create_lag_features, print_metrics_table


class Problem2Prediction:
    """
    问题2：多模型需求预测
    
    预测目标（6个变量）：
    - total_orders: 每日就餐人数
    - total_sales: 每日销售总额
    - total_calories: 每日热量需求
    - total_protein: 每日蛋白质需求
    - total_fat: 每日脂肪需求
    - total_carbohydrates: 每日碳水化合物需求
    """
    
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
        """初始化并准备数据"""
        print('\n' + '=' * 60)
        print('问题2：需求预测模型')
        print('=' * 60)
        
        if loader is None:
            self.loader = load_all_data()
        else:
            self.loader = loader
        
        self.df_daily = self.loader.get_daily_data().copy()
        self.df_daily = self.df_daily.set_index('date').sort_index()
        
        # 确保日期间无缺失（填充节假日/停业日）
        full_idx = pd.date_range(
            start=self.df_daily.index.min(),
            end=self.df_daily.index.max(),
            freq='D'
        )
        self.df_daily = self.df_daily.reindex(full_idx)
        
        # 填充缺失的日期特征
        self.df_daily['day_of_week'] = self.df_daily.index.dayofweek
        self.df_daily['is_weekend'] = self.df_daily['day_of_week'].isin([5, 6]).astype(int)
        self.df_daily['month'] = self.df_daily.index.month
        self.df_daily['day'] = self.df_daily.index.day
        
        # 对于缺失值（停业日），用0填充关键变量
        for col in self.TARGET_COLS:
            self.df_daily[col] = self.df_daily[col].fillna(0)
        
        # 对零值日进行标记
        self.df_daily['is_closed'] = (self.df_daily['total_orders'] == 0).astype(int)
        
        self.results = {}
        
    def run(self):
        """运行完整预测流程"""
        print('\n>>> 2.1 时间序列特征分析')
        self._time_series_analysis()
        
        print('\n>>> 2.2 构建预测模型')
        predictions = self._build_and_evaluate_models()
        
        print('\n>>> 2.3 模型比较与选择')
        self._model_comparison()
        
        print('\n>>> 2.4 预测2025年5月工作日')
        may_predictions = self._predict_may_2025()
        
        print('\n问题2预测完成！')
        return self.results
    
    def _time_series_analysis(self):
        """时间序列特征分析：趋势、季节性、平稳性检验"""
        df = self.df_daily[self.df_daily['is_closed'] == 0].copy()
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        
        for i, (col, name) in enumerate(self.TARGET_NAMES.items()):
            ax = axes[i // 2, i % 2]
            series = df[col].dropna()
            
            # ADF平稳性检验
            if len(series) > 20:
                adf_result = adfuller(series.dropna(), autolag='AIC')
                is_stationary = adf_result[1] < 0.05
                
                # 绘制序列
                ax.plot(series.index, series.values, color=COLORS['primary'], 
                       linewidth=1, alpha=0.8)
                ax.set_title(f'{name}\n(ADF p={adf_result[1]:.4f}, '
                            f'{"Stationary" if is_stationary else "Non-stationary"})',
                            fontsize=10, fontweight='bold')
                ax.set_ylabel(name)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.tick_params(axis='x', rotation=45, labelsize=8)
            else:
                ax.plot(series.index, series.values, color=COLORS['primary'])
                ax.set_title(name, fontsize=10)
        
        plt.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/p2_time_series_overview.png', dpi=150)
        plt.close()
        print('  已保存: p2_time_series_overview.png')
        
        # ACF/PACF分析（以orders为例）
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        plot_acf(df['total_orders'].dropna(), lags=30, ax=axes[0])
        axes[0].set_title('ACF - Daily Orders', fontweight='bold')
        plot_pacf(df['total_orders'].dropna(), lags=30, ax=axes[1])
        axes[1].set_title('PACF - Daily Orders', fontweight='bold')
        plt.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/p2_acf_pacf.png', dpi=150)
        plt.close()
        print('  已保存: p2_acf_pacf.png')
        
    def _build_features(self, df, target_col):
        """构建预测特征"""
        data = df.copy()
        
        # 时间特征
        data['day_of_week'] = data.index.dayofweek
        data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
        data['month'] = data.index.month
        data['day'] = data.index.day
        data['week_of_year'] = data.index.isocalendar().week.astype(int)
        
        # 工作日独热编码
        for d in range(7):
            data[f'dow_{d}'] = (data['day_of_week'] == d).astype(int)
        
        # 滞后特征
        data = create_lag_features(data, target_col, lags=[1, 2, 3, 7, 14])
        
        # 滑动窗口统计
        for w in [3, 7, 14]:
            data[f'{target_col}_ma{w}'] = data[target_col].rolling(window=w, min_periods=1).mean()
            data[f'{target_col}_std{w}'] = data[target_col].rolling(window=w, min_periods=1).std()
        
        return data
    
    def _build_and_evaluate_models(self):
        """
        构建并比较多种预测模型
        
        模型列表：
        1. 基准模型 - 历史同星期均值
        2. SARIMA - 季节性ARIMA
        3. XGBoost - 梯度提升树
        
        评估指标：MAE, RMSE, MAPE
        使用时间序列交叉验证
        """
        # 使用完整数据（排除零值日）
        df = self.df_daily[self.df_daily['is_closed'] == 0].copy()
        
        # 只对核心目标变量运行完整模型
        # 其他营养变量使用XGBoost
        all_predictions = {}
        all_metrics = {}
        
        for target_col in self.TARGET_COLS:
            print(f'\n  --- 预测目标: {self.TARGET_NAMES[target_col]} ---')
            
            # ---- 模型1: 基准模型（历史同星期均值） ----
            baseline_preds = self._baseline_forecast(df, target_col)
            
            # ---- 模型2: SARIMA ----
            try:
                sarima_preds = self._sarima_forecast(df, target_col)
            except Exception as e:
                print(f'    SARIMA失败: {e}，使用NaN填充')
                sarima_preds = pd.Series(np.nan, index=df.index)
            
            # ---- 模型3: XGBoost ----
            xgb_preds, xgb_feat_importance = self._xgboost_forecast(df, target_col)
            
            # ---- 组合预测：加权平均 ----
            # 在交叉验证中选择最优权重
            ensemble_preds = self._ensemble_forecast(
                df, target_col, baseline_preds, sarima_preds, xgb_preds
            )
            
            # 存储结果
            pred_df = pd.DataFrame({
                'actual': df[target_col],
                'baseline': baseline_preds,
                'sarima': sarima_preds,
                'xgboost': xgb_preds,
                'ensemble': ensemble_preds,
            }, index=df.index)
            
            all_predictions[target_col] = pred_df
            
            # 计算各模型指标
            metrics = {}
            actual = df[target_col].values
            
            for model_name in ['baseline', 'sarima', 'xgboost', 'ensemble']:
                pred = pred_df[model_name].values
                valid = ~np.isnan(pred) & ~np.isnan(actual)
                if valid.sum() > 0:
                    metrics[model_name] = {
                        'MAE': mean_absolute_error(actual[valid], pred[valid]),
                        'RMSE': np.sqrt(mean_squared_error(actual[valid], pred[valid])),
                        'MAPE': mape_score(actual[valid], pred[valid])
                    }
            
            all_metrics[target_col] = metrics
            print_metrics_table(metrics, target_col)
            
            if xgb_feat_importance is not None:
                self.results[f'feat_imp_{target_col}'] = xgb_feat_importance
        
        self.results['predictions'] = all_predictions
        self.results['metrics'] = all_metrics
        
        return all_predictions
    
    def _baseline_forecast(self, df, target_col):
        """基准模型：历史同星期均值预测"""
        preds = pd.Series(index=df.index, dtype=float)
        dow_values = df.index.dayofweek  # 预提取星期值
        
        for i, date in enumerate(df.index):
            # 获取历史上同星期的均值
            target_dow = date.dayofweek  # 属性而非方法
            past_same_dow = (dow_values == target_dow) & (df.index < date)
            if past_same_dow.sum() > 0:
                preds.iloc[i] = df.loc[past_same_dow, target_col].mean()
            else:
                preds.iloc[i] = df[target_col].mean()
        
        return preds
    
    def _sarima_forecast(self, df, target_col):
        """
        SARIMA预测
        
        使用SARIMA(1,1,1)(1,1,1,7)模型：
        - 非季节性阶数(p,d,q) = (1,1,1) — 处理趋势
        - 季节性阶数(P,D,Q,s) = (1,1,1,7) — 以7天为周期
        """
        series = df[target_col].dropna().values
        
        # 尝试拟合SARIMA
        try:
            model = SARIMAX(
                series,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False, maxiter=100)
            
            # 获取拟合值
            fitted_values = fitted.fittedvalues
            
            # 创建预测序列
            preds = pd.Series(index=df.index, dtype=float)
            n_fitted = len(fitted_values)
            if n_fitted > 0 and n_fitted <= len(df):
                # SARIMAX拟合值可能短于原序列（差分导致）
                start_idx = len(df) - n_fitted
                for j in range(n_fitted):
                    preds.iloc[start_idx + j] = fitted_values[j]
            
            return preds
        except Exception as e:
            print(f'    SARIMA拟合失败: {e}')
            return pd.Series(np.nan, index=df.index)
    
    def _xgboost_forecast(self, df, target_col):
        """
        XGBoost预测
        
        特征包括：
        - 时间特征（星期、月份、日期）
        - 滞后值（1、2、3、7、14天前）
        - 移动平均（3、7、14天窗口）
        
        使用时间序列交叉验证评估
        """
        # 构建特征
        data = self._build_features(df, target_col)
        
        # 特征列
        feat_cols = [c for c in data.columns if c not in 
                     self.TARGET_COLS + ['is_closed', 'weekday_name']
                     and data[c].dtype in ['float64', 'int64', 'int32', 'float32', 'bool']
                     and not c.startswith('total_fiber')]
        
        # 去除无穷值和NaN
        data_clean = data.replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols + [target_col])
        
        if len(data_clean) < 30:
            return pd.Series(np.nan, index=df.index), None
        
        X = data_clean[feat_cols].values
        y = data_clean[target_col].values
        
        # 时间序列交叉验证
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
        
        # 特征重要性
        importance = dict(zip(feat_cols, model.feature_importances_))
        importance = {k: v for k, v in sorted(importance.items(), 
                                               key=lambda x: x[1], reverse=True)[:10]}
        
        # 创建完整预测序列
        preds = pd.Series(np.nan, index=df.index)
        preds.loc[data_clean.index] = all_preds
        
        return preds, importance
    
    def _ensemble_forecast(self, df, target_col, baseline, sarima, xgb):
        """
        组合预测：加权平均
        
        权重按交叉验证性能分配（MAPE越小权重越大）
        """
        actual = df[target_col].values
        weights = {}
        
        for name, pred in [('baseline', baseline), ('sarima', sarima), ('xgboost', xgb)]:
            valid = ~np.isnan(pred.values) & ~np.isnan(actual)
            if valid.sum() > 20:
                mape = mape_score(actual[valid], pred.values[valid])
                # 权重 = 1/MAPE（MAPE越小，权重越大）
                weights[name] = 1.0 / max(mape, 0.001) if mape > 0 else 1.0
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
        for name, pred in [('baseline', baseline), ('sarima', sarima), ('xgboost', xgb)]:
            if weights[name] > 0:
                valid_idx = ~pred.isna()
                ensemble[valid_idx] += weights[name] * pred[valid_idx]
        
        return ensemble
    
    def _model_comparison(self):
        """模型比较可视化"""
        metrics = self.results.get('metrics', {})
        if not metrics:
            return
        
        # 提取各模型的MAPE
        models = ['baseline', 'sarima', 'xgboost', 'ensemble']
        model_labels = ['Baseline', 'SARIMA', 'XGBoost', 'Ensemble']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        for i, (target_col, name) in enumerate(self.TARGET_NAMES.items()):
            ax = axes[i // 3, i % 3]
            
            target_metrics = metrics.get(target_col, {})
            mape_values = []
            for m in models:
                mape_values.append(target_metrics.get(m, {}).get('MAPE', np.nan))
            
            bars = ax.bar(model_labels, mape_values, 
                         color=[COLORS['primary'], COLORS['secondary'], 
                                COLORS['accent'], COLORS['success']],
                         edgecolor='white', linewidth=1)
            ax.set_title(name, fontweight='bold')
            ax.set_ylabel('MAPE (%)')
            ax.tick_params(axis='x', rotation=30)
            
            for bar, val in zip(bars, mape_values):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/p2_model_comparison.png', dpi=150)
        plt.close()
        print('  已保存: p2_model_comparison.png')
        
    def _predict_may_2025(self):
        """
        预测2025年5月工作日
        
        使用最优模型（组合预测）对未来进行外推预测
        """
        print('\n  预测2025年5月（工作日）...')
        
        # 生成2025年5月所有日期
        may_dates = pd.date_range(
            start=f'{PREDICTION_YEAR}-{PREDICTION_MONTH}-01',
            end=f'{PREDICTION_YEAR}-{PREDICTION_MONTH}-31',
            freq='D'
        )
        
        # 筛选工作日（周一至周五）
        may_workdays = may_dates[may_dates.dayofweek < 5]
        
        # 准备预测DataFrame
        pred_df = pd.DataFrame(index=may_workdays)
        pred_df['day_of_week'] = pred_df.index.dayofweek
        pred_df['is_weekend'] = 0
        pred_df['month'] = pred_df.index.month
        pred_df['day'] = pred_df.index.day
        pred_df['week_of_year'] = pred_df.index.isocalendar().week.astype(int)
        
        for d in range(7):
            pred_df[f'dow_{d}'] = (pred_df['day_of_week'] == d).astype(int)
        
        # 对每个目标变量进行预测
        # 使用历史同星期均值作为外推基准（对于没有滞后特征的未来预测）
        df = self.df_daily[self.df_daily['is_closed'] == 0]
        
        for target_col in self.TARGET_COLS:
            # 方法：使用历史同月份工作日的均值 + 趋势调整
            # 获取每年5月工作日的平均值
            may_historical = df[
                (df.index.month == 5) & (df['is_weekend'] == 0)
            ]
            
            for i, date in enumerate(may_workdays):
                dow = date.dayofweek
                # 同星期+同月份的历史均值
                same_condition = (
                    (df.index.month == 5) & 
                    (df.index.dayofweek == dow) &
                    (df['is_weekend'] == 0)
                )
                historical_same = df.loc[same_condition, target_col]
                
                if len(historical_same) > 0:
                    pred_df.loc[date, target_col] = historical_same.mean()
                else:
                    # 回退：同星期所有历史均值
                    same_dow = (df.index.dayofweek == dow) & (df['is_weekend'] == 0)
                    pred_df.loc[date, target_col] = df.loc[same_dow, target_col].mean()
            
            # 添加趋势调整：使用XGBoost对最近日期的预测作为校准
            try:
                # 对最后30天进行XGBoost预测以估计近期偏差
                recent_data = self._build_features(
                    df.iloc[-60:].copy(), target_col
                )
                feat_cols = [c for c in recent_data.columns if c not in 
                            self.TARGET_COLS + ['is_closed', 'weekday_name']
                            and recent_data[c].dtype in ['float64', 'int64', 'int32']
                            and not c.startswith('total_fiber')]
                recent_clean = recent_data.replace([np.inf, -np.inf], np.nan).dropna()
                
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
                    bias = np.mean(y_recent) / max(np.mean(pred_train), 0.001)
                    bias = np.clip(bias, 0.8, 1.2)  # 限制修正范围
                    
                    # 应用偏差修正
                    pred_df[target_col] = pred_df[target_col] * bias
                    
                    print(f'    {target_col}: 偏差修正因子={bias:.3f}')
            except Exception as e:
                print(f'    {target_col}: 偏差修正失败: {e}')
        
        # 整数化就餐人数
        pred_df['total_orders'] = pred_df['total_orders'].round().astype(int)
        
        # 保存预测结果
        self.results['may_2025_predictions'] = pred_df
        
        # ---- 预测结果可视化 ----
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        for i, (col, name) in enumerate(self.TARGET_NAMES.items()):
            ax = axes[i // 3, i % 3]
            
            # 预测值柱状图
            bars = ax.bar(range(len(may_workdays)), pred_df[col].values,
                         color=COLORS['primary'], alpha=0.7, edgecolor='white')
            ax.set_xticks(range(0, len(may_workdays), 5))
            ax.set_xticklabels([may_workdays[d].strftime('%m/%d') 
                               for d in range(0, len(may_workdays), 5)],
                              rotation=45, fontsize=8)
            ax.set_title(f'{name} - May 2025 Workdays', fontweight='bold')
            ax.set_ylabel(name)
            ax.grid(axis='y', alpha=0.3)
            
            # 添加均值线
            mean_val = pred_df[col].mean()
            ax.axhline(y=mean_val, color=COLORS['danger'], linestyle='--',
                      linewidth=1, alpha=0.7,
                      label=f'Mean: {mean_val:.0f}')
            ax.legend(fontsize=8)
        
        plt.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/p2_may2025_predictions.png', dpi=150)
        plt.close()
        print('  已保存: p2_may2025_predictions.png')
        
        # 输出预测表格
        print('\n  === 2025年5月工作日预测结果 ===')
        print(pred_df[self.TARGET_COLS].round(0).to_string())
        
        # 保存CSV
        pred_df[self.TARGET_COLS].round(0).to_csv(
            f'{OUTPUT_DIR}/p2_may2025_predictions.csv',
            encoding='utf-8-sig'
        )
        print(f'\n  已保存到: p2_may2025_predictions.csv')
        
        return pred_df


if __name__ == '__main__':
    pred = Problem2Prediction()
    results = pred.run()
