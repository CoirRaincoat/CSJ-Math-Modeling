"""utils.py — 通用工具函数模块"""


import numpy as np
import pandas as pd


def mape_score(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # 过滤零值 (避免除零) 和 NaN
    mask = (y_true != 0) & (~np.isnan(y_true)) & (~np.isnan(y_pred))

    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape_score(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # 过滤 NaN
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask2 = denominator > 0

    if mask2.sum() == 0:
        return np.nan

    return np.mean(np.abs(y_true[mask2] - y_pred[mask2]) / denominator[mask2]) * 100


def create_lag_features(df, col, lags):
    df = df.copy()
    for lag in lags:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df


def create_rolling_features(df, col, windows):
    df = df.copy()
    for w in windows:
        df[f'{col}_ma{w}'] = df[col].rolling(window=w, min_periods=1).mean()
        df[f'{col}_std{w}'] = df[col].rolling(window=w, min_periods=1).std()
    return df


def print_metrics_table(metrics_dict, target_name=''):
    if target_name:
        print(f'\n  目标: {target_name}')

    # 表头
    print(f'  {"Model":<15} {"MAE":>10} {"RMSE":>10} {"MAPE(%)":>10}')
    print(f'  {"-"*45}')

    best_mape = float('inf')
    best_model = ''

    for model_name, metrics in metrics_dict.items():
        mae = metrics.get('MAE', np.nan)
        rmse = metrics.get('RMSE', np.nan)
        mape = metrics.get('MAPE', np.nan)

        mae_str = f'{mae:.2f}' if not np.isnan(mae) else 'N/A'
        rmse_str = f'{rmse:.2f}' if not np.isnan(rmse) else 'N/A'
        mape_str = f'{mape:.2f}' if not np.isnan(mape) else 'N/A'

        marker = ''
        if not np.isnan(mape) and mape < best_mape:
            best_mape = mape
            best_model = model_name
            marker = ' *'

        print(f'  {model_name+marker:<15} {mae_str:>10} '
              f'{rmse_str:>10} {mape_str:>10}')

    if best_model:
        print(f'  {"* Best model":<15}')


def classify_dish_by_nutrition(calories, protein, fat, carbs, fiber):
    if carbs > 20 and protein < 5 and fat < 3:
        return '主食'
    elif protein > 8 or fat > 8:
        return '荤菜'
    elif protein > 3 or fat > 3:
        return '半荤半素'
    else:
        return '素菜'


def calculate_calorie_breakdown(protein, fat, carbs):
    protein_cal = protein * 4
    fat_cal = fat * 9
    carbs_cal = carbs * 4
    total = protein_cal + fat_cal + carbs_cal

    return {
        'protein_cal': protein_cal,
        'fat_cal': fat_cal,
        'carbs_cal': carbs_cal,
        'total_cal': total,
        'protein_pct': protein_cal / total * 100 if total > 0 else 0,
        'fat_pct': fat_cal / total * 100 if total > 0 else 0,
        'carbs_pct': carbs_cal / total * 100 if total > 0 else 0,
    }


def format_prediction_table(pred_df, columns, rename_map=None):
    df = pred_df[columns].copy()
    if rename_map:
        df = df.rename(columns=rename_map)
    return df.to_string()


def check_nutrition_balance(calories, protein, fat, carbs, fiber):
    # 计算各营养素提供的热量
    total_cal = protein * 4 + fat * 9 + carbs * 4

    if total_cal > 0:
        protein_ratio = (protein * 4) / total_cal
        fat_ratio = (fat * 9) / total_cal
        carbs_ratio = (carbs * 4) / total_cal
    else:
        protein_ratio = fat_ratio = carbs_ratio = 0

    # 距离推荐区间的偏差评分
    def ratio_score(val, low, high):
        if low <= val <= high:
            return 1.0
        elif val < low:
            return max(0, 1 - (low - val) / low)
        else:
            return max(0, 1 - (val - high) / high)

    scores = {
        'protein_score': ratio_score(protein_ratio, 0.10, 0.15),
        'fat_score': ratio_score(fat_ratio, 0.20, 0.30),
        'carbs_score': ratio_score(carbs_ratio, 0.50, 0.65),
        'overall_balance': 0.0,
    }

    # 综合均衡度 = 三项评分的平均值
    scores['overall_balance'] = (
        scores['protein_score'] + scores['fat_score'] + scores['carbs_score']
    ) / 3

    # 附加信息
    scores.update({
        'protein_ratio': protein_ratio,
        'fat_ratio': fat_ratio,
        'carbs_ratio': carbs_ratio,
        'total_cal': total_cal,
        'fiber_per_1000kcal': (
            fiber / (calories / 1000) if calories > 0 else 0
        ),
    })

    return scores
