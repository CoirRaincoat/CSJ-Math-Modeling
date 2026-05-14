"""
utils.py — 通用工具函数模块
==========================
提供所有模块共享的辅助函数：
- 评估指标计算（MAPE、MAE、RMSE）
- 时间序列特征工程（滞后特征、滑动窗口）
- 结果格式化输出
- 营养学相关计算
"""

import numpy as np
import pandas as pd


def mape_score(y_true, y_pred):
    """
    计算MAPE（平均绝对百分比误差）
    
    MAPE = (1/n) * sum(|(y_i - y_hat_i) / y_i|) * 100
    
    对零值进行处理：跳过y_true=0的情况
    
    Args:
        y_true: 真实值数组
        y_pred: 预测值数组
        
    Returns:
        float: MAPE百分比值
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    # 过滤零值和NaN
    mask = (y_true != 0) & (~np.isnan(y_true)) & (~np.isnan(y_pred))
    
    if mask.sum() == 0:
        return np.nan
    
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape_score(y_true, y_pred):
    """
    计算sMAPE（对称平均绝对百分比误差）
    
    sMAPE = (1/n) * sum(|y_i - y_hat_i| / ((|y_i| + |y_hat_i|) / 2)) * 100
    
    比MAPE更稳健，避免零值问题
    
    Args:
        y_true: 真实值数组
        y_pred: 预测值数组
        
    Returns:
        float: sMAPE百分比值
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask2 = denominator > 0
    
    if mask2.sum() == 0:
        return np.nan
    
    return np.mean(np.abs(y_true[mask2] - y_pred[mask2]) / denominator[mask2]) * 100


def create_lag_features(df, col, lags):
    """
    创建滞后特征
    
    为时间序列创建指定滞后的特征列。
    例如 lag=1 表示前一天的值，lag=7 表示七天前的值。
    
    Args:
        df: DataFrame（index必须是日期）
        col: 源列名
        lags: 滞后天数列表，如 [1, 7, 14]
        
    Returns:
        DataFrame: 添加了滞后列的数据框
    """
    df = df.copy()
    for lag in lags:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df


def print_metrics_table(metrics_dict, target_name=''):
    """
    格式化打印模型评估指标
    
    Args:
        metrics_dict: {model_name: {'MAE': val, 'RMSE': val, 'MAPE': val}}
        target_name: 目标变量名称
    """
    if target_name:
        print(f'\n  目标: {target_name}')
    
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
        
        print(f'  {model_name+marker:<15} {mae_str:>10} {rmse_str:>10} {mape_str:>10}')
    
    if best_model:
        print(f'  {"* Best model":<15}')


def classify_dish_by_nutrition(calories, protein, fat, carbs, fiber):
    """
    基于营养成分的菜品分类（补充关键词分类的不足）
    
    对于无法通过关键词分类的菜品，使用营养成分特征进行辅助分类。
    
    分类逻辑：
    - 高蛋白+高脂 → 荤菜
    - 中蛋白+中脂 → 半荤半素
    - 低蛋白+低脂+高纤维 → 素菜
    - 高碳水+低蛋白 → 主食
    
    Args:
        calories: 热量 (kcal/100g)
        protein: 蛋白质 (g/100g)
        fat: 脂肪 (g/100g)
        carbs: 碳水化合物 (g/100g)
        fiber: 膳食纤维 (g/100g)
        
    Returns:
        str: 菜品类别
    """
    if carbs > 20 and protein < 5 and fat < 3:
        return '主食'
    elif protein > 8 or fat > 8:
        return '荤菜'
    elif protein > 3 or fat > 3:
        return '半荤半素'
    else:
        return '素菜'


def calculate_calorie_breakdown(protein, fat, carbs):
    """
    计算热量来源分布
    
    每克营养素的热量：
    - 蛋白质：4 kcal/g
    - 脂肪：9 kcal/g
    - 碳水化合物：4 kcal/g
    
    Args:
        protein: 蛋白质克数
        fat: 脂肪克数
        carbs: 碳水化合物克数
        
    Returns:
        dict: {'protein_cal': ..., 'fat_cal': ..., 'carbs_cal': ..., 
               'total_cal': ...}
    """
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
    """
    格式化预测结果为表格字符串
    
    Args:
        pred_df: 预测DataFrame
        columns: 要显示的列
        rename_map: 列名映射字典
        
    Returns:
        str: 格式化表格
    """
    df = pred_df[columns].copy()
    if rename_map:
        df = df.rename(columns=rename_map)
    return df.to_string()


def check_nutrition_balance(calories, protein, fat, carbs, fiber):
    """
    检查营养均衡度
    
    参考《中国居民膳食指南(2022)》及DRIs标准：
    - 碳水供能比：50-65%
    - 脂肪供能比：20-30%
    - 蛋白质供能比：10-15%
    
    Args:
        calories: 热量 (kcal)
        protein: 蛋白质 (g)
        fat: 脂肪 (g)
        carbs: 碳水化合物 (g)
        fiber: 膳食纤维 (g)
        
    Returns:
        dict: 各项指标的评分和均衡度
    """
    total_cal = protein * 4 + fat * 9 + carbs * 4
    
    if total_cal > 0:
        protein_ratio = (protein * 4) / total_cal
        fat_ratio = (fat * 9) / total_cal
        carbs_ratio = (carbs * 4) / total_cal
    else:
        protein_ratio = fat_ratio = carbs_ratio = 0
    
    # 评分（距离推荐区间的偏差）
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
    
    scores['overall_balance'] = (
        scores['protein_score'] + scores['fat_score'] + scores['carbs_score']
    ) / 3
    
    scores.update({
        'protein_ratio': protein_ratio,
        'fat_ratio': fat_ratio,
        'carbs_ratio': carbs_ratio,
        'total_cal': total_cal,
        'fiber_per_1000kcal': fiber / (calories / 1000) if calories > 0 else 0,
    })
    
    return scores
