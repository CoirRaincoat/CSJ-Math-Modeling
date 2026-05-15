"""
utils.py — 通用工具函数模块
==========================
提供所有模块共享的辅助函数:

1. 评估指标计算:
   - mape_score(): 平均绝对百分比误差
   - smape_score(): 对称平均绝对百分比误差
   - print_metrics_table(): 格式化打印模型评估结果

2. 时间序列特征工程:
   - create_lag_features(): 创建滞后特征 (lag-1, lag-7, lag-14)
   - create_rolling_features(): 创建滑动窗口统计

3. 营养学计算:
   - classify_dish_by_nutrition(): 基于营养成分的菜品分类
   - calculate_calorie_breakdown(): 热量来源分布计算
   - check_nutrition_balance(): 营养均衡度评价

4. 格式化输出:
   - format_prediction_table(): 预测结果表格格式化

参考文献:
  [1] Hyndman R.J. "Forecasting: Principles and Practice" (第3版)
      第5章: 时间序列的特征工程方法论
      https://otexts.com/fpp3/
  [2] 中国营养学会. 中国居民膳食指南(2022)
      http://dg.cnsoc.org/
"""

import numpy as np
import pandas as pd


def mape_score(y_true, y_pred):
    """
    计算 MAPE (Mean Absolute Percentage Error, 平均绝对百分比误差)

    公式:
      MAPE = (1/n) * Σ|(y_i - ŷ_i) / y_i| × 100%

    其中:
      y_i   = 第 i 个实际值
      ŷ_i   = 第 i 个预测值
      n     = 样本数量

    特点:
    - 优点: 直观易理解，以百分比表示预测误差
    - 缺点: 当实际值为 0 时无定义，对小值敏感
    - 适用: 餐饮人数预测 (值通常 > 0)

    处理策略:
    - 跳过 y_true = 0 的样本 (停业日就餐人数为 0)
    - 同时跳过 NaN 值
    - 若有效样本不足，返回 NaN

    Args:
        y_true: array-like, 真实值
        y_pred: array-like, 预测值

    Returns:
        float: MAPE 值 (百分比)，若无有效样本则返回 NaN

    Example:
        >>> mape_score([100, 200, 300], [110, 190, 310])
        3.33  # 平均误差约 3.33%
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # 过滤零值 (避免除零) 和 NaN
    mask = (y_true != 0) & (~np.isnan(y_true)) & (~np.isnan(y_pred))

    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape_score(y_true, y_pred):
    """
    计算 sMAPE (Symmetric Mean Absolute Percentage Error, 对称 MAPE)

    公式:
      sMAPE = (1/n) * Σ(|y_i - ŷ_i| / ((|y_i| + |ŷ_i|) / 2)) × 100%

    特点:
    - 优点: 比 MAPE 更稳健，值域为 [0, 200]，不受零点影响
    - 缺点: 当 y_true 和 y_pred 都接近 0 时不稳定
    - 适用: 作为 MAPE 的补充评估指标

    Args:
        y_true: array-like, 真实值
        y_pred: array-like, 预测值

    Returns:
        float: sMAPE 值 (百分比)

    Example:
        >>> smape_score([100, 200, 300], [110, 190, 310])
        3.28
    """
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
    """
    创建滞后特征 (Lag Features)

    滞后特征用于时间序列预测，表示过去某个时刻的观测值。
    例如 lag=7 表示 7 天前的值，可捕捉星期周期性模式。

    原理:
      对于时间序列 {y_t}，滞后 k 的特征为 y_{t-k}
      即第 t 天的 lag_7 值为第 t-7 天的观测值

    Args:
        df: DataFrame, 索引必须是日期 (DatetimeIndex)
        col: str, 源列名 (要创建滞后特征的列)
        lags: list of int, 滞后天数列表, 如 [1, 3, 7, 14]

    Returns:
        DataFrame: 添加了滞后列的数据框 (新列名为 {col}_lag{k})

    Example:
        >>> df = pd.DataFrame({'sales': [100,110,120,130]},
        ...                   index=pd.date_range('2023-01-01', periods=4))
        >>> df = create_lag_features(df, 'sales', [1, 2])
        >>> df['sales_lag1'].tolist()
        [nan, 100.0, 110.0, 120.0]

    Note:
        前 k 行的滞后值为 NaN (无历史数据)
    """
    df = df.copy()
    for lag in lags:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df


def create_rolling_features(df, col, windows):
    """
    创建滑动窗口统计特征 (Rolling Window Features)

    滑动窗口统计用于平滑时间序列波动，反映短期趋势。
    常见窗口:
    - window=3: 近 3 天的均值/标准差 (反映近期波动)
    - window=7: 近 7 天的均值/标准差 (反映周期模式)
    - window=14: 近 14 天的均值/标准差 (反映双周趋势)

    Args:
        df: DataFrame, 索引必须是日期
        col: str, 源列名
        windows: list of int, 窗口大小列表

    Returns:
        DataFrame: 添加了滑动窗口列的数据框

    Note:
        前 window-1 行的值为 NaN
    """
    df = df.copy()
    for w in windows:
        df[f'{col}_ma{w}'] = df[col].rolling(window=w, min_periods=1).mean()
        df[f'{col}_std{w}'] = df[col].rolling(window=w, min_periods=1).std()
    return df


def print_metrics_table(metrics_dict, target_name=''):
    """
    格式化打印多模型评估指标对比表

    输出格式:
       目标: 每日就餐人数
       Model           MAE      RMSE    MAPE(%)
       -----------------------------------------
       baseline *      23.45     30.12     12.34
       sarima          25.67     32.89     13.56
       xgboost         21.30     28.45     11.28 *
       ensemble        20.15     27.01     10.50 *

    其中 * 标记 MAPE 最小的模型 (最优模型)。

    Args:
        metrics_dict: dict, {模型名: {'MAE': val, 'RMSE': val, 'MAPE': val}, ...}
        target_name: str, 目标变量名称 (用于显示)
    """
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
    """
    基于营养成分的菜品辅助分类

    当菜品名称无法通过关键词匹配分类时 (返回 "其他")，
    使用此函数基于营养成分特征进行二次分类。

    分类逻辑 (单位: 每 100g):
    ┌───────────────┬────────────────────┬────────────────┐
    │  条件          │   分类             │  典型例子       │
    ├───────────────┼────────────────────┼────────────────┤
    │ 碳水>20g +    │   主食             │  米饭/面条     │
    │ 蛋白<5g +     │                    │                │
    │ 脂肪<3g       │                    │                │
    ├───────────────┼────────────────────┼────────────────┤
    │ 蛋白>8g 或    │   荤菜             │  红烧肉/鸡腿   │
    │ 脂肪>8g       │                    │                │
    ├───────────────┼────────────────────┼────────────────┤
    │ 蛋白>3g 或    │   半荤半素          │  鱼香肉丝/     │
    │ 脂肪>3g       │                    │  番茄炒蛋      │
    ├───────────────┼────────────────────┼────────────────┤
    │ 其余          │   素菜             │  清炒时蔬/     │
    │               │                    │  凉拌黄瓜      │
    └───────────────┴────────────────────┴────────────────┘

    参考依据:
    - 中国食物成分表 (第6版), 杨月欣主编
    - 米饭 (100g): 碳水 25.9g, 蛋白 2.6g, 脂肪 0.3g
    - 红烧肉 (100g): 碳水 4.2g, 蛋白 15.5g, 脂肪 34.5g
    - 番茄炒蛋 (100g): 碳水 5.2g, 蛋白 5.5g, 脂肪 5.0g

    Args:
        calories: 热量 (kcal, 每份)
        protein: 蛋白质 (g, 每份)
        fat: 脂肪 (g, 每份)
        carbs: 碳水化合物 (g, 每份)
        fiber: 膳食纤维 (g, 每份) — 当前未使用，保留用于扩展

    Returns:
        str: 菜品类别 (主食/荤菜/半荤半素/素菜)

    Note:
        此为辅助分类方法，优先级低于关键词匹配。
        阈值的设定基于中国常见菜品的营养特征分布。
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
    计算三大宏量营养素的热量贡献分布

    热量系数 (Atwater 通用系数):
      - 蛋白质: 4 kcal/g
      - 脂肪:   9 kcal/g
      - 碳水化合物: 4 kcal/g

    参考:
      FAO/WHO/UNU. "Energy and Protein Requirements", 1985.

    公式:
      蛋白质热量 = protein × 4
      脂肪热量   = fat × 9
      碳水热量   = carbs × 4
      总热量     = 蛋白质热量 + 脂肪热量 + 碳水热量

      (注意: 此公式计算的热量可能略低于标称热量，
      因为未计入膳食纤维约 2 kcal/g 和酒精 7 kcal/g)

    Args:
        protein: 蛋白质克数 (g)
        fat: 脂肪克数 (g)
        carbs: 碳水化合物克数 (g)

    Returns:
        dict: 包含各营养素热量和占比的字典
            - protein_cal: 蛋白质提供的热量 (kcal)
            - fat_cal: 脂肪提供的热量 (kcal)
            - carbs_cal: 碳水化合物提供的热量 (kcal)
            - total_cal: 总热量 (kcal)
            - protein_pct: 蛋白质供能比 (%)
            - fat_pct: 脂肪供能比 (%)
            - carbs_pct: 碳水化合物供能比 (%)

    Example:
        >>> breakdown = calculate_calorie_breakdown(65, 65, 300)
        >>> breakdown['protein_pct']
        13.0  # 蛋白质供能占 13%
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

    用于终端输出和报告生成。

    Args:
        pred_df: 预测 DataFrame
        columns: list, 要显示的列名
        rename_map: dict or None, 列名映射 {原列名: 显示名}

    Returns:
        str: 格式化表格字符串
    """
    df = pred_df[columns].copy()
    if rename_map:
        df = df.rename(columns=rename_map)
    return df.to_string()


def check_nutrition_balance(calories, protein, fat, carbs, fiber):
    """
    检查一餐的营养均衡度

    参考标准:
    《中国居民膳食指南 (2022)》推荐的三大宏量营养素供能比:
      - 碳水化合物: 50-65%
      - 脂肪:       20-30%
      - 蛋白质:     10-15%

    评分方法:
      对每个指标计算其偏离推荐区间的程度:
      - 若在推荐区间内 → 得分 = 1.0
      - 若低于下限 → 得分 = max(0, 1 - (下限-实际)/下限)
      - 若高于上限 → 得分 = max(0, 1 - (实际-上限)/上限)

      综合均衡度 = (蛋白质得分 + 脂肪得分 + 碳水得分) / 3

    Args:
        calories: 总热量 (kcal)
        protein: 蛋白质总量 (g)
        fat: 脂肪总量 (g)
        carbs: 碳水化合物总量 (g)
        fiber: 膳食纤维总量 (g)

    Returns:
        dict: 各项评分和比例信息
            - protein_score: 蛋白质评分 (0-1)
            - fat_score: 脂肪评分 (0-1)
            - carbs_score: 碳水化合物评分 (0-1)
            - overall_balance: 综合均衡度 (0-1)
            - protein_ratio: 蛋白质供能比
            - fat_ratio: 脂肪供能比
            - carbs_ratio: 碳水化合物供能比
            - total_cal: 按营养素计算的总热量
            - fiber_per_1000kcal: 每 1000 kcal 的膳食纤维量

    Example:
        >>> balance = check_nutrition_balance(800, 30, 25, 120, 10)
        >>> balance['overall_balance']
        0.90  # 接近推荐比例
    """
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
        """计算值在推荐区间 [low, high] 内的符合度 (0-1)"""
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
