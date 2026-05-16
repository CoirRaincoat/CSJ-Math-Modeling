# 第 2 轮优化报告：自助餐厅需求预测与运营优化

## 1. 题目重述

同 Iteration 1。

## 2. 本轮开始前的项目状态 (v1)

- P2: SARIMA.get_forecast() 样本外预测 + 中国假日过滤 + 95% CI ✓
- P1: Bootstrap 生存率标注 ✓
- P3: 仍使用独立的历史均值估算人数（未与 P2 串联）
- P4: 套餐搜索存在重复菜品风险

## 3. 程序输出完整分析

### P2→P3 数据流断裂（核心问题）

v1 中 Problem 2 消耗约 5 分钟训练 SARIMA 并生成 May 2025 预测 CSV，但 Problem 3 完全不读取这个结果，而是独立使用 `_get_predicted_diners(dow)` 从历史数据重新估算人数。两套预测值存在差异：

| 日期 | P2 SARIMA 预测 | P3 独立估算 | 差异 |
|------|---------------|-------------|------|
| 05-06 Tue | 284 | 287 | -3 |
| 05-07 Wed | 294 | 284 | +10 |
| 05-08 Thu | 300 | 291 | +9 |
| 05-09 Fri | 300 | 276 | +24 |
| 05-12 Mon | 288 | 294 | -6 |

理论依据：Rodrigues et al. (2024) 提出的"预测驱动备菜优化"框架要求优化模型的输入直接来自预测模型的输出，而非独立重新估算。

### P4 套餐重复菜品

v1 中 10 元套餐偶尔出现同一菜品被选中 2 次（如白饭出现于"主食"和"素菜"两个类别槽位），根因是 `used_names` 集合在跨类别边界时未充分检查。

## 4. 参考论文、文档和项目

| 类型 | 标题 | 来源 | 关系 | 支持 |
|------|------|------|------|------|
| 论文 | Rodrigues et al. "ML for short-term demand forecasting in food catering" (2024) | J. Cleaner Production | 预测驱动备菜优化框架 | M4 |
| 论文 | Padovan et al. "Optimized menu formulation" (2023) | BMC Nutrition | 营养约束 MILP | M4 |

## 5. 本轮修改内容

| ID | 文件 | 修改 | 原因 | 依据 |
|----|------|------|------|------|
| M4 | problem3_optimization.py | `run()` 新增 `prediction_csv` 参数; `_get_predicted_diners(date, pred_df)` 优先读取 P2 预测; `_get_predicted_nutrition(pred_df)` 优先读取 P2 营养均值 | P2→P3 数据流打通 | Rodrigues et al. (2024) |
| M5 | problem4_combos.py | `_greedy_search()` 添加去重检查: `if len(names) != len(set(names)): continue` | 防止套餐重复菜品 | 组合优化标准做法 |

### M4 修改细节

```python
# v1: 独立估算
predicted_diners = self._get_predicted_diners(dow)  # 历史同星期均值

# v2: 读取 P2 SARIMA 预测
def run(self, prediction_csv=None):
    if prediction_csv and os.path.exists(prediction_csv):
        pred_df = pd.read_csv(prediction_csv, index_col=0, parse_dates=True)
    ...
    predicted_diners = self._get_predicted_diners(date, pred_df)

def _get_predicted_diners(self, date, pred_df=None):
    if pred_df is not None:
        date_str = date.strftime('%Y-%m-%d')
        if date_str in pred_df.index.astype(str):
            return float(pred_df.loc[date_str, 'total_orders'])
    return self._fallback_diners(date)  # 回退
```

## 6. 实验设置

- 随机种子: RANDOM_SEED=42
- P2 预测: SARIMA(1,1,1)(1,1,1,7), 19 个 May 2025 工作日
- P3 MILP: CBC solver, timeLimit=120s, 50 菜品
- P4 套餐: 200 次贪心 × 100 次局部优化
- 运行命令: `python problem3_optimization.py` (读取 output/p2_may2025_predictions.csv)

## 7. 与历史版本的指标对比

### P2→P3 串联效果对比

| 日期 | v1 独立估算(人) | v2 P2预测(人) | v1 利润(元) | v2 利润(元) | 变化 |
|------|----------------|---------------|-------------|-------------|------|
| 05-06 | 287 | 284 | 731 | 712 | -19 |
| 05-07 | 284 | 294 | 721 | 751 | +30 |
| 05-08 | 291 | 300 | 761 | 763 | +2 |
| 05-09 | 276 | 300 | 703 | 763 | +60 |
| 05-12 | 294 | 288 | 756 | 728 | -28 |

### P4 去重效果

| 版本 | 10元重复 | 15元重复 | 20元重复 |
|------|----------|----------|----------|
| v1 | 偶尔 (白饭×2) | 无 | 无 |
| v2 | False | False | False |

### 综合指标对比

| 指标 | v0 Baseline | v1 (P2修复) | v2 (P2→P3串联+P4去重) |
|------|-------------|-------------|----------------------|
| P2 预测方法 | 历史均值 | SARIMA forecast | SARIMA forecast |
| 假日过滤 | 无 | chinese_calendar | chinese_calendar |
| 预测 CI | 无 | 95% CI | 95% CI |
| P3 数据源 | 独立估算 | 独立估算 | **P2 SARIMA预测** |
| P4 重复菜品 | 偶尔 | 偶尔 | **已修复** |
| 模块串联 | 断裂 | 断裂 | **已打通** |

## 8. 可视化结果说明

### v1 已有图表（保持不变）
- output/p1_*.png (5 张), output/p2_*.png (4 张)

### v2 更新图表
- output/p3_meal_plans.png: 午餐备菜方案，数据源标注为"问题2 SARIMA预测"
- output/p4_combo_results.png: 套餐结果，无重复菜品

### 数据流示意图
```
Problem 2 (SARIMA forecast)
    │
    ├─ output/p2_may2025_predictions.csv
    │       │
    │       └─ total_orders, total_sales, total_calories, ...
    │
    └── Problem 3 (MILP optimization)
            │
            ├─ reads: total_orders → predicted_diners
            ├─ reads: total_calories/protein/fat/carbs → nutrition targets
            │
            └─ output/p3_meal_plan_detail.csv (备菜方案)
```

## 9. 负优化检查

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 指标退化 | 无 | 某日利润因人数减少而轻微下降(-19元)，某日因人数增加而上升(+60元)，为正常波动 |
| 复杂度增加 | 轻微 | P3 新增 `prediction_csv` 参数和回退逻辑 |
| 运行时间增加 | 无 | CSV 读取 <0.1s |
| 解释性下降 | 无 | 明确标注数据来源 |
| 数据泄露 | 无 | P2 预测基于 2025-04 前数据，不包含 P3 目标周信息 |

**结论: 本轮为正向优化，无负优化。**

## 10. 当前仍可优化之处

| 优先级 | 问题 | 计划迭代 |
|--------|------|----------|
| P0 | Walk-forward validation 替代固定切分 | Iteration 3 |
| P1 | 预测残差自相关检验 (Ljung-Box) | Iteration 3 |
| P1 | 按星期分组 MAPE 分析 | Iteration 3 |
| P2 | 菜品分类准确率提升 | Iteration 4 |
| P2 | MILP 成本率差异化 | Iteration 4 |
| P3 | 套餐方案加入 Bootstrap 稳定规则 | Iteration 4 |

## 11. 下一轮优化计划 (Iteration 3)

1. **Walk-forward validation**: 对 XGBoost 使用 expanding window 滚动预测，替代 TimeSeriesSplit
2. **残差诊断**: Ljung-Box 白噪声检验 + 残差 ACF 图
3. **分组误差**: 按星期/月份/餐次分组计算 MAPE
4. **预测评估增强**: 使用最近 30 天作为 hold-out 验证集，检验 SARIMA 外推精度

---

*报告生成时间: 2026-05-15 | 迭代: 2/5 | 状态: 正向优化*
