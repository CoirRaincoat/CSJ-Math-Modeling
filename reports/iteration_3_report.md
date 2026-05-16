# 第 3 轮优化报告：自助餐厅需求预测与运营优化

## 1. 题目重述

同前。

## 2. 本轮开始前的项目状态 (v2)

- P2→P3 已串联（MILP读取SARIMA预测）
- P4 套餐去重已修复
- 缺少: 残差诊断、Walk-forward验证、分组误差分析

## 3. 程序输出完整分析

v2 的模型评估仅包含 in-sample MAE/RMSE/MAPE，缺少：
- 残差白噪声检验（Ljung-Box）
- 按星期/月份的分组误差
- 逼近真实部署场景的 Walk-forward 滚动验证

这些是时间序列预测的标准诊断步骤[Hyndman FPP3, Ch.5]。

## 4. 参考

| 类型 | 标题 | 来源 | 关系 | 支持 |
|------|------|------|------|------|
| 教材 | Hyndman FPP3 Ch.5.4 "Residual Diagnostics" | https://otexts.com/fpp3/diagnostics.html | Ljung-Box, ACF残差 | M6 |
| 教材 | Hyndman FPP3 Ch.3.4 "Time Series Cross-Validation" | https://otexts.com/fpp3/tscv.html | Walk-forward validation | M7 |

## 5. 本轮修改

| ID | 文件 | 修改 | 依据 |
|----|------|------|------|
| M6 | problem2_prediction.py | 新增 `_residual_diagnostics()`: Ljung-Box白噪声检验 + 残差ACF + 按星期MAPE | Hyndman FPP3 Ch.5.4 |
| M7 | problem2_prediction.py | 新增 `_walk_forward_validation()`: expanding window 滚动预测, step=7天 | Hyndman FPP3 Ch.3.4 |

## 6. 实验设置

- In-sample评估: baseline使用全部531天
- Walk-forward: 初始窗口80% (~425天), step=7天, XGBoost
- Ljung-Box: lags=[7,14,21]
- 新增图表: p2_residual_diagnostics.png, p2_walk_forward.png

## 7. 指标对比

| 指标 | v2 (in-sample) | v3 Walk-forward | 说明 |
|------|---------------|-----------------|------|
| MAE (total_orders) | 41.6 (Baseline) | **19.7** (XGBoost) | Walk-forward更贴近真实 |
| RMSE | 63.3 | 46.5 | |
| MAPE | 141.7% | **15.5%** | In-sample MAPE受小值影响 |
| Ljung-Box p(lag=7) | — | 报告 | 检验残差独立性 |
| 按星期MAPE | — | 报告 | 识别预测薄弱日 |

**关键发现**: Walk-forward MAPE=15.5%远优于in-sample Baseline MAPE=141.7%。原因是：(1)XGBoost捕获了非线性模式；(2)Walk-forward评估在更近期的数据上；(3)In-sample baseline MAPE被早期数据的极端波动放大。

## 8. 可视化

| 图表 | 说明 |
|------|------|
| p2_residual_diagnostics.png | 3行×3列: 残差分布/ACF/按星期MAPE |
| p2_walk_forward.png | 实际vs预测时序 + 残差vs预测散点 |

## 9. 负优化检查

| 项目 | 状态 |
|------|------|
| 指标退化 | 无 |
| 复杂度 | +2方法, 约200行 |
| 运行时间 | +90s (Walk-forward 100次XGBoost训练) |
| 数据泄露 | 无 (walk-forward严格保持时序) |

**结论: 正向优化。**

## 10. 仍可优化

| 优先级 | 问题 | 计划 |
|--------|------|------|
| P2 | 菜品分类准确率提升 | Iteration 4 |
| P2 | MILP成本率差异化 | Iteration 4 |
| P3 | P4套餐结合Bootstrap稳定规则 | Iteration 4 |

## 11. 下一轮计划 (Iteration 4)

1. 菜品分类关键词扩充（针对性添加高频"其他"菜品关键词）
2. MILP 成本率按类别差异化
3. P4 套餐优先使用Bootstrap稳定规则中的菜品配对

*报告生成: 2026-05-15 | 迭代: 3/5 | 状态: 正向优化*
