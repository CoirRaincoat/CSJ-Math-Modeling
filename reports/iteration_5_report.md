# 第 5 轮优化报告（最终轮）：自助餐厅需求预测与运营优化

## 1. 题目重述

同前。

## 2. 本轮开始前状态 (v4)

- 菜品分类准确率 99.1% ✓
- 成本率按类别差异化 ✓
- P2→P3 串联 ✓
- Walk-forward + 残差诊断 ✓
- P4 套餐去重 ✓

## 3. 本轮修改

| ID | 文件 | 修改 | 依据 |
|----|------|------|------|
| M11 | problem4_combos.py | `_score_combo()` 新增 Bootstrap 稳定规则奖励: 若套餐包含生存率>80%的关联配对, +0.05分 | Bootstrap 验证报告(Iter 1): 9/19规则稳定 |

## 4. 五轮迭代总览

| 迭代 | 核心修改 | 文件 | 分类 | 预测 | 优化 | 套餐 | 策略 |
|------|----------|------|------|------|------|------|------|
| Iter 1 | SARIMA forecast + 假日 + CI | problem2_prediction.py | — | ✓ | — | — | — |
| Iter 2 | P2→P3 数据流 + 套餐去重 | problem3/4 | — | — | ✓ | ✓ | — |
| Iter 3 | Walk-forward + 残差诊断 | problem2_prediction.py | — | ✓ | — | — | — |
| Iter 4 | 分类关键词 + 成本率差异化 | config.py, data_loader.py | ✓ | — | ✓ | — | — |
| Iter 5 | 套餐Bootstrap规则集成 | problem4_combos.py | — | — | — | ✓ | — |

## 5. 最终指标对比

| 指标 | v0 Baseline | v5 Final | 改善 |
|------|-------------|----------|------|
| P2 预测方法 | 历史均值(无变化) | SARIMA forecast + 95%CI | 真正样本外预测 |
| P2 Walk-forward MAPE | — | 15.5% (XGBoost) | 新增验证 |
| 假日过滤 | 无 | chinese_calendar | May1-5排除 |
| 分类准确率 | ~46% | 99.1% | +53% |
| 成本建模 | 统一45% | 按类别(28-60%) | 差异化 |
| P2→P3 串联 | 断裂 | 已打通 | 预测驱动优化 |
| P4 重复菜品 | 偶尔 | 已修复 | — |
| P4 规则可信度 | 无 | Bootstrap稳定规则奖励 | +0.05 bonus |
| 残差诊断 | 无 | Ljung-Box + ACF + 按星期MAPE | 新增 |
| 图表数量 | 14 | 18 (+4张诊断图) | — |

## 6. 运行命令

```powershell
& "C:\Users\CoirRaincoat\AppData\Local\Programs\Python\Python313\python.exe" main.py
```

环境: Python 3.13, RANDOM_SEED=42, 所有依赖见 AGENTS.md

## 7. 负优化检查（全局）

遍历 5 轮迭代的所有修改，无指标退化、无数据泄露、无可解释性下降。所有修改均有文献或官方文档依据。

## 8. 达到的停止条件

1. 核心缺陷已修复（P2 预测、P2→P3 串联、分类准确率）
2. 诊断体系已建立（残差/分组误差/Walk-forward）
3. 成本建模已差异化
4. 剩余问题（价格单位验证、NLP分类）需要外部数据或超出建模范围

## 9. 最终项目文件清单

```
CSJ-MathModeling/
├── config.py, data_loader.py, utils.py         # 基础设施
├── problem1_analysis.py ~ problem5_strategy.py # 五题求解
├── main.py                                      # 主入口
├── validate_reliability.py                      # 可靠性验证套件
├── baseline_eval.py                             # Baseline评估
├── generate_paper.py                            # Word论文生成
├── output/ (18 PNG + 3 CSV)                     # 可视化输出
├── reports/
│   ├── iteration_1_report.md ~ iteration_5_report.md  # 迭代报告
├── final_report.md, solution_summary.md         # 总结文档
├── worklog.md, phase_review.md                  # 工程日志
├── AGENTS.md                                    # 开发指南
└── 赛题B论文_*.docx                              # 生成的Word论文
```

*报告生成: 2026-05-15 | 迭代: 5/5 (最终轮) | 状态: 全部正向优化, 达到停止条件*
