# 代码功能介绍与逻辑链

> 项目：自助量贩餐厅菜量需求预测与运营优化设计  
> 路径：`C:\Users\CoirRaincoat\PyCharmMiscProject\CSJ-MathModeling`

---

## 一、文件清单与功能

| 文件 | 类型 | 功能 |
|------|------|------|
| `config.py` | 配置 | 路径/餐次/分类键/营养标准/NPG配色/所有全局常量 |
| `data_loader.py` | 数据 | 全量Sheet加载→清洗→特征→日级/餐次级汇总→购物篮矩阵 |
| `utils.py` | 工具 | MAPE/sMAPE、滞后特征、滑动窗口、营养均衡度、热量分解 |
| `problem1_analysis.py` | 问题1 | 描述性统计 + ABC/Pareto + Apriori关联规则 + 5张可视化 |
| `problem2_prediction.py` | 问题2 | SARIMA + XGBoost + Ensemble预测 + Walk-forward + May2025外推 |
| `problem3_optimization.py` | 问题3 | MILP午餐备菜优化(PuLP+CBC)，读取P2预测结果 |
| `problem4_combos.py` | 问题4 | 贪心搜索+局部优化三层套餐设计(10/15/20元) |
| `problem5_strategy.py` | 问题5 | 五维度经营策略分析 + 6子图框架图 |
| `main.py` | 入口 | 串联全部模块(--skip/--only CLI) |
| `validate_reliability.py` | 验证 | 5项可靠性检验：覆盖率偏差/Bootstrap/SHAP/敏感性/营养一致性 |
| `generate_paper.py` | 论文 | 仿优秀论文格式生成Word文档 |
| `data_generator.py` | 备用 | 仿真数据生成器(42道菜×6个月) |

---

## 二、全局逻辑链

```
                    ┌─────────────────┐
                    │   config.py     │  全局常量、路径、颜色
                    └───────┬─────────┘
                            ↓
                    ┌─────────────────┐
                    │  data_loader.py │  全量Sheet加载→清洗→汇总→购物篮
                    │   DataLoader()  │  一次实例化，所有模块共享
                    └───────┬─────────┘
                            ↓
        ┌───────────────────┼───────────────────┐
        ↓                   ↓                   ↓
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ problem1     │   │ problem2     │   │ problem4     │
│ EDA+Apriori  │   │ SARIMA+XGBoost│  │ 套餐设计     │
│ 5张图        │   │ +Ensemble    │   │ 贪心+局部    │
└──────────────┘   └──────┬───────┘   └──────────────┘
                          │ 输出 CSV
                          ↓
                  ┌──────────────┐
                  │ problem3     │  读取P2预测→MILP午餐备菜
                  │ PuLP+CBC     │
                  └──────┬───────┘
                         │
                         ↓
                  ┌──────────────┐
                  │ problem5     │  五维度策略+数据依据
                  │ 策略分析     │
                  └──────────────┘
```

---

## 三、各模块核心逻辑

### 3.1 config.py

**职责**：所有模块共享的全局配置。

**关键常量**：
- `ATTACHMENT1/2`：按文件大小自动匹配附件路径（解决中文编码）
- `LUNCH_START/END=10/14, DINNER_START/END=16/20`：餐次划分
- `CATEGORY_KEYWORDS`：主食/荤菜/半荤半素/素菜四类关键词词典
- `COST_RATIO_BY_CATEGORY`：主食0.28/荤菜0.60/半荤0.45/素菜0.30
- `NUTRITION_PER_MEAL`：午餐热量880kcal/蛋白质26g/脂肪26g/碳水120g/纤维10g
- `COLORS`：Nature NPG期刊配色(#3C5488深蓝/#00A087青绿/#E64B35红等)
- `RANDOM_SEED=42`：全局随机种子

### 3.2 data_loader.py (DataLoader类)

**入口**：`load_all_data()` → `DataLoader()`

**流水线**（5步自动执行）：
1. `_load_data()` → 遍历每个xlsx的所有Sheet，`pd.concat()`全量拼接
2. `_clean_and_preprocess()` → 日期解析/餐次划分/异常值标记(IQR)/空列剔除
3. `_feature_engineering()` → 两轮菜品分类(关键词匹配→营养特征辅助) + dish_info构建(成本=price×category_ratio) + 附件1↔2融合(df_trans)
4. `_build_aggregations()` → df_daily(531天) + df_meal(570条)
5. `_build_basket()` → pivot_table → 二值化 → 过滤频次<50 → 12,944×223矩阵

**数据接口**：`get_daily_data()`, `get_meal_data()`, `get_dish_info()`, `get_basket_data()`

### 3.3 problem1_analysis.py (Problem1Analysis类)

**逻辑链**：DataLoader → 预处理摘要 → 销量分布(5图) → 时间模式(4子图) → 餐次对比(3子图) → 营养分析(4子图) → Apriori规则(2子图)

**关键方法**：
- `_sales_distribution_analysis()` → Top20订单/销售额柱状图 + Pareto/ABC分析 + 类别饼图
- `_temporal_pattern_analysis()` → 日趋势 + 星期箱线图 + 月度趋势 + Welch's t检验(工作日vs周末)
- `_association_rule_mining()` → 三级阈值Apriori(0.01→0.005→0.003) + confidence≥0.25 + lift≥1.15

**输出**：p1_sales_distribution.png, p1_temporal_patterns.png, p1_meal_comparison.png, p1_nutrition_analysis.png, p1_association_rules.png

### 3.4 problem2_prediction.py (Problem2Prediction类)

**逻辑链**：日级数据→日期索引→补全缺失→标记is_closed → ADF检验(6目标) → ACF/PACF → 4模型训练(Baseline/SARIMA/XGBoost/Ensemble) → 模型比较图 → 残差诊断(Ljung-Box+ACF+按星期MAPE) → Walk-forward验证(expanding window, step=7) → May2025 SARIMA.get_forecast()

**关键方法**：
- `_baseline_forecast()` → 历史同星期均值(朴素基线)
- `_sarima_forecast()` → SARIMA(1,1,1)(1,1,1,7)，maxiter=100
- `_xgboost_forecast()` → ~30维特征 + TimeSeriesSplit(3折)
- `_ensemble_forecast()` → 按1/MAPE加权融合
- `_predict_may_2025()` → chinese_calendar排除假日 → SARIMA.get_forecast(steps=19) → 95%CI
- `_walk_forward_validation()` → 初始80%→step=7天→expanding window
- `_residual_diagnostics()` → 残差分布+Ljung-Box+按星期MAPE

**输出**：p2_time_series_overview.png, p2_acf_pacf.png, p2_model_comparison.png, p2_residual_diagnostics.png, p2_walk_forward.png, p2_may2025_predictions.png, p2_may2025_predictions.csv

### 3.5 problem3_optimization.py (Problem3Optimization类)

**逻辑链**：DataLoader → 加载P2预测CSV(或回退历史均值) → 选择50种午餐菜品 → 逐日MILP求解

**核心方法**：`optimize_meal(dishes, predicted_diners)`

**MILP模型**（PuLP + CBC求解器）：
- **决策变量**：x_i ∈ Z⁺（50种菜品的整数备菜份数）
- **目标**：max Z = Σp_i·s_i - Σc_i·x_i - Σh·w_i + 0.1·Σpop_i·x_i
- **约束**：(a)0.85D≤Σx_i≤1.30D (b)营养供给±20% (c)每类≥最小份数 (d)5≤x_i≤0.25D
- **线性化**：s_i = min(x_i,d_i)通过s_i≤x_i且s_i≤d_i线性化

**输出**：p3_meal_plans.png, p3_meal_plan_detail.csv

### 3.6 problem4_combos.py (Problem4Combos类)

**逻辑链**：dish_info → 构建共现矩阵(3000订单采样) → 对每个价位(10/15/20)：贪心200次+局部优化100次 → 五维评分 → 选择最优

**五维评分**：0.30×偏好 + 0.30×营养 + 0.25×利润 + 0.15×共购 + 0.15×价格符合度 + 类别多样性奖励

**关键方法**：
- `_greedy_search(target_price)` → 按结构模板逐类选择(70%确定性+30%随机性)
- `_local_optimization()` → 替换/添加/移除(爬山法)，仅接受更优
- `_score_combo()` → 含去重检查和Bootstrap稳定规则奖励

**输出**：p4_combo_results.png

### 3.7 problem5_strategy.py (Problem5Strategy类)

**逻辑链**：DataLoader → 备菜策略(CV分析+ABC) → 菜品结构(销量×单价矩阵) → 套餐推广 → 数字化运营 → 营养ESG(脂肪比+浪费估算) → 6子图框架

**每项建议公式**：`data_basis`字段标注定量来源（如"基于问题1的ABC分析：A类76种贡献80%销量"）

**输出**：p5_strategy_summary.png

### 3.8 main.py

**功能**：统一编排入口，支持选择性执行。
```powershell
python main.py              # 全部5题
python main.py --only 3     # 仅问题3
python main.py --skip 2,5   # 跳过问题2和5
```

### 3.9 validate_reliability.py

**5项独立验证**：
1. V1 覆盖率偏差 → KS检验附件2子集vs全集的分布一致性
2. V2 Bootstrap稳定性 → 500次Apriori重采样，追踪规则生存率
3. V3 XGBoost特征重要性 → gain-based top 15特征排名
4. V4 MILP敏感性 → 200次Monte Carlo参数扰动(需求/成本/浪费)
5. V5 营养一致性 → 附件1 vs 附件2同订单营养汇总对比(MAD/MAPE/相关系数)

---

## 四、数据流向图

```
附件1.xlsx (3 Sheets)           附件2.xlsx (15 Sheets)
  indent_1: 65,534                indent_details_1~15: 72,129
  indent_2: 65,536
  indent_3: 19,503                      ↓
       ↓                         df2 (菜品级明细)
  df1 (订单级流水)                      ↓
       ↓                  ┌──── pivot_table(indent_id, dish_name)
       ↓                  │
  pd.concat() → 150,573行  │     购物篮矩阵(12,944 × 314)
       ↓                  │           ↓
  清洗/特征/餐次          │     过滤频次≥50
       ↓                  │           ↓
  df_daily (531天)         │     223种菜品矩阵
  df_meal (570条)          │           ↓
       ↓                  │     Apriori (问题1)
  问题2 预测模型           │     Bootstrap验证
       ↓                  │
  p2_may2025_predictions.csv    套餐共现矩阵 (问题4)
       ↓                          
  问题3 MILP优化 ←────────┘
```

---

## 五、运行环境

```powershell
# Python 3.13 (位于 C:\Users\CoirRaincoat\AppData\Local\Programs\Python\Python313\python.exe)
# 依赖安装:
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost statsmodels pulp mlxtend networkx openpyxl python-docx chinesecalendar

# 运行:
python main.py
```
