# 赛题B：自助量贩餐厅菜量需求预测与运营优化设计

## 完整解题总结

> **竞赛**: 2026年第六届长三角高校数学建模竞赛  
> **数据文件**: 附件1(3 sheet, 150,573行) + 附件2(15 sheet, 72,129行) + 附件3(2 sheet)  
> **时间跨度**: 2022-09-02 至 2025-04-30（31个月, 531个营业日, 149,626个订单）  
> **核心数据**: 日均282人, 日均销售额3,187元, 客单价11.39元, 314种菜品, 午餐占99.2%

---

## 一、问题1 — 数据预处理、统计分析与菜品关联关系

### 题干解读

对附件中自助量贩餐厅的历史交易数据进行预处理、统计和可视化分析，分析不同菜品销售量的分布规律以及它们之间可能存在的关联关系。

### 使用模型

| 模型 | 作用 | 支撑文献 |
|------|------|----------|
| **描述性统计** | 销量排名、日均指标、餐次分布、营养结构 | — |
| **Pareto/ABC分析** | 识别头部菜品（A类58种→80%销量，B类70种→15%，C类109种→5%） | [1] Rodrigues et al. (2024) — 餐饮库存分级管理 |
| **Welch's t-test** | 工作日vs周末订单量差异显著性检验（不等方差假设） | [4] Hyndman (FPP3) — 时间序列假设检验 |
| **Apriori关联规则** | 挖掘菜品共购关系（support→confidence→lift三级筛选） | [5] Agrawal & Srikant (1994) — 算法基础 |
| **菜品共现网络** (NetworkX) | 力导向布局可视化菜品关联结构 | [6] 余滔滔等 (2019) — 中文餐饮场景验证 |

### 关键参数

- Apriori三级阈值策略: min_support = 0.01 → 0.005 → 0.003
- 规则筛选: confidence ≥ 0.25, lift ≥ 1.15
- 最终产出: **19条关联规则**，最强规则 lift = 8.78（米饭+酱鸭 → 豆芽/木耳）

---

## 二、问题2 — 就餐人数、营养需求与销售总额预测

### 题干解读

根据餐厅销售记录，对该餐厅每天就餐人数、各类营养素需求量以及销售总额进行预测研究，并讨论预测模型的合理性和结果的可靠性。给出2025年5月份工作日的就餐人数、各类营养素需求量以及销售总额预测结果。

### 使用模型

| 模型 | 作用 | 支撑文献 |
|------|------|----------|
| **Baseline（历史同星期均值）** | 基准对比，捕捉星期周期性 | [4] Hyndman (FPP3) — 朴素预测方法 |
| **SARIMA(1,1,1)(1,1,1,7)** | 处理趋势+以7天为周期的季节性 | [4] Hyndman (FPP3) — 第9章ARIMA, 第11章季节性 |
| **XGBoost**（~30维特征） | 捕捉非线性关系：时间特征10维+滞后特征5维+滑动窗口6维 | [12] Chen & Guestrin (2016) — 算法基础 |
| **Ensemble组合预测** | 按1/MAPE加权融合三种模型 | [1] Rodrigues et al. (2024) — 多模型对比融合 |
| **TimeSeriesSplit**（3折） | 保持时间顺序的交叉验证 | [3] Thomassey et al. (2022) — 机器学习餐厅预测 |
| **ADF平稳性检验 + ACF/PACF** | 时间序列预分析 | [4] Hyndman (FPP3) — 第8章 |

### 预测目标（6个变量）

| 变量 | 预测日均值（May 2025） |
|------|----------------------|
| 就餐人数 | 287人 |
| 销售总额 | 3,115元 |
| 总热量 | 201,590 kcal |
| 总蛋白质 | 10,791 g |
| 总脂肪 | 7,009 g |
| 总碳水化合物 | 23,107 g |

### 评估指标

MAE、RMSE、MAPE，SARIMA在多数目标上表现最优。

### 补充文献

- [2] Posch K. et al. (2022) — 贝叶斯餐饮销售预测，多重季节性
- [3] Thomassey S. et al. (2022) — 特征工程+模型比较
- [11] Breiman L. (2001) — 随机森林，集成学习方法基础

---

## 三、问题3 — 菜品备菜优化（仅午餐）

### 题干解读

为提高餐厅营业利润，综合考虑各类营养素需求、该餐厅消费群体的消费习惯以及菜品多样性等因素，建立餐厅菜品优化模型，并给出2025年5月6日至5月12日期间每个工作日的备菜方案。

### 使用模型

| 模型 | 作用 | 支撑文献 |
|------|------|----------|
| **MILP（混合整数线性规划）** | 利润最大化，决策变量x_i为菜品i的备菜整数份数 | [7] 黄健等 (2018) — 双目标整数线性规划 |
| **营养约束建模** | 基于DRIs 2023的五大营养素供给上下限（±20%浮动） | [8] Padovan et al. (2023) — 营养推荐+成本约束 |
| **多样性约束** | 按菜品类别设最少份数（主食≥10，荤菜≥30，素菜≥30） | [9] Cohen et al. (2023) — 多目标菜单优化 |
| **安全库存模型** | 安全库存=需求×15%（Z=1.65, 95%服务水平） | [1] Rodrigues et al. (2024) — 预测驱动备菜 |
| **浪费成本建模** | w_i = max(x_i-d_i, 0)，浪费成本 = 成本×30% | [10] Gazendam et al. (2018) — LP膳食优化框架 |

### 数学公式

**决策变量**: x_i ∈ Z⁺（菜品i的备菜份数）

**目标函数**:
```
max Z = Σ(p_i·s_i) - Σ(c_i·x_i) - Σ(h·w_i) + γ·Σ(pop_i·x_i)
```
- p_i: 售价, c_i: 成本, s_i = min(x_i, d_i): 期望销售量
- w_i = max(x_i - d_i, 0): 浪费量, h = c_i × 0.3: 浪费成本系数
- pop_i: 午餐偏好度, γ = 0.1: 偏好奖励权重

**约束条件**:
1. 总份量: Σx_i ∈ [0.85D, 1.30D]（D为预测总需求）
2. 营养供给: Σ(a_ij·x_i) ∈ [R_j×(1-0.20), R_j×(1+0.20)]（j为5种营养素）
3. 类别多样性: 每类≥(最小菜品数×10)份
4. 单品上下限: 5 ≤ x_i ≤ D×0.25

**求解器**: PuLP + CBC，50菜品×5工作日

### 结果

| 日期 | 预估人数 | 备菜份数 | 预期利润 |
|------|----------|----------|----------|
| 05-06 Tue | 287 | 1,817 | 731元 |
| 05-07 Wed | 284 | 1,796 | 721元 |
| 05-08 Thu | 291 | 1,841 | 761元 |
| 05-09 Fri | 276 | 1,744 | 703元 |
| 05-12 Mon | 294 | 1,859 | 756元 |

> 营养均衡度 0.93-0.94, 蛋白质供能比17.9-18.0%, 脂肪22.6-23.5%, 碳水58.6-59.5%

### 关于晚餐的说明

晚餐仅占0.8%订单量（41天/529天），数据极度稀疏，无法支持可靠MILP建模。晚餐备菜建议采用启发式方法（精简菜单+午餐剩余利用），不作为优化模型输出。

### 补充文献

- [13] 中国营养学会. 《中国居民膳食营养素参考摄入量(DRIs)》(2023版) — 营养目标值来源
- [14] 中国营养学会. 《中国居民膳食指南》(2022) — 宏量营养素供能比推荐

---

## 四、问题4 — 不同价位套餐优化设计

### 题干解读

基于该餐厅消费群体的消费习惯以及营养搭配科学性，建立数学模型，优化设计不同价位的套餐，并分别给出10元、15元和20元三个价位的套餐方案。

### 使用模型

| 模型 | 作用 | 支撑文献 |
|------|------|----------|
| **贪心搜索**（200次采样） | 按价位结构模板逐类选择，70%确定性+30%随机性 | — |
| **爬山法局部优化**（100次迭代） | 替换/添加/移除操作寻找更优组合 | — |
| **五维评分函数** | 偏好(0.30)+营养均衡(0.30)+利润(0.25)+共购关联(0.15)+价格符合度(0.15) | [7] 黄健等 (2018) — 菜品配置框架 |
| **共购关联矩阵** | 基于购物篮数据的菜品共现概率（3,000订单采样） | [5] Agrawal (1994) — 关联规则; [6] 余滔滔等 (2019) — 菜品关联 |
| **营养均衡评估** | 三大宏量营养素供能比是否在推荐区间 | [14] 中国营养学会 (2022) — 膳食指南标准 |
| **套餐结构设计** | 阶梯定价+类别约束+菜品数量约束 | [8] Padovan et al. (2023) — 营养配餐约束; [9] Cohen et al. (2023) — 多目标优化 |

### 套餐方案

| 价位 | 定位 | 结构 | 实际总价 | 利润率 | 营养均衡度 | 热量 |
|------|------|------|----------|--------|------------|------|
| 10元 | 经济基础型 | 主食×1+荤菜×1+素菜×1 | 10.00元 | 55% | 0.98 | 683 kcal |
| 15元 | 均衡实用型 | 主食×1+荤菜×1+半荤×1+素菜×2 | 14.31元 | 55% | 0.98 | 669 kcal |
| 20元 | 丰富营养型 | 主食×1+荤菜×2+半荤×1+素菜×2+其他×1 | 19.69元 | 55% | 0.99 | 1,173 kcal |

---

## 五、问题5 — 经营情况分析与策略建议

### 题干解读

根据研究，综合分析该餐厅的运营情况，给出该餐厅优化经营的策略和建议。

### 策略体系

| 维度 | 核心策略 | 数据依据 | 支撑文献 |
|------|----------|----------|----------|
| **备菜策略** | ABC分级(A×1.15/B×1.05/C轮换)，预测→备菜→复盘闭环，Z=1.65安全库存 | 日需求波动CV=22.7%，星期波动16.3% | [1] Rodrigues et al. (2024) |
| **菜品结构** | 销量×利润率双维矩阵（推广76种/维持146种/替换79种），午餐99%专用策略 | 问题1菜品排序+问题3利润分析 | [1][7] |
| **套餐推广** | 10/15/20元三层阶梯套餐，动态更新，营养标识 | 问题4套餐评分+客单价11.39元 | [5][8][9] |
| **数字化运营** | 数据看板、模型周期迭代(MAPE>20%触发审查)、后厨智能集成 | 问题2预测框架+问题3MILP模型 | [2][3] |
| **营养ESG** | 优化脂肪供能比(32.9%→<30%)、食物浪费控制(日均319元→目标减半)、碳足迹管理 | 实际营养数据+10%剩余率估算 | [8][9][10][13][14] |

### 关键量化数据

- 人均热量: 721 kcal（午餐标准DRIs: 880 kcal）
- 脂肪供能比: 32.9%（推荐20-30%，略偏高）
- 日均浪费估算: 319元（按10%剩余率），年约11.6万元

---

## 六、文献支撑总表

| 编号 | 文献 | 来源地址 |
|------|------|----------|
| [1] | Rodrigues M. et al. "Machine learning models for short-term demand forecasting in food catering services: A solution to reduce food waste." *J. Cleaner Production*, 2024. | https://www.sciencedirect.com/science/article/pii/S0959652623044232 |
| [2] | Posch K. et al. "A Bayesian Approach for Predicting Food and Beverage Sales in Staff Canteens and Restaurants." *Intl. J. Forecasting*, 2022. | https://www.sciencedirect.com/science/article/pii/S0169207021001011 |
| [3] | Thomassey S. et al. "Machine Learning Based Restaurant Sales Forecasting." *Mach. Learn. Knowl. Extr.*, 2022. | https://www.mdpi.com/2504-4990/4/1/6 |
| [4] | Hyndman R.J., Athanasopoulos G. "Forecasting: Principles and Practice." 3rd ed., OTexts. | https://otexts.com/fpp3/ |
| [5] | Agrawal R., Srikant R. "Fast Algorithms for Mining Association Rules." *VLDB*, 1994. | https://www.vldb.org/conf/1994/P487.PDF |
| [6] | 余滔滔, 张革伕, 胡朝晖. "基于Apriori算法的菜品配置规则研究." *服务科学和管理*, 2019. | https://www.hanspub.org/journal/PaperInformation?paperID=32795 |
| [7] | 黄健等. "中国海洋大学食堂菜谱的优化模型研究." *应用数学进展*, 2018. | https://www.hanspub.org/journal/PaperInformation?paperID=23869 |
| [8] | Padovan M. et al. "Optimized menu formulation to enhance nutritional goals." *BMC Nutrition*, 2023. | https://link.springer.com/article/10.1186/s40795-023-00705-0 |
| [9] | Cohen J.F.W. et al. "Improving school lunch menus with multi-objective optimisation." *Public Health Nutrition*, 2023. | https://www.cambridge.org/core/journals/public-health-nutrition/article/3F4AABC5CDAD37717DEF164C4490DBAC |
| [10] | Gazendam A. et al. "A Review of the Use of Linear Programming to Optimize Diets, Nutritiously, Economically and Environmentally." *Frontiers in Nutrition*, 2018. | https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2018.00048/full |
| [11] | Breiman L. "Random Forests." *Machine Learning*, 2001. | https://link.springer.com/article/10.1023/A:1010933404324 |
| [12] | Chen T., Guestrin C. "XGBoost: A Scalable Tree Boosting System." *KDD*, 2016. | https://arxiv.org/abs/1603.02754 |
| [13] | 中国营养学会. 《中国居民膳食营养素参考摄入量(DRIs)》(2023版) | http://www.cnsoc.org/ |
| [14] | 中国营养学会. 《中国居民膳食指南》(2022) | http://dg.cnsoc.org/ |

---

## 七、代码模块总览

| 文件 | 功能 | 核心类/函数 |
|------|------|-------------|
| `config.py` | 全局配置、路径匹配、Nature NPG配色、营养标准 | `_find_attachments()` |
| `data_loader.py` | 3+15 sheet加载、数据清洗、特征工程、购物篮构建 | `DataLoader` |
| `utils.py` | MAPE/sMAPE、滞后特征、滑动窗口、营养均衡计算 | `mape_score()`, `check_nutrition_balance()` |
| `problem1_analysis.py` | EDA + ABC分析 + Apriori关联规则 + 5张可视化 | `Problem1Analysis` |
| `problem2_prediction.py` | SARIMA + XGBoost + Ensemble预测 + May2025外推 | `Problem2Prediction` |
| `problem3_optimization.py` | MILP午餐备菜优化（PuLP+CBC） | `Problem3Optimization` |
| `problem4_combos.py` | 贪心+局部搜索三层套餐设计 | `Problem4Combos` |
| `problem5_strategy.py` | 五维度经营策略分析 | `Problem5Strategy` |
| `main.py` | 主入口（支持 --skip/--only CLI参数） | `run_all()` |

### 输出文件（output/目录）

| 类型 | 文件 | 说明 |
|------|------|------|
| 问题1 | p1_sales_distribution.png | 销量Top20 + Pareto + ABC + 类别占比 |
| 问题1 | p1_temporal_patterns.png | 日趋势 + 星期箱线图 + 月度 + Welch's t-test |
| 问题1 | p1_meal_comparison.png | 午/晚餐消费分布 + 时段分布 + 营养对比 |
| 问题1 | p1_nutrition_analysis.png | 营养趋势 + 热量来源 + 相关矩阵 + 客单价 |
| 问题1 | p1_association_rules.png | 支持度-置信度散点图 + 菜品共现网络 |
| 问题2 | p2_time_series_overview.png | 6目标ADF平稳性检验 |
| 问题2 | p2_acf_pacf.png | ACF/PACF自相关分析 |
| 问题2 | p2_model_comparison.png | 4模型×6目标MAPE对比 |
| 问题2 | p2_may2025_predictions.png | 2025年5月22工作日预测柱状图 |
| 问题2 | p2_may2025_predictions.csv | 预测数据表 |
| 问题3 | p3_meal_plans.png | 午餐备菜方案（5天）+利润率+类别分布+雷达图 |
| 问题3 | p3_meal_plan_detail.csv | 50菜×5天详细备菜方案 |
| 问题4 | p4_combo_results.png | 三价位指标对比 + 营养雷达图 |
| 问题5 | p5_strategy_summary.png | 6子图策略框架 |

---

*文档生成时间: 2026-05-15*  
*项目路径: `C:\Users\CoirRaincoat\PyCharmMiscProject\CSJ-MathModeling`*
