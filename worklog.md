# 工作日志：2026长三角数学建模赛题B全流程

> 日期：2026-05-15
> 项目：自助量贩餐厅菜量需求预测与运营优化设计
> 路径：C:\Users\CoirRaincoat\PyCharmMiscProject\MathModeling

---

## 一、任务复盘时间线

```
14:00  接收任务，阅读PDF题干 + 方向建议文档
14:05  探索数据文件，发现中文编码问题
14:15  安装依赖（pandas, matplotlib, xgboost, statsmodels, pulp, mlxtend等）
14:20  编写数据探索脚本，理解数据结构
14:25  创建config.py → 文件路径动态匹配（解决中文名乱码）
14:30  创建data_loader.py → 数据加载/清洗/特征工程/购物篮
14:45  创建problem1_analysis.py → EDA + Apriori + 5张图
15:00  调试Apriori（min_support从0.02逐步降至0.01获25条规则）
15:10  创建problem2_prediction.py → SARIMA + XGBoost + Ensemble
15:25  遇到statsmodels未安装 → pip install解决
15:30  修复baseline_forecast的dayofweek()→dayofweek bug
15:35  修复print_metrics_table列的mae/mape打印顺序bug
15:40  创建utils.py → MAPE、滞后特征、营养均衡等工具函数
15:45  创建problem3_optimization.py → MILP备菜优化
15:55  修复df2/df2_with_meal变量名bug（3次迭代）
16:00  修复is_closed列不存在（df_daily来自loader无此列）
16:05  晚餐MILP infeasible → 添加is_small_meal分支放宽约束
16:10  创建problem4_combos.py → 贪心+局部搜索套餐设计
16:15  ¥符号GBK编码错误 → 改用"yuan"文本
16:20  创建problem5_strategy.py → 五维策略+可视化
16:25  创建main.py → 主入口（--skip/--only CLI参数）
16:30  创建final_report.md → 完整工作报告
16:35  测试全流水线端到端
16:40  创建本工作日志
```

---

## 二、核心架构决策

### 2.1 模块设计

采用「共享数据源 + 独立问题求解」架构：

```
config.py (全局配置)
    ↓
data_loader.py (DataLoader类，一次性加载，各模块复用)
    ↓
┌──────────┬──────────┬──────────┬──────────┬──────────┐
│problem1  │problem2  │problem3  │problem4  │problem5  │
│Analysis  │Prediction│Optimize  │Combos    │Strategy  │
└──────────┴──────────┴──────────┴──────────┴──────────┘
    ↓
main.py (编排器，支持选择性执行)
```

**关键设计原则**：
- DataLoader只实例化一次，通过参数传递给各问题模块（避免重复加载12MB的xlsx）
- 每个问题模块是独立的类，可单独运行或组合运行
- 输出统一写入`output/`目录
- 随机种子固定(RANDOM_SEED=42)保证可复现性

### 2.2 数据流

```
附件1.xlsx (65,534行订单级)
附件2.xlsx (65,535行菜品级)
        ↓ DataLoader._load_data()
    原始pandas DataFrame
        ↓ _clean_and_preprocess()
    + 日期/餐次/星期特征
    + 异常值标记(IQR)
        ↓ _feature_engineering()
    + 菜品分类(关键词匹配)
    + 附件1+2融合 (df_trans)
    + 菜品信息表 (dish_info)
        ↓ _build_aggregations()
    → df_daily (236天日级汇总)
    → df_meal (245条餐次级汇总)
        ↓ _build_basket()
    → basket_binary (11,828订单 × 237菜品的0/1矩阵)
```

---

## 三、遇到的问题与解决方案

### 3.1 文件编码问题 ⭐ 关键教训

**现象**：中文字符在终端输出均为乱码，Python中文字符串与文件系统文件名不匹配导致FileNotFoundError

**根因**：Windows系统使用GBK编码，而Python源码/终端使用UTF-8，Mojibake导致文件名匹配失败

**解决**：
```python
# 错误做法：硬编码中文路径
ATTACHMENT1 = '附件1餐厅销售流水信息表.xlsx'  # 文件系统存的不是这个编码

# 正确做法：按文件大小特征动态匹配
def _find_attachments():
    xlsx_files = []
    for f in os.listdir(DATA_DIR):
        if f.endswith('.xlsx'):
            xlsx_files.append((os.path.join(DATA_DIR, f), os.path.getsize(f)))
    xlsx_files.sort(key=lambda x: x[1], reverse=True)
    return xlsx_files[0][0], xlsx_files[1][0]  # 最大=附件1, 第二=附件2
```

**教训**：在中文Windows环境下，永远不要硬编码中文文件名，使用文件大小、修改时间、或通配符匹配

### 3.2 Apriori无关联规则

**现象**：初始min_support=0.02时产生138个频繁项集但0条关联规则

**根因**：237种菜品+11,828个订单，单个菜品最高支持度0.96（米饭），其余菜品远低于0.02。频繁项集中size-2以上的项集太少，无法生成满足confidence≥0.3的规则

**解决**：采用三级阈值逐步降低策略：
```python
min_support_levels = [0.01, 0.005, 0.003]
for min_support in min_support_levels:
    # ...apriori...
    if len(rules_filtered) >= 5: break
```
最终min_support=0.01 + confidence≥0.25 + lift≥1.15 → 25条规则

### 3.3 MILP晚餐Infeasible

**现象**：晚餐（21-23人）优化求解返回Infeasible，或解出荒谬结果（备菜29万份，利润-83万）

**根因**：就餐人数极少时，营养约束（热量/蛋白质/脂肪下限）与菜品多样性约束（各类别至少n种）矛盾。例如：21人的午餐热量需求≈21×880=18,480kcal，但要覆盖主食+荤菜+素菜等5类，每类至少10份，导致必然超量

**解决**：添加`is_small_meal`标志（predicted_diners < 50），对晚餐：
- 总份量约束放宽至0.8x-2.0x
- 营养约束仅设下限（0.3-0.5x），去掉上限
- 单菜品上限从demand×0.25提升至demand×0.5

但即使如此，晚餐仍有时infeasible。**根本原因是数据本身——该餐厅99%业务在午餐，晚餐数据几乎不存在。** 在论文中应诚实说明这一局限性。

### 3.4 其他小Bug

| Bug | 原因 | 修复 |
|-----|------|------|
| `date.dayofweek()` TypeError | pandas Timestamp中dayofweek是属性非方法 | 去掉括号 |
| print_metrics_table列错位 | 格式化字符串MAE位置写了mape_str | 改为mae_str |
| df2/df2_with_meal变量混淆 | merge后用回原始df2导致meal_period列缺失 | 统一使用df2_with_meal |
| `¥`符号GBK编码错误 | print输出时终端GBK无法编码¥ | 改用"yuan"文本 |

---

## 四、技术笔记（供日后参考）

### 4.1 购物篮分析标准流程

```python
# 1. 构建二值矩阵（行=交易，列=商品，值=0/1）
basket = df.pivot_table(index='order_id', columns='item', aggfunc='count', fill_value=0)
basket_binary = (basket > 0).astype(int)

# 2. 过滤低频商品（减少噪音和计算量）
frequent = basket_binary.sum()[basket_binary.sum() >= 50].index
basket_filt = basket_binary[frequent]

# 3. Apriori挖掘
from mlxtend.frequent_patterns import apriori, association_rules
itemsets = apriori(basket_filt, min_support=0.01, use_colnames=True, max_len=3)
rules = association_rules(itemsets, metric='lift', min_threshold=1.0)

# 4. 筛选有意义的规则
rules_filtered = rules[(rules['lift'] > 1.2) & (rules['confidence'] > 0.3)]
```

### 4.2 时间序列预测pipeline

```python
# 特征工程三件套
df['day_of_week'] = df.index.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
for lag in [1, 3, 7]:
    df[f'lag_{lag}'] = df['target'].shift(lag)
for w in [3, 7]:
    df[f'ma_{w}'] = df['target'].rolling(w).mean()

# 模型比较框架
models = {
    'baseline': 历史同星期均值,
    'sarima': SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,7)),
    'xgboost': XGBRegressor(特征=时间+滞后+滑动),
}
# 组合预测：按1/MAPE加权
```

### 4.3 MILP建模模板（PuLP）

```python
from pulp import LpProblem, LpMaximize, LpVariable, LpInteger, lpSum

prob = LpProblem("name", LpMaximize)

# 决策变量
x = {i: LpVariable(f"x_{i}", lowBound=0, cat=LpInteger) for i in range(n)}

# 目标函数
prob += lpSum([revenue[i] * x[i] for i in range(n)]) - lpSum([cost[i] * x[i]])

# 约束
prob += lpSum([x[i] for i in range(n)]) >= total_demand  # 需求约束
prob += lpSum([nutrient[i] * x[i] for i in range(n)]) >= target  # 营养约束

prob.solve(PULP_CBC_CMD(msg=False))
result = {i: int(value(x[i])) for i in range(n)}
```

### 4.4 套餐搜索模板

```python
def score_combo(dishes, target_price):
    """五维评分：偏好 + 营养 + 利润 + 关联 + 价格符合度"""
    popularity = mean(d.popularity_score for d in dishes)
    nutrition = check_balance(dishes)  # 蛋白质/脂肪/碳水比例
    profit = (total_price - total_cost) / total_price
    association = mean(cooccurrence.get((d1,d2), 0) for d1,d2 in pairs(dishes))
    price_fit = max(0, 1 - abs(total_price - target_price) / target_price * 3)
    return 0.3*popularity + 0.3*nutrition + 0.25*profit + 0.15*association + 0.15*price_fit

# 贪心采样 + 局部搜索
candidates = [greedy_sample() for _ in range(200)]
best = max(candidates, key=score)
for _ in range(100):
    neighbor = mutate(best)  # 替换/添加/移除一个菜品
    if score(neighbor) > score(best): best = neighbor
```

---

## 五、能力自评

### 做得好的
- 数据处理稳健（编码问题、缺失值、异常值全覆盖）
- 多模型比较意识（预测用了3种模型+组合）
- 可视化丰富（14张专业级图表）
- 工程项目化（模块分离、配置集中、CLI接口）
- 文献引用规范（14篇中外文献含URL）

### 下次改进
- 菜品分类应用NLP/聚类替代硬编码关键词（当前46%覆盖率）
- 预测模型加入Prophet处理节假日
- 晚餐优化应用完全不同的策略（数据不够时不要强行建模）
- 套餐搜索用遗传算法替代贪心
- 增加灵敏度分析和假设检验
- 代码单元测试覆盖

### 关键数字记忆
- 日均274单、3033元、客单价11.36元
- 237种菜品，ABC分类A类58种(80%销量)
- 午餐99.2% vs 晚餐0.8%
- Apriori: min_support≥0.01, 25条规则
- MILP: 40种菜品午餐优化，晚餐需特殊处理
- 套餐: 10/15/20元三层，贪心200次+局部100次

---

## 六、文件清单

```
MathModeling/
├── config.py              # 全局配置
├── data_loader.py         # 数据加载与预处理
├── utils.py               # 通用工具函数
├── problem1_analysis.py   # 问题1：EDA + 关联规则
├── problem2_prediction.py # 问题2：多模型预测
├── problem3_optimization.py # 问题3：MILP备菜优化
├── problem4_combos.py     # 问题4：套餐设计
├── problem5_strategy.py   # 问题5：经营策略
├── main.py                # 主入口
├── final_report.md        # 完整工作报告
├── worklog.md             # 本工作日志
├── output/                # 输出目录（14个文件）
│   ├── p1_*.png           # 问题1 × 5张图
│   ├── p2_*.png/csv       # 问题2 × 4图1表
│   ├── p3_*.png/csv       # 问题3 × 1图1表
│   ├── p4_*.png           # 问题4 × 1图
│   └── p5_*.png           # 问题5 × 1图
├── 附件1餐厅销售流水信息表.xlsx (~12.8MB)
├── 附件2部分消费订单菜品具体信息表.xlsx (~5.5MB)
├── 附件3数据说明.xlsx (~12KB)
└── 赛题B_方向与文献建议整理.md
```
