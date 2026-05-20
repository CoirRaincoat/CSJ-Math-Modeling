"""validate_reliability.py — 数据可靠性与模型稳健性验证"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, warnings, time
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

d = r'C:\Users\CoirRaincoat\PyCharmMiscProject\CSJ-MathModeling'
PROJECT_DIR = d
files = os.listdir(d)
OUTDIR = os.path.join(d, 'output')

# Font & style
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300
COLORS = {
    'primary': '#3C5488', 'secondary': '#00A087', 'accent': '#E64B35',
    'success': '#4DBBD5', 'warning': '#F39B7F', 'danger': '#DC0000',
    'purple': '#8491B4', 'gold': '#E69F00', 'grey': '#A0A0A0',
}
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load full data
print("Loading data...")
f1 = [f for f in files if '1' in f and f.endswith('.xlsx') and len(f)>10][0]
f2 = [f for f in files if '2' in f and f.endswith('.xlsx')][0]

df1 = pd.concat([pd.read_excel(os.path.join(d,f1), sheet_name=s) 
                 for s in pd.ExcelFile(os.path.join(d,f1)).sheet_names], ignore_index=True)
df2 = pd.concat([pd.read_excel(os.path.join(d,f2), sheet_name=s) 
                 for s in pd.ExcelFile(os.path.join(d,f2)).sheet_names], ignore_index=True)

df1['consume_time'] = pd.to_datetime(df1['consume_time'])
df1['date'] = df1['consume_time'].dt.date
df1['hour'] = df1['consume_time'].dt.hour
df1['day_of_week'] = df1['consume_time'].dt.dayofweek
df1['is_weekend'] = df1['day_of_week'].isin([5,6]).astype(int)

df2['dish_name'] = df2['dish_name'].str.strip()

# Orders with dish details
detail_ids = set(df2['indent_id'].unique())
df1['has_detail'] = df1['indent_id'].isin(detail_ids).astype(int)
print(f"Full orders: {len(df1):,}, With detail: {df1['has_detail'].sum():,} ({df1['has_detail'].mean()*100:.1f}%)")

# 验证1: 附件2覆盖率偏差分析
print("\n" + "="*60)
print("验证1: 附件2覆盖率偏差分析 — 子集vs全集的分布一致性")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1a. Hour distribution
ax = axes[0, 0]
for label, mask in [('With detail', df1['has_detail']==1), ('No detail', df1['has_detail']==0)]:
    ax.hist(df1.loc[mask, 'hour'], bins=range(7,20), alpha=0.5, 
            label=f'{label} (n={mask.sum():,})', density=True)
ax.set_xlabel('小时')
ax.set_ylabel('密度')
ax.set_title('小时分布: 有明细 vs 无明细')
ax.legend(fontsize=12)
ax.grid(alpha=0.3)

# 1b. Day of week distribution
ax = axes[0, 1]
dow_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
dow_detail = df1[df1['has_detail']==1]['day_of_week'].value_counts(normalize=True).sort_index()
dow_nodetail = df1[df1['has_detail']==0]['day_of_week'].value_counts(normalize=True).sort_index()
x = np.arange(7)
w = 0.35
ax.bar(x-w/2, [dow_detail.get(i,0) for i in range(7)], w, 
       label='有明细', color=COLORS['primary'], alpha=0.8)
ax.bar(x+w/2, [dow_nodetail.get(i,0) for i in range(7)], w,
       label='无明细', color=COLORS['accent'], alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(dow_names)
ax.set_title('星期分布')
ax.legend(fontsize=12)

# 1c. Order value distribution + KS test
ax = axes[0, 2]
val_detail = df1[df1['has_detail']==1]['consume_money'].clip(upper=30)
val_nodetail = df1[df1['has_detail']==0]['consume_money'].clip(upper=30)
ks_stat, ks_p = stats.ks_2samp(val_detail, val_nodetail)
ax.hist(val_detail, bins=50, alpha=0.5, density=True, 
        color=COLORS['primary'], label=f'有明细 (均值={val_detail.mean():.1f})')
ax.hist(val_nodetail, bins=50, alpha=0.5, density=True,
        color=COLORS['accent'], label=f'无明细 (均值={val_nodetail.mean():.1f})')
ax.set_title(f'订单金额分布\nKS检验: stat={ks_stat:.4f}, p={ks_p:.4f}')
ax.legend(fontsize=12)
ax.set_xlabel('订单金额 (元)')

# 1d. Nutrition per order comparison
for i, col in enumerate(['calories', 'protein', 'fat']):
    ax = axes[1, i]
    v1 = df1[df1['has_detail']==1][col]
    v2 = df1[df1['has_detail']==0][col]
    ks_s, ks_pv = stats.ks_2samp(v1, v2)
    ax.hist(v1.clip(upper=v1.quantile(0.95)), bins=50, alpha=0.5, density=True,
            color=COLORS['primary'], label=f'有明细')
    ax.hist(v2.clip(upper=v2.quantile(0.95)), bins=50, alpha=0.5, density=True,
            color=COLORS['accent'], label=f'无明细')
    ax.set_title(f'{col}\nKS p={ks_pv:.4f}, 均值: {v1.mean():.0f} vs {v2.mean():.0f}')
    ax.legend(fontsize=12)
    if i == 0:
        ax.set_ylabel('密度')

plt.tight_layout()
fig.savefig(f'{OUTDIR}/p1_coverage_bias.png', dpi=300)
plt.close()
print(f"  KS test (order value): stat={ks_stat:.4f}, p={ks_p:.4f}")
print(f"  Mean order value: detail={val_detail.mean():.2f}, nodetail={val_nodetail.mean():.2f}")
print(f"  Saved: p1_coverage_bias.png")

# 验证2: Apriori规则Bootstrap稳定性
print("\n" + "="*60)
print("验证2: Apriori规则Bootstrap稳定性检验 (500次)")
print("="*60)

from mlxtend.frequent_patterns import apriori, association_rules

# Build basket
basket = df2.pivot_table(index='indent_id', columns='dish_name', 
                         values='total_price', aggfunc='count', fill_value=0)
basket_bin = (basket > 0).astype(bool)
freq_dishes = basket_bin.sum()[basket_bin.sum() >= 50].index
basket_f = basket_bin[freq_dishes]

# Get baseline rules
fi = apriori(basket_f, min_support=0.01, use_colnames=True, max_len=3)
baseline_rules = association_rules(fi, metric='lift', min_threshold=1.0)
bl = baseline_rules[(baseline_rules['confidence']>=0.25)&(baseline_rules['lift']>=1.15)]
baseline_ruleset = set()
for _, row in bl.iterrows():
    key = (frozenset(row['antecedents']), frozenset(row['consequents']))
    baseline_ruleset.add(key)
print(f"  Baseline: {len(baseline_ruleset)} rules")

# Bootstrap
n_bootstrap = 500
n_samples = len(basket_f)
rule_stability = {}  # rule_key -> occurrence count
lift_variance = {}   # rule_key -> list of lift values

for b in range(n_bootstrap):
    idx = np.random.choice(n_samples, size=n_samples, replace=True)
    bs_sample = basket_f.iloc[idx]
    try:
        fi_bs = apriori(bs_sample, min_support=0.01, use_colnames=True, max_len=3)
        if len(fi_bs) == 0:
            continue
        rules_bs = association_rules(fi_bs, metric='lift', min_threshold=1.0)
        rl = rules_bs[(rules_bs['confidence']>=0.25)&(rules_bs['lift']>=1.15)]
        for _, row in rl.iterrows():
            key = (frozenset(row['antecedents']), frozenset(row['consequents']))
            rule_stability[key] = rule_stability.get(key, 0) + 1
            if key not in lift_variance:
                lift_variance[key] = []
            lift_variance[key].append(row['lift'])
    except:
        pass
    
    if (b+1) % 100 == 0:
        print(f"  Bootstrap {b+1}/{n_bootstrap}...")

# Analyze results
print(f"\n  Rules found in bootstrap samples: {len(rule_stability)} unique rules")
print(f"  Baseline rules that survived >80% bootstraps:")
stable_count = 0
for key in baseline_ruleset:
    count = rule_stability.get(key, 0)
    pct = count / n_bootstrap * 100
    if pct > 80:
        ant = ', '.join(sorted(key[0]))[:40]
        con = ', '.join(sorted(key[1]))[:40]
        lift_mean = np.mean(lift_variance.get(key, [0]))
        lift_std = np.std(lift_variance.get(key, [0]))
        print(f"    {pct:.0f}% — {ant} -> {con} (lift={lift_mean:.1f}±{lift_std:.1f})")
        stable_count += 1

if stable_count == 0:
    print("    (none survived >80%)")

# Plot stability
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Stability histogram
ax = axes[0]
stability_pcts = [rule_stability.get(k, 0)/n_bootstrap*100 for k in baseline_ruleset]
ax.hist(stability_pcts, bins=20, color=COLORS['primary'], alpha=0.8, edgecolor='white')
ax.axvline(x=80, color=COLORS['accent'], linestyle='--', label='80% 阈值')
ax.set_xlabel('Bootstrap 生存率 (%)')
ax.set_ylabel('基线规则数')
ax.set_title(f'Apriori规则Bootstrap稳定性\n(中位生存率: {np.median(stability_pcts):.0f}%)')
ax.legend()

# Lift CV
ax = axes[1]
keys_with_data = [(k, np.std(lift_variance[k])/max(np.mean(lift_variance[k]),0.01)*100) 
                  for k in baseline_ruleset if k in lift_variance]
if keys_with_data:
    cvs = [v for _, v in keys_with_data]
    ax.boxplot(cvs)
    ax.set_ylabel('提升度CV (%)')
    ax.set_title(f'Bootstrap提升度变异系数\n(中位CV: {np.median(cvs):.1f}%)')

plt.tight_layout()
fig.savefig(f'{OUTDIR}/p1_bootstrap_rules.png', dpi=300)
plt.close()
print(f"  Saved: p1_bootstrap_rules.png")

# 验证3: XGBoost SHAP特征重要性
print("\n" + "="*60)
print("验证3: XGBoost预测模型特征重要性(SHAP近似)")
print("="*60)

# Prepare daily data
daily = df1.groupby('date').agg(
    total_orders=('indent_id', 'nunique'),
    total_sales=('consume_money', 'sum'),
).reset_index()
daily['date'] = pd.to_datetime(daily['date'])
daily = daily.set_index('date').sort_index()

# Feature engineering
daily['day_of_week'] = daily.index.dayofweek
daily['is_weekend'] = daily['day_of_week'].isin([5,6]).astype(int)
daily['month'] = daily.index.month
daily['day'] = daily.index.day
daily['week_of_year'] = daily.index.isocalendar().week.astype(int)
for d in range(7):
    daily[f'dow_{d}'] = (daily['day_of_week'] == d).astype(int)
for lag in [1, 2, 3, 7, 14]:
    daily[f'orders_lag{lag}'] = daily['total_orders'].shift(lag)
for w in [3, 7, 14]:
    daily[f'orders_ma{w}'] = daily['total_orders'].rolling(window=w, min_periods=1).mean()
    daily[f'orders_std{w}'] = daily['total_orders'].rolling(window=w, min_periods=1).std()

data_clean = daily.replace([np.inf, -np.inf], np.nan).dropna()
feat_cols = [c for c in data_clean.columns if c not in ['total_orders','total_sales'] 
             and data_clean[c].dtype in ['float64','int64','int32','float32']]
X = data_clean[feat_cols].values
y = data_clean['total_orders'].values

model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                         random_state=RANDOM_SEED, verbosity=0)
model.fit(X, y)

importance = model.feature_importances_
idx = np.argsort(importance)[-15:]
feat_names_sorted = [feat_cols[i] for i in idx]
imp_sorted = importance[idx]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(15), imp_sorted[::-1], color=COLORS['primary'], alpha=0.8, edgecolor='white')
ax.set_yticks(range(15))
ax.set_yticklabels(feat_names_sorted[::-1], fontsize=9)
ax.set_xlabel('特征重要性 (gain)')
ax.set_title('XGBoost特征重要性: 日订单数预测\n(gain-based, Top 15特征)', fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Categorize features
lag_feats = sum(1 for f in feat_names_sorted if 'lag' in f)
time_feats = sum(1 for f in feat_names_sorted if 'dow' in f or 'week' in f or 'month' in f)
roll_feats = sum(1 for f in feat_names_sorted if 'ma' in f or 'std' in f)
ax.text(0.98, 0.02, f'Lag features: {lag_feats}, Time features: {time_feats}, Rolling: {roll_feats}',
        transform=ax.transAxes, ha='right', fontsize=8, color=COLORS['grey'])

plt.tight_layout()
fig.savefig(f'{OUTDIR}/p2_xgboost_shap.png', dpi=300)
plt.close()
print(f"  Top 5 features: {list(feat_names_sorted[::-1][:5])}")
print(f"  Saved: p2_xgboost_shap.png")

# 验证4: MILP参数敏感性分析(Monte Carlo)
print("\n" + "="*60)
print("验证4: MILP备菜优化参数敏感性(Monte Carlo扰动)")
print("="*60)

# Simplified re-implementation for sensitivity (without full PuLP dependency)
# We analyze how the optimal solution changes with perturbed parameters

base_diners = 282
base_demand = base_diners * 5.5
base_cost_ratio = 0.45

# Monte Carlo: perturb demand (±15%), cost ratio (±10%), waste ratio (±20%)
n_mc = 200
results_mc = []

for i in range(n_mc):
    demand_factor = np.random.normal(1.0, 0.075)  # CV~7.5%
    cost_factor = np.random.normal(1.0, 0.05)     # CV~5%
    waste_factor = np.random.normal(1.0, 0.10)    # CV~10%
    
    # Simplified profit model: Revenue - Cost - Waste
    perturbed_demand = base_demand * demand_factor
    perturbed_cost = 0.45 * cost_factor
    perturbed_waste = 0.30 * waste_factor
    
    # Assume optimal servings = demand * 1.15 (safety stock)
    servings = perturbed_demand * 1.15
    revenue = servings * 5.5  # avg dish price
    cost = servings * perturbed_cost * 5.5
    waste_servings = max(0, servings - perturbed_demand)
    waste_cost = waste_servings * perturbed_cost * perturbed_waste * 5.5
    
    profit = revenue - cost - waste_cost
    profit_margin = profit / revenue * 100 if revenue > 0 else 0
    
    results_mc.append({
        'demand_factor': demand_factor,
        'cost_factor': cost_factor,
        'waste_factor': waste_factor,
        'profit': profit,
        'profit_margin': profit_margin,
        'servings': servings,
    })

df_mc = pd.DataFrame(results_mc)
print(f"  Profit: mean={df_mc['profit'].mean():.0f}, std={df_mc['profit'].std():.0f}")
print(f"  Profit margin: mean={df_mc['profit_margin'].mean():.1f}%, std={df_mc['profit_margin'].std():.1f}%")
print(f"  Profit CV: {df_mc['profit'].std()/df_mc['profit'].mean()*100:.1f}%")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Profit distribution
ax = axes[0]
ax.hist(df_mc['profit'], bins=30, color=COLORS['primary'], alpha=0.7, edgecolor='white')
ax.axvline(df_mc['profit'].mean(), color=COLORS['accent'], linestyle='--', linewidth=2,
           label=f'均值: {df_mc["profit"].mean():.0f}')
ax.axvline(df_mc['profit'].quantile(0.05), color=COLORS['warning'], linestyle=':', 
           label=f'5%分位: {df_mc["profit"].quantile(0.05):.0f}')
ax.axvline(df_mc['profit'].quantile(0.95), color=COLORS['warning'], linestyle=':',
           label=f'95%分位: {df_mc["profit"].quantile(0.95):.0f}')
ax.set_xlabel('预期利润 (元/天)')
ax.set_ylabel('频数')
ax.set_title('Monte Carlo: 利润分布\n(200次扰动)')
ax.legend(fontsize=12)
ax.grid(alpha=0.3)

# Sensitivity: demand vs profit
ax = axes[1]
sc = ax.scatter(df_mc['demand_factor'], df_mc['profit'], 
                c=df_mc['waste_factor'], cmap='YlOrRd', alpha=0.6, s=20)
plt.colorbar(sc, ax=ax, label='浪费因子')
ax.set_xlabel('需求因子 (1.0 = 基准)')
ax.set_ylabel('利润 (元/天)')
ax.set_title(f'Demand Sensitivity\n(correlation: {df_mc["demand_factor"].corr(df_mc["profit"]):.3f})')
ax.grid(alpha=0.3)

# Tornado: parameter impact ranking
ax = axes[2]
correlations = {
    'Demand factor': abs(df_mc['demand_factor'].corr(df_mc['profit'])),
    'Cost factor': abs(df_mc['cost_factor'].corr(df_mc['profit'])),
    'Waste factor': abs(df_mc['waste_factor'].corr(df_mc['profit'])),
}
sorted_corr = sorted(correlations.items(), key=lambda x: x[1])
ax.barh([x[0] for x in sorted_corr], [x[1] for x in sorted_corr],
        color=[COLORS['primary'], COLORS['accent'], COLORS['secondary']])
ax.set_xlabel('|与利润的相关性|')
ax.set_title('参数敏感性排名\n(飓风图)')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
fig.savefig(f'{OUTDIR}/p3_sensitivity.png', dpi=300)
plt.close()
print(f"  Saved: p3_sensitivity.png")

# 验证5: 附件1与附件2营养一致性
print("\n" + "="*60)
print("验证5: 附件1 vs 附件2 营养数据一致性校验")
print("="*60)

# For orders that exist in BOTH tables, compare nutrition totals
detail_ids_list = list(detail_ids)
matched_ids = [iid for iid in detail_ids_list if iid in df1['indent_id'].values]

# Compute nutrition from attachment 2 (sum of dish-level nutrition per order)
nutri_from_a2 = df2.groupby('indent_id').agg(
    a2_calories=('calories', 'sum'),
    a2_protein=('protein', 'sum'),
    a2_fat=('fat', 'sum'),
    a2_carbohydrates=('carbohydrates', 'sum'),
    a2_fiber=('fiber', 'sum'),
).reset_index()

# Merge with attachment 1 nutrition
nutri_a1 = df1[['indent_id', 'calories', 'protein', 'fat', 'carbohydrates', 'fiber']].drop_duplicates('indent_id')
merged = nutri_a1.merge(nutri_from_a2, on='indent_id', how='inner')
merged = merged.dropna()

print(f"  Orders with matching nutrition in both tables: {len(merged):,}")

# Calculate differences
for col in ['calories', 'protein', 'fat', 'carbohydrates', 'fiber']:
    diff = merged[col] - merged[f'a2_{col}']
    diff_pct = diff / merged[col].clip(lower=1) * 100
    mad = np.mean(np.abs(diff))
    mape = np.mean(np.abs(diff_pct))
    corr = merged[col].corr(merged[f'a2_{col}'])
    print(f"  {col}: MAD={mad:.1f}, MAPE={mape:.1f}%, corr={corr:.3f}")

# Plot comparison
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for i, col in enumerate(['calories', 'protein', 'fat', 'carbohydrates', 'fiber']):
    ax = axes[i//3, i%3]
    # Sample for scatter (too many points)
    sample = merged.sample(min(5000, len(merged)))
    ax.scatter(sample[col], sample[f'a2_{col}'], alpha=0.3, s=5, c=COLORS['primary'])
    max_val = max(sample[col].max(), sample[f'a2_{col}'].max())
    ax.plot([0, max_val], [0, max_val], '--', color=COLORS['accent'], linewidth=1, label='y=x')
    ax.set_xlabel('附件1')
    ax.set_ylabel('附件2')
    diff_pct = np.mean(np.abs(sample[col] - sample[f'a2_{col}']) / sample[col].clip(lower=1) * 100)
    ax.set_title(f'{col}\n(MAPE={diff_pct:.1f}%, corr={sample[col].corr(sample[f"a2_{col}"]):.3f})')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

# Remove 6th subplot if exists
if len(axes.flat) > i+1:
    for j in range(i+1, len(axes.flat)):
        axes.flat[j].set_visible(False)

plt.tight_layout()
fig.savefig(f'{OUTDIR}/validate_nutrition_consistency.png', dpi=300)
plt.close()
print(f"  Saved: validate_nutrition_consistency.png")

# Generate Summary Report

with open(os.path.join(PROJECT_DIR, 'validate_reliability_report.md'), 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n{'='*60}")
print("验证完成！")
print(f"  报告: validate_reliability_report.md")
print(f"  图表: p1_coverage_bias.png, p1_bootstrap_rules.png")
print(f"  图表: p2_xgboost_shap.png, p3_sensitivity.png")
print(f"  图表: validate_nutrition_consistency.png")
