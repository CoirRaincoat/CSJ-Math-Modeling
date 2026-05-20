"""拆分 p1_temporal_patterns.png 为 4 张独立正方形图表"""
import matplotlib.pyplot as plt, numpy as np, os, sys, warnings
from scipy import stats
warnings.filterwarnings('ignore')

sys.path.insert(0, r'C:\Users\CoirRaincoat\PyCharmMiscProject\CSJ-MathModeling')
from data_loader import load_all_data
from config import OUTPUT_DIR, COLORS

plt.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300

loader = load_all_data()
daily = loader.get_daily_data().copy()
daily['weekday_name'] = daily['date'].dt.day_name()

# ====== 图 A: 日订单数与销售额趋势 + 7日移动平均 ======
fig, ax1 = plt.subplots(figsize=(8, 8))
ax2 = ax1.twinx()
ax1.fill_between(range(len(daily)), daily['total_orders'].values,
                 alpha=0.3, color=COLORS['primary'], label='日订单数')
ax2.plot(range(len(daily)), daily['total_sales'].values,
         color=COLORS['accent'], linewidth=1.5, label='日销售额')
ma = daily['total_orders'].rolling(window=7).mean()
ax1.plot(range(len(daily)), ma.values, color=COLORS['danger'],
         linewidth=2, linestyle='--', label='7日移动平均')
ax1.set_xlabel('天数序号'); ax1.set_ylabel('日订单数', color=COLORS['primary'])
ax2.set_ylabel('日销售额 (元)', color=COLORS['accent'])
ax1.set_title('日订单数与销售额趋势', fontweight='bold', fontsize=13)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc='upper left', fontsize=9)
ax1.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/p2a_daily_trend.png', dpi=300)
plt.close()

# ====== 图 B: 星期订单分布箱线图 ======
fig, ax = plt.subplots(figsize=(8, 8))
weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
box_data = [daily[daily['weekday_name']==d]['total_orders'].values for d in weekday_order]
weekday_colors = [COLORS['weekday']]*5 + [COLORS['weekend']]*2
bp = ax.boxplot(box_data, labels=['周一','周二','周三','周四','周五','周六','周日'], patch_artist=True)
for patch, color in zip(bp['boxes'], weekday_colors): patch.set_facecolor(color); patch.set_alpha(0.7)
ax.set_ylabel('日订单数')
ax.set_title('一周各天订单分布箱线图', fontweight='bold', fontsize=13)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/p2b_weekday_boxplot.png', dpi=300)
plt.close()

# ====== 图 C: 月度销售趋势 ======
fig, ax1 = plt.subplots(figsize=(8, 8))
monthly = daily.groupby(daily['date'].dt.to_period('M')).agg(
    total_orders=('total_orders','sum'), avg_orders=('total_orders','mean')
).reset_index()
monthly['month_label'] = monthly['date'].astype(str)
ax2 = ax1.twinx()
ax1.bar(range(len(monthly)), monthly['total_orders'].values, color=COLORS['primary'], alpha=0.7)
ax2.plot(range(len(monthly)), monthly['avg_orders'].values, color=COLORS['accent'], linewidth=2, marker='o', markersize=6)
ax1.set_xticks(range(0, len(monthly), 3))
ax1.set_xticklabels(monthly['month_label'].values[::3], rotation=45, fontsize=8)
ax1.set_ylabel('月总订单数', color=COLORS['primary'])
ax2.set_ylabel('日均订单数', color=COLORS['accent'])
ax1.set_title('月度销售趋势', fontweight='bold', fontsize=13)
ax1.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/p2c_monthly_trend.png', dpi=300)
plt.close()

# ====== 图 D: 工作日 vs 周末 (Welch's t 检验) ======
fig, ax = plt.subplots(figsize=(8, 8))
wd = daily[daily['is_weekend']==0]['total_orders']
we = daily[daily['is_weekend']==1]['total_orders']
means = [wd.mean(), we.mean()]
stds = [wd.std(), we.std()]
bars = ax.bar(['工作日','周末'], means, yerr=stds, capsize=10,
             color=[COLORS['weekday'], COLORS['weekend']], edgecolor='white', linewidth=1.5, width=0.5)
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x()+bar.get_width()/2, m+s+2, f'{m:.0f}\n±{s:.0f}', ha='center', fontweight='bold', fontsize=10)
t_stat, p_val = stats.ttest_ind(wd, we, equal_var=False)
sig = '***' if p_val<0.001 else ('**' if p_val<0.01 else ('*' if p_val<0.05 else 'n.s.'))
ax.text(0.5, means[0]-15, f"Welch's t 检验: t={t_stat:.2f}, p={p_val:.4f} ({sig})",
        ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor=COLORS['beige'], alpha=0.3))
ax.set_ylabel('日均订单数')
ax.set_title('工作日 vs 周末订单对比', fontweight='bold', fontsize=13)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/p2d_weekday_vs_weekend.png', dpi=300)
plt.close()

for f in ['p2a_daily_trend.png','p2b_weekday_boxplot.png','p2c_monthly_trend.png','p2d_weekday_vs_weekend.png']:
    print(f'  {f} ({os.path.getsize(os.path.join(OUTPUT_DIR, f)):,} bytes)')
