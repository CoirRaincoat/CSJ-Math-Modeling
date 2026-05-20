"""生成 p1_sales_distribution.png 的 4 个子图为独立的高清图表"""
import matplotlib.pyplot as plt, numpy as np, os, sys, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, r'C:\Users\CoirRaincoat\PyCharmMiscProject\CSJ-MathModeling')
from data_loader import load_all_data
from config import OUTPUT_DIR, COLORS

plt.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300

loader = load_all_data()
dish_info = loader.get_dish_info().copy()
df2 = loader.df2_raw

# ====== 图 A: 销量 Top20 柱状图 ======
fig, ax = plt.subplots(figsize=(8, 8))
top20_orders = dish_info.nlargest(20, 'total_orders')
colors_top = [COLORS['primary']]*5 + [COLORS['success']]*10 + [COLORS['purple']]*5
ax.barh(range(20), top20_orders['total_orders'].values[::-1],
        color=colors_top[::-1], edgecolor='white', linewidth=0.5)
ax.set_yticks(range(20))
ax.set_yticklabels(top20_orders['dish_name'].values[::-1], fontsize=9)
ax.set_xlabel('订单数'); ax.invert_yaxis()
ax.set_title('菜品销量 Top 20', fontweight='bold', fontsize=13)
ax.grid(axis='x', alpha=0.3)
for i, (v, n) in enumerate(zip(top20_orders['total_orders'].values[::-1], top20_orders['dish_name'].values[::-1])):
    ax.text(v+10, i, str(v), va='center', fontsize=7)
plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/p1a_top20_orders.png', dpi=300)
plt.close()

# ====== 图 B: 销售额 Top20 柱状图 ======
fig, ax = plt.subplots(figsize=(8, 8))
top20_rev = dish_info.nlargest(20, 'total_revenue')
ax.barh(range(20), top20_rev['total_revenue'].values[::-1],
        color=colors_top[::-1], edgecolor='white', linewidth=0.5)
ax.set_yticks(range(20))
ax.set_yticklabels(top20_rev['dish_name'].values[::-1], fontsize=9)
ax.set_xlabel('销售额 (元)'); ax.invert_yaxis()
ax.set_title('菜品销售额 Top 20', fontweight='bold', fontsize=13)
ax.grid(axis='x', alpha=0.3)
for i, (v, n) in enumerate(zip(top20_rev['total_revenue'].values[::-1], top20_rev['dish_name'].values[::-1])):
    ax.text(v+10, i, f'{v:.0f}', va='center', fontsize=7)
plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/p1b_top20_revenue.png', dpi=300)
plt.close()

# ====== 图 C: ABC 分类 (Pareto 图) ======
fig, ax = plt.subplots(figsize=(8, 8))
dish_sorted = dish_info.sort_values('total_orders', ascending=False)
dish_sorted['cumsum_pct'] = dish_sorted['total_orders'].cumsum() / dish_sorted['total_orders'].sum() * 100
dish_sorted['pct'] = dish_sorted['total_orders'] / dish_sorted['total_orders'].sum() * 100

x = range(len(dish_sorted))
ax.bar(x, dish_sorted['pct'].values, color=COLORS['primary'], alpha=0.7, width=1)
ax2 = ax.twinx()
ax2.plot(x, dish_sorted['cumsum_pct'].values, color=COLORS['accent'], linewidth=2)
ax2.axhline(y=80, color=COLORS['warning'], linestyle='--', alpha=0.7, linewidth=1.2, label='80% 阈值')
ax2.axhline(y=95, color=COLORS['danger'], linestyle='--', alpha=0.5, linewidth=1.2, label='95% 阈值')

a_count = (dish_sorted['cumsum_pct'] <= 80).sum()
b_count = (dish_sorted['cumsum_pct'] <= 95).sum() - a_count
c_count = len(dish_sorted) - a_count - b_count

ax.axvspan(0, a_count, alpha=0.05, color=COLORS['primary'])
ax.axvspan(a_count, a_count+b_count, alpha=0.05, color=COLORS['warning'])
ax.axvspan(a_count+b_count, len(dish_sorted), alpha=0.05, color=COLORS['grey'])
ax.text(a_count/2, max(dish_sorted['pct'])*0.8, f'A 类\n{a_count} 道菜\n80% 销量', ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor=COLORS['teal'], alpha=0.15))
ax.text(a_count+b_count/2, max(dish_sorted['pct'])*0.5, f'B 类\n{b_count} 道菜', ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor=COLORS['purple'], alpha=0.15))
ax.text(a_count+b_count+c_count/2, max(dish_sorted['pct'])*0.35, f'C 类\n{c_count} 道菜\n5% 销量', ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor=COLORS['grey'], alpha=0.15))

ax.set_xlabel('菜品排名 (按订单数)')
ax.set_ylabel('占总订单百分比 (%)', color=COLORS['primary'])
ax2.set_ylabel('累计百分比 (%)', color=COLORS['accent'])
ax.set_title('菜品 ABC 分类 (帕累托图)', fontweight='bold', fontsize=13)
ax2.legend(loc='lower right', fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/p1c_abc_pareto.png', dpi=300)
plt.close()

# ====== 图 D: 菜品类别占比饼图 ======
fig, ax = plt.subplots(figsize=(8, 8))
cat_order = df2.groupby('category')['indent_details_id'].count().sort_values(ascending=False)
colors_cat = [COLORS['primary'], COLORS['success'], COLORS['accent'], COLORS['warning'], COLORS['purple']]
wedges, texts, autotexts = ax.pie(
    cat_order.values, labels=cat_order.index, autopct='%1.1f%%',
    colors=colors_cat[:len(cat_order)], startangle=90,
    explode=[0.05]*len(cat_order), textprops={'fontsize': 11}
)
for at in autotexts: at.set_fontsize(10)
ax.set_title('菜品类别分布 (按订单条目)', fontweight='bold', fontsize=13)
plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/p1d_category_pie.png', dpi=300)
plt.close()

print('4 individual charts saved:')
for f in ['p1a_top20_orders.png','p1b_top20_revenue.png','p1c_abc_pareto.png','p1d_category_pie.png']:
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f'  {f} ({size:,} bytes)')
