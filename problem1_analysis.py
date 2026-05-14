"""
problem1_analysis.py — 问题1：数据预处理、统计分析与关联规则挖掘
============================================================
题目要求：
  对附件中自助量贩餐厅的历史交易数据进行预处理、统计和可视化分析，
  分析不同菜品销售量的分布规律以及它们之间可能存在的关联关系。

解题思路：
  1. 数据预处理：缺失值处理、异常值检测、餐次划分、日期特征提取
  2. 描述性统计分析：
     - 菜品销量排名（Pareto分析）
     - 菜品类别销售占比
     - 日销售额/订单量时间序列
     - 星期消费规律
     - 午餐/晚餐对比
     - 人均营养摄入分析
  3. 关联规则挖掘（Apriori算法）：
     - 构建购物篮数据
     - 计算支持度、置信度、提升度
     - 提取有意义的关联规则
     - 菜品共现网络可视化

参考文献：
  [5] Agrawal R, Srikant R. "Fast Algorithms for Mining Association Rules"
      Proceedings of the 20th VLDB Conference, 1994.
      https://www.vldb.org/conf/1994/P487.PDF
  [6] 余滔滔等. "基于Apriori算法的菜品配置规则研究"
      服务科学和管理, 2019.
      https://www.hanspub.org/journal/PaperInformation?paperID=32795

核心模型：Apriori关联规则挖掘
  - 支持度 Support(A→B) = P(A∪B) = count(A∪B) / N
  - 置信度 Confidence(A→B) = P(B|A) = count(A∪B) / count(A)
  - 提升度 Lift(A→B) = P(B|A) / P(B) = Confidence(A→B) / Support(B)
    提升度 > 1 表示正相关，< 1 表示负相关，= 1 表示独立
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from collections import Counter
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

from data_loader import load_all_data
from config import (OUTPUT_DIR, COLORS, RANDOM_SEED, 
                     LUNCH_START, LUNCH_END, DINNER_START, DINNER_END)

np.random.seed(RANDOM_SEED)


class Problem1Analysis:
    """
    问题1完整分析类
    
    封装了从数据预处理到关联规则挖掘的全流程，
    提供丰富的可视化输出和统计表格。
    """
    
    def __init__(self, loader=None):
        """初始化，加载数据"""
        print('\n' + '=' * 60)
        print('问题1：数据预处理、统计与关联分析')
        print('=' * 60)
        
        if loader is None:
            self.loader = load_all_data()
        else:
            self.loader = loader
        
        self.df_daily = self.loader.get_daily_data()
        self.df_meal = self.loader.get_meal_data()
        self.df_trans = self.loader.get_transaction_data()
        self.dish_info = self.loader.get_dish_info()
        self.basket = self.loader.get_basket_data()
        
        # 存储分析结果
        self.results = {}
        
    def run(self):
        """运行完整的问题1分析流程"""
        print('\n>>> 1.1 数据预处理摘要')
        self._data_preprocessing_summary()
        
        print('\n>>> 1.2 菜品销售量分布分析')
        self._sales_distribution_analysis()
        
        print('\n>>> 1.3 时间维度销售规律分析')
        self._temporal_pattern_analysis()
        
        print('\n>>> 1.4 餐次差异分析')
        self._meal_period_analysis()
        
        print('\n>>> 1.5 营养摄入分析')
        self._nutrition_analysis()
        
        print('\n>>> 1.6 关联规则挖掘')
        self._association_rule_mining()
        
        print('\n问题1分析完成！可视化结果已保存至 output/ 目录')
        
        return self.results
    
    def _data_preprocessing_summary(self):
        """数据预处理摘要"""
        df1 = self.loader.df1_raw
        df2 = self.loader.df2_raw
        
        summary = {
            '总交易记录数': len(df1),
            '总订单数': df1['indent_id'].nunique(),
            '日期范围': f'{df1["date"].min()} 至 {df1["date"].max()}',
            '营业天数': df1['date'].nunique(),
            '日均订单数': f'{df1.groupby("date")["indent_id"].nunique().mean():.0f}',
            '日均销售额': f'{df1.groupby("date")["consume_money"].sum().mean():.0f} 元',
            '人均消费': f'{df1["consume_money"].mean():.2f} 元',
            '午餐占比': f'{len(df1[df1["meal_period"]=="lunch"])/len(df1)*100:.1f}%',
            '晚餐占比': f'{len(df1[df1["meal_period"]=="dinner"])/len(df1)*100:.1f}%',
            '唯一菜品数': df2['dish_name'].nunique(),
            '菜品总类别': df1['calories'].count(),
        }
        
        self.results['preprocessing_summary'] = summary
        for k, v in summary.items():
            print(f'  {k}: {v}')
            
    def _sales_distribution_analysis(self):
        """
        菜品销售量分布分析
        
        分析方法：
        1. Pareto分析（ABC分类）：识别头部菜品
        2. 长尾分布检验
        3. 菜品类别销售占比
        4. 单位价格分布
        """
        dish_info = self.dish_info.copy()
        
        # ====== 图1：菜品销量Top20柱状图 ======
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 子图1: 销量Top20柱状图
        ax1 = axes[0, 0]
        top20_orders = dish_info.nlargest(20, 'total_orders')
        colors_top = [COLORS['primary'] if i < 5 else 
                       COLORS['secondary'] if i < 10 else 
                       COLORS['accent'] for i in range(20)]
        bars = ax1.barh(range(20), top20_orders['total_orders'].values[::-1], 
                        color=colors_top[::-1], edgecolor='white', linewidth=0.5)
        ax1.set_yticks(range(20))
        ax1.set_yticklabels(top20_orders['dish_name'].values[::-1], fontsize=8)
        ax1.set_xlabel('Orders Count')
        ax1.set_title('Top 20 Dishes by Order Frequency', fontweight='bold')
        ax1.invert_yaxis()
        
        # 子图2: 销售额Top20柱状图
        ax2 = axes[0, 1]
        top20_rev = dish_info.nlargest(20, 'total_revenue')
        bars2 = ax2.barh(range(20), top20_rev['total_revenue'].values[::-1],
                        color=colors_top[::-1], edgecolor='white', linewidth=0.5)
        ax2.set_yticks(range(20))
        ax2.set_yticklabels(top20_rev['dish_name'].values[::-1], fontsize=8)
        ax2.set_xlabel('Total Revenue (Yuan)')
        ax2.set_title('Top 20 Dishes by Revenue', fontweight='bold')
        ax2.invert_yaxis()
        
        # 子图3: ABC分类（Pareto图）
        ax3 = axes[1, 0]
        dish_sorted = dish_info.sort_values('total_orders', ascending=False)
        dish_sorted['cumsum_pct'] = dish_sorted['total_orders'].cumsum() / dish_sorted['total_orders'].sum() * 100
        dish_sorted['pct'] = dish_sorted['total_orders'] / dish_sorted['total_orders'].sum() * 100
        
        x = range(len(dish_sorted))
        ax3.bar(x, dish_sorted['pct'].values, color=COLORS['primary'], alpha=0.7, width=1)
        ax3_2 = ax3.twinx()
        ax3_2.plot(x, dish_sorted['cumsum_pct'].values, color=COLORS['danger'], linewidth=2)
        ax3_2.axhline(y=80, color=COLORS['warning'], linestyle='--', alpha=0.7, label='80% threshold')
        ax3.set_xlabel('Dish Rank')
        ax3.set_ylabel('Percentage of Total Orders (%)', color=COLORS['primary'])
        ax3_2.set_ylabel('Cumulative Percentage (%)', color=COLORS['danger'])
        ax3.set_title('ABC Analysis (Pareto Chart) of Dish Sales', fontweight='bold')
        ax3_2.legend(loc='lower right')
        
        # 标记ABC区间
        a_count = (dish_sorted['cumsum_pct'] <= 80).sum()
        b_count = (dish_sorted['cumsum_pct'] <= 95).sum() - a_count
        ax3.text(a_count/2, 5, f'Class A\n{a_count} dishes\n80% sales', 
                ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax3.text(a_count + b_count/2, 3, f'Class B\n{b_count} dishes', 
                ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 子图4: 菜品类别占比饼图
        ax4 = axes[1, 1]
        df2 = self.loader.df2_raw
        cat_order = df2.groupby('category')['indent_details_id'].count().sort_values(ascending=False)
        colors_cat = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], 
                      COLORS['success'], COLORS['warning']]
        wedges, texts, autotexts = ax4.pie(
            cat_order.values, labels=cat_order.index, autopct='%1.1f%%',
            colors=colors_cat[:len(cat_order)], startangle=90,
            explode=[0.05] * len(cat_order), textprops={'fontsize': 10}
        )
        ax4.set_title('Dish Category Distribution', fontweight='bold')
        
        plt.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/p1_sales_distribution.png', dpi=150)
        plt.close()
        print('  已保存: p1_sales_distribution.png')
        
        # 保存Top数据
        self.results['top20_orders'] = top20_orders[['dish_name', 'total_orders']].head(10)
        self.results['abc_analysis'] = {
            'a_count': a_count, 'b_count': b_count, 
            'c_count': len(dish_sorted) - a_count - b_count
        }
        
    def _temporal_pattern_analysis(self):
        """时间维度销售规律分析"""
        daily = self.df_daily.copy()
        
        # ====== 图2：日销售趋势 + 星期模式 ======
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 子图1: 日订单数与销售额趋势
        ax1 = axes[0, 0]
        ax1_2 = ax1.twinx()
        ax1.fill_between(range(len(daily)), daily['total_orders'].values, 
                         alpha=0.3, color=COLORS['primary'], label='Orders')
        ax1_2.plot(range(len(daily)), daily['total_sales'].values, 
                   color=COLORS['danger'], linewidth=1.5, label='Sales')
        ax1.set_xlabel('Day Index')
        ax1.set_ylabel('Daily Orders', color=COLORS['primary'])
        ax1_2.set_ylabel('Daily Sales (Yuan)', color=COLORS['danger'])
        ax1.set_title('Daily Orders and Sales Trend', fontweight='bold')
        
        # 添加移动平均线
        window = 7
        ma_orders = daily['total_orders'].rolling(window=window).mean()
        ax1.plot(range(len(daily)), ma_orders.values, color=COLORS['secondary'], 
                linewidth=2, linestyle='--', label=f'{window}-day MA')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 子图2: 星期销售模式（箱线图）
        ax2 = axes[0, 1]
        daily['weekday_name'] = daily['date'].dt.day_name()
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        box_data = [daily[daily['weekday_name'] == d]['total_orders'].values for d in weekday_order]
        bp = ax2.boxplot(box_data, labels=[d[:3] for d in weekday_order], patch_artist=True)
        for patch, color in zip(bp['boxes'], [COLORS['primary']]*5 + [COLORS['warning']]*2):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.set_ylabel('Daily Orders')
        ax2.set_title('Orders Distribution by Day of Week', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # 子图3: 月度销售趋势
        ax3 = axes[1, 0]
        monthly = daily.groupby(daily['date'].dt.to_period('M')).agg(
            total_orders=('total_orders', 'sum'),
            total_sales=('total_sales', 'sum'),
            avg_orders=('total_orders', 'mean')
        ).reset_index()
        monthly['month_label'] = monthly['date'].astype(str)
        
        ax3_2 = ax3.twinx()
        ax3.bar(range(len(monthly)), monthly['total_orders'].values, 
               color=COLORS['primary'], alpha=0.7)
        ax3_2.plot(range(len(monthly)), monthly['avg_orders'].values, 
                  color=COLORS['danger'], linewidth=2, marker='o', markersize=6)
        ax3.set_xticks(range(len(monthly)))
        ax3.set_xticklabels(monthly['month_label'].values, rotation=45, fontsize=8)
        ax3.set_ylabel('Total Monthly Orders', color=COLORS['primary'])
        ax3_2.set_ylabel('Avg Daily Orders', color=COLORS['danger'])
        ax3.set_title('Monthly Sales Trend', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # 子图4: 工作日vs周末对比
        ax4 = axes[1, 1]
        weekday_data = daily[daily['is_weekend'] == 0]['total_orders']
        weekend_data = daily[daily['is_weekend'] == 1]['total_orders']
        
        categories = ['Weekdays', 'Weekends']
        means = [weekday_data.mean(), weekend_data.mean()]
        stds = [weekday_data.std(), weekend_data.std()]
        
        x_pos = np.arange(len(categories))
        bars = ax4.bar(x_pos, means, yerr=stds, capsize=10, 
                       color=[COLORS['primary'], COLORS['warning']], 
                       edgecolor='white', linewidth=1.5, width=0.5)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(categories, fontsize=11)
        ax4.set_ylabel('Average Daily Orders')
        ax4.set_title('Weekday vs Weekend Orders', fontweight='bold')
        
        # 显示数值
        for bar, mean in zip(bars, means):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{mean:.0f}', ha='center', fontweight='bold', fontsize=11)
        
        # 进行t检验判断差异显著性
        from scipy import stats
        t_stat, p_val = stats.ttest_ind(weekday_data, weekend_data)
        ax4.text(0.5, ax4.get_ylim()[1]*0.9, 
                f't-test p-value: {p_val:.4f}', ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/p1_temporal_patterns.png', dpi=150)
        plt.close()
        print('  已保存: p1_temporal_patterns.png')
        
        # 保存统计
        self.results['weekday_vs_weekend'] = {
            'weekday_mean': f'{weekday_data.mean():.0f}',
            'weekend_mean': f'{weekend_data.mean():.0f}',
            'difference_pct': f'{(weekend_data.mean()/weekday_data.mean()-1)*100:.1f}%',
            't_stat': f'{t_stat:.3f}',
            'p_value': f'{p_val:.4f}'
        }
        
    def _meal_period_analysis(self):
        """午餐vs晚餐差异分析"""
        df1 = self.loader.df1_raw
        lunch = df1[df1['meal_period'] == 'lunch']
        dinner = df1[df1['meal_period'] == 'dinner']
        
        # ====== 图3：午餐/晚餐对比 ======
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # 子图1: 消费金额分布对比
        ax1 = axes[0]
        ax1.hist(lunch['consume_money'].clip(upper=30), bins=40, alpha=0.6, 
                color=COLORS['lunch'], label=f'Lunch (n={len(lunch):,})', density=True)
        ax1.hist(dinner['consume_money'].clip(upper=30), bins=40, alpha=0.6,
                color=COLORS['dinner'], label=f'Dinner (n={len(dinner):,})', density=True)
        ax1.set_xlabel('Consumption Amount (Yuan)')
        ax1.set_ylabel('Density')
        ax1.set_title('Consumption Distribution: Lunch vs Dinner', fontweight='bold')
        ax1.legend()
        ax1.axvline(lunch['consume_money'].median(), color=COLORS['lunch'], 
                    linestyle='--', alpha=0.8, label=f'Lunch Median: {lunch["consume_money"].median():.1f}')
        ax1.axvline(dinner['consume_money'].median(), color=COLORS['dinner'], 
                    linestyle='--', alpha=0.8, label=f'Dinner Median: {dinner["consume_money"].median():.1f}')
        
        # 子图2: 小时消费分布
        ax2 = axes[1]
        hourly = df1.groupby('hour').size()
        hours = range(7, 19)
        counts = [hourly.get(h, 0) for h in hours]
        colors_bar = []
        for h in hours:
            if LUNCH_START <= h < LUNCH_END:
                colors_bar.append(COLORS['lunch'])
            elif DINNER_START <= h < DINNER_END:
                colors_bar.append(COLORS['dinner'])
            else:
                colors_bar.append('#CCCCCC')
        ax2.bar(hours, counts, color=colors_bar, edgecolor='white', linewidth=0.5)
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Transaction Count')
        ax2.set_title('Hourly Transaction Distribution', fontweight='bold')
        ax2.axvspan(LUNCH_START, LUNCH_END, alpha=0.15, color=COLORS['lunch'], label='Lunch Period')
        ax2.axvspan(DINNER_START, DINNER_END, alpha=0.15, color=COLORS['dinner'], label='Dinner Period')
        ax2.legend(fontsize=8)
        ax2.set_xticks(hours)
        
        # 子图3: 人均营养摄入对比
        ax3 = axes[2]
        metrics = ['calories', 'protein', 'fat', 'carbohydrates', 'fiber']
        lunch_means = [lunch[m].mean() for m in metrics]
        dinner_means = [dinner[m].mean() for m in metrics]
        metric_labels = ['Calories\n(kcal)', 'Protein\n(g)', 'Fat\n(g)', 'Carbs\n(g)', 'Fiber\n(g)']
        
        x = np.arange(len(metrics))
        width = 0.35
        bars1 = ax3.bar(x - width/2, lunch_means, width, label='Lunch', 
                        color=COLORS['lunch'], alpha=0.8)
        bars2 = ax3.bar(x + width/2, dinner_means, width, label='Dinner',
                        color=COLORS['dinner'], alpha=0.8)
        ax3.set_xticks(x)
        ax3.set_xticklabels(metric_labels, fontsize=9)
        ax3.set_ylabel('Average per Order')
        ax3.set_title('Nutrition Comparison: Lunch vs Dinner', fontweight='bold')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/p1_meal_comparison.png', dpi=150)
        plt.close()
        print('  已保存: p1_meal_comparison.png')
        
    def _nutrition_analysis(self):
        """营养摄入分析"""
        daily = self.df_daily.copy()
        
        # ====== 图4：营养分析 ======
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 子图1: 每日营养素趋势
        ax1 = axes[0, 0]
        nutri_cols = ['total_calories', 'total_protein', 'total_fat', 'total_carbohydrates']
        nutri_labels = ['Calories (kcal)', 'Protein (g)', 'Fat (g)', 'Carbs (g)']
        nutri_colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['success']]
        
        # 归一化处理以在同一图中显示不同量纲
        for col, label, color in zip(nutri_cols, nutri_labels, nutri_colors):
            normalized = (daily[col] - daily[col].mean()) / daily[col].std()
            ax1.plot(range(len(daily)), normalized.values, color=color, 
                    linewidth=1, alpha=0.7, label=label)
        ax1.set_xlabel('Day Index')
        ax1.set_ylabel('Normalized Value (Z-score)')
        ax1.set_title('Daily Nutritional Intake Trends (Normalized)', fontweight='bold')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.grid(axis='y', alpha=0.3)
        
        # 子图2: 营养摄入结构饼图
        ax2 = axes[0, 1]
        # 食物热量来源估算：蛋白4kcal/g, 碳水4kcal/g, 脂肪9kcal/g
        avg_protein_cal = daily['total_protein'].mean() * 4
        avg_carb_cal = daily['total_carbohydrates'].mean() * 4
        avg_fat_cal = daily['total_fat'].mean() * 9
        total_cal_avg = avg_protein_cal + avg_carb_cal + avg_fat_cal
        
        sizes = [avg_protein_cal/total_cal_avg*100, avg_fat_cal/total_cal_avg*100, 
                 avg_carb_cal/total_cal_avg*100]
        labels_nutri = ['Protein', 'Fat', 'Carbohydrates']
        colors_nutri = [COLORS['secondary'], COLORS['accent'], COLORS['success']]
        explode = (0.03, 0.03, 0.03)
        
        wedges, texts, autotexts = ax2.pie(
            sizes, explode=explode, labels=labels_nutri, colors=colors_nutri,
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10}
        )
        ax2.set_title('Average Calorie Source Distribution', fontweight='bold')
        
        # 子图3: 人均营养素相关矩阵
        ax3 = axes[1, 0]
        person_nutri = daily[[
            'avg_calories_per_person', 'avg_protein_per_person',
            'avg_fat_per_person', 'avg_carbohydrates_per_person',
            'avg_fiber_per_person', 'avg_order_value'
        ]].rename(columns={
            'avg_calories_per_person': 'Calories',
            'avg_protein_per_person': 'Protein', 
            'avg_fat_per_person': 'Fat',
            'avg_carbohydrates_per_person': 'Carbs',
            'avg_fiber_per_person': 'Fiber',
            'avg_order_value': 'Order Value'
        })
        corr = person_nutri.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, square=True, linewidths=0.5, ax=ax3,
                   vmin=-1, vmax=1)
        ax3.set_title('Nutrition & Spending Correlation Matrix', fontweight='bold')
        
        # 子图4: 客单价分布直方图
        ax4 = axes[1, 1]
        order_values = self.loader.df1_raw['consume_money'].clip(upper=30)
        ax4.hist(order_values, bins=60, color=COLORS['primary'], alpha=0.7, 
                edgecolor='white', linewidth=0.3)
        ax4.axvline(order_values.mean(), color=COLORS['danger'], linestyle='--', 
                   linewidth=2, label=f'Mean: {order_values.mean():.1f} yuan')
        ax4.axvline(order_values.median(), color=COLORS['accent'], linestyle='--',
                   linewidth=2, label=f'Median: {order_values.median():.1f} yuan')
        ax4.set_xlabel('Order Value (Yuan)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Order Value Distribution', fontweight='bold')
        ax4.legend()
        
        plt.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/p1_nutrition_analysis.png', dpi=150)
        plt.close()
        print('  已保存: p1_nutrition_analysis.png')
        
    def _association_rule_mining(self):
        """
        Apriori关联规则挖掘
        
        步骤：
        1. 数据准备：二值化购物篮数据（0/1表示菜品是否被购买）
        2. 频繁项集挖掘：使用Apriori算法
        3. 关联规则生成：过滤满足最小支持度和置信度的规则
        4. 规则分析与可视化
        """
        print('\n  运行Apriori关联规则挖掘...')
        
        basket = self.basket  # 212种高频菜品，11828个订单
        
        # ====== 步骤1：频繁项集挖掘 ======
        # 由于数据中菜品有237种，单个菜品最高支持度约0.96（米饭）
        # 其余菜品支持度较低，需设置较小的min_support才能捕获共现关系
        # 采用多级阈值策略：先用较高阈值，逐步降低
        min_support_levels = [0.01, 0.005, 0.003]
        rules_filtered = pd.DataFrame()
        
        for min_support in min_support_levels:
            frequent_itemsets = apriori(basket, min_support=min_support, 
                                         use_colnames=True, max_len=3)
            n_itemsets = len(frequent_itemsets)
            n_size1 = len(frequent_itemsets[frequent_itemsets['itemsets'].apply(len) == 1])
            n_size2 = len(frequent_itemsets[frequent_itemsets['itemsets'].apply(len) >= 2])
            
            print(f'  min_support={min_support}: {n_itemsets} itemsets '
                  f'(size-1: {n_size1}, size>=2: {n_size2})')
            
            if n_size2 < 5:
                continue  # 需要更多项对才能生成有意义的规则
                
            rules = association_rules(frequent_itemsets, metric='lift', 
                                       min_threshold=1.0)
            rules_filtered = rules[
                (rules['confidence'] >= 0.25) & 
                (rules['lift'] >= 1.15)
            ].sort_values('lift', ascending=False)
            
            if len(rules_filtered) >= 5:
                print(f'  找到 {len(rules_filtered)} 条关联规则 (min_support={min_support})')
                break
        
        if len(rules_filtered) == 0:
            # 最终备选：使用最低阈值
            min_support = 0.002
            frequent_itemsets = apriori(basket, min_support=min_support, 
                                         use_colnames=True, max_len=2)
            rules = association_rules(frequent_itemsets, metric='lift', 
                                       min_threshold=1.0)
            rules_filtered = rules[rules['lift'] >= 1.1].sort_values('lift', ascending=False)
            print(f'  备选方案：找到 {len(rules_filtered)} 条规则 (min_support={min_support})')
        
        # ====== 步骤2：关联规则可视化 ======
        if len(rules_filtered) > 0:
            self._plot_association_rules(rules_filtered, frequent_itemsets)
        else:
            print('  警告：无法生成足够的关联规则，跳过可视化')
        
        # 保存规则
        top_rules = rules_filtered.head(20)[['antecedents', 'consequents', 
                                               'support', 'confidence', 'lift']]
        top_rules['antecedents'] = top_rules['antecedents'].apply(
            lambda x: ', '.join(list(x)[:3]))
        top_rules['consequents'] = top_rules['consequents'].apply(
            lambda x: ', '.join(list(x)))
        
        self.results['association_rules'] = top_rules
        self.results['num_rules'] = len(rules_filtered)
        
    def _plot_association_rules(self, rules, frequent_itemsets):
        """绘制关联规则可视化图表"""
        rules_top = rules.head(30)
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # 子图1: 支持度-置信度-提升度散点图
        ax1 = axes[0]
        scatter = ax1.scatter(
            rules_top['support'], rules_top['confidence'],
            c=rules_top['lift'], cmap='YlOrRd', 
            s=rules_top['lift'] * 80, alpha=0.7, edgecolors='grey', linewidth=0.5
        )
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Lift')
        ax1.set_xlabel('Support')
        ax1.set_ylabel('Confidence')
        ax1.set_title('Association Rules: Support vs Confidence\n(point size & color = Lift)', 
                     fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # 标注top规则
        for i, (idx, row) in enumerate(rules_top.head(5).iterrows()):
            ant = ', '.join(list(row['antecedents'])[:2])
            con = ', '.join(list(row['consequents'])[:1])
            ax1.annotate(f'{ant}->{con}', 
                        (row['support'], row['confidence']),
                        fontsize=6, alpha=0.8,
                        arrowprops=dict(arrowstyle='->', alpha=0.5))
        
        # 子图2: 菜品共现网络图
        ax2 = axes[1]
        self._plot_dish_network(ax2, rules_top, frequent_itemsets)
        
        plt.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/p1_association_rules.png', dpi=150)
        plt.close()
        print('  已保存: p1_association_rules.png')
        
    def _plot_dish_network(self, ax, rules, frequent_itemsets):
        """
        绘制菜品共现网络图
        
        使用NetworkX构建图：
        - 节点：菜品
        - 边：关联规则（提升度>1.2），边宽表示提升度大小
        - 节点大小：菜品出现频率
        - 节点颜色：菜品类别
        """
        # 获取菜品频率
        dish_freq = frequent_itemsets[frequent_itemsets['itemsets'].apply(len) == 1].copy()
        dish_freq['dish'] = dish_freq['itemsets'].apply(lambda x: list(x)[0])
        freq_dict = dict(zip(dish_freq['dish'], dish_freq['support']))
        
        # 构建图
        G = nx.Graph()
        
        # 添加节点
        top_dishes = set()
        for _, row in rules.iterrows():
            for d in row['antecedents']:
                top_dishes.add(d)
            for d in row['consequents']:
                top_dishes.add(d)
        
        for dish in top_dishes:
            if dish in freq_dict:
                G.add_node(dish, weight=freq_dict[dish] * 100)
        
        # 添加边
        for _, row in rules.iterrows():
            for a in row['antecedents']:
                for c in row['consequents']:
                    if a in G and c in G and a != c:
                        G.add_edge(a, c, weight=row['lift'])
        
        # 布局
        if len(G) > 0:
            pos = nx.spring_layout(G, k=1.5, iterations=50, seed=RANDOM_SEED)
            
            # 节点大小
            node_sizes = [G.nodes[n].get('weight', 1) * 200 for n in G.nodes()]
            node_sizes = [max(50, min(800, s)) for s in node_sizes]
            
            # 边宽
            edge_widths = [G.edges[e].get('weight', 1) * 1.5 for e in G.edges()]
            
            # 绘制
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                                   node_color=COLORS['primary'], alpha=0.8)
            nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                                   edge_color=COLORS['accent'], alpha=0.4)
            
            # 标签
            labels = {n: n[:6] for n in G.nodes()}  # 截断过长名称
            nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=6)
        
        ax.set_title('Dish Co-occurrence Network\n(edges = association rules with lift>1.2)', 
                    fontweight='bold')
        ax.axis('off')
    
    def get_results_table(self):
        """获取结果摘要表格以供报告使用"""
        return self.results


if __name__ == '__main__':
    # 测试运行
    analysis = Problem1Analysis()
    results = analysis.run()
    print('\n=== 问题1结果摘要 ===')
    print(f'关联规则数量: {results.get("num_rules", 0)}')
    if 'association_rules' in results:
        print('\nTop 10 关联规则:')
        print(results['association_rules'].head(10).to_string())
