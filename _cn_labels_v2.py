"""全面批量替换所有英文图表标签为中文"""
import os, re

PROJECT = r'C:\Users\CoirRaincoat\PyCharmMiscProject\CSJ-MathModeling'
FILES = ['problem1_analysis.py','problem2_prediction.py','problem3_optimization.py',
         'problem4_combos.py','problem5_strategy.py','validate_reliability.py',
         'split_p1_charts.py','split_p2_charts.py']

REPLACEMENTS = {
    # ===== problem1 =====
    "label='Orders'": "label='日订单数'",
    "label='Sales'": "label='日销售额'",
    "label='80% threshold'": "label='80% 阈值'",
    "label='95% threshold'": "label='95% 阈值'",
    "label='Lunch'": "label='午餐'",
    "label='Dinner'": "label='晚餐'",
    "label='Lunch Period'": "label='午餐时段'",
    "label='Dinner Period'": "label='晚餐时段'",
    f"label=f'{{window}}-day MA'": f"label=f'{{window}}日移动平均'",
    "label=f'Lunch (n=": "label=f'午餐 (n=",
    "label=f'Dinner (n=": "label=f'晚餐 (n=",
    
    "'Top 20 Dishes by Order Frequency'": "'菜品销量 Top 20'",
    "'Top 20 Dishes by Revenue'": "'菜品销售额 Top 20'",
    "'ABC Analysis (Pareto Chart) of Dish Sales'": "'菜品 ABC 分类 (帕累托图)'",
    "'Dish Category Distribution (by order items)'": "'菜品类别分布 (按订单条目)'",
    
    "'Daily Orders and Sales Trend'": "'日订单数与销售额趋势'",
    "'Orders Distribution by Day of Week'": "'一周各天订单分布箱线图'",
    "'Monthly Sales Trend'": "'月度销售趋势'",
    "'Weekday vs Weekend Orders'": "'工作日 vs 周末订单对比'",
    
    "'Consumption Distribution:\\nLunch vs Dinner'": "'消费金额分布:\\n午餐 vs 晚餐'",
    "'Hourly Transaction Distribution'": "'小时交易分布'",
    "'Nutrition Comparison:\\nLunch vs Dinner'": "'营养对比:\\n午餐 vs 晚餐'",
    
    "'Daily Nutritional Intake Trends (Z-score)'": "'日营养素摄入趋势 (Z-score)'",
    "'Average Calorie Source Distribution\\n'": "'日均热量来源构成\\n'",
    "'Nutrition & Spending Correlation Matrix'": "'营养与消费相关性矩阵'",
    "'Order Value Distribution'": "'订单金额分布'",
    
    "'Total Order Count'": "'订单数'",
    "'Total Revenue (Yuan)'": "'销售额 (元)'",
    "'Dish Rank (by order count)'": "'菜品排名 (按订单数)'",
    "'Percentage of Total Orders (%)'": "'占总订单百分比 (%)'",
    "'Cumulative Percentage (%)'": "'累计百分比 (%)'",
    "'Day Index'": "'天数序号'",
    "'Daily Orders'": "'日订单数'",
    "'Daily Sales (Yuan)'": "'日销售额 (元)'",
    "'Total Monthly Orders'": "'月总订单数'",
    "'Avg Daily Orders'": "'日均订单数'",
    "'Average Daily Orders'": "'日均订单数'",
    "'Consumption Amount (Yuan)'": "'消费金额 (元)'",
    "'Density'": "'密度'",
    "'Hour of Day'": "'小时'",
    "'Transaction Count'": "'交易数'",
    "'Average per Order'": "'每单均值'",
    "'Normalized Value (Z-score)'": "'归一化值 (Z-score)'",
    "'Frequency'": "'频数'",
    "'Order Value (Yuan)'": "'订单金额 (元)'",
    
    "label=f'Mean: {order_values.mean():.1f} yuan'": "label=f'均值: {order_values.mean():.1f} 元'",
    "label=f'Median: {order_values.median():.1f} yuan'": "label=f'中位数: {order_values.median():.1f} 元'",
    
    "f'Lunch median: {l_med:.1f}'": "f'午餐中位数: {l_med:.1f}'",
    "f'Dinner median: {d_med:.1f}'": "f'晚餐中位数: {d_med:.1f}'",
    "'*Dinner: small sample, ref. only'": "'* 晚餐样本量小，仅供参考'",
    
    # ===== problem2 =====
    "label='Actual'": "label='实际值'",
    "label='Predicted'": "label='预测值'",
    "label='Forecast'": "label='预测值'",
    "label='95% CI'": "label='95% 置信区间'",
    f"label=f'Mean: {{mean_val:.0f}}'": f"label=f'均值: {{mean_val:.0f}}'",
    
    "'ACF - Daily Orders'": "'ACF — 日订单数'",
    "'PACF - Daily Orders'": "'PACF — 日订单数'",
    "'Residual'": "'残差'",
    "'Predicted'": "'预测值'",
    "'MAPE (%)'": "'MAPE (%)'",
    "'Frequency'": "'频数'",
    "'Residual'": "'残差'",
    
    f"f'Residual Dist: {{self.TARGET_NAMES[col]}}\\n'": f"f'残差分布: {{self.TARGET_NAMES[col]}}\\n'",
    f"f'Residual ACF: {{self.TARGET_NAMES[col]}}\\n'": f"f'残差 ACF: {{self.TARGET_NAMES[col]}}\\n'",
    f"f'MAPE by Weekday: {{self.TARGET_NAMES[col]}}'": f"f'按星期 MAPE: {{self.TARGET_NAMES[col]}}'",
    f"f'Walk-Forward Validation: {{self.TARGET_NAMES[target_col]}}\\n'": f"f'Walk-Forward 验证: {{self.TARGET_NAMES[target_col]}}\\n'",
    f"f'Residual vs Predicted\\n(bias={{np.mean(resid):.1f}})'": f"f'残差 vs 预测值\\n(偏差={{np.mean(resid):.1f}})'",
    f"f'{{name}} - May 2025 Workdays'": f"f'{{name}} — 2025年5月工作日预测'",
    
    f"f'{{name}}\\n(ADF p={{adf_result[1]:.4f}}, '": f"f'{{name}}\\n(ADF p={{adf_result[1]:.4f}}, '",
    f"f'{{\\\"平稳\\\" if is_stationary else \\\"非平稳\\\"}})'": f"f'{{\\\"平稳\\\" if is_stationary else \\\"非平稳\\\"}})'",
    
    # ===== problem3 =====
    "'Total Servings'": "'总备菜份数'",
    "'Daily Lunch Preparation Quantities'": "'每日午餐备菜量'",
    "'Profit Margin (%)'": "'利润率 (%)'",
    "'Expected Lunch Profit Margin'": "'预期午餐利润率'",
    "label='30% target'": "label='30% 目标'",
    "label='DRIs Target'": "label='DRIs 标准'",
    "label='Actual Supply'": "label='实际供给'",
    
    # ===== problem4 =====
    "'Value'": "'数值'",
    "'Combo Comparison Across Price Levels'": "'各价位套餐指标对比'",
    "'Nutrition Comparison (Normalized)'": "'营养成分对比 (归一化)'",
    f"label=f'{{price}} Yuan'": f"label=f'{{price}} 元'",
    f"'{{p}} Yuan'": f"'{{p}} 元'",
    
    # ===== problem5 =====
    "'Total Orders'": "'总订单数'",
    "'Profit Margin (%)'": "'利润率 (%)'",
    "'Dish Evaluation Matrix\\n(Sales × Profit)'": "'菜品评估矩阵\\n(销量 × 利润)'",
    "label='Promote'": "label='重点推广'",
    "label='Maintain'": "label='维持'",
    "label='Replace'": "label='考虑替换'",
    
    # ===== split_p2 =====
    "label='Orders'": "label='订单数'",
    "label='Sales'": "label='销售额'",
    "label='Weekdays'": "label='工作日'",
    "label='Weekends'": "label='周末'",
    
    # ===== validate =====
    "label='With detail'": "label='有明细'",
    "label='No detail'": "label='无明细'",
    "label='y=x'": "label='y=x'",
    "'Attachment 1'": "'附件1'",
    "'Attachment 2'": "'附件2'",
    "'Hour of Day'": "'小时'",
    "'Order Value (yuan)'": "'订单金额 (元)'",
    "'Bootstrap Survival Rate (%)'": "'Bootstrap 生存率 (%)'",
    "'Feature Importance (gain)'": "'特征重要性 (gain)'",
    "'Expected Profit (yuan/day)'": "'预期利润 (元/天)'",
    "'Demand Factor (1.0 = baseline)'": "'需求因子 (1.0 = 基准)'",
    "'|Correlation with Profit|'": "'|与利润的相关性|'",
    
    "'Hourly Distribution: With vs Without Detail'": "'小时分布: 有明细 vs 无明细'",
    "'Day-of-Week Distribution'": "'星期分布'",
}

for fname in FILES:
    fp = os.path.join(PROJECT, fname)
    if not os.path.exists(fp): continue
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read()
    count = 0
    for old, new in REPLACEMENTS.items():
        if old in content:
            content = content.replace(old, new)
            count += 1
    if count > 0:
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'{fname}: {count} replacements')
    else:
        print(f'{fname}: no changes')
