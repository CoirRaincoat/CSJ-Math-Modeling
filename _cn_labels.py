"""批量替换所有可视化文件中的英文注记为中文，然后重新生成图表"""
import os, re

PROJECT = r'C:\Users\CoirRaincoat\PyCharmMiscProject\CSJ-MathModeling'

# 中英文映射表: 按文件分类
replacements = {
    'problem1_analysis.py': [
        # Legend labels
        ("label='80% threshold'", "label='80% 阈值'"),
        ("label='95% threshold'", "label='95% 阈值'"),
        ("label='Orders'", "label='订单数'"),
        ("label='Sales'", "label='销售额'"),
        ("label='Lunch'", "label='午餐'"),
        ("label='Dinner'", "label='晚餐'"),
        ("label='Lunch Period'", "label='午餐时段'"),
        ("label='Dinner Period'", "label='晚餐时段'"),
        ("label='Lunch (n=", "label='午餐 (n="),
        ("label='Dinner (n=", "label='晚餐 (n="),
        # Titles
        ("Welch's t-test", "Welch's t 检验"),
        ("(Significance", "(显著性"),
        ("n.s.", "不显著"),
    ],
    'problem2_prediction.py': [
        ("label='Actual'", "label='实际值'"),
        ("label='Predicted'", "label='预测值'"),
        ("label='Forecast'", "label='预测值'"),
        ("label='95% CI'", "label='95% 置信区间'"),
        ("label='Mean:", "label='均值:"),
    ],
    'problem3_optimization.py': [
        ("label='30% target'", "label='30% 目标'"),
        ("label='DRIs Target'", "label='DRIs 标准'"),
        ("label='Actual Supply'", "label='实际供给'"),
    ],
    'problem4_combos.py': [
        ("'Total Price (Yuan)'", "'总价格 (元)'"),
        ("'Profit Margin (%)'", "'利润率 (%)'"),
        ("'Nutrition Balance (%)'", "'营养均衡度 (%)'"),
        ("'Overall Score (%)'", "'综合得分 (%)'"),
        ("'Calories\\n(kcal)'", "'热量\\n(kcal)'"),
        ("'Protein\\n(g)'", "'蛋白质\\n(g)'"),
        ("'Fat\\n(g)'", "'脂肪\\n(g)'"),
        ("'Carbs\\n(g)'", "'碳水\\n(g)'"),
        ("'Fiber\\n(g)'", "'纤维\\n(g)'"),
        ("label=f'{price} Yuan'", "label=f'{price} 元'"),
        ("f'{p} Yuan'", "f'{p} 元'"),
    ],
    'problem5_strategy.py': [
        ("label='Promote'", "label='重点推广'"),
        ("label='Maintain'", "label='维持'"),
        ("label='Replace'", "label='考虑替换'"),
        ("'Data\nCollection'", "'数据\n采集'"),
        ("'Prediction\nModel'", "'预测\n模型'"),
        ("'Prep\nExecution'", "'备菜\n执行'"),
        ("'Performance\nReview'", "'效果\n评估'"),
        ("'Continuous\nImprovement\nLoop'", "'持续\n改进\n循环'"),
        ("'Strategy:\\nFull preparation\\n1.15x buffer'", "'策略: 充分备货\\n1.15倍缓冲'"),
        ("'Strategy:\\nModerate prep\\n1.05x buffer'", "'策略: 适度备货\\n1.05倍缓冲'"),
        ("'Strategy:\\nSmall batch\\nrotation supply'", "'策略: 小批量\\n轮换供应'"),
        ("'Basic'", "'基础型'"),
        ("'Balanced'", "'均衡型'"),
        ("'Premium'", "'品质型'"),
        ("'Demand\\nPrediction'", "'需求\\n预测'"),
        ("'Prep\\nEfficiency'", "'备菜\\n效率'"),
        ("'Waste\\nReduction'", "'浪费\\n减少'"),
        ("'Revenue\\nGrowth'", "'收入\\n增长'"),
        ("'Customer\\nSatisfaction'", "'顾客\\n满意度'"),
        ("'Expected Improvement (%)'", "'预期改善 (%)'"),
        ("'Digital\nOperation'", "'数字化\n运营'"),
        ("'Data\nFeedback'", "'数据\n反馈'"),
        ("'Nutrition\nESG'", "'营养\nESG'"),
        ("'Menu\nManagement'", "'菜品\n管理'"),
        ("'Combo\nMarketing'", "'套餐\n营销'"),
        ("'Prediction-driven\nPrep Optimization'", "'预测驱动\n备菜优化'"),
    ],
    'validate_reliability.py': [
        ("label='With detail'", "label='有明细'"),
        ("label='No detail'", "label='无明细'"),
        ("label='80% threshold'", "label='80% 阈值'"),
        ("label='Waste factor'", "label='浪费因子'"),
        ("label='y=x'", "label='y=x'"),
        ("'Attachment 1'", "'附件1'"),
        ("'Attachment 2'", "'附件2'"),
        ("'Hour of Day'", "'小时'"),
        ("'Order Value (yuan)'", "'订单金额 (元)'"),
        ("'Bootstrap Survival Rate (%)'", "'Bootstrap 生存率 (%)'"),
        ("'Feature Importance (gain)'", "'特征重要性 (gain)'"),
        ("'Expected Profit (yuan/day)'", "'预期利润 (元/天)'"),
        ("'Demand Factor (1.0 = baseline)'", "'需求因子 (1.0 = 基准)'"),
        ("'|Correlation with Profit|'", "'|与利润的相关性|'"),
    ],
}

for fname, reps in replacements.items():
    fp = os.path.join(PROJECT, fname)
    if not os.path.exists(fp):
        continue
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read()
    count = 0
    for old, new in reps:
        if old in content:
            content = content.replace(old, new)
            count += 1
    if count > 0:
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'{fname}: {count} replacements made')
    else:
        print(f'{fname}: no changes (already up to date)')

print('\nDone. Run main.py to regenerate all images.')
