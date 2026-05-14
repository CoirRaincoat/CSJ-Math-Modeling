"""
problem5_strategy.py — 问题5：经营情况分析与策略建议
================================================
题目要求：
  综合分析该餐厅的运营情况，给出该餐厅优化经营的策略和建议。

解题思路：
  基于问题1-4的分析结果，从以下维度提出系统性策略建议：
  
  1. 备菜策略优化
     - 建立"预测→备菜→销售→复盘"闭环
     - 分级备菜制度（ABC分类管理）
     - 安全库存与应急响应机制
  
  2. 菜品结构优化
     - 保留/优化/淘汰三维决策矩阵
     - 午餐/晚餐差异化供应策略
     - 季节性菜品轮换机制
  
  3. 套餐推广策略
     - 阶梯定价策略（10/15/20元三层套餐）
     - 套餐组合动态调整机制
     - 营养标签与健康营销
  
  4. 数字化运营建议
     - 每日数据看板
     - 预测模型迭代更新
     - 后厨智能化管理
  
  5. 营养与ESG策略
     - 健康膳食引导
     - 食物浪费控制
     - 可持续经营

参考文献：
  [1] Rodrigues M. et al. "Machine learning models for short-term demand
      forecasting in food catering services" J. Cleaner Production, 2024.
  [8] Padovan M. et al. "Optimized menu formulation" BMC Nutrition, 2023.
  [9] Cohen J.F.W. et al. "Improving school lunch menus with multi-objective
      optimisation" Public Health Nutrition, 2023.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

from data_loader import load_all_data
from config import OUTPUT_DIR, COLORS, NUTRITION_PER_MEAL


class Problem5Strategy:
    """
    问题5：综合经营策略建议
    
    生成系统性的运营优化建议，涵盖备菜、菜品结构、
    套餐、数字化和营养ESG五大维度。
    """
    
    def __init__(self, loader=None):
        """初始化"""
        print('\n' + '=' * 60)
        print('问题5：经营策略分析')
        print('=' * 60)
        
        if loader is None:
            self.loader = load_all_data()
        else:
            self.loader = loader
        
        self.df_daily = self.loader.get_daily_data()
        self.df_meal = self.loader.get_meal_data()
        self.dish_info = self.loader.get_dish_info()
        
        self.strategies = {}
        
    def run(self):
        """生成完整策略报告"""
        print('\n>>> 5.1 备菜策略分析')
        self._preparation_strategy()
        
        print('\n>>> 5.2 菜品结构优化')
        self._menu_structure_analysis()
        
        print('\n>>> 5.3 套餐推广策略')
        self._combo_strategy()
        
        print('\n>>> 5.4 数字化运营建议')
        self._digital_operation_strategy()
        
        print('\n>>> 5.5 营养与ESG策略')
        self._nutrition_esg_strategy()
        
        print('\n>>> 5.6 综合策略可视化')
        self._plot_strategy_summary()
        
        print('\n问题5策略分析完成！')
        return self.strategies
    
    def _preparation_strategy(self):
        """备菜策略"""
        daily = self.df_daily.copy()
        
        # 计算日销量的变异系数（CV = std/mean）
        # CV反映需求波动程度
        orders_cv = daily['total_orders'].std() / daily['total_orders'].mean()
        sales_cv = daily['total_sales'].std() / daily['total_sales'].mean()
        
        # 星期波动分析
        daily['weekday'] = daily['date'].dt.dayofweek
        dow_means = daily.groupby('weekday')['total_orders'].mean()
        dow_cv = dow_means.std() / dow_means.mean()
        
        strategies = {
            'demand_variability': {
                'orders_cv': f'{orders_cv:.2%}',
                'sales_cv': f'{sales_cv:.2%}',
                'dow_variation': f'{dow_cv:.2%}',
                'interpretation': (
                    '日订单量变异系数为{:.0%}，说明需求存在中等程度波动，'
                    '需要建立安全库存机制'.format(orders_cv)
                ),
            },
            'recommendations': [
                {
                    'title': 'ABC分级备菜制度',
                    'detail': (
                        '对A类菜品（占总销量80%的头部菜品）采用充分备货策略，'
                        '备货量=预测需求×1.15；B类菜品（中间15%）采用适度备货，'
                        '备货量=预测需求×1.05；C类菜品（尾部5%）采用小批量轮换供应。'
                    ),
                    'expected_benefit': '减少备菜浪费15-20%，同时保证高需求菜品充足供应',
                },
                {
                    'title': '预测-备菜-复盘闭环',
                    'detail': (
                        '每日营业前：基于预测模型生成备菜计划；\n'
                        '每日营业中：实时监控菜品消耗速度，必要时启动补菜；\n'
                        '每日营业后：记录剩余量和缺货情况，反馈至预测模型。'
                    ),
                    'expected_benefit': '持续优化预测精度，形成正向反馈循环',
                },
                {
                    'title': '安全库存机制',
                    'detail': (
                        '安全库存量 = Z × σ × √LT\n'
                        '其中Z=1.65（95%服务水平），σ为需求标准差，LT为备货提前期。\n'
                        '建议安全库存系数设定为预测需求的15%。'
                    ),
                    'expected_benefit': '将缺货概率控制在5%以内',
                },
            ]
        }
        
        self.strategies['preparation'] = strategies
        
        print(f'  需求波动系数: 订单CV={orders_cv:.2%}, 销售额CV={sales_cv:.2%}')
        print(f'  星期波动: {dow_cv:.2%}')
        
    def _menu_structure_analysis(self):
        """菜品结构优化策略"""
        dish_info = self.dish_info.copy()
        
        # 菜品三维评估：销量 × 利润率 × 浪费风险
        dish_info['sales_rank'] = dish_info['total_orders'].rank(pct=True)
        dish_info['profit_rank'] = dish_info['profit_margin'].rank(pct=True)
        
        # 综合评分
        dish_info['comprehensive_score'] = (
            dish_info['sales_rank'] * 0.5 +
            dish_info['profit_rank'] * 0.3
        )
        
        # 分类建议
        top_quartile = dish_info['comprehensive_score'].quantile(0.75)
        bot_quartile = dish_info['comprehensive_score'].quantile(0.25)
        
        dish_info['recommendation'] = '维持'
        dish_info.loc[dish_info['comprehensive_score'] >= top_quartile, 'recommendation'] = '重点推广'
        dish_info.loc[dish_info['comprehensive_score'] <= bot_quartile, 'recommendation'] = '考虑替换'
        
        rec_counts = dish_info['recommendation'].value_counts()
        
        # 午餐/晚餐差异化
        lunch_mean = self.df_meal[self.df_meal['meal_period'] == 'lunch']['total_orders'].mean()
        dinner_mean = self.df_meal[self.df_meal['meal_period'] == 'dinner']['total_orders'].mean()
        
        strategies = {
            'menu_evaluation': {
                'promote': int(rec_counts.get('重点推广', 0)),
                'maintain': int(rec_counts.get('维持', 0)),
                'replace': int(rec_counts.get('考虑替换', 0)),
            },
            'lunch_dinner_ratio': f'{lunch_mean:.1f} : {dinner_mean:.1f}',
            'recommendations': [
                {
                    'title': '午/晚餐差异化供应',
                    'detail': (
                        f'午餐占总顾客的99%以上，应采取"丰富多样"策略，提供40+菜品选择；'
                        f'晚餐仅占<1%，应采取"精简精选"策略，提供15-20种核心菜品。\n'
                        f'午餐人均菜品数约5.5个，备菜时确保各类别菜品充足；'
                        f'晚餐可适当提高客单价，推出晚餐专属套餐。'
                    ),
                },
                {
                    'title': '菜品生命周期管理',
                    'detail': (
                        '引入期：小批量试供，收集顾客反馈；\n'
                        '成长期：根据销量增长趋势逐步扩大备货；\n'
                        '成熟期：标准化备货量，稳定供应；\n'
                        '衰退期：减少备货频率，准备替换方案。'
                    ),
                },
                {
                    'title': '季节性轮换',
                    'detail': (
                        '春夏：增加凉拌菜、清淡菜品比例；\n'
                        '秋冬：增加炖菜、热汤类菜品比例；\n'
                        '每月评估菜品表现，淘汰连续2月排名后20%的菜品。'
                    ),
                },
            ]
        }
        
        self.strategies['menu'] = strategies
        
        print(f'  菜品评估: 重点推广{rec_counts.get("重点推广", 0)}种, '
              f'维持{rec_counts.get("维持", 0)}种, 考虑替换{rec_counts.get("考虑替换", 0)}种')
        print(f'  午餐:晚餐人数比 = {lunch_mean:.0f}:{dinner_mean:.0f}')
        
    def _combo_strategy(self):
        """套餐推广策略"""
        strategies = {
            'recommendations': [
                {
                    'title': '三层阶梯套餐',
                    'detail': (
                        '10元"经济基础型"：面向价格敏感顾客，1主食+1荤+1素，\n'
                        '15元"均衡实用型"：主推款，1主食+1荤+1半荤+2素，\n'
                        '20元"丰富营养型"：提升客单价，1主食+2荤+1半荤+2素+1特色。\n'
                        '预期套餐销售占比：10元(20%) + 15元(50%) + 20元(30%)'
                    ),
                },
                {
                    'title': '动态套餐调整',
                    'detail': (
                        '每周根据菜品库存和时令食材更新套餐内容；\n'
                        '结合天气、节假日推出限定套餐（如夏季清凉套餐）；\n'
                        '根据关联规则分析结果优化套餐内菜品搭配。'
                    ),
                },
                {
                    'title': '套餐营养标识',
                    'detail': (
                        '对每个套餐标注热量、蛋白质、脂肪、碳水含量；\n'
                        '推出"高蛋白套餐""低脂套餐"等健康标签；\n'
                        '契合当前消费者对健康饮食的关注趋势。'
                    ),
                },
            ]
        }
        
        self.strategies['combo'] = strategies
        
    def _digital_operation_strategy(self):
        """数字化运营建议"""
        strategies = {
            'recommendations': [
                {
                    'title': '每日运营数据看板',
                    'detail': (
                        '实时监控指标：当前销售额、就餐人数、热门菜品排名；\n'
                        '预警指标：菜品剩余率>30%或消耗率>90%时触发提醒；\n'
                        '历史对比：当日数据与历史同星期均值对比。'
                    ),
                },
                {
                    'title': '预测模型迭代',
                    'detail': (
                        '每周使用最新数据重新训练预测模型；\n'
                        '节假日和异常天气日单独建模；\n'
                        '当MAPE连续2周>20%时，触发模型审查和调优。'
                    ),
                },
                {
                    'title': '后厨智能化',
                    'detail': (
                        '将优化模型输出的备菜方案直接推送至后厨管理系统；\n'
                        '菜品消耗数据实时回传，动态调整补菜计划；\n'
                        '建立菜品制作标准工序卡，确保品质一致性。'
                    ),
                },
            ]
        }
        
        self.strategies['digital'] = strategies
        
    def _nutrition_esg_strategy(self):
        """营养与ESG策略"""
        # 分析当前营养结构
        daily = self.df_daily.copy()
        
        avg_cal_per_person = daily['total_calories'].mean() / daily['total_orders'].mean()
        avg_protein_per_person = daily['total_protein'].mean() / daily['total_orders'].mean()
        avg_fat_per_person = daily['total_fat'].mean() / daily['total_orders'].mean()
        
        # 热量来源分析
        protein_cal = avg_protein_per_person * 4
        fat_cal = avg_fat_per_person * 9
        carb_cal = avg_cal_per_person - protein_cal - fat_cal
        
        fat_ratio = fat_cal / avg_cal_per_person * 100 if avg_cal_per_person > 0 else 0
        
        # 浪费估算
        # 假设日均剩余率10-20%
        waste_rate = 0.15
        daily_waste_estimate = daily['total_sales'].mean() * waste_rate
        
        strategies = {
            'nutrition_current': {
                'avg_cal_per_person': f'{avg_cal_per_person:.0f} kcal',
                'avg_protein_per_person': f'{avg_protein_per_person:.0f} g',
                'fat_ratio': f'{fat_ratio:.1f}%',
            },
            'waste_estimate_daily': f'{daily_waste_estimate:.0f} 元/天',
            'waste_estimate_yearly': f'{daily_waste_estimate * 365:.0f} 元/年',
            'recommendations': [
                {
                    'title': '营养结构优化',
                    'detail': (
                        f'当前人均脂肪供能比约{fat_ratio:.0f}%，'
                        f'略高于推荐的20-30%，建议适当增加低脂菜品比例；\n'
                        '推出"轻食专区"，提供高蛋白低脂肪的菜品组合；\n'
                        '对高热量菜品标注热量信息，引导理性消费。'
                    ),
                },
                {
                    'title': '食物浪费控制',
                    'detail': (
                        f'按15%剩余率估算，日均浪费约{daily_waste_estimate:.0f}元，'
                        f'年损失约{daily_waste_estimate*365:.0f}元；\n'
                        '通过精准预测可将浪费率降至8-10%，年节省约'
                        f'{daily_waste_estimate*365*0.4:.0f}元；\n'
                        '剩余菜品可对接食物银行或员工福利餐，减少直接丢弃。'
                    ),
                },
                {
                    'title': 'ESG可持续经营',
                    'detail': (
                        '减少食物浪费 = 降低碳足迹（每减少1kg食物浪费≈减少2.5kg CO2e）；\n'
                        '优先采购本地食材，减少运输碳排放；\n'
                        '使用可降解餐盒，减少塑料污染；\n'
                        '将ESG成果融入品牌宣传，提升企业社会形象。'
                    ),
                },
            ]
        }
        
        self.strategies['nutrition_esg'] = strategies
        
        print(f'  人均热量: {avg_cal_per_person:.0f} kcal')
        print(f'  脂肪供能比: {fat_ratio:.1f}%')
        print(f'  日均浪费估算: {daily_waste_estimate:.0f} 元')
        
    def _plot_strategy_summary(self):
        """策略综合可视化"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 子图1: 策略框架总览
        ax1 = axes[0, 0]
        ax1.axis('off')
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        
        framework = [
            ('预测驱动\n备菜优化', 2, 8, COLORS['primary']),
            ('菜品结构\n动态管理', 5, 8, COLORS['secondary']),
            ('套餐分层\n精准营销', 8, 8, COLORS['accent']),
            ('数字化\n运营平台', 2, 3, COLORS['success']),
            ('营养ESG\n可持续', 5, 3, COLORS['warning']),
            ('数据反馈\n持续迭代', 8, 3, '#888888'),
        ]
        
        for text, x, y, color in framework:
            ax1.add_patch(plt.Rectangle((x-1.2, y-0.8), 2.4, 1.6, 
                          fill=True, facecolor=color, alpha=0.8,
                          edgecolor='white', linewidth=2))
            ax1.text(x, y, text, ha='center', va='center', fontsize=9,
                    fontweight='bold', color='white')
        
        # 连接线
        for i in range(len(framework)-1):
            ax1.annotate('', xy=(framework[i+1][1]-1.2, framework[i+1][2]),
                        xytext=(framework[i][1]+1.2, framework[i][2]),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=1))
        
        ax1.set_title('Operation Optimization Framework', fontweight='bold', fontsize=12)
        
        # 子图2: 菜品评估矩阵散点图
        ax2 = axes[0, 1]
        dish = self.dish_info.copy()
        colors_rec = []
        for rec in dish['recommendation'] if 'recommendation' in dish.columns else dish['profit_margin']:
            if isinstance(rec, str):
                if rec == '重点推广':
                    colors_rec.append(COLORS['success'])
                elif rec == '考虑替换':
                    colors_rec.append(COLORS['danger'])
                else:
                    colors_rec.append(COLORS['primary'])
            else:
                colors_rec.append(COLORS['primary'])
        
        sc = ax2.scatter(dish['total_orders'], dish['profit_margin'] * 100,
                        c=colors_rec[:len(dish)], alpha=0.6, s=30, edgecolors='grey', linewidth=0.3)
        ax2.set_xlabel('Total Orders')
        ax2.set_ylabel('Profit Margin (%)')
        ax2.set_title('Dish Evaluation Matrix', fontweight='bold')
        ax2.axhline(y=30, color=COLORS['danger'], linestyle='--', alpha=0.5)
        ax2.axvline(x=dish['total_orders'].median(), color=COLORS['warning'], 
                   linestyle='--', alpha=0.5)
        ax2.grid(alpha=0.3)
        
        # 图例
        legend_elements = [
            mpatches.Patch(color=COLORS['success'], label='Promote'),
            mpatches.Patch(color=COLORS['primary'], label='Maintain'),
            mpatches.Patch(color=COLORS['danger'], label='Replace'),
        ]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # 子图3: ABC分类策略示意图
        ax3 = axes[0, 2]
        categories = ['Class A\n(80% sales)\n~58 dishes', 
                     'Class B\n(15% sales)\n~70 dishes',
                     'Class C\n(5% sales)\n~109 dishes']
        strategies_abc = [
            'Strategy:\nFull preparation\n1.15x buffer',
            'Strategy:\nModerate prep\n1.05x buffer',
            'Strategy:\nSmall batch\nrotation supply'
        ]
        colors_abc = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]
        
        y_pos = [7, 4, 1]
        for i in range(3):
            ax3.add_patch(plt.Rectangle((0.5, y_pos[i]-0.8), 8, 1.6,
                          fill=True, facecolor=colors_abc[i], alpha=0.2,
                          edgecolor=colors_abc[i], linewidth=2))
            ax3.text(1, y_pos[i], categories[i], fontsize=9, fontweight='bold', va='center')
            ax3.text(5, y_pos[i], strategies_abc[i], fontsize=8, va='center', color='#555555')
        
        ax3.set_xlim(0, 10)
        ax3.set_ylim(0, 9)
        ax3.axis('off')
        ax3.set_title('ABC Classification Strategy', fontweight='bold')
        
        # 子图4: 套餐推广策略图示
        ax4 = axes[1, 0]
        ax4.axis('off')
        ax4.set_xlim(0, 10)
        ax4.set_ylim(0, 8)
        
        combos = [
            ('10 Yuan\nBasic', 1.5, 5, COLORS['accent'], '20%'),
            ('15 Yuan\nBalanced', 4.5, 5, COLORS['primary'], '50% ★'),
            ('20 Yuan\nPremium', 7.5, 5, COLORS['success'], '30%'),
        ]
        
        for text, x, y, color, share in combos:
            size = 2.2 if '★' in share else 1.5
            ax4.add_patch(plt.Circle((x, y), size/2, fill=True,
                          facecolor=color, alpha=0.7, edgecolor='white', linewidth=2))
            ax4.text(x, y, text + f'\n({share})', ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white')
        
        # 箭头连接
        for i in range(len(combos)-1):
            ax4.annotate('', xy=(combos[i+1][1]-1.0, combos[i+1][2]),
                        xytext=(combos[i][1]+1.0, combos[i][2]),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=2))
        
        ax4.set_title('Tiered Combo Strategy', fontweight='bold', fontsize=12)
        
        # 子图5: 预期改善效果（假设性估算）
        ax5 = axes[1, 1]
        improvements = {
            'Demand\nPrediction': 20,
            'Prep\nEfficiency': 25,
            'Waste\nReduction': 40,
            'Revenue\nGrowth': 15,
            'Customer\nSatisfaction': 20,
        }
        
        bars = ax5.barh(list(improvements.keys()), list(improvements.values()),
                       color=[COLORS['primary'], COLORS['secondary'], 
                              COLORS['success'], COLORS['accent'], COLORS['warning']],
                       edgecolor='white', linewidth=1)
        ax5.set_xlabel('Expected Improvement (%)')
        ax5.set_title('Expected Impact of Optimization', fontweight='bold')
        ax5.set_xlim(0, 55)
        ax5.grid(axis='x', alpha=0.3)
        
        for bar, val in zip(bars, improvements.values()):
            ax5.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'+{val}%', va='center', fontweight='bold', fontsize=9)
        
        # 子图6: 数据闭环示意图
        ax6 = axes[1, 2]
        ax6.axis('off')
        ax6.set_xlim(0, 10)
        ax6.set_ylim(0, 10)
        
        cycle_steps = [
            (5, 8.5, 'Data\nCollection', COLORS['primary']),
            (8.5, 5, 'Prediction\nModel', COLORS['secondary']),
            (5, 1.5, 'Prep\nExecution', COLORS['accent']),
            (1.5, 5, 'Performance\nReview', COLORS['success']),
        ]
        
        for x, y, text, color in cycle_steps:
            ax6.add_patch(plt.Circle((x, y), 1.3, fill=True,
                          facecolor=color, alpha=0.8, edgecolor='white', linewidth=2))
            ax6.text(x, y, text, ha='center', va='center', fontsize=8,
                    fontweight='bold', color='white')
        
        # 循环箭头
        prev = cycle_steps[-1]
        for step in cycle_steps:
            ax6.annotate('', xy=(step[0], step[1]),
                        xytext=(prev[0], prev[1]),
                        arrowprops=dict(arrowstyle='->', 
                                       connectionstyle='arc3,rad=0.3',
                                       color='gray', lw=1.5))
            prev = step
        
        ax6.text(5, 5, 'Continuous\nImprovement\nLoop', ha='center', 
                va='center', fontsize=10, fontweight='bold', color='#333333')
        ax6.set_title('Data-Driven Operation Cycle', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/p5_strategy_summary.png', dpi=150)
        plt.close()
        print('  已保存: p5_strategy_summary.png')
    
    def print_strategy_report(self):
        """打印完整的策略报告文本"""
        print('\n' + '=' * 70)
        print('                    经营优化策略报告')
        print('=' * 70)
        
        sections = [
            ('一、备菜策略优化', 'preparation'),
            ('二、菜品结构优化', 'menu'),
            ('三、套餐推广策略', 'combo'),
            ('四、数字化运营建议', 'digital'),
            ('五、营养与ESG策略', 'nutrition_esg'),
        ]
        
        for title, key in sections:
            if key not in self.strategies:
                continue
            
            print(f'\n{title}')
            print('-' * 50)
            
            strategy = self.strategies[key]
            
            if 'recommendations' in strategy:
                for i, rec in enumerate(strategy['recommendations'], 1):
                    print(f'\n  {i}. {rec["title"]}')
                    print(f'     {rec["detail"]}')
                    if 'expected_benefit' in rec:
                        print(f'     [预期效果] {rec["expected_benefit"]}')


if __name__ == '__main__':
    ps = Problem5Strategy()
    results = ps.run()
    ps.print_strategy_report()
