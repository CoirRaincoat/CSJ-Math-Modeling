"""problem5_strategy.py — 问题5: 经营情况分析与策略建议"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from data_loader import load_all_data
from config import OUTPUT_DIR, COLORS, COLOR_CYCLE, NUTRITION_PER_MEAL


class Problem5Strategy:

    def __init__(self, loader=None):
        print('\n' + '=' * 60)
        print('问题5: 经营策略分析')
        print('=' * 60)

        if loader is None:
            self.loader = load_all_data()
        else:
            self.loader = loader

        self.df_daily = self.loader.get_daily_data()
        self.df_meal = self.loader.get_meal_data()
        self.dish_info = self.loader.get_dish_info()

        # 在初始化时运行菜品评估
        self._run_menu_evaluation()

        self.strategies = {}

    def _run_menu_evaluation(self):
        dish_info = self.dish_info.copy()

        # 销量排名 (百分位)
        dish_info['sales_rank'] = dish_info['total_orders'].rank(pct=True)
        # 利润率排名 (百分位)
        dish_info['profit_rank'] = dish_info['profit_margin'].rank(pct=True)

        # 综合评分 = 销量权重 × 50% + 利润权重 × 30%
        dish_info['comprehensive_score'] = (
            dish_info['sales_rank'] * 0.5 +
            dish_info['profit_rank'] * 0.3
        )

        # 按四分位分类
        top_q = dish_info['comprehensive_score'].quantile(0.75)
        bot_q = dish_info['comprehensive_score'].quantile(0.25)

        dish_info['recommendation'] = '维持'
        dish_info.loc[
            dish_info['comprehensive_score'] >= top_q, 'recommendation'
        ] = '重点推广'
        dish_info.loc[
            dish_info['comprehensive_score'] <= bot_q, 'recommendation'
        ] = '考虑替换'

        self.dish_eval = dish_info

    def run(self):
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

        print('\n问题5 策略分析完成!')
        return self.strategies

    def _preparation_strategy(self):
        daily = self.df_daily.copy()

        # 需求波动性分析
        orders_cv = daily['total_orders'].std() / daily['total_orders'].mean()
        sales_cv = daily['total_sales'].std() / daily['total_sales'].mean()

        # 星期波动分析
        daily['weekday'] = daily['date'].dt.dayofweek
        dow_means = daily.groupby('weekday')['total_orders'].mean()
        dow_cv = dow_means.std() / dow_means.mean()

        # 波动等级判定
        cv_level = (
            '低波动' if orders_cv < 0.20
            else ('中等波动' if orders_cv < 0.30 else '较高波动')
        )

        strategies = {
            'data_insights': {
                'orders_cv': f'{orders_cv:.1%}',
                'sales_cv': f'{sales_cv:.1%}',
                'dow_variation': f'{dow_cv:.1%}',
                'cv_level': cv_level,
            },
            'recommendations': [
                {
                    'title': 'ABC 分级备菜制度',
                    'detail': (
                        'A 类菜品 (占总销量 80% 的头部菜品): '
                        '充分备货策略，备菜量 = 预测需求 × 1.15\n'
                        'B 类菜品 (中间 15%): 适度备货策略，'
                        '备菜量 = 预测需求 × 1.05\n'
                        'C 类菜品 (尾部 5%): 小批量轮换供应策略，'
                        '降低滞销风险'
                    ),
                    'data_basis': (
                        f'基于问题1的ABC分析: A类约58种菜品贡献80%销量，'
                        f'B类约70种贡献15%，C类约109种贡献5%'
                    ),
                },
                {
                    'title': '预测-备菜-复盘闭环机制',
                    'detail': (
                        '每日营业前: 基于预测模型 (问题2) 生成备菜计划\n'
                        '每日营业中: 监控菜品消耗速度，必要时启动补菜流程\n'
                        '每日营业后: 记录剩余量和缺货情况，反馈至预测模型\n'
                        '每周复盘: 评估预测精度，优化模型参数'
                    ),
                    'data_basis': (
                        f'当前需求变异系数 CV={orders_cv:.1%}，'
                        f'属于{cv_level}，安全库存系数建议 {cv_level}'
                    ),
                },
                {
                    'title': '星期差异化备菜',
                    'detail': (
                        '根据问题1中发现的星期消费规律 (工作日 vs '
                        '周末差异)，制定差异化的备菜量基线\n'
                        '工作日: 基于历史同星期均值 + 趋势修正\n'
                        '周末: 使用专用预测模型，考虑周末消费模式'
                    ),
                    'data_basis': (
                        f'星期波动系数 DOW_CV={dow_cv:.1%}，'
                        f'不同星期之间的需求差异显著'
                    ),
                },
            ]
        }

        self.strategies['preparation'] = strategies

        print(f'  需求波动: 订单CV={orders_cv:.1%}, 销量CV={sales_cv:.1%}')
        print(f'  波动等级: {cv_level}')
        print(f'  星期波动: {dow_cv:.1%}')

    def _menu_structure_analysis(self):
        rec_counts = self.dish_eval['recommendation'].value_counts()

        # 午餐晚餐人数比 (用于差异化策略)
        lunch_mean = self.df_meal[
            self.df_meal['meal_period'] == 'lunch'
        ]['total_orders'].mean()
        dinner_mean = self.df_meal[
            self.df_meal['meal_period'] == 'dinner'
        ]['total_orders'].mean()

        strategies = {
            'menu_evaluation': {
                'promote': int(rec_counts.get('重点推广', 0)),
                'maintain': int(rec_counts.get('维持', 0)),
                'replace': int(rec_counts.get('考虑替换', 0)),
            },
            'lunch_stats': f'午餐日均 {lunch_mean:.0f} 人',
            'recommendations': [
                {
                    'title': '菜品生命周期管理',
                    'detail': (
                        '引入期: 小批量试供，收集顾客反馈\n'
                        '成长期: 根据需求增长趋势逐步扩大备货\n'
                        '成熟期: 标准化备货量，稳定供应\n'
                        '衰退期: 减少备货频率，准备替换方案'
                    ),
                    'data_basis': (
                        f'菜品评估结果: 重点推广{rec_counts.get("重点推广",0)}种，'
                        f'维持{rec_counts.get("维持",0)}种，'
                        f'考虑替换{rec_counts.get("考虑替换",0)}种'
                    ),
                },
                {
                    'title': '午餐专属供应策略',
                    'detail': (
                        '午餐占该餐厅 99%+ 的业务量，应采用"丰富多样"策略\n'
                        '提供 40-50 种菜品选择，确保各类别菜品充足\n'
                        '按 ABC 等级分级备菜 (A类充足, B类适度, C类轮换)'
                    ),
                    'data_basis': (
                        f'午餐日均 {lunch_mean:.0f} 人，'
                        f'占营业总量的 99.2%'
                    ),
                },
                {
                    'title': '季节性菜品轮换机制',
                    'detail': (
                        '春夏季: 增加凉拌菜、清淡类菜品比例\n'
                        '秋冬季: 增加炖菜、热汤类菜品比例\n'
                        '每月评估菜品表现，淘汰连续 2 月排名后 20% 的菜品\n'
                        '每周引入 1-2 种新品测试市场反应'
                    ),
                    'data_basis': '基于行业最佳实践的运营建议',
                },
            ]
        }

        self.strategies['menu'] = strategies

        print(f'  菜品评估: '
              f'重点推广{rec_counts.get("重点推广",0)}种, '
              f'维持{rec_counts.get("维持",0)}种, '
              f'考虑替换{rec_counts.get("考虑替换",0)}种')
        print(f'  午餐日均: {lunch_mean:.0f} 人 (占 99.2%)')

    def _combo_strategy(self):
        strategies = {
            'recommendations': [
                {
                    'title': '三层阶梯套餐体系',
                    'detail': (
                        '10 元"经济基础型": 面向价格敏感顾客，'
                        '1主食+1荤+1素\n'
                        '15 元"均衡实用型": 主推款，'
                        '1主食+1荤+1半荤+2素\n'
                        '20 元"丰富营养型": 提升客单价，'
                        '1主食+2荤+1半荤+2素+1特色\n'
                        '建议套餐销售占比: 10元(20-25%) + '
                        '15元(45-55%) + 20元(20-30%)'
                    ),
                    'data_basis': (
                        '基于问题4的贪心+局部优化搜索结果，'
                        '当前客单价均值为 11.36 元'
                    ),
                },
                {
                    'title': '动态套餐内容更新',
                    'detail': (
                        '每周根据菜品库存和时令食材调整套餐内容\n'
                        '结合关联规则分析结果 (问题1) 优化菜品搭配\n'
                        '推出"今日特惠套餐"以消化当日冗余库存'
                    ),
                    'data_basis': (
                        '基于问题1的25条关联规则，'
                        '可用于优化套餐内菜品搭配'
                    ),
                },
                {
                    'title': '套餐营养信息标识',
                    'detail': (
                        '对每个套餐标注热量、蛋白质、脂肪、碳水含量\n'
                        '符合消费者日益增长的健康饮食关注趋势\n'
                        '参考《中国居民膳食指南(2022)》'
                        '的营养推荐标准'
                    ),
                    'data_basis': (
                        '基于问题4计算的三价位套餐营养均衡度'
                    ),
                },
            ]
        }

        self.strategies['combo'] = strategies

    def _digital_operation_strategy(self):
        strategies = {
            'recommendations': [
                {
                    'title': '每日运营数据看板',
                    'detail': (
                        '实时监控: 当日销售额、就餐人数、Top 5 热门菜品\n'
                        '智能预警: 菜品剩余率 > 30% 或消耗率 > 90% 触发提醒\n'
                        '历史对比: 当日数据与历史同星期均值自动对比'
                    ),
                    'data_basis': '基于问题1和问题2的数据分析能力构建',
                },
                {
                    'title': '预测模型持续迭代',
                    'detail': (
                        '每周使用最新运营数据重新训练预测模型\n'
                        '节假日和极端天气日单独建模 (扩展特征)\n'
                        '当 MAPE 连续 2 周 > 20% 时触发模型审查'
                    ),
                    'data_basis': (
                        '基于问题2的多模型比较框架，'
                        '当前最优模型可集成到生产环境'
                    ),
                },
                {
                    'title': '后厨智能排产',
                    'detail': (
                        '将 MILP 优化模型 (问题3) 输出的备菜方案'
                        '直接推送至后厨管理系统\n'
                        '菜品消耗数据实时回传，触发动态补菜\n'
                        '建立菜品标准工序卡，确保品质一致性'
                    ),
                    'data_basis': (
                        '基于问题3的午餐备菜优化模型，'
                        '可扩展为实时决策支持系统'
                    ),
                },
            ]
        }

        self.strategies['digital'] = strategies

    def _nutrition_esg_strategy(self):
        daily = self.df_daily.copy()

        # 人均营养现状
        avg_cal = daily['total_calories'].mean() / daily['total_orders'].mean()
        avg_protein = daily['total_protein'].mean() / daily['total_orders'].mean()
        avg_fat = daily['total_fat'].mean() / daily['total_orders'].mean()
        avg_carbs = daily['total_carbohydrates'].mean() / daily['total_orders'].mean()

        # 热量来源分析
        protein_cal = avg_protein * 4
        fat_cal = avg_fat * 9
        carb_cal = avg_carbs * 4
        total = protein_cal + fat_cal + carb_cal
        fat_ratio = fat_cal / total * 100 if total > 0 else 0
        carbs_ratio = carb_cal / total * 100 if total > 0 else 0
        protein_ratio = protein_cal / total * 100 if total > 0 else 0

        # 浪费估算
        # 基于安全库存系数 0.15 估算日均浪费
        waste_rate_estimate = 0.10  # 保守估计 10% 剩余率
        daily_waste = daily['total_sales'].mean() * waste_rate_estimate

        # 脂肪供能比评估
        fat_warning = (
            '偏高 (推荐 20-30%)'
            if fat_ratio > 30
            else ('偏低' if fat_ratio < 20 else '在推荐范围内')
        )

        strategies = {
            'nutrition_current': {
                'avg_cal': f'{avg_cal:.0f} kcal/人',
                'avg_protein': f'{avg_protein:.1f} g/人',
                'avg_fat': f'{avg_fat:.1f} g/人',
                'avg_carbs': f'{avg_carbs:.1f} g/人',
                'fat_ratio': f'{fat_ratio:.1f}% ({fat_warning})',
                'carbs_ratio': f'{carbs_ratio:.1f}%',
                'protein_ratio': f'{protein_ratio:.1f}%',
            },
            'waste_estimate': {
                'daily': f'{daily_waste:.0f} 元/天',
                'yearly': f'{daily_waste * 365:.0f} 元/年',
            },
            'recommendations': [
                {
                    'title': '营养结构优化',
                    'detail': (
                        f'当前人均脂肪供能比 {fat_ratio:.1f}%'
                        f'({fat_warning})，'
                        f'建议适当增加低脂菜品比例\n'
                        '推出"轻食专区"提供高蛋白低脂组合\n'
                        '对高热量菜品标注热量信息，引导理性消费'
                    ),
                    'data_basis': (
                        f'基于实际营养数据分析, '
                        f'参考 DRIs 2023 版推荐标准'
                    ),
                },
                {
                    'title': '食物浪费控制方案',
                    'detail': (
                        f'按保守估计 {waste_rate_estimate:.0%} 剩余率，'
                        f'日均浪费约 {daily_waste:.0f} 元\n'
                        '通过精准预测和合理备菜 (问题2+3)，'
                        '可将剩余率降至 5-8%\n'
                        '剩余菜品可对接: 员工福利餐、打折促销、食物捐赠'
                    ),
                    'data_basis': (
                        f'浪费估算基于日均销售额 '
                        f'{daily["total_sales"].mean():.0f} 元 '
                        f'× 剩余率 {waste_rate_estimate:.0%}'
                    ),
                },
                {
                    'title': 'ESG 可持续经营',
                    'detail': (
                        '食物浪费减排 = 碳足迹降低 '
                        '(每减少 1kg 食物浪费 ~ 减少 2.5 kg CO2eq)\n'
                        '优先采购本地/当季食材，减少运输碳排\n'
                        '使用可降解餐盒包装\n'
                        '将 ESG 成果融入品牌宣传'
                    ),
                    'data_basis': '基于联合国粮农组织 (FAO) '
                                  '食物浪费碳足迹系数',
                },
            ]
        }

        self.strategies['nutrition_esg'] = strategies

        print(f'  人均热量: {avg_cal:.0f} kcal')
        print(f'  脂肪供能比: {fat_ratio:.1f}% ({fat_warning})')
        print(f'  日均浪费估算: {daily_waste:.0f} 元')

    def _plot_strategy_summary(self):
        fig, axes = plt.subplots(2, 3, figsize=(22, 12))

        ax1 = axes[0, 0]
        ax1.axis('off')
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)

        framework = [
            ('Prediction-driven\nPrep Optimization\n(Problem 2+3)', 2.5, 8.0,
             COLORS['primary']),
            ('Menu Structure\nLifecycle Mgmt\n(Problem 1)', 7.5, 8.0,
             COLORS['secondary']),
            ('Tiered Combo\nMarketing\n(Problem 4)', 5.0, 5.0,
             COLORS['accent']),
            ('Digital\nOperations\nPlatform', 2.5, 2.0, COLORS['success']),
            ('Nutrition & ESG\nSustainability', 7.5, 2.0, COLORS['warning']),
        ]

        for text, x, y, color in framework:
            rect = plt.Rectangle((x-1.5, y-1.0), 3.0, 2.0,
                                 fill=True, facecolor=color, alpha=0.85,
                                 edgecolor='white', linewidth=2,
                                 transform=ax1.transData)
            ax1.add_patch(rect)
            ax1.text(x, y, text, ha='center', va='center', fontsize=8,
                    fontweight='bold', color='white')

        ax1.set_title('Operation Optimization Framework',
                     fontweight='bold', fontsize=12)

        ax2 = axes[0, 1]
        dish = self.dish_eval.copy()

        colors_rec = []
        for _, row in dish.iterrows():
            if row['recommendation'] == '重点推广':
                colors_rec.append(COLORS['success'])
            elif row['recommendation'] == '考虑替换':
                colors_rec.append(COLORS['danger'])
            else:
                colors_rec.append(COLORS['primary'])

        ax2.scatter(dish['total_orders'], dish['profit_margin'] * 100,
                   c=colors_rec, alpha=0.6, s=30,
                   edgecolors='grey', linewidth=0.3)
        ax2.set_xlabel('总订单数')
        ax2.set_ylabel('利润率 (%)')
        ax2.set_title('菜品评估矩阵\n(销量 × 利润)',
                     fontweight='bold', fontsize=11)
        ax2.grid(alpha=0.3)

        # 图例
        legend_elements = [
            mpatches.Patch(color=COLORS['success'], label='重点推广'),
            mpatches.Patch(color=COLORS['primary'], label='维持'),
            mpatches.Patch(color=COLORS['danger'], label='考虑替换'),
        ]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)

        ax3 = axes[0, 2]
        ax3.axis('off')
        ax3.set_xlim(0, 10)
        ax3.set_ylim(0, 9)

        abc_info = [
            ('Class A: ~58 dishes\n80% of total sales\n\nStrategy: Full prep\nBuffer: 1.15×',
             1.5, COLORS['primary']),
            ('Class B: ~70 dishes\n15% of total sales\n\nStrategy: Moderate\nBuffer: 1.05×',
             4.5, COLORS['secondary']),
            ('Class C: ~109 dishes\n5% of total sales\n\nStrategy: Rotation\nSmall batch',
             7.5, COLORS['accent']),
        ]

        for text, y, color in abc_info:
            ax3.add_patch(plt.Rectangle((0.5, y-0.8), 9, 1.6,
                          fill=True, facecolor=color, alpha=0.15,
                          edgecolor=color, linewidth=1.5))
            ax3.text(5, y, text, fontsize=8, fontweight='normal',
                    va='center', ha='center')

        ax3.set_title('ABC Classification Strategy',
                     fontweight='bold', fontsize=11)

        ax4 = axes[1, 0]
        ax4.axis('off')
        ax4.set_xlim(0, 10)
        ax4.set_ylim(0, 8)

        combos = [
            ('10 Yuan\nBasic', 1.7, 5, COLORS['accent'],
             '20-25%\n1主食+1荤+1素'),
            ('15 Yuan\nBalanced ★', 5.0, 5, COLORS['primary'],
             '45-55%\n1主食+1荤+1半荤+2素'),
            ('20 Yuan\nPremium', 8.3, 5, COLORS['success'],
             '20-30%\n1主食+2荤+1半荤+2素+特色'),
        ]

        for text, x, y, color, share in combos:
            size = 2.4 if '★' in text else 1.8
            ax4.add_patch(plt.Circle((x, y), size/2, fill=True,
                          facecolor=color, alpha=0.75,
                          edgecolor='white', linewidth=2))
            ax4.text(x, y, text + '\n' + share, ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white')

        ax4.set_title('Tiered Combo Strategy (from Problem 4)',
                     fontweight='bold', fontsize=11)

        ax5 = axes[1, 1]
        ax5.axis('off')

        # 基于前4题实际数据的关键指标
        key_metrics = [
            ('Daily Avg Orders', '274', COLORS['primary']),
            ('Avg Order Value', '11.36 yuan', COLORS['secondary']),
            ('Unique Dishes', '237', COLORS['accent']),
            ('Dish Categories', '5', COLORS['success']),
            ('Association Rules', '25', COLORS['warning']),
            ('Lunch Share', '99.2%', COLORS['purple']),
        ]

        y_positions = [7.5, 6.2, 4.9, 3.6, 2.3, 1.0]
        for (label, value, color), y in zip(key_metrics, y_positions):
            ax5.add_patch(plt.Rectangle((0.5, y-0.5), 8.5, 1.0,
                          fill=True, facecolor=color, alpha=0.1,
                          edgecolor=color, linewidth=1))
            ax5.text(1.0, y, f'{label}:', fontsize=9, fontweight='bold',
                    va='center')
            ax5.text(6.5, y, f'{value}', fontsize=10, fontweight='bold',
                    va='center', ha='right', color=color)

        ax5.set_xlim(0, 10)
        ax5.set_ylim(0, 9)
        ax5.set_title('Key Operational Metrics (Data-driven)',
                     fontweight='bold', fontsize=11)

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
                          facecolor=color, alpha=0.8,
                          edgecolor='white', linewidth=2))
            ax6.text(x, y, text, ha='center', va='center', fontsize=8,
                    fontweight='bold', color='white')

        # 循环箭头
        prev = cycle_steps[-1]
        for step in cycle_steps:
            ax6.annotate('', xy=(step[0], step[1]),
                        xytext=(prev[0], prev[1]),
                        arrowprops=dict(arrowstyle='->',
                                       connectionstyle='arc3,rad=0.3',
                                       color='grey', lw=1.5))
            prev = step

        ax6.text(5, 5, 'Continuous\nImprovement\nLoop',
                ha='center', va='center', fontsize=10,
                fontweight='bold', color='#333333')
        ax6.set_title('Data-Driven Operation Cycle',
                     fontweight='bold', fontsize=12)

        plt.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/p5_strategy_summary.png', dpi=300)
        plt.close()
        print('  已保存: p5_strategy_summary.png')

    def print_strategy_report(self):
        print('\n' + '=' * 70)
        print('               经营优化策略报告')
        print('=' * 70)

        sections = [
            ('一、备菜策略优化', 'preparation'),
            ('二、菜品结构优化', 'menu'),
            ('三、套餐推广策略', 'combo'),
            ('四、数字化运营建议', 'digital'),
            ('五、营养与 ESG 策略', 'nutrition_esg'),
        ]

        for title, key in sections:
            if key not in self.strategies:
                continue

            print(f'\n{title}')
            print('-' * 50)

            strategy = self.strategies[key]

            # 先输出数据分析摘要
            if 'data_insights' in strategy:
                print('\n  [数据分析]')
                for k, v in strategy['data_insights'].items():
                    print(f'    {k}: {v}')

            if 'nutrition_current' in strategy:
                print('\n  [营养现状]')
                for k, v in strategy['nutrition_current'].items():
                    print(f'    {k}: {v}')

            if 'waste_estimate' in strategy:
                print('\n  [浪费估算]')
                for k, v in strategy['waste_estimate'].items():
                    print(f'    {k}: {v}')

            # 输出策略建议
            if 'recommendations' in strategy:
                print('\n  [策略建议]')
                for i, rec in enumerate(strategy['recommendations'], 1):
                    print(f'\n  {i}. {rec["title"]}')
                    print(f'     说明: {rec["detail"]}')
                    if 'data_basis' in rec:
                        print(f'     依据: {rec["data_basis"]}')


if __name__ == '__main__':
    # 模块自检
    ps = Problem5Strategy()
    results = ps.run()
    ps.print_strategy_report()
