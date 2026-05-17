"""problem3_optimization.py — 问题3: 餐厅菜品优化模型与午餐备菜方案"""


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pulp import (LpProblem, LpMaximize, LpVariable, LpInteger,
                  LpStatus, lpSum, value, PULP_CBC_CMD)
import warnings
warnings.filterwarnings('ignore')

from data_loader import load_all_data
from config import (OUTPUT_DIR, COLORS, COLOR_CYCLE, RANDOM_SEED,
                     NUTRITION_PER_MEAL, NUTRITION_TOLERANCE,
                     WASTE_COST_RATIO, SHORTAGE_PENALTY_RATIO,
                     SAFETY_STOCK_FACTOR, MEAL_PLAN_START, MEAL_PLAN_END)
from utils import check_nutrition_balance


class Problem3Optimization:

    def __init__(self, loader=None):
        print('\n' + '=' * 60)
        print('问题3: 菜品备菜优化模型 (午餐)')
        print('=' * 60)

        if loader is None:
            self.loader = load_all_data()
        else:
            self.loader = loader

        # 获取已处理的数据
        self.dish_info = self.loader.get_dish_info()
        self.df_meal = self.loader.get_meal_data()
        self.df_trans = self.loader.get_transaction_data()
        self.df_daily = self.loader.get_daily_data()

        # 准备菜品统计 (含餐次偏好)
        self._prepare_dish_stats()

        self.results = {}

    def _prepare_dish_stats(self):
        df2 = self.loader.df2_raw
        df1 = self.loader.df1_raw

        # 融合餐次信息到菜品明细
        meal_info = df1[['indent_id', 'meal_period', 'date']].copy()
        meal_info = meal_info.drop_duplicates(subset='indent_id')
        df2_with_meal = df2.merge(meal_info, on='indent_id', how='left')

        # 检查合并结果
        if 'meal_period' not in df2_with_meal.columns:
            print('  警告: merge 后缺少 meal_period 列，添加默认值')
            df2_with_meal['meal_period'] = 'lunch'

        # 每种菜品的餐次统计
        dish_stats = df2_with_meal.groupby('dish_name').agg(
            total_count=('indent_details_id', 'count'),
            avg_price=('total_price', 'mean'),
            lunch_count=('meal_period', lambda x: (x == 'lunch').sum()),
            dinner_count=('meal_period', lambda x: (x == 'dinner').sum()),
        ).reset_index()

        # 按出现频率降序排列
        dish_stats = dish_stats.sort_values('total_count', ascending=False)

        # 计算偏好度 (归一化)
        total_meals = dish_stats['total_count'].sum()
        dish_stats['popularity'] = dish_stats['total_count'] / total_meals

        # 午餐偏好度 (用于优化模型)
        lunch_total = dish_stats['lunch_count'].sum()
        dish_stats['lunch_popularity'] = (
            dish_stats['lunch_count'] / max(lunch_total, 1)
        )

        # 晚餐偏好度 (仅用于参考)
        dinner_total = dish_stats['dinner_count'].sum()
        dish_stats['dinner_popularity'] = (
            dish_stats['dinner_count'] / max(dinner_total, 1)
        )

        self.dish_stats = dish_stats

        # 建立菜品名称到营养信息的映射 (用于高效查询)
        self.dish_nutrition = {}
        for _, row in self.dish_info.iterrows():
            self.dish_nutrition[row['dish_name']] = {
                'unit_price': row['unit_price'],
                'unit_cost': row['unit_cost'],
                'calories': row['calories'],
                'carbohydrates': row['carbohydrates'],
                'protein': row['protein'],
                'fat': row['fat'],
                'fiber': row['fiber'],
                'category': row['category'],
                'total_orders': row['total_orders'],
            }

    def _select_dishes_for_optimization(self, n_dishes=50):
        stats = self.dish_stats.copy()
        stats = stats.sort_values('lunch_count', ascending=False)

        selected = []
        categories = {}

        for _, row in stats.iterrows():
            dish_name = row['dish_name']
            if dish_name not in self.dish_nutrition:
                continue

            cat = self.dish_nutrition[dish_name].get('category', '其他')

            # 确保每类至少 3 种
            if cat not in categories:
                categories[cat] = 0

            if categories[cat] < 3 or len(selected) < n_dishes:
                if len(selected) < n_dishes:
                    selected.append(dish_name)
                    categories[cat] = categories.get(cat, 0) + 1

        return selected[:n_dishes]

    def optimize_meal(self, dishes, predicted_diners,
                      predicted_nutrition=None):
        n_dishes = len(dishes)

        prob = LpProblem("LunchPrepOptimization", LpMaximize)

        x = {}
        for i, dish in enumerate(dishes):
            x[i] = LpVariable(f"x_{i}", lowBound=0, cat=LpInteger)

        prices = []
        costs = []
        calories_list = []
        protein_list = []
        fat_list = []
        carbs_list = []
        fiber_list = []
        categories = []
        popularities = []

        for dish in dishes:
            nutri = self.dish_nutrition.get(dish, {})
            prices.append(nutri.get('unit_price', 5))
            costs.append(nutri.get('unit_cost', 2.5))
            calories_list.append(nutri.get('calories', 100))
            protein_list.append(nutri.get('protein', 5))
            fat_list.append(nutri.get('fat', 3))
            carbs_list.append(nutri.get('carbohydrates', 15))
            fiber_list.append(nutri.get('fiber', 1))
            categories.append(nutri.get('category', '其他'))

            # 午餐偏好度
            stats_row = self.dish_stats[
                self.dish_stats['dish_name'] == dish
            ]
            if len(stats_row) > 0:
                pop = stats_row['lunch_popularity'].values[0]
            else:
                pop = 0.01
            popularities.append(pop)

        # 人均期望菜品数 (每个顾客平均选 5-6 种菜品)
        avg_dishes_per_person = 5.5
        total_demand = predicted_diners * avg_dishes_per_person

        # 安全库存缓冲
        safe_buffer = SAFETY_STOCK_FACTOR * total_demand

        # 营养需求标准 (来自 config.py 的 NUTRITION_PER_MEAL['lunch'])
        if predicted_nutrition is None:
            nutrition_std = NUTRITION_PER_MEAL['lunch']
            target_calories = nutrition_std['calories'] * predicted_diners
            target_protein = nutrition_std['protein'] * predicted_diners
            target_fat = nutrition_std['fat'] * predicted_diners
            target_carbs = nutrition_std['carbohydrates'] * predicted_diners
            target_fiber = nutrition_std['fiber'] * predicted_diners
        else:
            target_calories = predicted_nutrition.get('calories', 200000)
            target_protein = predicted_nutrition.get('protein', 10000)
            target_fat = predicted_nutrition.get('fat', 7000)
            target_carbs = predicted_nutrition.get('carbohydrates', 23000)
            target_fiber = predicted_nutrition.get('fiber', 1500)

        # 对每种菜品构建线性表达式

        revenue_terms = []     # 期望销售收入
        cost_terms = []        # 备菜成本
        waste_terms = []       # 浪费成本
        popularity_terms = []  # 偏好奖励

        for i in range(n_dishes):
            # 成本项
            cost_terms.append(costs[i] * x[i])

            # 收入项: s_i = min(x_i, d_i) 的线性化
            # d_i = total_demand × pop_i 为菜品 i 的期望需求量
            expected_demand_i = total_demand * popularities[i]

            # 辅助变量 s_i: 实际销售数量 ≤ min(备菜量, 需求)
            s = LpVariable(f"s_{i}", lowBound=0, cat=LpInteger)
            prob += s <= x[i]                     # s ≤ 备菜量
            prob += s <= expected_demand_i         # s ≤ 期望需求
            revenue_terms.append(prices[i] * s)

            # 浪费成本: w_i ≥ x_i - d_i 且 w_i ≥ 0
            w = LpVariable(f"w_{i}", lowBound=0, cat=LpInteger)
            prob += w >= x[i] - expected_demand_i
            waste_terms.append(costs[i] * WASTE_COST_RATIO * w)

            # 偏好奖励项 (鼓励准备受欢迎的菜品)
            popularity_terms.append(popularities[i] * x[i])

        # 目标函数加权
        gamma_popularity = 0.1  # 偏好奖励权值

        prob += (
            lpSum(revenue_terms)                              # 收入
            - lpSum(cost_terms)                               # 成本
            - lpSum(waste_terms)                              # 浪费
            + gamma_popularity * lpSum(popularity_terms)      # 偏好奖励
        )


        # (a) 总份量约束
        prob += (
            lpSum([x[i] for i in range(n_dishes)])
            >= total_demand + safe_buffer
        )
        prob += (
            lpSum([x[i] for i in range(n_dishes)])
            <= total_demand * (1 + SAFETY_STOCK_FACTOR * 2)
        )

        # (b) 营养供给约束
        # 热量
        prob += (
            lpSum([calories_list[i] * x[i] for i in range(n_dishes)])
            >= target_calories * (1 - NUTRITION_TOLERANCE)
        )
        prob += (
            lpSum([calories_list[i] * x[i] for i in range(n_dishes)])
            <= target_calories * (1 + NUTRITION_TOLERANCE)
        )
        # 蛋白质
        prob += (
            lpSum([protein_list[i] * x[i] for i in range(n_dishes)])
            >= target_protein * (1 - NUTRITION_TOLERANCE)
        )
        # 脂肪
        prob += (
            lpSum([fat_list[i] * x[i] for i in range(n_dishes)])
            >= target_fat * (1 - NUTRITION_TOLERANCE)
        )
        # 碳水化合物
        prob += (
            lpSum([carbs_list[i] * x[i] for i in range(n_dishes)])
            >= target_carbs * (1 - NUTRITION_TOLERANCE)
        )
        # 膳食纤维
        prob += (
            lpSum([fiber_list[i] * x[i] for i in range(n_dishes)])
            >= target_fiber * (1 - NUTRITION_TOLERANCE)
        )

        # (c) 菜品多样性约束 (按类别)
        cat_indices = {}
        for i, cat in enumerate(categories):
            if cat not in cat_indices:
                cat_indices[cat] = []
            cat_indices[cat].append(i)

        min_dishes_per_category = {
            '主食': 1, '荤菜': 3, '半荤半素': 2, '素菜': 3
        }

        for cat, indices in cat_indices.items():
            min_count = min_dishes_per_category.get(cat, 1)
            # 该类别至少准备 min_count × 10 份
            prob += (
                lpSum([x[i] for i in indices]) >= min_count * 10
            )

        # (d) 单菜品备菜上下限
        min_per_dish = max(5, int(total_demand * 0.005))
        max_per_dish = int(total_demand * 0.25)

        for i in range(n_dishes):
            prob += x[i] >= min_per_dish
            prob += x[i] <= max_per_dish

        prob.solve(PULP_CBC_CMD(msg=False, timeLimit=120))

        status = LpStatus[prob.status]

        if status != 'Optimal':
            print(f'    警告: 求解状态 = {status}, '
                  f'目标值 = {value(prob.objective):.0f}')

        solution = {
            'status': status,
            'objective_value': value(prob.objective),
            'dishes': [],
            'total_servings': 0,
            'total_cost': 0,
            'total_expected_revenue': 0,
            'nutrition_summary': {},
            'waste_estimate': 0,
        }

        total_cal = 0
        total_protein = 0
        total_fat = 0
        total_carbs = 0
        total_fiber = 0

        for i, dish in enumerate(dishes):
            servings = int(value(x[i]))
            if servings > 0:
                expected_demand_i = total_demand * popularities[i]
                expected_sales = min(servings, expected_demand_i)
                revenue = prices[i] * expected_sales
                cost = costs[i] * servings
                waste = (
                    max(0, servings - expected_demand_i)
                    * costs[i] * WASTE_COST_RATIO
                )

                solution['dishes'].append({
                    'name': dish,
                    'category': categories[i],
                    'servings': servings,
                    'unit_price': prices[i],
                    'unit_cost': costs[i],
                    'expected_sales': expected_sales,
                    'expected_revenue': revenue,
                    'waste_risk': max(0, servings - expected_sales),
                    'popularity': popularities[i],
                    'calories': calories_list[i] * servings,
                    'protein': protein_list[i] * servings,
                    'fat': fat_list[i] * servings,
                    'carbs': carbs_list[i] * servings,
                })

                solution['total_servings'] += servings
                solution['total_cost'] += cost
                solution['total_expected_revenue'] += revenue
                solution['waste_estimate'] += waste

                total_cal += calories_list[i] * servings
                total_protein += protein_list[i] * servings
                total_fat += fat_list[i] * servings
                total_carbs += carbs_list[i] * servings
                total_fiber += fiber_list[i] * servings

        # 营养汇总
        solution['nutrition_summary'] = {
            'calories': total_cal,
            'protein': total_protein,
            'fat': total_fat,
            'carbohydrates': total_carbs,
            'fiber': total_fiber,
        }

        # 预期利润
        solution['expected_profit'] = (
            solution['total_expected_revenue']
            - solution['total_cost']
            - solution['waste_estimate']
        )

        # 营养均衡度评分
        balance = check_nutrition_balance(
            total_cal / max(predicted_diners, 1),
            total_protein / max(predicted_diners, 1),
            total_fat / max(predicted_diners, 1),
            total_carbs / max(predicted_diners, 1),
            total_fiber / max(predicted_diners, 1),
        )
        solution['nutrition_balance'] = balance

        return solution

    def run(self, prediction_csv=None):
        print('\n>>> 3.1 加载预测数据')

        if prediction_csv is not None and os.path.exists(prediction_csv):
            pred_df = pd.read_csv(prediction_csv, index_col=0, parse_dates=True)
            print(f'  从问题2加载预测: {prediction_csv}')
            print(f'  预测可用日期: {len(pred_df)} 天')
            use_p2_predictions = True
        else:
            pred_df = None
            use_p2_predictions = False
            if prediction_csv is not None:
                print(f'  警告: 预测文件不存在 ({prediction_csv})，回退到历史均值')

        # 确定优化日期范围
        plan_dates = pd.date_range(
            start=MEAL_PLAN_START,
            end=MEAL_PLAN_END,
            freq='D'
        )
        plan_dates = plan_dates[plan_dates.dayofweek < 5]

        print(f'  计划日期: {len(plan_dates)} 个工作日 (仅午餐)')
        for d in plan_dates:
            print(f'    {d.strftime("%Y-%m-%d")} ({d.day_name()})')

        # 获取预测的营养需求
        predicted_nutrition = self._get_predicted_nutrition(pred_df)

        print('\n>>> 3.2 选择优化菜品')
        lunch_dishes = self._select_dishes_for_optimization(n_dishes=50)
        print(f'  午餐可选菜品: {len(lunch_dishes)} 种')

        cats_included = {}
        for dish in lunch_dishes:
            cat = self.dish_nutrition.get(dish, {}).get('category', '其他')
            cats_included[cat] = cats_included.get(cat, 0) + 1
        for cat, cnt in cats_included.items():
            print(f'    {cat}: {cnt} 种')

        print(f'\n>>> 3.3 逐日优化午餐备菜方案')
        print(f'  预测来源: {"问题2 SARIMA预测" if use_p2_predictions else "历史同星期均值"}')

        all_plans = []

        for date in plan_dates:
            # 获取预测就餐人数
            predicted_diners = self._get_predicted_diners(date, pred_df)

            print(f'\n  {date.strftime("%Y-%m-%d")} '
                  f'({date.day_name()}) 午餐 '
                  f'(预估 {predicted_diners:.0f} 人)')

            lunch_solution = self.optimize_meal(
                lunch_dishes, predicted_diners,
                predicted_nutrition=predicted_nutrition
            )
            lunch_solution['date'] = date.strftime('%Y-%m-%d')
            lunch_solution['meal'] = 'lunch'
            lunch_solution['predicted_diners'] = int(predicted_diners)
            all_plans.append(lunch_solution)

            print(f'    求解状态: {lunch_solution["status"]}')
            print(f'    备菜总份数: {lunch_solution["total_servings"]:.0f}')
            print(f'    预期利润: {lunch_solution["expected_profit"]:.0f} 元')
            print(f'    营养均衡度: '
                  f'{lunch_solution["nutrition_balance"]["overall_balance"]:.2f}')

        self.results['meal_plans'] = all_plans

        self._plot_meal_plans(all_plans, plan_dates)
        self._print_detailed_plans(all_plans)

        print('\n问题3 午餐备菜优化完成!')
        return self.results

    def _get_predicted_diners(self, date, pred_df=None):
        if pred_df is not None:
            date_str = date.strftime('%Y-%m-%d')
            if date_str in pred_df.index.astype(str):
                val = pred_df.loc[date_str, 'total_orders']
                return float(val) if not pd.isna(val) else self._fallback_diners(date)

        return self._fallback_diners(date)

    def _fallback_diners(self, date):
        df = self.df_daily[self.df_daily['total_orders'] > 0].copy()
        dow = date.dayofweek
        same_dow = df[df['day_of_week'] == dow]
        if len(same_dow) > 0:
            return same_dow['total_orders'].mean()
        return df['total_orders'].mean()

    def _get_predicted_nutrition(self, pred_df=None):
        if pred_df is not None and 'total_calories' in pred_df.columns:
            return {
                'calories': pred_df['total_calories'].mean(),
                'protein': pred_df['total_protein'].mean(),
                'fat': pred_df['total_fat'].mean(),
                'carbohydrates': pred_df['total_carbohydrates'].mean(),
                'fiber': self.df_daily[self.df_daily['total_orders'] > 0]['total_fiber'].mean(),
            }

        df = self.df_daily[self.df_daily['total_orders'] > 0].copy()
        return {
            'calories': df['total_calories'].mean(),
            'protein': df['total_protein'].mean(),
            'fat': df['total_fat'].mean(),
            'carbohydrates': df['total_carbohydrates'].mean(),
            'fiber': df['total_fiber'].mean(),
        }

    def _plot_meal_plans(self, all_plans, plan_dates):
        fig = plt.figure(figsize=(16, 10))

        # 获取午餐计划
        lunch_plans = all_plans
        x = range(len(lunch_plans))
        labels = [p['date'][-5:] for p in lunch_plans]

        ax1 = fig.add_subplot(2, 2, 1)

        bars = ax1.bar(x, [p['total_servings'] for p in lunch_plans],
                      color=COLORS['primary'], alpha=0.8,
                      edgecolor='white', linewidth=1)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=30)
        ax1.set_ylabel('总备菜份数')
        ax1.set_title('每日午餐备菜量',
                      fontweight='bold', fontsize=11)
        ax1.grid(axis='y', alpha=0.3)

        # 添加数值标注
        for bar, plan in zip(bars, lunch_plans):
            ax1.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 10,
                    f'{plan["total_servings"]:.0f}',
                    ha='center', fontsize=8, fontweight='bold')

        ax2 = fig.add_subplot(2, 2, 2)
        margins = []
        for p in lunch_plans:
            rev = p['total_expected_revenue']
            cost = p['total_cost']
            margin = (rev - cost) / rev * 100 if rev > 0 else 0
            margins.append(margin)

        bars2 = ax2.bar(labels, margins, color=COLORS['success'], alpha=0.7,
                       edgecolor='white', linewidth=1)
        ax2.set_ylabel('利润率 (%)')
        ax2.set_title('预期午餐利润率', fontweight='bold',
                      fontsize=11)
        ax2.axhline(y=30, color=COLORS['accent'], linestyle='--', alpha=0.5,
                   linewidth=1, label='30% 目标')
        ax2.legend(fontsize=9)
        ax2.tick_params(axis='x', rotation=30)

        # 添加数值标注
        for bar, m in zip(bars2, margins):
            ax2.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f'{m:.1f}%', ha='center', fontsize=8)

        ax3 = fig.add_subplot(2, 2, 3)
        if lunch_plans:
            sample = lunch_plans[0]['dishes']
            cats = {}
            for d in sample:
                cat = d['category']
                cats[cat] = cats.get(cat, 0) + d['servings']

            colors_pie = [
                COLORS['primary'], COLORS['success'], COLORS['accent'],
                COLORS['warning'], COLORS['purple']
            ]
            wedges, texts, autotexts = ax3.pie(
                cats.values(), labels=cats.keys(), autopct='%1.1f%%',
                colors=colors_pie[:len(cats)], startangle=90,
                explode=[0.03]*len(cats),
                textprops={'fontsize': 9}
            )
            ax3.set_title(
                f'Category Distribution\n({lunch_plans[0]["date"]} Lunch)',
                fontweight='bold', fontsize=11
            )

        # 使用中间日期的方案 (第3天或第1天)
        mid_idx = min(2, len(lunch_plans) - 1)
        plan_for_radar = lunch_plans[mid_idx]
        nutrition = plan_for_radar['nutrition_summary']
        diners = plan_for_radar.get('predicted_diners', 274)

        # 实际人均营养
        actual_per_person = {
            'calories': nutrition['calories'] / max(diners, 1),
            'protein': nutrition['protein'] / max(diners, 1),
            'fat': nutrition['fat'] / max(diners, 1),
            'carbohydrates': nutrition['carbohydrates'] / max(diners, 1),
            'fiber': nutrition['fiber'] / max(diners, 1),
        }

        # 标准人均营养 (来自 DRIs)
        targets_per_person = NUTRITION_PER_MEAL['lunch']

        # 归一化到 [0, 1] 区间
        nutrient_keys = ['calories', 'protein', 'fat', 'carbohydrates', 'fiber']
        nutrient_labels_zh = ['Calories', 'Protein', 'Fat', 'Carbs', 'Fiber']

        ax4 = fig.add_subplot(2, 2, 4, projection='polar')
        angles = np.linspace(0, 2 * np.pi, len(nutrient_keys),
                            endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        target_values = [1.0] * len(nutrient_keys)
        actual_values = [
            min(1.5, actual_per_person.get(k, 0)
                / max(targets_per_person.get(k, 1), 1))
            for k in nutrient_keys
        ]
        target_values += target_values[:1]
        actual_values += actual_values[:1]

        ax4.fill(angles, target_values, alpha=0.15, color=COLORS['primary'],
                label='DRIs 标准')
        ax4.plot(angles, actual_values, 'o-', color=COLORS['accent'],
                linewidth=2, label='实际供给')
        ax4.fill(angles, actual_values, alpha=0.2, color=COLORS['accent'])
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(nutrient_labels_zh, fontsize=9)
        ax4.set_title(
            f'Nutrition Satisfaction per Person\n'
            f'({plan_for_radar["date"]} Lunch, '
            f'{diners:.0f} diners)',
            fontweight='bold', fontsize=10, pad=20
        )
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)

        plt.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/p3_meal_plans.png', dpi=300)
        plt.close()
        print('  已保存: p3_meal_plans.png')

    def _print_detailed_plans(self, all_plans):
        print('\n' + '=' * 80)
        print('详细备菜方案 (2025年5月6日-12日 工作日 午餐)')
        print('=' * 80)

        for plan in all_plans:
            print(f'\n--- {plan["date"]} 午餐 ---')
            print(f'预计就餐人数: {plan.get("predicted_diners", "N/A")}')
            print(f'备菜总份数: {plan["total_servings"]:.0f}')
            print(f'预期收入: {plan["total_expected_revenue"]:.0f} 元')
            print(f'备菜成本: {plan["total_cost"]:.0f} 元')
            print(f'预期利润: {plan["expected_profit"]:.0f} 元')
            print(f'营养均衡度: {plan["nutrition_balance"]["overall_balance"]:.2f}')

            print(f'\n{"菜品名称":<22} {"类别":<8} {"备菜":>5} '
                  f'{"单价":>5} {"预期销量":>7} {"预期收入":>7}')
            print('-' * 60)

            for d in plan['dishes'][:15]:
                print(f'{d["name"]:<22} {d["category"]:<8} '
                      f'{d["servings"]:>5} {d["unit_price"]:>5.1f} '
                      f'{d["expected_sales"]:>7.0f} '
                      f'{d["expected_revenue"]:>7.0f}')

            if len(plan['dishes']) > 15:
                print(f'  ... (共 {len(plan["dishes"])} 种菜品)')

            # 营养汇总
            nutri = plan['nutrition_summary']
            print(f'\n营养汇总: '
                  f'热量={nutri["calories"]:.0f} kcal, '
                  f'蛋白={nutri["protein"]:.0f} g, '
                  f'脂肪={nutri["fat"]:.0f} g, '
                  f'碳水={nutri["carbohydrates"]:.0f} g, '
                  f'纤维={nutri["fiber"]:.0f} g')

            # 营养均衡详情
            bal = plan['nutrition_balance']
            print(f'营养均衡: '
                  f'蛋白供能比={bal["protein_ratio"]:.1%}, '
                  f'脂肪供能比={bal["fat_ratio"]:.1%}, '
                  f'碳水供能比={bal["carbs_ratio"]:.1%}')

        all_rows = []
        for plan in all_plans:
            for d in plan['dishes']:
                all_rows.append({
                    '日期': plan['date'],
                    '餐次': '午餐',
                    '菜品名称': d['name'],
                    '类别': d['category'],
                    '备菜份数': d['servings'],
                    '单价_元': round(d['unit_price'], 2),
                    '预期销量': round(d['expected_sales'], 0),
                    '预期收入_元': round(d['expected_revenue'], 2),
                    '浪费风险_份': round(d['waste_risk'], 0),
                })

        plan_df = pd.DataFrame(all_rows)
        plan_df.to_csv(
            f'{OUTPUT_DIR}/p3_meal_plan_detail.csv',
            index=False, encoding='utf-8-sig'
        )
        print(f'\n完整方案已保存: p3_meal_plan_detail.csv')


if __name__ == '__main__':
    # 模块自检
    opt = Problem3Optimization()
    results = opt.run()
