"""
problem3_optimization.py — 问题3：餐厅菜品优化模型与备菜方案
==========================================================
题目要求：
  为提高餐厅营业利润，综合考虑各类营养素需求、该餐厅消费群体的消费习惯
  以及菜品多样性等因素，建立餐厅菜品优化模型，并给出2025年5月6日至5月12日
  期间每个工作日的备菜方案（午餐、晚餐需分别给出）。

解题思路：
  1. 模型类型：混合整数线性规划（MILP）
  2. 决策变量：
     x_{i,t,m} — 第t天第m餐次中菜品i的备菜份数（整数）
  3. 目标函数：
     max 利润 = 销售收入 - 制作成本 - 浪费成本 - 缺货惩罚
  4. 约束条件：
     - 总份量约束（满足预测需求）
     - 营养供给约束（各类营养素达到标准）
     - 菜品多样性约束（各类别菜品数量上下限）
     - 单菜品备菜上下限约束
     - 整数约束
  5. 求解方法：使用PuLP调用CBC求解器

数学模型（参考 [7][8][9]）：
  max Z = sum(p_i * min(x_i, d_i)) - sum(c_i * x_i) 
         - sum(h_i * max(x_i - d_i, 0)) - sum(b_i * max(d_i - x_i, 0))
  
  s.t.
    sum(x_i) >= D_total           (总需求满足)
    sum(a_ij * x_i) >= R_j        (营养需求满足)
    L_k <= sum_{i in C_k}(x_i) <= U_k  (类别多样性)
    l_i <= x_i <= u_i             (单品上下限)
    x_i ∈ Z+                      (整数约束)

参考文献：
  [7] 黄健等. "中国海洋大学食堂菜谱的优化模型研究"
      应用数学进展, 2018. https://www.hanspub.org/journal/PaperInformation?paperID=23869
  [8] Padovan M. et al. "Optimized menu formulation to enhance nutritional goals"
      BMC Nutrition, 2023. https://link.springer.com/article/10.1186/s40795-023-00705-0
  [9] Cohen J.F.W. et al. "Improving school lunch menus with multi-objective optimisation"
      Public Health Nutrition, 2023.
  [10] Gazendam A. et al. "A Review of the Use of Linear Programming to Optimize Diets"
       Frontiers in Nutrition, 2018. https://www.frontiersin.org/articles/10.3389/fnut.2018.00048
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pulp import (LpProblem, LpMaximize, LpVariable, LpInteger, 
                  LpStatus, lpSum, value, PULP_CBC_CMD)
import warnings
warnings.filterwarnings('ignore')

from data_loader import load_all_data
from config import (OUTPUT_DIR, COLORS, RANDOM_SEED,
                    NUTRITION_PER_MEAL, NUTRITION_TOLERANCE,
                    WASTE_COST_RATIO, SHORTAGE_PENALTY_RATIO,
                    SAFETY_STOCK_FACTOR, MEAL_PLAN_START, MEAL_PLAN_END,
                    LUNCH_START, LUNCH_END, DINNER_START, DINNER_END)
from utils import check_nutrition_balance


class Problem3Optimization:
    """
    问题3：备菜优化模型
    
    使用混合整数线性规划（MILP）为每一天的午餐和晚餐
    分别生成最优备菜方案。
    """
    
    def __init__(self, loader=None):
        """初始化"""
        print('\n' + '=' * 60)
        print('问题3：菜品备菜优化模型')
        print('=' * 60)
        
        if loader is None:
            self.loader = load_all_data()
        else:
            self.loader = loader
        
        self.dish_info = self.loader.get_dish_info()
        self.df_meal = self.loader.get_meal_data()
        self.df_trans = self.loader.get_transaction_data()
        self.df_daily = self.loader.get_daily_data()
        
        # 获取有附件2明细的订单的菜品偏好统计
        self._prepare_dish_stats()
        
        self.results = {}
        
    def _prepare_dish_stats(self):
        """
        准备菜品统计数据
        
        从附件2的交易明细中提取：
        - 每种菜品的日销量（午餐/晚餐分别）
        - 菜品偏好概率（用于估计各菜品的需求比例）
        """
        df2 = self.loader.df2_raw
        df1 = self.loader.df1_raw
        
        # 融合餐次信息到菜品明细
        # 使用 df1 的已处理数据获取餐次信息
        meal_info = df1[['indent_id', 'meal_period', 'date']].copy()
        meal_info = meal_info.drop_duplicates(subset='indent_id')
        df2_with_meal = df2.merge(meal_info, on='indent_id', how='left')
        
        # 验证列存在
        if 'meal_period' not in df2_with_meal.columns:
            print('  警告: merge后缺少meal_period列，添加默认值')
            df2_with_meal['meal_period'] = 'lunch'

        # 每种菜品的总体统计
        dish_stats = df2_with_meal.groupby('dish_name').agg(
            total_count=('indent_details_id', 'count'),
            avg_price=('total_price', 'mean'),
            lunch_count=('meal_period', lambda x: (x == 'lunch').sum()),
            dinner_count=('meal_period', lambda x: (x == 'dinner').sum()),
        ).reset_index()
        
        # 按菜品出现频率降序排列
        dish_stats = dish_stats.sort_values('total_count', ascending=False)
        
        # 计算偏好概率
        total_meals = dish_stats['total_count'].sum()
        dish_stats['popularity'] = dish_stats['total_count'] / total_meals
        
        # 午餐偏好
        lunch_total = dish_stats['lunch_count'].sum()
        dish_stats['lunch_popularity'] = dish_stats['lunch_count'] / max(lunch_total, 1)
        
        # 晚餐偏好
        dinner_total = dish_stats['dinner_count'].sum()
        dish_stats['dinner_popularity'] = dish_stats['dinner_count'] / max(dinner_total, 1)
        
        self.dish_stats = dish_stats
        
        # 建立菜品名称到营养信息的映射
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
        
    def _select_dishes_for_optimization(self, meal_period='lunch', n_dishes=50):
        """
        选择参与优化的菜品列表
        
        优先选择：
        1. 该餐次历史销量高的菜品
        2. 覆盖各类别（主食、荤菜、半荤半素、素菜）
        
        Args:
            meal_period: 'lunch' or 'dinner'
            n_dishes: 最多选择的菜品数量
            
        Returns:
            list: 菜品名称列表
        """
        stats = self.dish_stats.copy()
        
        if meal_period == 'lunch':
            stats = stats.sort_values('lunch_count', ascending=False)
        else:
            stats = stats.sort_values('dinner_count', ascending=False)
        
        # 确保每个类别至少有一定数量的菜品
        selected = []
        categories = {}
        
        for _, row in stats.iterrows():
            dish_name = row['dish_name']
            if dish_name not in self.dish_nutrition:
                continue
            
            cat = self.dish_nutrition[dish_name].get('category', '其他')
            
            # 每类至少选2个
            if cat not in categories:
                categories[cat] = 0
            
            if categories[cat] < 3 or len(selected) < n_dishes:
                if len(selected) < n_dishes:
                    selected.append(dish_name)
                    categories[cat] = categories.get(cat, 0) + 1
        
        return selected[:n_dishes]
    
    def optimize_meal(self, dishes, meal_period, predicted_diners, 
                      predicted_nutrition=None, min_dishes_per_category=None):
        """
        核心优化函数：使用MILP求解最优备菜方案
        
        数学公式见文件头部说明
        
        Args:
            dishes: 可选菜品列表
            meal_period: 'lunch' or 'dinner'
            predicted_diners: 预测的就餐人数
            predicted_nutrition: 预测的营养素需求总量 dict
            min_dishes_per_category: 每类最少的菜品数量
            
        Returns:
            dict: 优化结果，包含备菜方案和关键指标
        """
        n_dishes = len(dishes)
        
        # ---- 创建优化问题 ----
        prob = LpProblem("MealPrepOptimization", LpMaximize)
        
        # ---- 决策变量：每种菜品的备菜份数 ----
        x = {}
        for i, dish in enumerate(dishes):
            x[i] = LpVariable(f"x_{i}", lowBound=0, cat=LpInteger)
        
        # ---- 参数准备 ----
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
            
            # 偏好度
            stats_row = self.dish_stats[self.dish_stats['dish_name'] == dish]
            if len(stats_row) > 0:
                if meal_period == 'lunch':
                    pop = stats_row['lunch_popularity'].values[0]
                else:
                    pop = stats_row['dinner_popularity'].values[0]
            else:
                pop = 0.01
            popularities.append(pop)
        
        # ---- 预测需求参数 ----
        # 人均期望菜品数（每个顾客平均选5-6个菜）
        avg_dishes_per_person = 5.5
        total_demand = predicted_diners * avg_dishes_per_person
        
        # 安全库存缓冲
        safe_buffer = SAFETY_STOCK_FACTOR * total_demand
        
        # 营养需求标准
        if predicted_nutrition is None:
            nutrition_std = NUTRITION_PER_MEAL[meal_period]
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
        
        # ---- 目标函数 ----
        # max: sum(p_i * min(x_i, d_i*pop_i)) - sum(c_i * x_i) 
        #       - waste_cost - shortage_penalty
        
        # 线性化处理：收入项用期望销量 min(x_i, 预测需求量_i)
        # 由于MILP中min函数需要线性化，这里采用简化处理：
        # 假设实际销售 = min(x_i, 预计需求量)，其中预计需求量 = total_demand * pop_i
        
        revenue_terms = []
        cost_terms = []
        waste_terms = []
        shortage_terms = []
        popularity_terms = []
        
        for i in range(n_dishes):
            # 成本项
            cost_terms.append(costs[i] * x[i])
            
            # 收入项：简化为 min(备菜量, 预计需求) * 价格
            # 对于MILP，使用辅助变量和约束来线性化
            expected_demand_i = total_demand * popularities[i]
            
            # 销售数量 s_i = min(x_i, d_i)
            s = LpVariable(f"s_{i}", lowBound=0, cat=LpInteger)
            prob += s <= x[i]
            prob += s <= expected_demand_i  # 浮点数约束，PuLP自动处理
            
            revenue_terms.append(prices[i] * s)
            
            # 剩余浪费 = max(x_i - d_i, 0)，线性化为 w_i >= x_i - d_i, w_i >= 0
            w = LpVariable(f"w_{i}", lowBound=0, cat=LpInteger)
            prob += w >= x[i] - expected_demand_i
            
            waste_terms.append(costs[i] * WASTE_COST_RATIO * w)
            
            # 缺货惩罚 = max(d_i - x_i, 0)（仅在目标函数中处理）
            # 通过收入函数中的 s_i = min(x_i, d_i) 间接体现
            
            # 偏好得分（鼓励准备受欢迎的菜品）
            popularity_terms.append(popularities[i] * x[i])
        
        # 综合目标函数
        gamma_revenue = 1.0
        gamma_cost = 1.0
        gamma_waste = 1.0
        gamma_popularity = 0.1
        
        prob += (
            gamma_revenue * lpSum(revenue_terms)
            - gamma_cost * lpSum(cost_terms)
            - gamma_waste * lpSum(waste_terms)
            + gamma_popularity * lpSum(popularity_terms)
        )
        
        # ---- 约束条件 ----
        
        # 根据就餐人数规模调整约束策略
        is_small_meal = (predicted_diners < 50)
        
        # 1. 总份量约束
        # 小规模餐次（晚餐）：放松上下限
        if is_small_meal:
            prob += lpSum([x[i] for i in range(n_dishes)]) >= total_demand * 0.8
            prob += lpSum([x[i] for i in range(n_dishes)]) <= total_demand * 2.0
        else:
            prob += lpSum([x[i] for i in range(n_dishes)]) >= total_demand + safe_buffer
            prob += lpSum([x[i] for i in range(n_dishes)]) <= total_demand * (1 + SAFETY_STOCK_FACTOR * 2)
        
        # 2. 营养供给约束
        if is_small_meal:
            # 小规模餐次：使用宽松的软约束（营养目标作为参考而非硬约束）
            # 只做下限约束，不做上限约束，允许较大偏差
            prob += lpSum([calories_list[i] * x[i] for i in range(n_dishes)]) >= target_calories * 0.5
            prob += lpSum([protein_list[i] * x[i] for i in range(n_dishes)]) >= target_protein * 0.5
            prob += lpSum([fat_list[i] * x[i] for i in range(n_dishes)]) >= target_fat * 0.3
            prob += lpSum([carbs_list[i] * x[i] for i in range(n_dishes)]) >= target_carbs * 0.5
            prob += lpSum([fiber_list[i] * x[i] for i in range(n_dishes)]) >= target_fiber * 0.3
        else:
            prob += lpSum([calories_list[i] * x[i] for i in range(n_dishes)]) >= target_calories * (1 - NUTRITION_TOLERANCE)
            prob += lpSum([calories_list[i] * x[i] for i in range(n_dishes)]) <= target_calories * (1 + NUTRITION_TOLERANCE)
            prob += lpSum([protein_list[i] * x[i] for i in range(n_dishes)]) >= target_protein * (1 - NUTRITION_TOLERANCE)
            prob += lpSum([fat_list[i] * x[i] for i in range(n_dishes)]) >= target_fat * (1 - NUTRITION_TOLERANCE)
            prob += lpSum([carbs_list[i] * x[i] for i in range(n_dishes)]) >= target_carbs * (1 - NUTRITION_TOLERANCE)
            prob += lpSum([fiber_list[i] * x[i] for i in range(n_dishes)]) >= target_fiber * (1 - NUTRITION_TOLERANCE)
        
        # 3. 菜品多样性约束（按类别）
        cat_indices = {}
        for i, cat in enumerate(categories):
            if cat not in cat_indices:
                cat_indices[cat] = []
            cat_indices[cat].append(i)
        
        cat_constraints = {}
        if min_dishes_per_category is None:
            min_dishes_per_category = {'主食': 1, '荤菜': 3, '半荤半素': 2, '素菜': 3}
        
        for cat, indices in cat_indices.items():
            min_count = min_dishes_per_category.get(cat, 1)
            # 至少有min_count种菜品的备菜量 > 0
            prob += lpSum([x[i] for i in indices]) >= min_count * 10  # 至少10份
        
        # 4. 单菜品备菜上下限
        if is_small_meal:
            min_per_dish = 1
            max_per_dish = max(10, int(total_demand * 0.5))
        else:
            min_per_dish = max(5, int(total_demand * 0.005))
            max_per_dish = int(total_demand * 0.25)
        
        for i in range(n_dishes):
            prob += x[i] >= min_per_dish
            prob += x[i] <= max_per_dish
        
        # ---- 求解 ----
        prob.solve(PULP_CBC_CMD(msg=False, timeLimit=120))
        
        # ---- 提取结果 ----
        status = LpStatus[prob.status]
        
        if status != 'Optimal':
            print(f'    警告: 求解状态 = {status}, 目标值 = {value(prob.objective):.0f}')
        
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
                waste = max(0, servings - expected_demand_i) * costs[i] * WASTE_COST_RATIO
                
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
        
        solution['nutrition_summary'] = {
            'calories': total_cal,
            'protein': total_protein,
            'fat': total_fat,
            'carbohydrates': total_carbs,
            'fiber': total_fiber,
        }
        
        solution['expected_profit'] = (
            solution['total_expected_revenue'] 
            - solution['total_cost'] 
            - solution['waste_estimate']
        )
        
        # 计算营养均衡度
        balance = check_nutrition_balance(
            total_cal / max(predicted_diners, 1),
            total_protein / max(predicted_diners, 1),
            total_fat / max(predicted_diners, 1),
            total_carbs / max(predicted_diners, 1),
            total_fiber / max(predicted_diners, 1),
        )
        solution['nutrition_balance'] = balance
        
        return solution
    
    def run(self):
        """运行完整的备菜优化流程"""
        print('\n>>> 3.1 准备优化数据')
        
        # 获取2025年5月6-12日的工作日
        plan_dates = pd.date_range(
            start=MEAL_PLAN_START,
            end=MEAL_PLAN_END,
            freq='D'
        )
        # 筛选工作日（周一至周五）
        plan_dates = plan_dates[plan_dates.dayofweek < 5]
        
        print(f'  计划日期: {len(plan_dates)} 个工作日')
        for d in plan_dates:
            print(f'    {d.strftime("%Y-%m-%d")} ({d.day_name()})')
        
        # 使用问题2的预测结果（模拟预测数据）
        # 在实际项目中，应直接调用Problem2的结果
        predicted_nutrition = self._get_predicted_nutrition()
        
        print('\n>>> 3.2 选择优化菜品')
        lunch_dishes = self._select_dishes_for_optimization('lunch', n_dishes=40)
        dinner_dishes = self._select_dishes_for_optimization('dinner', n_dishes=30)
        print(f'  午餐可选菜品: {len(lunch_dishes)} 种')
        print(f'  晚餐可选菜品: {len(dinner_dishes)} 种')
        
        print('\n>>> 3.3 逐日逐餐优化备菜方案')
        
        all_plans = []
        
        for date in plan_dates:
            dow = date.dayofweek
            
            # 获取该星期的预测就餐人数
            predicted_diners = self._get_predicted_diners(dow)
            
            # ---- 午餐优化 ----
            print(f'\n  {date.strftime("%Y-%m-%d")} 午餐 (预估{predicted_diners:.0f}人)')
            lunch_solution = self.optimize_meal(
                lunch_dishes, 'lunch', predicted_diners,
                predicted_nutrition=predicted_nutrition
            )
            lunch_solution['date'] = date.strftime('%Y-%m-%d')
            lunch_solution['meal'] = 'lunch'
            all_plans.append(lunch_solution)
            
            print(f'    目标值: {lunch_solution["objective_value"]:.0f}')
            print(f'    备菜总份数: {lunch_solution["total_servings"]:.0f}')
            print(f'    预期利润: {lunch_solution["expected_profit"]:.0f} 元')
            print(f'    菜品数: {len(lunch_solution["dishes"])}')
            
            # ---- 晚餐优化 ----
            # 晚餐人数约为午餐的5-10%
            dinner_diners = predicted_diners * 0.08
            print(f'\n  {date.strftime("%Y-%m-%d")} 晚餐 (预估{dinner_diners:.0f}人)')
            dinner_solution = self.optimize_meal(
                dinner_dishes, 'dinner', dinner_diners,
                predicted_nutrition=predicted_nutrition
            )
            dinner_solution['date'] = date.strftime('%Y-%m-%d')
            dinner_solution['meal'] = 'dinner'
            all_plans.append(dinner_solution)
            
            print(f'    目标值: {dinner_solution["objective_value"]:.0f}')
            print(f'    备菜总份数: {dinner_solution["total_servings"]:.0f}')
            print(f'    预期利润: {dinner_solution["expected_profit"]:.0f} 元')
        
        self.results['meal_plans'] = all_plans
        
        # ---- 可视化 ----
        self._plot_meal_plans(all_plans, plan_dates)
        
        # ---- 输出详细方案表格 ----
        self._print_detailed_plans(all_plans)
        
        print('\n问题3备菜优化完成！')
        return self.results
    
    def _get_predicted_diners(self, dow):
        """
        获取预测的就餐人数
        
        基于历史数据中的星期模式进行估算。
        在实际项目中，应使用问题2的预测结果。
        """
        df = self.df_daily[self.df_daily['total_orders'] > 0].copy()
        
        # 同星期历史均值
        same_dow = df[df['day_of_week'] == dow]
        if len(same_dow) > 0:
            return same_dow['total_orders'].mean()
        return df['total_orders'].mean()
    
    def _get_predicted_nutrition(self):
        """
        获取预测的营养素需求
        
        使用历史人均营养摄入量 × 预测人数
        """
        df = self.df_daily[self.df_daily['total_orders'] > 0].copy()
        
        return {
            'calories': df['total_calories'].mean(),
            'protein': df['total_protein'].mean(),
            'fat': df['total_fat'].mean(),
            'carbohydrates': df['total_carbohydrates'].mean(),
            'fiber': df['total_fiber'].mean(),
        }
    
    def _plot_meal_plans(self, all_plans, plan_dates):
        """可视化备菜方案"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 子图1: 每日备菜总份数（午餐vs晚餐）
        ax1 = axes[0, 0]
        lunch_plans = [p for p in all_plans if p['meal'] == 'lunch']
        dinner_plans = [p for p in all_plans if p['meal'] == 'dinner']
        
        x = range(len(lunch_plans))
        labels = [p['date'][-5:] for p in lunch_plans]
        
        width = 0.35
        ax1.bar([i - width/2 for i in x], 
                [p['total_servings'] for p in lunch_plans],
                width, label='Lunch', color=COLORS['lunch'], alpha=0.8)
        ax1.bar([i + width/2 for i in x],
                [p['total_servings'] for p in dinner_plans],
                width, label='Dinner', color=COLORS['dinner'], alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=30)
        ax1.set_ylabel('Total Servings')
        ax1.set_title('Daily Meal Preparation Quantities', fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 子图2: 预期利润
        ax2 = axes[0, 1]
        margins = []
        for p in lunch_plans:
            rev = p['total_expected_revenue']
            cost = p['total_cost']
            margins.append((rev - cost) / rev * 100 if rev > 0 else 0)
        
        ax2.bar(labels, margins, color=COLORS['primary'], alpha=0.7)
        ax2.set_ylabel('Profit Margin (%)')
        ax2.set_title('Expected Lunch Profit Margin', fontweight='bold')
        ax2.axhline(y=30, color=COLORS['danger'], linestyle='--', 
                   alpha=0.5, label='30% target')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=30)
        
        # 子图3: 菜品类别分布（某一天为例）
        ax3 = axes[1, 0]
        if lunch_plans:
            sample = lunch_plans[0]['dishes']
            cats = {}
            for d in sample:
                cat = d['category']
                cats[cat] = cats.get(cat, 0) + d['servings']
            
            wedges, texts, autotexts = ax3.pie(
                cats.values(), labels=cats.keys(), autopct='%1.1f%%',
                colors=[COLORS['primary'], COLORS['secondary'], 
                       COLORS['accent'], COLORS['success'], COLORS['warning']],
                startangle=90
            )
            ax3.set_title(f'Category Distribution ({lunch_plans[0]["date"]} Lunch)', 
                         fontweight='bold')
        
        # 子图4: 营养满足度
        ax4 = axes[1, 1]
        nutrient_targets = list(NUTRITION_PER_MEAL['lunch'].keys())
        nutrient_labels = ['Calories', 'Protein', 'Fat', 'Carbs', 'Fiber']
        
        if lunch_plans:
            nutrition = lunch_plans[2]['nutrition_summary'] if len(lunch_plans) > 2 else lunch_plans[0]['nutrition_summary']
            diners = 274  # 估算
            actual_per_person = {
                'calories': nutrition['calories'] / diners,
                'protein': nutrition['protein'] / diners,
                'fat': nutrition['fat'] / diners,
                'carbohydrates': nutrition['carbohydrates'] / diners,
                'fiber': nutrition['fiber'] / diners,
            }
            
            targets_per_person = NUTRITION_PER_MEAL['lunch']
            
            x_radar = np.arange(len(nutrient_targets))
            angles = np.linspace(0, 2 * np.pi, len(nutrient_targets), endpoint=False).tolist()
            angles += angles[:1]
            
            actual_values = [actual_per_person.get(k, 0) / max(targets_per_person.get(k, 1), 1) 
                           for k in nutrient_targets]
            target_values = [1.0] * len(nutrient_targets)
            
            actual_values += actual_values[:1]
            target_values += target_values[:1]
            
            ax4 = plt.subplot(2, 2, 4, projection='polar')
            ax4.fill(angles, target_values, alpha=0.2, color=COLORS['primary'], label='Target')
            ax4.plot(angles, actual_values, 'o-', color=COLORS['danger'], 
                    linewidth=2, label='Actual')
            ax4.fill(angles, actual_values, alpha=0.3, color=COLORS['danger'])
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(nutrient_labels, fontsize=9)
            ax4.set_title('Nutrition Satisfaction (per person)', fontweight='bold', pad=20)
            ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/p3_meal_plans.png', dpi=150)
        plt.close()
        print('  已保存: p3_meal_plans.png')
    
    def _print_detailed_plans(self, all_plans):
        """打印详细备菜方案表格"""
        print('\n' + '=' * 80)
        print('详细备菜方案（2025年5月6日-12日 工作日）')
        print('=' * 80)
        
        for plan in all_plans[:4]:  # 只展示前2天（午餐+晚餐）
            print(f'\n--- {plan["date"]} {plan["meal"]} ---')
            print(f'预计就餐人数: {plan.get("predicted_diners", "N/A")}')
            print(f'备菜总份数: {plan["total_servings"]:.0f}')
            print(f'预期收入: {plan["total_expected_revenue"]:.0f} 元')
            print(f'备菜成本: {plan["total_cost"]:.0f} 元')
            print(f'预期利润: {plan["expected_profit"]:.0f} 元')
            
            print(f'\n{"菜品名称":<20} {"类别":<8} {"备菜份数":>8} {"单价":>6} {"预期销量":>8} {"预期收入":>8}')
            print('-' * 65)
            
            for d in plan['dishes'][:15]:  # 展示前15个菜品
                print(f'{d["name"]:<20} {d["category"]:<8} {d["servings"]:>8} '
                      f'{d["unit_price"]:>6.1f} {d["expected_sales"]:>8.0f} '
                      f'{d["expected_revenue"]:>8.0f}')
            
            if len(plan['dishes']) > 15:
                print(f'  ... (共{len(plan["dishes"])}种菜品)')
            
            # 营养汇总
            nutri = plan['nutrition_summary']
            print(f'\n营养汇总: 热量={nutri["calories"]:.0f}kcal, '
                  f'蛋白质={nutri["protein"]:.0f}g, '
                  f'脂肪={nutri["fat"]:.0f}g, '
                  f'碳水={nutri["carbohydrates"]:.0f}g, '
                  f'纤维={nutri["fiber"]:.0f}g')
        
        # 保存完整方案到CSV
        all_rows = []
        for plan in all_plans:
            for d in plan['dishes']:
                all_rows.append({
                    '日期': plan['date'],
                    '餐次': plan['meal'],
                    '菜品名称': d['name'],
                    '类别': d['category'],
                    '备菜份数': d['servings'],
                    '单价': d['unit_price'],
                    '预期销量': d['expected_sales'],
                    '预期收入': d['expected_revenue'],
                    '浪费风险份数': d['waste_risk'],
                })
        
        plan_df = pd.DataFrame(all_rows)
        plan_df.to_csv(f'{OUTPUT_DIR}/p3_meal_plan_detail.csv', 
                       index=False, encoding='utf-8-sig')
        print(f'\n完整方案已保存到: p3_meal_plan_detail.csv')


if __name__ == '__main__':
    opt = Problem3Optimization()
    results = opt.run()
