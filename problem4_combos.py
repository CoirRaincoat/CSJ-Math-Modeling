"""
problem4_combos.py — 问题4：不同价位套餐优化设计
============================================
题目要求：
  基于该餐厅消费群体的消费习惯以及营养搭配科学性，建立数学模型，
  优化设计不同价位的套餐，并分别给出10元、15元和20元三个价位的套餐方案。

解题思路：
  1. 套餐设计原则：
     - 10元：经济基础型 — 米饭+1荤+1素，控制成本
     - 15元：均衡实用型 — 米饭+2荤+1素+1汤，兼顾多样与营养
     - 20元：丰富营养型 — 米饭+2荤+2素+1特色菜，强调品质
  
  2. 数学模型：
     使用组合优化（遗传算法/启发式搜索）从菜品库中筛选最优组合
     
     目标函数 max Z = α·偏好得分 + β·营养均衡得分 + γ·利润得分 + δ·共购关联得分
     
     约束条件：
     - 价格约束: |Σp_i - B| ≤ ε
     - 菜品数量约束: L ≤ Σy_i ≤ U
     - 类别多样性约束
     - 营养均衡约束
  
  3. 评估维度：
     - 营养评分：热量、蛋白质、脂肪、碳水均衡度
     - 消费者偏好：基于历史销量和共购关系
     - 利润评分：套餐利润率
     - 多样性评分：菜品种类和搭配合理性

参考文献：
  [5] Agrawal R. et al. "Fast Algorithms for Mining Association Rules" VLDB, 1994.
  [6] 余滔滔等. "基于Apriori算法的菜品配置规则研究" 服务科学和管理, 2019.
  [7] 黄健等. "中国海洋大学食堂菜谱的优化模型研究" 应用数学进展, 2018.
  [8] Padovan M. et al. "Optimized menu formulation" BMC Nutrition, 2023.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import random
import warnings
warnings.filterwarnings('ignore')

from data_loader import load_all_data
from config import (OUTPUT_DIR, COLORS, RANDOM_SEED, COMBO_PRICE_LEVELS,
                    COMBO_MIN_DISHES, COMBO_MAX_DISHES, COMBO_WEIGHTS,
                    NUTRITION_PER_MEAL)
from utils import check_nutrition_balance, calculate_calorie_breakdown

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


class Problem4Combos:
    """
    问题4：套餐优化设计
    
    使用启发式搜索 + 贪心策略，从菜品库中筛选最优套餐组合。
    """
    
    def __init__(self, loader=None):
        """初始化"""
        print('\n' + '=' * 60)
        print('问题4：不同价位套餐优化设计')
        print('=' * 60)
        
        if loader is None:
            self.loader = load_all_data()
        else:
            self.loader = loader
        
        self.dish_info = self.loader.get_dish_info()
        self.df_trans = self.loader.get_transaction_data()
        self.basket = self.loader.get_basket_data()
        
        # 构建菜品数据库
        self._build_dish_database()
        
        # 从问题1的结果获取关联规则（如果有）
        self.association_pairs = self._extract_associations()
        
        self.results = {}
    
    def _build_dish_database(self):
        """
        构建菜品数据库
        
        为每种菜品提取关键属性：
        - 营养信息、价格、成本
        - 历史销量和偏好度
        - 类别标签
        """
        dishes = []
        
        for _, row in self.dish_info.iterrows():
            dish = {
                'name': row['dish_name'],
                'serial': row['dish_serial'],
                'category': row['category'],
                'price': row['unit_price'],
                'cost': row['unit_cost'],
                'profit': row['unit_profit'],
                'profit_margin': row['profit_margin'],
                'calories': row['calories'],
                'protein': row['protein'],
                'fat': row['fat'],
                'carbs': row['carbohydrates'],
                'fiber': row['fiber'],
                'total_orders': row['total_orders'],
                'popularity_score': 0.0,  # 稍后计算
            }
            dishes.append(dish)
        
        self.dish_db = pd.DataFrame(dishes)
        
        # 计算偏好得分（归一化到0-1）
        max_orders = self.dish_db['total_orders'].max()
        if max_orders > 0:
            self.dish_db['popularity_score'] = self.dish_db['total_orders'] / max_orders
        
        # 过滤掉价格异常或过高的菜品
        self.dish_db = self.dish_db[
            (self.dish_db['price'] > 0.1) & 
            (self.dish_db['price'] < 15)
        ]
        
        print(f'  菜品数据库: {len(self.dish_db)} 种可用菜品')
        
    def _extract_associations(self):
        """
        从购物篮数据中提取菜品共现关系
        
        计算每对菜品的共现次数，用于套餐设计中的搭配评分。
        """
        basket = self.basket
        
        dishes = basket.columns.tolist()
        n = len(dishes)
        
        pairs = {}
        
        # 计算共现矩阵（采样计算以加速）
        sample_size = min(3000, len(basket))
        basket_sample = basket.sample(n=sample_size, random_state=RANDOM_SEED)
        
        for i, d1 in enumerate(dishes):
            col1 = basket_sample[d1].values
            for j, d2 in enumerate(dishes):
                if i < j:
                    col2 = basket_sample[d2].values
                    co_occurrence = np.sum(col1 & col2)
                    if co_occurrence > 3:
                        pairs[(d1, d2)] = co_occurrence / sample_size
        
        # 保留共现次数高的配对
        self.cooccurrence = pairs
        
        print(f'  共现配对: {len(pairs)} 对')
        
        return pairs
    
    def _score_combo(self, combo_dishes, target_price):
        """
        套餐综合评分
        
        评分维度：
        1. 消费者偏好评分 (0-1)
        2. 营养均衡评分 (0-1)
        3. 利润评分 (0-1)
        4. 共购关联评分 (0-1)
        5. 价格符合度 (0-1) — 惩罚价格偏离
        
        Args:
            combo_dishes: 菜品字典列表
            target_price: 目标价位（10/15/20元）
            
        Returns:
            float: 综合得分
        """
        if len(combo_dishes) == 0:
            return 0.0
        
        w = COMBO_WEIGHTS
        
        # 1. 偏好评分
        pop_scores = [d['popularity_score'] for d in combo_dishes]
        popularity = np.mean(pop_scores) if pop_scores else 0
        
        # 2. 营养均衡评分
        total_cal = sum(d['calories'] for d in combo_dishes)
        total_protein = sum(d['protein'] for d in combo_dishes)
        total_fat = sum(d['fat'] for d in combo_dishes)
        total_carbs = sum(d['carbs'] for d in combo_dishes)
        total_fiber = sum(d['fiber'] for d in combo_dishes)
        
        balance = check_nutrition_balance(total_cal, total_protein, total_fat, 
                                          total_carbs, total_fiber)
        nutrition_score = balance.get('overall_balance', 0.5)
        
        # 3. 利润评分
        total_price = sum(d['price'] for d in combo_dishes)
        total_cost = sum(d['cost'] for d in combo_dishes)
        profit_margin = (total_price - total_cost) / max(total_price, 0.01)
        # 理想利润率30-50%
        profit_score = max(0, 1 - abs(profit_margin - 0.4) / 0.4)
        
        # 4. 共购关联评分
        assoc_score = 0.0
        if len(combo_dishes) >= 2:
            pair_count = 0
            for i, d1 in enumerate(combo_dishes):
                for j, d2 in enumerate(combo_dishes):
                    if i < j:
                        name1, name2 = d1['name'], d2['name']
                        key1 = (name1, name2)
                        key2 = (name2, name1)
                        if key1 in self.cooccurrence:
                            assoc_score += self.cooccurrence[key1]
                            pair_count += 1
                        elif key2 in self.cooccurrence:
                            assoc_score += self.cooccurrence[key2]
                            pair_count += 1
            if pair_count > 0:
                assoc_score = min(1.0, assoc_score / pair_count * 50)
        
        # 5. 价格符合度
        price_deviation = abs(total_price - target_price) / max(target_price, 1)
        price_fit = max(0, 1 - price_deviation * 3)
        
        # 6. 类别多样性（鼓励包含不同类别）
        categories = set(d['category'] for d in combo_dishes)
        diversity_bonus = min(1.0, len(categories) / 3) * 0.1
        
        # 综合得分
        score = (
            w['popularity'] * popularity +
            w['nutrition'] * nutrition_score +
            w['profit'] * profit_score +
            w['association'] * max(0, assoc_score) +
            0.15 * price_fit +
            diversity_bonus
        )
        
        # 惩罚：菜品数量太少或太多
        n_dishes = len(combo_dishes)
        if n_dishes < COMBO_MIN_DISHES:
            score *= 0.5
        elif n_dishes > COMBO_MAX_DISHES:
            score *= 0.7
        
        return score
    
    def _greedy_search(self, target_price, max_dishes=None):
        """
        贪心搜索最优套餐组合
        
        算法：
        1. 按类别筛选候选菜品
        2. 使用贪心策略逐步添加菜品
        3. 对top候选方案进行局部优化
        
        Args:
            target_price: 目标价位
            max_dishes: 最大菜品数
            
        Returns:
            list: 最优套餐组合
        """
        if max_dishes is None:
            max_dishes = COMBO_MAX_DISHES
        
        # 按价位筛选候选菜品
        affordable = self.dish_db[
            self.dish_db['price'] <= target_price * 0.7
        ].copy()
        
        if len(affordable) < 5:
            affordable = self.dish_db[self.dish_db['price'] <= target_price].copy()
        
        # 分策略设计套餐结构
        if target_price <= 10:
            # 10元套餐：1主食 + 1荤 + 1素
            structure = {'主食': 1, '荤菜': 1, '半荤半素': 0, '素菜': 1, '其他': 0}
            max_dishes = 3
        elif target_price <= 15:
            # 15元套餐：1主食 + 1荤 + 1半荤 + 1素 + 1其他
            structure = {'主食': 1, '荤菜': 1, '半荤半素': 1, '素菜': 2, '其他': 0}
            max_dishes = 5
        else:
            # 20元套餐：1主食 + 2荤 + 1半荤 + 2素 + 1特色
            structure = {'主食': 1, '荤菜': 2, '半荤半素': 1, '素菜': 2, '其他': 1}
            max_dishes = 7
        
        # 生成多个候选方案
        candidates = []
        
        for trial in range(200):
            combo = []
            remaining_budget = target_price
            
            # 按结构选取菜品
            used_names = set()
            
            for cat, count in structure.items():
                if count <= 0:
                    continue
                
                cat_dishes = affordable[affordable['category'] == cat].copy()
                if len(cat_dishes) == 0:
                    cat_dishes = affordable.copy()
                
                # 按受欢迎度 + 随机性选择
                cat_dishes['score'] = (
                    cat_dishes['popularity_score'] * 0.7 + 
                    np.random.random(len(cat_dishes)) * 0.3
                )
                cat_dishes = cat_dishes.sort_values('score', ascending=False)
                
                selected_count = 0
                for _, row in cat_dishes.iterrows():
                    if selected_count >= count:
                        break
                    if row['name'] not in used_names and row['price'] <= remaining_budget * 0.8:
                        combo.append(row.to_dict())
                        used_names.add(row['name'])
                        remaining_budget -= row['price']
                        selected_count += 1
            
            if len(combo) >= 2:
                total = sum(d['price'] for d in combo)
                score = self._score_combo(combo, target_price)
                candidates.append({
                    'combo': combo,
                    'total_price': total,
                    'score': score,
                    'n_dishes': len(combo),
                })
        
        # 选择得分最高的方案
        if not candidates:
            return []
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return candidates[0]
    
    def _local_optimization(self, combo_data, target_price, iterations=50):
        """
        局部搜索优化
        
        对贪心搜索结果进行微调：
        - 尝试替换单个菜品
        - 尝试添加/移除菜品
        - 保持总价在预算范围内
        """
        best_combo = combo_data['combo'].copy()
        best_score = combo_data['score']
        best_price = combo_data['total_price']
        
        for _ in range(iterations):
            # 随机选择操作类型
            op = random.choice(['replace', 'add', 'remove'])
            
            if op == 'replace' and len(best_combo) >= 1:
                # 替换一个菜品
                idx = random.randint(0, len(best_combo) - 1)
                old_dish = best_combo[idx]
                old_price = old_dish['price']
                
                candidates = self.dish_db[
                    (self.dish_db['category'] == old_dish['category']) &
                    (self.dish_db['name'] != old_dish['name'])
                ]
                
                if len(candidates) > 0:
                    new_dish = candidates.sample(1).iloc[0].to_dict()
                    new_price = new_dish['price']
                    new_total = best_price - old_price + new_price
                    
                    if abs(new_total - target_price) <= target_price * 0.3:
                        new_combo = best_combo.copy()
                        new_combo[idx] = new_dish
                        new_score = self._score_combo(new_combo, target_price)
                        
                        if new_score > best_score:
                            best_combo = new_combo
                            best_score = new_score
                            best_price = new_total
            
            elif op == 'add' and len(best_combo) < COMBO_MAX_DISHES + 2:
                # 添加一个菜品
                if best_price < target_price * 0.9:
                    remaining = target_price - best_price
                    candidates = self.dish_db[
                        self.dish_db['price'] <= remaining * 1.1
                    ]
                    
                    if len(candidates) > 0:
                        new_dish = candidates.sample(1).iloc[0].to_dict()
                        new_combo = best_combo + [new_dish]
                        new_total = best_price + new_dish['price']
                        
                        if abs(new_total - target_price) <= target_price * 0.3:
                            new_score = self._score_combo(new_combo, target_price)
                            if new_score > best_score:
                                best_combo = new_combo
                                best_score = new_score
                                best_price = new_total
            
            elif op == 'remove' and len(best_combo) > COMBO_MIN_DISHES:
                # 移除一个低价值菜品
                idx = random.randint(0, len(best_combo) - 1)
                new_combo = best_combo[:idx] + best_combo[idx+1:]
                new_total = best_price - best_combo[idx]['price']
                new_score = self._score_combo(new_combo, target_price)
                
                if new_score > best_score:
                    best_combo = new_combo
                    best_score = new_score
                    best_price = new_total
        
        return {
            'combo': best_combo,
            'total_price': best_price,
            'score': best_score,
            'n_dishes': len(best_combo),
        }
    
    def run(self):
        """运行套餐优化"""
        print('\n>>> 4.1 套餐设计策略')
        
        # 套餐定位描述
        combo_positions = {
            10: {
                'name': '经济基础型',
                'description': '满足基础饱腹需求，1主食+1荤+1素，严格控制成本',
                'target_customers': '价格敏感型顾客、学生群体',
                'structure': '主食 ×1 + 荤菜 ×1 + 素菜 ×1',
            },
            15: {
                'name': '均衡实用型',
                'description': '兼顾荤素搭配与营养均衡，1主食+1荤+1半荤半素+2素',
                'target_customers': '追求性价比的上班族',
                'structure': '主食 ×1 + 荤菜 ×1 + 半荤半素 ×1 + 素菜 ×2',
            },
            20: {
                'name': '丰富营养型',
                'description': '强调菜品多样性与高蛋白摄入，1主食+2荤+1半荤+2素+1特色',
                'target_customers': '追求品质体验的白领顾客',
                'structure': '主食 ×1 + 荤菜 ×2 + 半荤半素 ×1 + 素菜 ×2 + 其他 ×1',
            },
        }
        
        all_combos = {}
        
        print('\n>>> 4.2 各价位套餐优化')
        
        for price in COMBO_PRICE_LEVELS:
            print(f'\n  --- {price}元套餐 ({combo_positions[price]["name"]}) ---')
            
            # 贪心搜索
            greedy_result = self._greedy_search(price)
            
            if not greedy_result:
                print(f'    未找到可行组合')
                continue
            
            # 局部优化
            opt_result = self._local_optimization(greedy_result, price, iterations=100)
            
            combo = opt_result['combo']
            total = opt_result['total_price']
            score = opt_result['score']
            
            # 计算营养汇总
            total_cal = sum(d['calories'] for d in combo)
            total_protein = sum(d['protein'] for d in combo)
            total_fat = sum(d['fat'] for d in combo)
            total_carbs = sum(d['carbs'] for d in combo)
            total_fiber = sum(d['fiber'] for d in combo)
            total_cost = sum(d['cost'] for d in combo)
            
            balance = check_nutrition_balance(total_cal, total_protein, total_fat,
                                              total_carbs, total_fiber)
            
            print(f'    套餐总分: {score:.3f}')
            print(f'    实际总价: {total:.2f}元 (目标: {price}元)')
            print(f'    菜品数量: {len(combo)}')
            print(f'    利润率: {(total - total_cost) / total * 100:.0f}%')
            print(f'    营养均衡度: {balance["overall_balance"]:.2f}')
            print(f'    热量: {total_cal:.0f} kcal, 蛋白: {total_protein:.0f}g, '
                  f'脂肪: {total_fat:.0f}g, 碳水: {total_carbs:.0f}g')
            print(f'    菜品组成:')
            for d in combo:
                print(f'      - {d["name"]} ({d["category"]}) {d["price"]:.2f} yuan')
            
            all_combos[price] = {
                'position': combo_positions[price],
                'combo': combo,
                'total_price': total,
                'score': score,
                'nutrition': {
                    'calories': total_cal,
                    'protein': total_protein,
                    'fat': total_fat,
                    'carbs': total_carbs,
                    'fiber': total_fiber,
                },
                'balance': balance,
                'profit_margin': (total - total_cost) / total * 100,
            }
        
        self.results['combos'] = all_combos
        
        # ---- 可视化 ----
        self._plot_combo_results(all_combos)
        
        print('\n问题4套餐设计完成！')
        return self.results
    
    def _plot_combo_results(self, all_combos):
        """可视化套餐结果"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 子图1: 各价位套餐对比柱状图
        ax1 = axes[0]
        prices = sorted(all_combos.keys())
        
        metrics = {
            '总价格 (元)': [all_combos[p]['total_price'] for p in prices],
            '利润率 (%)': [all_combos[p]['profit_margin'] for p in prices],
            '营养均衡度 (%)': [all_combos[p]['balance']['overall_balance'] * 100 for p in prices],
            '综合得分 (%)': [all_combos[p]['score'] * 100 for p in prices],
        }
        
        x = np.arange(len(prices))
        width = 0.2
        colors_m = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['success']]
        
        for i, (metric_name, values) in enumerate(metrics.items()):
            bars = ax1.bar(x + i * width - width * 1.5, values, width,
                          label=metric_name, color=colors_m[i], alpha=0.8)
        
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{p}元' for p in prices], fontsize=11)
        ax1.set_ylabel('Value')
        ax1.set_title('Combo Comparison Across Price Points', fontweight='bold')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(axis='y', alpha=0.3)
        
        # 子图2: 营养雷达图
        ax2 = fig.add_subplot(1, 2, 2, projection='polar')
        
        nutri_keys = ['calories', 'protein', 'fat', 'carbs', 'fiber']
        nutri_labels = ['Calories\n(kcal)', 'Protein\n(g)', 'Fat\n(g)', 
                        'Carbs\n(g)', 'Fiber\n(g)']
        
        angles = np.linspace(0, 2 * np.pi, len(nutri_keys), endpoint=False).tolist()
        angles += angles[:1]
        
        color_cycle = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]
        
        for i, price in enumerate(prices):
            combo = all_combos[price]
            values = [combo['nutrition'][k] for k in nutri_keys]
            
            # 归一化到0-1
            max_vals = [max(all_combos[p]['nutrition'][k] for p in prices) 
                       for k in nutri_keys]
            norm_values = [v / max(mv, 1) for v, mv in zip(values, max_vals)]
            norm_values += norm_values[:1]
            
            ax2.fill(angles, norm_values, alpha=0.15, color=color_cycle[i])
            ax2.plot(angles, norm_values, 'o-', linewidth=2, color=color_cycle[i],
                    label=f'{price}元')
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(nutri_labels, fontsize=8)
        ax2.set_title('Nutrition Comparison (Normalized)', fontweight='bold', pad=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/p4_combo_results.png', dpi=150)
        plt.close()
        print('  已保存: p4_combo_results.png')


if __name__ == '__main__':
    combos = Problem4Combos()
    results = combos.run()
