"""
problem4_combos.py — 问题4: 不同价位套餐优化设计
============================================
题目要求:
  基于该餐厅消费群体的消费习惯以及营养搭配科学性，建立数学模型，
  优化设计不同价位的套餐，并分别给出10元、15元和20元三个价位的套餐方案。

解题思路:
  1. 套餐设计原则:
     - 10元"经济基础型": 1主食 + 1荤 + 1素，控制成本
     - 15元"均衡实用型": 1主食 + 1荤 + 1半荤 + 2素
     - 20元"丰富营养型": 1主食 + 2荤 + 1半荤 + 2素 + 1其他
  2. 数学模型: 组合优化 + 启发式搜索
     目标函数:
       max Z = w1·偏好得分 + w2·营养均衡得分 + w3·利润得分
              + w4·共购关联得分 + w5·价格符合度 - 惩罚项
     约束:
     - 价格约束: |Σp_i - B| ≤ ε (总价在目标价位附近)
     - 菜品数量约束: 2 ≤ n ≤ 5
     - 类别多样性约束: 包含主食
     - 营养均衡约束: 三大营养素供能比在推荐区间
  3. 搜索算法:
     (a) 贪心构建: 按套餐结构逐类选择高偏好度菜品 (200次采样)
     (b) 局部优化: 替换/添加/移除单个菜品 (100次迭代)
     (c) 最终选择: 各价位得分最高的组合
  4. 评估维度:
     - 营养评分: 碳水/脂肪/蛋白质供能比均衡度
     - 消费偏好: 基于历史销量的 popularity_score
     - 利润评分: 套餐利润率 (目标 40%)
     - 关联评分: 基于共现矩阵的搭配契合度

核心数据结构:
  dish_db: 菜品数据库 DataFrame
    列: name, category, price, cost, profit, profit_margin,
         calories, protein, fat, carbs, fiber, total_orders, popularity_score

参考文献:
  [5]  Agrawal R. et al. "Fast Algorithms for Mining Association Rules"
       VLDB 1994. https://www.vldb.org/conf/1994/P487.PDF
  [6]  余滔滔等. "基于Apriori算法的菜品配置规则研究"
       服务科学和管理, 2019.
  [7]  黄健等. "中国海洋大学食堂菜谱的优化模型研究"
       应用数学进展, 2018.
  [8]  Padovan M. et al. "Optimized menu formulation to enhance nutritional
       goals" BMC Nutrition, 2023.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings('ignore')

from data_loader import load_all_data
from config import (OUTPUT_DIR, COLORS, COLOR_CYCLE, RANDOM_SEED,
                     COMBO_PRICE_LEVELS, COMBO_MIN_DISHES,
                     COMBO_MAX_DISHES, COMBO_WEIGHTS,
                     NUTRITION_PER_MEAL)
from utils import check_nutrition_balance, calculate_calorie_breakdown

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


class Problem4Combos:
    """
    问题4: 套餐优化设计

    使用贪心搜索 + 局部优化策略，从菜品数据库中
    为每个价位筛选出最优套餐组合。

    算法流程:
    1. 构建菜品数据库 (_build_dish_database)
    2. 提取菜品共现关系 (_extract_associations)
    3. 对每个价位:
       a. 贪心搜索生成候选套餐 (_greedy_search)
       b. 局部优化改进 (_local_optimization)
       c. 选择最优方案
    4. 可视化比较 (_plot_combo_results)
    """

    def __init__(self, loader=None):
        """
        初始化套餐设计模块

        Args:
            loader: DataLoader 实例或 None
        """
        print('\n' + '=' * 60)
        print('问题4: 不同价位套餐优化设计')
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

        # 提取菜品共现关系 (基于购物篮)
        self.association_pairs = self._extract_associations()

        self.results = {}

    def _build_dish_database(self):
        """
        构建菜品数据库 (dish_db)

        从 dish_info 中提取每种菜品的核心属性:
        - 基本信息: name, category, price, cost
        - 营养信息: calories, protein, fat, carbs, fiber
        - 商业指标: total_orders, profit_margin
        - 偏好评分: popularity_score (归一化到 [0, 1])

        过滤规则:
        - 排除价格 ≤ 0.1 元的异常菜品
        - 排除价格 ≥ 15 元的高价菜品 (非套餐目标)
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
                'popularity_score': 0.0,
            }
            dishes.append(dish)

        self.dish_db = pd.DataFrame(dishes)

        # 计算偏好得分 (归一化到 [0, 1])
        max_orders = self.dish_db['total_orders'].max()
        if max_orders > 0:
            self.dish_db['popularity_score'] = (
                self.dish_db['total_orders'] / max_orders
            )

        # 过滤异常菜品
        self.dish_db = self.dish_db[
            (self.dish_db['price'] > 0.1) &
            (self.dish_db['price'] < 15)
        ]

        print(f'  菜品数据库: {len(self.dish_db)} 种可用菜品')

    def _extract_associations(self):
        """
        从购物篮数据中提取菜品共现关系

        方法: 采样计算每对菜品的共现频次
        - 采样 3000 个订单 (加速, 全量 11828 个)
        - 对每对菜品 (i, j), 计算 order 中同时出现的次数
        - 保留共现次数 > 3 的配对
        - 共现值归一化为共现概率

        共现关系用于套餐设计中的搭配评分:
        - 经常被一起购买的菜品在同一套餐中得分更高
        - 例如 {米饭, 红烧肉} 的共现次数高 → 搭配合理

        Returns:
            dict: {(dish1, dish2): co_occurrence_prob, ...}
        """
        basket = self.basket
        dishes = basket.columns.tolist()

        pairs = {}

        # 采样加速
        sample_size = min(3000, len(basket))
        basket_sample = basket.sample(
            n=sample_size, random_state=RANDOM_SEED
        )

        # 逐对计算共现次数
        for i, d1 in enumerate(dishes):
            col1 = basket_sample[d1].values
            for j, d2 in enumerate(dishes):
                if i < j:
                    col2 = basket_sample[d2].values
                    co_occurrence = np.sum(col1 & col2)
                    if co_occurrence > 3:
                        pairs[(d1, d2)] = co_occurrence / sample_size

        self.cooccurrence = pairs
        print(f'  共现配对: {len(pairs)} 对')

        return pairs

    def _score_combo(self, combo_dishes, target_price):
        """
        套餐综合评分函数

        评分维度 (5 项, 满分约 1.15):
        1. 消费者偏好评分 (w=0.30):
           popularity = mean(d.popularity_score for d in combo)
           反映菜品的历史受欢迎程度

        2. 营养均衡评分 (w=0.30):
           使用 check_nutrition_balance() 中的 overall_balance
           评估碳水/脂肪/蛋白质供能比是否符合推荐区间

        3. 利润评分 (w=0.25):
           profit_score = max(0, 1 - |利润率 - 0.40| / 0.40)
           理想利润率约 40% (餐厅行业中位)

        4. 共购关联评分 (w=0.15):
           对套餐中每对菜品的共现概率求平均
           反映菜品的搭配合理性

        5. 价格符合度 (w=0.15):
           price_fit = max(0, 1 - |实际价 - 目标价| / 目标价 × 3)
           总价越接近目标价位，得分越高

        额外: 类别多样性奖励 (+0.10)
           套餐覆盖 3+ 个类别则加分，鼓励类别多样性

        Args:
            combo_dishes: list of dict, 菜品列表
            target_price: float, 目标价位

        Returns:
            float: 综合得分 (越高越好)
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

        balance = check_nutrition_balance(
            total_cal, total_protein, total_fat, total_carbs, total_fiber
        )
        nutrition_score = balance.get('overall_balance', 0.5)

        # 3. 利润评分
        total_price = sum(d['price'] for d in combo_dishes)
        total_cost = sum(d['cost'] for d in combo_dishes)
        profit_margin = (
            (total_price - total_cost) / max(total_price, 0.01)
        )
        # 理想利润率约 40%
        profit_score = max(0, 1 - abs(profit_margin - 0.40) / 0.40)

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
                # 放大系数 50: 将 0.02 左右的共现概率映射到 [0, 1]
                assoc_score = min(1.0, assoc_score / pair_count * 50)

        # 5. 价格符合度
        price_deviation = abs(total_price - target_price) / max(target_price, 1)
        price_fit = max(0, 1 - price_deviation * 3)

        # 6. 类别多样性奖励
        categories = set(d['category'] for d in combo_dishes)
        diversity_bonus = min(1.0, len(categories) / 3) * 0.10

        # ---- 综合得分 ----
        score = (
            w['popularity'] * popularity +
            w['nutrition'] * nutrition_score +
            w['profit'] * profit_score +
            w['association'] * max(0, assoc_score) +
            0.15 * price_fit +
            diversity_bonus
        )

        # 惩罚: 菜品数量不合规
        n_dishes = len(combo_dishes)
        if n_dishes < COMBO_MIN_DISHES:
            score *= 0.5
        elif n_dishes > COMBO_MAX_DISHES:
            score *= 0.7

        return score

    def _greedy_search(self, target_price, max_dishes=None):
        """
        贪心搜索最优套餐组合

        算法:
        1. 根据价位选择套餐结构模板
           - 10元: 主食×1 + 荤菜×1 + 素菜×1
           - 15元: 主食×1 + 荤菜×1 + 半荤半素×1 + 素菜×2
           - 20元: 主食×1 + 荤菜×2 + 半荤半素×1 + 素菜×2 + 其他×1
        2. 对 200 次迭代:
           - 按结构逐类选择菜品
           - 每类内按 (popularity_score × 0.7 + random × 0.3) 排序
           - 引入随机性以增加候选多样性
        3. 从 200 个候选中选择 _score_combo() 最高的方案

        随机性控制:
        - 每次迭代使用 30% 的随机权重 (确定性 70%)
        - 固定 RANDOM_SEED 保证可复现

        Args:
            target_price: 目标价位 (10/15/20)
            max_dishes: 最大菜品数 (None → 自动设置)

        Returns:
            dict: {'combo': list, 'total_price': float, 'score': float,
                   'n_dishes': int}
        """
        if max_dishes is None:
            max_dishes = COMBO_MAX_DISHES

        # 筛选可负担的菜品
        affordable = self.dish_db[
            self.dish_db['price'] <= target_price * 0.7
        ].copy()

        if len(affordable) < 5:
            affordable = self.dish_db[
                self.dish_db['price'] <= target_price
            ].copy()

        # 价位对应的套餐结构
        if target_price <= 10:
            structure = {'主食': 1, '荤菜': 1, '半荤半素': 0,
                        '素菜': 1, '其他': 0}
            max_dishes = 3
        elif target_price <= 15:
            structure = {'主食': 1, '荤菜': 1, '半荤半素': 1,
                        '素菜': 2, '其他': 0}
            max_dishes = 5
        else:
            structure = {'主食': 1, '荤菜': 2, '半荤半素': 1,
                        '素菜': 2, '其他': 1}
            max_dishes = 6

        # 生成 200 个候选方案
        candidates = []

        for trial in range(200):
            combo = []
            remaining_budget = target_price
            used_names = set()

            # 按结构逐类选择
            for cat, count in structure.items():
                if count <= 0:
                    continue

                cat_dishes = affordable[
                    affordable['category'] == cat
                ].copy()
                if len(cat_dishes) == 0:
                    cat_dishes = affordable.copy()

                # 按 (受欢迎度 × 0.7 + 随机性 × 0.3) 排序
                cat_dishes['score'] = (
                    cat_dishes['popularity_score'] * 0.7 +
                    np.random.random(len(cat_dishes)) * 0.3
                )
                cat_dishes = cat_dishes.sort_values(
                    'score', ascending=False
                )

                selected_count = 0
                for _, row in cat_dishes.iterrows():
                    if selected_count >= count:
                        break
                    if (row['name'] not in used_names
                            and row['price'] <= remaining_budget * 0.8):
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

        # 返回得分最高的方案
        if not candidates:
            return []

        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[0]

    def _local_optimization(self, combo_data, target_price, iterations=100):
        """
        局部搜索优化

        对贪心搜索结果进行微调，尝试三类操作:
        1. replace: 随机替换一个菜品为同类别的另一个
        2. add: 在预算允许范围内添加一个新菜品
        3. remove: 删除一个菜品

        仅当新方案得分高于当前最佳时才接受 (爬山法)。

        Args:
            combo_data: 贪心搜索结果 dict
            target_price: 目标价位
            iterations: 优化迭代次数

        Returns:
            dict: 优化后的方案
        """
        best_combo = combo_data['combo'].copy()
        best_score = combo_data['score']
        best_price = combo_data['total_price']

        for _ in range(iterations):
            op = random.choice(['replace', 'add', 'remove'])

            if op == 'replace' and len(best_combo) >= 1:
                # 随机替换一个菜品为同类别的另一个
                idx = random.randint(0, len(best_combo) - 1)
                old_dish = best_combo[idx]
                old_price = old_dish['price']

                candidates = self.dish_db[
                    (self.dish_db['category'] == old_dish['category']) &
                    (self.dish_db['name'] != old_dish['name'])
                ]

                if len(candidates) > 0:
                    new_dish = candidates.sample(
                        1, random_state=RANDOM_SEED
                    ).iloc[0].to_dict()
                    new_price = new_dish['price']
                    new_total = best_price - old_price + new_price

                    if abs(new_total - target_price) <= target_price * 0.3:
                        new_combo = best_combo.copy()
                        new_combo[idx] = new_dish
                        new_score = self._score_combo(
                            new_combo, target_price
                        )

                        if new_score > best_score:
                            best_combo = new_combo
                            best_score = new_score
                            best_price = new_total

            elif op == 'add' and len(best_combo) < COMBO_MAX_DISHES + 2:
                # 在预算允许范围内添加新菜品
                if best_price < target_price * 0.9:
                    remaining = target_price - best_price
                    candidates = self.dish_db[
                        self.dish_db['price'] <= remaining * 1.1
                    ]

                    if len(candidates) > 0:
                        new_dish = candidates.sample(
                            1, random_state=RANDOM_SEED
                        ).iloc[0].to_dict()
                        new_combo = best_combo + [new_dish]
                        new_total = best_price + new_dish['price']

                        if (abs(new_total - target_price)
                                <= target_price * 0.3):
                            new_score = self._score_combo(
                                new_combo, target_price
                            )
                            if new_score > best_score:
                                best_combo = new_combo
                                best_score = new_score
                                best_price = new_total

            elif op == 'remove' and len(best_combo) > COMBO_MIN_DISHES:
                # 移除一个菜品
                idx = random.randint(0, len(best_combo) - 1)
                new_combo = (
                    best_combo[:idx] + best_combo[idx+1:]
                )
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
        """
        运行套餐优化全流程

        对 10/15/20 元三个价位分别:
        1. 贪心搜索生成初始方案
        2. 局部优化改进方案
        3. 计算营养和利润指标
        4. 可视化比较三个价位

        Returns:
            dict: 各价位套餐方案
        """
        print('\n>>> 4.1 套餐设计策略')

        # 套餐定位描述
        combo_positions = {
            10: {
                'name': '经济基础型',
                'description': '满足基础饱腹需求，严格控制成本',
                'structure': '主食×1 + 荤菜×1 + 素菜×1',
            },
            15: {
                'name': '均衡实用型',
                'description': '兼顾荤素搭配与营养均衡',
                'structure': '主食×1 + 荤菜×1 + 半荤半素×1 + 素菜×2',
            },
            20: {
                'name': '丰富营养型',
                'description': '强调菜品多样性与高蛋白摄入',
                'structure': '主食×1 + 荤菜×2 + 半荤半素×1 + 素菜×2 + 其他×1',
            },
        }

        all_combos = {}

        print('\n>>> 4.2 各价位套餐优化')

        for price in COMBO_PRICE_LEVELS:
            print(f'\n  --- {price}元套餐 '
                  f'({combo_positions[price]["name"]}) ---')

            # 步骤1: 贪心搜索
            greedy_result = self._greedy_search(price)

            if not greedy_result:
                print(f'    未找到可行组合')
                continue

            # 步骤2: 局部优化
            opt_result = self._local_optimization(
                greedy_result, price, iterations=100
            )

            combo = opt_result['combo']
            total = opt_result['total_price']
            score = opt_result['score']

            # 步骤3: 计算营养指标
            total_cal = sum(d['calories'] for d in combo)
            total_protein = sum(d['protein'] for d in combo)
            total_fat = sum(d['fat'] for d in combo)
            total_carbs = sum(d['carbs'] for d in combo)
            total_fiber = sum(d['fiber'] for d in combo)
            total_cost = sum(d['cost'] for d in combo)

            balance = check_nutrition_balance(
                total_cal, total_protein, total_fat,
                total_carbs, total_fiber
            )

            # 输出结果
            print(f'    套餐总分: {score:.3f}')
            print(f'    实际总价: {total:.2f}元 (目标: {price}元)')
            print(f'    菜品数量: {len(combo)}')
            print(f'    利润率: {(total - total_cost) / total * 100:.0f}%')
            print(f'    营养均衡度: {balance["overall_balance"]:.2f}')
            print(f'    热量: {total_cal:.0f} kcal, '
                  f'蛋白: {total_protein:.1f}g, '
                  f'脂肪: {total_fat:.1f}g, '
                  f'碳水: {total_carbs:.1f}g')

            # 菜品组成
            print(f'    菜品组成:')
            for d in combo:
                print(f'      - {d["name"]} ({d["category"]}) '
                      f'{d["price"]:.2f} 元')

            # 营养均衡详情
            print(f'    营养详情: '
                  f'蛋白供能比={balance["protein_ratio"]:.1%}, '
                  f'脂肪供能比={balance["fat_ratio"]:.1%}, '
                  f'碳水供能比={balance["carbs_ratio"]:.1%}')

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

        print('\n问题4 套餐设计完成!')
        return self.results

    def _plot_combo_results(self, all_combos):
        """
        可视化套餐结果 — 输出 p4_combo_results.png

        子图布局 (1 × 2):
        1. 各价位套餐指标对比柱状图:
           - 总价格, 利润率, 营养均衡度, 综合得分
        2. 营养成分雷达图:
           - 三价位归一化对比
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        prices = sorted(all_combos.keys())

        # ---- 子图1: 各价位套餐指标对比 ----
        ax1 = axes[0]

        metrics = {
            'Total Price (Yuan)': [
                all_combos[p]['total_price'] for p in prices
            ],
            'Profit Margin (%)': [
                all_combos[p]['profit_margin'] for p in prices
            ],
            'Nutrition Balance (%)': [
                all_combos[p]['balance']['overall_balance'] * 100
                for p in prices
            ],
            'Overall Score (%)': [
                all_combos[p]['score'] * 100 for p in prices
            ],
        }

        x = np.arange(len(prices))
        width = 0.2
        colors_m = [COLORS['primary'], COLORS['success'],
                    COLORS['accent'], COLORS['purple']]

        for i, (metric_name, values) in enumerate(metrics.items()):
            ax1.bar(x + i * width - width * 1.5, values, width,
                   label=metric_name, color=colors_m[i], alpha=0.8)

        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{p} Yuan' for p in prices], fontsize=10)
        ax1.set_ylabel('Value')
        ax1.set_title('Combo Comparison Across Price Levels',
                     fontweight='bold', fontsize=11)
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(axis='y', alpha=0.3)

        # ---- 子图2: 营养成分雷达图 ----
        ax2 = fig.add_subplot(1, 2, 2, projection='polar')

        nutri_keys = ['calories', 'protein', 'fat', 'carbs', 'fiber']
        nutri_labels = ['Calories\n(kcal)', 'Protein\n(g)',
                       'Fat\n(g)', 'Carbs\n(g)', 'Fiber\n(g)']

        angles = np.linspace(0, 2 * np.pi, len(nutri_keys),
                            endpoint=False).tolist()
        angles += angles[:1]

        color_cycle = [COLORS['primary'], COLORS['accent'],
                      COLORS['success']]

        for i, price in enumerate(prices):
            combo = all_combos[price]
            values = [combo['nutrition'][k] for k in nutri_keys]

            # 归一化到 [0, 1]
            max_vals = [max(all_combos[p]['nutrition'][k]
                           for p in prices) for k in nutri_keys]
            norm_values = [v / max(mv, 1) for v, mv
                          in zip(values, max_vals)]
            norm_values += norm_values[:1]

            ax2.fill(angles, norm_values, alpha=0.15,
                    color=color_cycle[i])
            ax2.plot(angles, norm_values, 'o-', linewidth=2,
                    color=color_cycle[i],
                    label=f'{price} Yuan')

        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(nutri_labels, fontsize=8)
        ax2.set_title('Nutrition Comparison (Normalized)',
                     fontweight='bold', fontsize=11, pad=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
                  fontsize=8)

        plt.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/p4_combo_results.png', dpi=300)
        plt.close()
        print('  已保存: p4_combo_results.png')


if __name__ == '__main__':
    # 模块自检
    combos = Problem4Combos()
    results = combos.run()
