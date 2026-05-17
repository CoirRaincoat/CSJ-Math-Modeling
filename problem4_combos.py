"""problem4_combos.py — 问题4: 不同价位套餐优化设计"""


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

    def __init__(self, loader=None):
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

        # Bootstrap 稳定规则配对 (Iteration 5)
        # 来自 validate_reliability.py 的 500 次 Bootstrap 验证:
        # 生存率>80%的规则均为素菜/半荤→酱鸭腿, 是数据中最可靠的关联
        self.stable_pairs = set()

        self.results = {}

    def _build_dish_database(self):
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
                assoc_score = min(1.0, assoc_score / pair_count * 50)

        # 5. 价格符合度
        price_deviation = abs(total_price - target_price) / max(target_price, 1)
        price_fit = max(0, 1 - price_deviation * 3)

        # 6. 类别多样性奖励
        categories = set(d['category'] for d in combo_dishes)
        diversity_bonus = min(1.0, len(categories) / 3) * 0.10

        # 验证报告表明: 生存率>80%的规则均涉及酱鸭腿作为后件
        # 若套餐包含此类稳定配对，给予额外奖励
        stable_bonus = 0.0
        if hasattr(self, 'stable_pairs') and self.stable_pairs:
            for i, d1 in enumerate(combo_dishes):
                for j, d2 in enumerate(combo_dishes):
                    if i < j:
                        pair = (d1['name'], d2['name'])
                        pair_rev = (d2['name'], d1['name'])
                        if pair in self.stable_pairs or pair_rev in self.stable_pairs:
                            stable_bonus = max(stable_bonus, 0.05)

        score = (
            w['popularity'] * popularity +
            w['nutrition'] * nutrition_score +
            w['profit'] * profit_score +
            w['association'] * max(0, assoc_score) +
            0.15 * price_fit +
            diversity_bonus +
            stable_bonus
        )

        # 惩罚: 菜品数量不合规
        n_dishes = len(combo_dishes)
        if n_dishes < COMBO_MIN_DISHES:
            score *= 0.5
        elif n_dishes > COMBO_MAX_DISHES:
            score *= 0.7

        return score

    def _greedy_search(self, target_price, max_dishes=None):
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
                names = [d['name'] for d in combo]
                if len(names) != len(set(names)):
                    continue  # 有重复，跳过此候选
                    
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

        self._plot_combo_results(all_combos)

        print('\n问题4 套餐设计完成!')
        return self.results

    def _plot_combo_results(self, all_combos):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        prices = sorted(all_combos.keys())

        ax1 = axes[0]

        metrics = {
            'Total Price (Yuan)': [
                all_combos[p]['total_price'] for p in prices
            ],
            '利润率 (%)': [
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
        ax1.set_xticklabels([f'{p} 元' for p in prices], fontsize=10)
        ax1.set_ylabel('数值')
        ax1.set_title('各价位套餐指标对比',
                     fontweight='bold', fontsize=11)
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(axis='y', alpha=0.3)

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
                    label=f'{price} 元')

        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(nutri_labels, fontsize=8)
        ax2.set_title('营养成分对比 (归一化)',
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
