"""data_loader.py — 数据加载、预处理与特征工程模块"""


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import *
from utils import classify_dish_by_nutrition


class DataLoader:

    def __init__(self):
        print('=' * 60)
        print('数据加载与预处理模块')
        print('=' * 60)

        # 存储所有处理后的数据
        self.df1_raw = None      # 附件1: 餐厅流水 (清洗后)
        self.df2_raw = None      # 附件2: 菜品详情 (清洗后)
        self.df_trans = None     # 融合后的交易明细表
        self.df_daily = None     # 日级汇总表
        self.df_meal = None      # 餐次级汇总表
        self.dish_info = None    # 菜品信息表
        self.basket_data = None  # 购物篮格式
        self.basket_binary = None # 二值化购物篮
        self.basket_filtered = None # 过滤低频菜品的购物篮

        # 执行 ETL 流水线
        self._load_data()
        self._clean_and_preprocess()
        self._feature_engineering()
        self._build_aggregations()
        self._build_basket()

    def _load_data(self):
        print('\n[1/5] 加载原始数据（所有 sheet）...')

        df1_sheets = []
        for sheet_name in pd.ExcelFile(ATTACHMENT1).sheet_names:
            df_sheet = pd.read_excel(ATTACHMENT1, sheet_name=sheet_name)
            df1_sheets.append(df_sheet)
            print(f'  附件1 [{sheet_name}]: {len(df_sheet):,} 行')
        self.df1_raw = pd.concat(df1_sheets, ignore_index=True)
        print(f'  附件1 合计加载完成: {self.df1_raw.shape[0]:,} 行, '
              f'{self.df1_raw.shape[1]} 列')

        df2_sheets = []
        for sheet_name in pd.ExcelFile(ATTACHMENT2).sheet_names:
            df_sheet = pd.read_excel(ATTACHMENT2, sheet_name=sheet_name)
            df2_sheets.append(df_sheet)
            if len(df_sheet) > 500:
                print(f'  附件2 [{sheet_name}]: {len(df_sheet):,} 行')
            else:
                pass  # 小型 sheet 不逐个打印
        self.df2_raw = pd.concat(df2_sheets, ignore_index=True)
        print(f'  附件2 合计加载完成: {self.df2_raw.shape[0]:,} 行, '
              f'{self.df2_raw.shape[1]} 列')
        print(f'  附件2 覆盖订单数: {self.df2_raw["indent_id"].nunique():,}')

        # 显示列名以供验证
        print(f'  附件1列名: {list(self.df1_raw.columns)}')
        print(f'  附件2列名: {list(self.df2_raw.columns)}')

    def _clean_and_preprocess(self):
        print('\n[2/5] 数据清洗与预处理...')

        df1 = self.df1_raw.copy()
        df2 = self.df2_raw.copy()


        # 步骤1: 提取日期时间特征
        # consume_time 字段格式: "YYYY-MM-DD HH:MM:SS"
        df1['consume_time'] = pd.to_datetime(df1['consume_time'])
        df1['date'] = df1['consume_time'].dt.date                    # 日期 (date 对象)
        df1['hour'] = df1['consume_time'].dt.hour                    # 小时 (0-23)
        df1['day_of_week'] = df1['consume_time'].dt.dayofweek        # 星期 (0=Monday)
        df1['is_weekend'] = df1['day_of_week'].isin([5, 6]).astype(int)  # 是否周末
        df1['month'] = df1['consume_time'].dt.month                  # 月份 (1-12)
        df1['year'] = df1['consume_time'].dt.year                    # 年份

        # 步骤2: 划分餐次 (lunch / dinner / other)
        # 依据 config.py 中的 LUNCH_START/END 和 DINNER_START/END
        df1['meal_period'] = 'other'
        df1.loc[(df1['hour'] >= LUNCH_START) & (df1['hour'] < LUNCH_END),
                'meal_period'] = 'lunch'
        df1.loc[(df1['hour'] >= DINNER_START) & (df1['hour'] < DINNER_END),
                'meal_period'] = 'dinner'

        # 步骤3: 异常值检测 (IQR 方法)
        # 对消费金额和营养成分使用 1%-99% 分位数标记潜在异常
        # 注意: 仅标记不剔除，因为餐饮场景中的高消费 (如团体订餐) 可能是合理的
        cols_to_check = ['consume_money', 'calories', 'protein',
                         'fat', 'carbohydrates']
        for col in cols_to_check:
            Q1 = df1[col].quantile(0.01)
            Q3 = df1[col].quantile(0.99)
            outliers = (df1[col] < Q1) | (df1[col] > Q3)
            if outliers.sum() > 0:
                print(f'  {col}: 标记 {outliers.sum()} 个潜在异常值 '
                      f'(1%-99%分位外, Q1={Q1:.1f}, Q3={Q3:.1f})')

        # 步骤4: 剔除无用的全缺失列
        # wallet_id, card_serial, user_phone_number, qr_code 在所有记录中均为空值
        # 这些字段对分析无贡献，直接删除
        df1 = df1.drop(columns=['wallet_id', 'card_serial',
                                 'user_phone_number', 'qr_code'], errors='ignore')

        # 步骤5: consume_way 分布分析
        # consume_way=3 占比 99.97% (65514条)，consume_way=2 仅 20条
        # 两者差异极小，后续分析中不作区分
        way_dist = df1['consume_way'].value_counts().to_dict()
        print(f'  consume_way分布: {way_dist}')

        self.df1_raw = df1


        # 步骤1: 数值列类型检查和转换
        # 确保价格、重量等数值字段为正确的数值类型
        df2['total_price'] = pd.to_numeric(df2['total_price'], errors='coerce')
        df2['weight'] = pd.to_numeric(df2['weight'], errors='coerce')
        df2['unit_price'] = pd.to_numeric(df2['unit_price'], errors='coerce')

        # 步骤2: 菜品名称去除首尾空格
        # 避免 " 米饭" 和 "米饭" 被视为不同菜品
        df2['dish_name'] = df2['dish_name'].str.strip()

        self.df2_raw = df2

        print(f'  清洗完成')

    def _classify_dish(self, name, nutrition=None):
        if not isinstance(name, str):
            return '其他'

        # 按优先级检查: 主食 > 荤菜 > 半荤半素 > 素菜
        # 高的优先级确保 "肉末茄子饭" → 主食 而非 半荤半素
        for cat in ['主食', '荤菜', '半荤半素', '素菜']:
            for kw in CATEGORY_KEYWORDS[cat]:
                if kw in name:
                    return cat
        return '其他'

    def _nutritional_feature_classify(self, row):
        # 标准化营养成分: 如果 weight 字段可用，计算每100g的含量
        weight = row.get('weight', np.nan)
        if pd.notna(weight) and weight > 0:
            # 换算为每100g的营养含量用于统一比较
            scale = 100.0 / weight
            cal = row.get('calories', 0) * scale
            protein = row.get('protein', 0) * scale
            fat = row.get('fat', 0) * scale
            carbs = row.get('carbohydrates', 0) * scale
            fiber = row.get('fiber', 0) * scale
        else:
            # 若无 weight 信息，使用原始值 (近似处理)
            cal = row.get('calories', 0)
            protein = row.get('protein', 0)
            fat = row.get('fat', 0)
            carbs = row.get('carbohydrates', 0)
            fiber = row.get('fiber', 0)

        return classify_dish_by_nutrition(cal, protein, fat, carbs, fiber)

    def _feature_engineering(self):
        print('\n[3/5] 特征工程...')

        df2 = self.df2_raw.copy()
        df2['category'] = df2['dish_name'].apply(self._classify_dish)

        # 打印第一轮分类统计
        cat_counts = df2.groupby('category')['dish_name'].nunique()
        total_dishes = df2['dish_name'].nunique()
        print('  菜品分类统计 (第一轮: 关键词匹配):')
        for cat, cnt in cat_counts.items():
            pct = cnt / total_dishes * 100
            print(f'    {cat}: {cnt} 种 ({pct:.1f}%)')

        # 对 "其他" 类菜品使用营养特征进行二次分类
        other_mask = df2['category'] == '其他'
        if other_mask.sum() > 0:
            # 获取每种 "其他" 菜品的营养平均值
            other_dishes = df2[other_mask].groupby('dish_name').agg({
                'calories': 'mean',
                'protein': 'mean',
                'fat': 'mean',
                'carbohydrates': 'mean',
                'fiber': 'mean',
                'weight': 'mean',
            }).reset_index()

            # 对每种菜品进行营养特征分类
            reclassified = 0
            for _, row in other_dishes.iterrows():
                nutr_cat = self._nutritional_feature_classify(row)
                if nutr_cat != '其他':
                    # 更新该菜品的分类
                    df2.loc[(df2['category'] == '其他') &
                            (df2['dish_name'] == row['dish_name']),
                            'category'] = nutr_cat
                    reclassified += 1

            # 打印第二轮分类统计
            cat_counts2 = df2.groupby('category')['dish_name'].nunique()
            print(f'\n  菜品分类统计 (第二轮: 营养特征补充, 重新分类 {reclassified} 种):')
            for cat, cnt in cat_counts2.items():
                pct = cnt / total_dishes * 100
                print(f'    {cat}: {cnt} 种 ({pct:.1f}%)')

        self.df2_raw = df2

        # 对每种菜品 (dish_serial) 进行聚合汇总
        self.dish_info = df2.groupby('dish_serial').agg(
            dish_name=('dish_name', 'first'),
            category=('category', 'first'),
            unit_price=('unit_price', 'mean'),           # 平均单价
            avg_weight=('weight', 'mean'),               # 平均重量
            calories=('calories', 'mean'),               # 平均热量
            carbohydrates=('carbohydrates', 'mean'),     # 平均碳水
            protein=('protein', 'mean'),                 # 平均蛋白质
            fat=('fat', 'mean'),                         # 平均脂肪
            fiber=('fiber', 'mean'),                     # 平均纤维
            total_orders=('indent_details_id', 'count'), # 总订单次数
            total_revenue=('total_price', 'sum'),        # 总销售收入
        ).reset_index()

        self.dish_info['cost_ratio'] = self.dish_info['category'].map(
            COST_RATIO_BY_CATEGORY
        ).fillna(0.45)
        self.dish_info['unit_cost'] = (
            self.dish_info['unit_price'] * self.dish_info['cost_ratio']
        )
        self.dish_info['unit_profit'] = (
            self.dish_info['unit_price'] - self.dish_info['unit_cost']
        )
        self.dish_info['profit_margin'] = (
            self.dish_info['unit_profit'] / self.dish_info['unit_price']
        )

        print(f'  菜品信息表构建完成: {len(self.dish_info)} 种菜品')

        # 附件1 提供订单级别的营养汇总 (覆盖100%订单)
        # 附件2 提供菜品级别的详细信息 (覆盖约18%订单)
        # 融合后可在订单粒度上同时看到宏观营养和微观菜品信息
        df1_sub = self.df1_raw[[
            'indent_id', 'consume_time', 'date', 'hour',
            'day_of_week', 'is_weekend', 'month', 'year',
            'meal_period', 'consume_money', 'consume_way',
            'calories', 'carbohydrates', 'protein', 'fat', 'fiber'
        ]]

        self.df_trans = df2.merge(df1_sub, on='indent_id', how='left',
                                   suffixes=('_detail', '_order'))

        print(f'  交易明细表构建完成: {len(self.df_trans):,} 行')
        print(f'  覆盖订单数: {self.df_trans["indent_id"].nunique():,}')

    def _build_aggregations(self):
        print('\n[4/5] 构建汇总表...')

        df1 = self.df1_raw

        daily_agg = df1.groupby('date').agg(
            # 就餐人数: 每个 indent_id 视为一位顾客
            total_orders=('indent_id', 'nunique'),
            # 销售总额: 当日所有消费金额之和
            total_sales=('consume_money', 'sum'),
            # 客单价: 人均消费金额
            avg_order_value=('consume_money', 'mean'),
            # 营养素总量: 当日所有订单的营养素之和
            total_calories=('calories', 'sum'),
            total_carbohydrates=('carbohydrates', 'sum'),
            total_protein=('protein', 'sum'),
            total_fat=('fat', 'sum'),
            total_fiber=('fiber', 'sum'),
            # 时间特征
            day_of_week=('day_of_week', 'first'),
            is_weekend=('is_weekend', 'first'),
            month=('month', 'first'),
            year=('year', 'first'),
        ).reset_index()

        # 日期格式化和排序
        daily_agg['date'] = pd.to_datetime(daily_agg['date'])
        daily_agg = daily_agg.sort_values('date').reset_index(drop=True)

        # 计算人均营养指标
        # 人均摄入量 = 总摄入量 / 就餐人数
        for nutri in ['calories', 'carbohydrates', 'protein', 'fat', 'fiber']:
            col = f'total_{nutri}'
            daily_agg[f'avg_{nutri}_per_person'] = (
                daily_agg[col] / daily_agg['total_orders']
            )

        self.df_daily = daily_agg
        print(f'  日级汇总表: {len(daily_agg)} 天')
        print(f'  日期范围: {daily_agg["date"].min().date()} 至 '
              f'{daily_agg["date"].max().date()}')
        print(f'  日均订单: {daily_agg["total_orders"].mean():.0f}, '
              f'日均销售额: {daily_agg["total_sales"].mean():.0f} 元')

        # 仅保留午餐和晚餐 (meal_period != 'other')
        meal_agg = df1[df1['meal_period'] != 'other'].groupby(
            ['date', 'meal_period']
        ).agg(
            total_orders=('indent_id', 'nunique'),
            total_sales=('consume_money', 'sum'),
            total_calories=('calories', 'sum'),
            total_carbohydrates=('carbohydrates', 'sum'),
            total_protein=('protein', 'sum'),
            total_fat=('fat', 'sum'),
            total_fiber=('fiber', 'sum'),
        ).reset_index()

        meal_agg['date'] = pd.to_datetime(meal_agg['date'])
        meal_agg = meal_agg.sort_values(['date', 'meal_period']).reset_index(drop=True)

        self.df_meal = meal_agg
        lunch_count = len(meal_agg[meal_agg['meal_period'] == 'lunch'])
        dinner_count = len(meal_agg[meal_agg['meal_period'] == 'dinner'])
        print(f'  餐次级汇总表: {len(meal_agg)} 条记录')
        print(f'  午餐记录: {lunch_count} 条')
        print(f'  晚餐记录: {dinner_count} 条')

    def _build_basket(self):
        print('\n[5/5] 构建购物篮数据...')

        df2 = self.df2_raw

        # 创建购物篮矩阵: 行=订单ID (indent_id), 列=菜品名称 (dish_name)
        # pivot_table 将长格式数据转换为宽格式矩阵
        basket = df2.pivot_table(
            index='indent_id',
            columns='dish_name',
            values='total_price',
            aggfunc='count',   # 计数每条菜品在订单中的出现次数
            fill_value=0       # 未购买的菜品填 0
        )

        # 转换为二值矩阵 (0/1): 不考虑购买数量，只关心是否购买
        self.basket_binary = (basket > 0).astype(int)

        # 计算每种菜品的出现频次
        dish_freq = self.basket_binary.sum(axis=0)

        # 过滤低频菜品: 至少出现 50 次才保留
        # 在 11,828 个订单中, 50 次对应约 0.42% 的支持度
        # 过滤后的菜品数量约 100-150 种 (远小于原始 237 种)
        frequent_dishes = dish_freq[dish_freq >= 50].index
        self.basket_filtered = self.basket_binary[frequent_dishes]

        print(f'  购物篮数据: {self.basket_binary.shape[0]:,} 个订单, '
              f'{self.basket_binary.shape[1]} 种菜品')
        print(f'  过滤后 (出现≥50次): {self.basket_filtered.shape[1]} 种菜品')

        self.basket_data = self.basket_filtered


    def get_daily_data(self):
        return self.df_daily

    def get_meal_data(self):
        return self.df_meal

    def get_transaction_data(self):
        return self.df_trans

    def get_dish_info(self):
        return self.dish_info

    def get_basket_data(self):
        return self.basket_data

    def get_lunch_data(self):
        return self.df1_raw[self.df1_raw['meal_period'] == 'lunch']

    def get_dinner_data(self):
        return self.df1_raw[self.df1_raw['meal_period'] == 'dinner']

    def print_summary(self):
        print('\n' + '=' * 60)
        print('数据摘要')
        print('=' * 60)

        df1 = self.df1_raw
        df2 = self.df2_raw
        daily = self.df_daily

        # 附件1 摘要
        print(f'\n附件1 (流水数据 — 3 sheet 合计):')
        print(f'  总订单数: {df1["indent_id"].nunique():,}')
        print(f'  总交易记录: {len(df1):,}')
        print(f'  日期跨度: {df1["date"].min()} 至 {df1["date"].max()}')
        print(f'  营业天数: {df1["date"].nunique()}')
        print(f'  日均订单数: {daily["total_orders"].mean():.0f}')
        print(f'  日均销售额: {daily["total_sales"].mean():.0f} 元')
        print(f'  平均客单价: {daily["avg_order_value"].mean():.2f} 元')
        print(f'  人均热量: {daily["avg_calories_per_person"].mean():.0f} kcal')

        # 附件2 摘要
        print(f'\n附件2 (菜品详情 — 15 sheet 合计):')
        print(f'  总记录数: {len(df2):,}')
        print(f'  涉及订单数: {df2["indent_id"].nunique():,}')
        print(f'  唯一菜品数: {df2["dish_name"].nunique()}')
        print(f'  平均每单菜品数: '
              f'{len(df2) / max(df2["indent_id"].nunique(), 1):.1f}')
        print(f'  平均菜品单价: {df2["unit_price"].mean():.2f} 元')

        # 餐次分布
        print(f'\n餐次分布:')
        for meal in ['lunch', 'dinner', 'other']:
            cnt = len(df1[df1['meal_period'] == meal])
            pct = cnt / len(df1) * 100
            print(f'  {meal}: {cnt:,} 条 ({pct:.1f}%)')


def load_all_data():
    return DataLoader()


if __name__ == '__main__':
    # 模块自检: 运行数据加载并打印摘要
    loader = load_all_data()
    loader.print_summary()
