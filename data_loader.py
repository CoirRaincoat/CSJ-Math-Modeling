"""
data_loader.py — 数据加载、预处理与特征工程模块
==============================================
负责：
1. 加载附件1（餐厅流水）和附件2（菜品消费详情）
2. 数据清洗：缺失值处理、异常值检测、数据类型转换
3. 特征工程：日期特征提取、餐次划分、菜品分类
4. 构建多层级数据集：交易明细表、日级汇总表、餐次级汇总表
5. 构建关联规则分析所需的购物篮格式数据

参考文献：
  [4] Hyndman, R.J., Athanasopoulos, G. "Forecasting: Principles and Practice"
     时间序列特征工程方法
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import *


class DataLoader:
    """
    数据加载与预处理类
    
    封装了所有数据加载、清洗、特征工程逻辑，
    对外提供统一的接口获取不同粒度的数据。
    """
    
    def __init__(self):
        """初始化，加载原始数据"""
        print('=' * 60)
        print('数据加载与预处理模块')
        print('=' * 60)
        
        self.df1_raw = None   # 附件1原始数据
        self.df2_raw = None   # 附件2原始数据
        self.df_trans = None  # 交易明细表（融合后）
        self.df_daily = None  # 日级汇总表
        self.df_meal = None   # 餐次级汇总表
        self.dish_info = None # 菜品信息表
        self.basket_data = None  # 购物篮格式（用于关联规则）
        
        self._load_data()
        self._clean_and_preprocess()
        self._feature_engineering()
        self._build_aggregations()
        self._build_basket()
        
    def _load_data(self):
        """加载附件1和附件2原始数据"""
        print('\n[1/5] 加载原始数据...')
        
        # 加载附件1：餐厅流水数据
        self.df1_raw = pd.read_excel(ATTACHMENT1)
        print(f'  附件1加载完成: {self.df1_raw.shape[0]} 行, {self.df1_raw.shape[1]} 列')
        
        # 加载附件2：菜品消费详情数据
        self.df2_raw = pd.read_excel(ATTACHMENT2)
        print(f'  附件2加载完成: {self.df2_raw.shape[0]} 行, {self.df2_raw.shape[1]} 列')
        
        # 显示列名
        print(f'  附件1列名: {list(self.df1_raw.columns)}')
        print(f'  附件2列名: {list(self.df2_raw.columns)}')
        
    def _clean_and_preprocess(self):
        """
        数据清洗与预处理
        
        处理步骤：
        1. 提取日期时间特征
        2. 识别并处理异常值（基于IQR方法）
        3. 处理缺失值（用户身份字段全缺失，需剔除）
        4. 将consume_way映射为午餐/晚餐
        """
        print('\n[2/5] 数据清洗与预处理...')
        
        df1 = self.df1_raw.copy()
        df2 = self.df2_raw.copy()
        
        # --- 附件1清洗 ---
        # 1. 提取日期时间字段
        df1['consume_time'] = pd.to_datetime(df1['consume_time'])
        df1['date'] = df1['consume_time'].dt.date
        df1['hour'] = df1['consume_time'].dt.hour
        df1['day_of_week'] = df1['consume_time'].dt.dayofweek  # 0=Monday
        df1['is_weekend'] = df1['day_of_week'].isin([5, 6]).astype(int)
        df1['month'] = df1['consume_time'].dt.month
        df1['year'] = df1['consume_time'].dt.year
        
        # 2. 划分餐次（午餐/晚餐）
        df1['meal_period'] = 'other'
        df1.loc[(df1['hour'] >= LUNCH_START) & (df1['hour'] < LUNCH_END), 'meal_period'] = 'lunch'
        df1.loc[(df1['hour'] >= DINNER_START) & (df1['hour'] < DINNER_END), 'meal_period'] = 'dinner'
        
        # 3. 异常值检测（消费金额和营养成分）
        # 使用IQR方法检测异常值，但不剔除（仅标记），因为在餐饮场景中
        # 大额消费（如团体订餐）可能是合理的
        for col in ['consume_money', 'calories', 'protein', 'fat', 'carbohydrates']:
            Q1 = df1[col].quantile(0.01)
            Q3 = df1[col].quantile(0.99)
            outliers = (df1[col] < Q1) | (df1[col] > Q3)
            if outliers.sum() > 0:
                print(f'  {col}: 标记 {outliers.sum()} 个潜在异常值 (1%-99%分位外)')
        
        # 4. 剔除无用的全缺失列
        # wallet_id, card_serial, user_phone_number, qr_code 全部缺失
        df1 = df1.drop(columns=['wallet_id', 'card_serial', 
                                 'user_phone_number', 'qr_code'], errors='ignore')
        
        # 5. consume_way处理
        # consume_way=2(20条), =3(65514条)，差异极小，合并处理
        print(f'  consume_way分布: {df1["consume_way"].value_counts().to_dict()}')
        
        self.df1_raw = df1
        
        # --- 附件2清洗 ---
        # 1. 数值列类型检查
        df2['total_price'] = pd.to_numeric(df2['total_price'], errors='coerce')
        df2['weight'] = pd.to_numeric(df2['weight'], errors='coerce')
        df2['unit_price'] = pd.to_numeric(df2['unit_price'], errors='coerce')
        
        # 2. 菜品名称去空格
        df2['dish_name'] = df2['dish_name'].str.strip()
        
        self.df2_raw = df2
        
        print(f'  清洗完成')
        
    def _classify_dish(self, name):
        """
        根据菜品名称关键词进行菜品分类
        
        分类逻辑：
        - 按优先级从高到低匹配关键词
        - 主食优先级最高（因"肉末茄子饭"应归为"主食"而非"半荤半素"）
        - 荤菜次之
        - 半荤半素再次
        - 最后匹配素菜
        
        Args:
            name: 菜品名称字符串
            
        Returns:
            str: 菜品类别（主食/荤菜/半荤半素/素菜/其他）
        """
        if not isinstance(name, str):
            return '其他'
        
        # 按优先级检查
        for cat in ['主食', '荤菜', '半荤半素', '素菜']:
            for kw in CATEGORY_KEYWORDS[cat]:
                if kw in name:
                    return cat
        return '其他'
    
    def _feature_engineering(self):
        """
        特征工程
        
        1. 为附件2的菜品添加分类标签
        2. 构建菜品信息字典（营养值、价格等）
        3. 融合附件1和附件2
        """
        print('\n[3/5] 特征工程...')
        
        # --- 菜品分类 ---
        df2 = self.df2_raw.copy()
        df2['category'] = df2['dish_name'].apply(self._classify_dish)
        
        # 打印分类统计
        cat_counts = df2.groupby('category')['dish_name'].nunique()
        print('  菜品分类统计（唯一菜品数）:')
        for cat, cnt in cat_counts.items():
            print(f'    {cat}: {cnt} 种')
        
        self.df2_raw = df2
        
        # --- 构建菜品信息表 ---
        # 按 dish_serial 聚合，取平均值和名称
        self.dish_info = df2.groupby('dish_serial').agg(
            dish_name=('dish_name', 'first'),
            category=('category', 'first'),
            unit_price=('unit_price', 'mean'),
            avg_weight=('weight', 'mean'),
            calories=('calories', 'mean'),
            carbohydrates=('carbohydrates', 'mean'),
            protein=('protein', 'mean'),
            fat=('fat', 'mean'),
            fiber=('fiber', 'mean'),
            total_orders=('indent_details_id', 'count'),
            total_revenue=('total_price', 'sum'),
        ).reset_index()
        
        self.dish_info['unit_cost'] = self.dish_info['unit_price'] * 0.45  # 估算成本率45%
        self.dish_info['unit_profit'] = self.dish_info['unit_price'] - self.dish_info['unit_cost']
        self.dish_info['profit_margin'] = self.dish_info['unit_profit'] / self.dish_info['unit_price']
        
        print(f'  菜品信息表构建完成: {len(self.dish_info)} 种菜品')
        
        # --- 融合附件1和附件2 ---
        # 以附件1为基础（包含所有订单的营养汇总），
        # 附件2提供菜品级别的详细信息（仅覆盖部分订单）
        df1_sub = self.df1_raw[['indent_id', 'consume_time', 'date', 'hour',
                                 'day_of_week', 'is_weekend', 'month', 'year',
                                 'meal_period', 'consume_money', 'consume_way',
                                 'calories', 'carbohydrates', 'protein', 'fat', 'fiber']]
        
        # 对于在附件2中有详细记录的订单，融合菜品信息
        self.df_trans = df2.merge(df1_sub, on='indent_id', how='left',
                                   suffixes=('_detail', '_order'))
        
        print(f'  交易明细表构建完成: {len(self.df_trans)} 行')
        
    def _build_aggregations(self):
        """构建日级和餐次级汇总表"""
        print('\n[4/5] 构建汇总表...')
        
        df1 = self.df1_raw
        
        # --- 日级汇总表 ---
        daily_agg = df1.groupby('date').agg(
            total_orders=('indent_id', 'nunique'),      # 就餐人数（每个indent_id视为一位顾客）
            total_sales=('consume_money', 'sum'),        # 销售总额
            avg_order_value=('consume_money', 'mean'),   # 客单价
            total_calories=('calories', 'sum'),          # 总热量
            total_carbohydrates=('carbohydrates', 'sum'),# 总碳水
            total_protein=('protein', 'sum'),            # 总蛋白质
            total_fat=('fat', 'sum'),                    # 总脂肪
            total_fiber=('fiber', 'sum'),                # 总纤维
            day_of_week=('day_of_week', 'first'),        # 星期
            is_weekend=('is_weekend', 'first'),          # 是否周末
            month=('month', 'first'),
            year=('year', 'first'),
        ).reset_index()
        
        daily_agg['date'] = pd.to_datetime(daily_agg['date'])
        daily_agg = daily_agg.sort_values('date').reset_index(drop=True)
        
        # 添加人均营养指标
        for nutri in ['calories', 'carbohydrates', 'protein', 'fat', 'fiber']:
            col = f'total_{nutri}'
            daily_agg[f'avg_{nutri}_per_person'] = daily_agg[col] / daily_agg['total_orders']
        
        self.df_daily = daily_agg
        print(f'  日级汇总表: {len(daily_agg)} 天')
        print(f'  日期范围: {daily_agg["date"].min().date()} 至 {daily_agg["date"].max().date()}')
        
        # --- 餐次级汇总表 ---
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
        print(f'  餐次级汇总表: {len(meal_agg)} 条记录')
        print(f'  午餐记录: {len(meal_agg[meal_agg["meal_period"]=="lunch"])}')
        print(f'  晚餐记录: {len(meal_agg[meal_agg["meal_period"]=="dinner"])}')
        
    def _build_basket(self):
        """
        构建购物篮格式数据（用于关联规则分析）
        
        每行 = 一个订单，每列 = 一种菜品的购买数量
        这种格式适用于Apriori/FP-Growth算法
        """
        print('\n[5/5] 构建购物篮数据...')
        
        # 使用附件2中有菜品明细的订单
        df2 = self.df2_raw
        
        # 创建购物篮矩阵：行=订单ID，列=菜品名称，值=购买数量
        basket = df2.pivot_table(
            index='indent_id',
            columns='dish_name',
            values='total_price',
            aggfunc='count',
            fill_value=0
        )
        
        # 转换为二值矩阵（0/1表示是否购买）
        self.basket_binary = (basket > 0).astype(int)
        
        # 过滤低频菜品（出现次数 < 50 的菜品排除，减少噪音）
        dish_freq = self.basket_binary.sum(axis=0)
        frequent_dishes = dish_freq[dish_freq >= 50].index
        self.basket_filtered = self.basket_binary[frequent_dishes]
        
        print(f'  购物篮数据: {self.basket_binary.shape[0]} 个订单, '
              f'{self.basket_binary.shape[1]} 种菜品')
        print(f'  过滤后: {self.basket_filtered.shape[1]} 种菜品（出现≥50次）')
        
        self.basket_data = self.basket_filtered
        
    def get_daily_data(self):
        """返回日级汇总数据"""
        return self.df_daily
    
    def get_meal_data(self):
        """返回餐次级汇总数据"""
        return self.df_meal
    
    def get_transaction_data(self):
        """返回交易明细数据（融合后）"""
        return self.df_trans
    
    def get_dish_info(self):
        """返回菜品信息表"""
        return self.dish_info
    
    def get_basket_data(self):
        """返回购物篮数据"""
        return self.basket_data
    
    def get_lunch_data(self):
        """返回午餐数据"""
        return self.df1_raw[self.df1_raw['meal_period'] == 'lunch']
    
    def get_dinner_data(self):
        """返回晚餐数据"""
        return self.df1_raw[self.df1_raw['meal_period'] == 'dinner']
    
    def print_summary(self):
        """打印数据摘要统计"""
        print('\n' + '=' * 60)
        print('数据摘要')
        print('=' * 60)
        
        df1 = self.df1_raw
        df2 = self.df2_raw
        daily = self.df_daily
        
        print(f'\n附件1（流水数据）:')
        print(f'  总订单数: {df1["indent_id"].nunique():,}')
        print(f'  总交易记录: {len(df1):,}')
        print(f'  日期跨度: {df1["date"].min()} 至 {df1["date"].max()}')
        print(f'  日均订单数: {daily["total_orders"].mean():.0f}')
        print(f'  日均销售额: {daily["total_sales"].mean():.0f} 元')
        print(f'  平均客单价: {daily["avg_order_value"].mean():.2f} 元')
        
        print(f'\n附件2（菜品详情）:')
        print(f'  总记录数: {len(df2):,}')
        print(f'  涉及订单数: {df2["indent_id"].nunique():,}')
        print(f'  唯一菜品数: {df2["dish_name"].nunique()}')
        print(f'  平均每单菜品数: {len(df2) / df2["indent_id"].nunique():.1f}')
        print(f'  平均菜品单价: {df2["unit_price"].mean():.2f} 元')
        
        print(f'\n餐次分布:')
        for meal in ['lunch', 'dinner', 'other']:
            cnt = len(df1[df1['meal_period'] == meal])
            print(f'  {meal}: {cnt:,} 条 ({cnt/len(df1)*100:.1f}%)')


def load_all_data():
    """
    便捷函数：一次性加载并预处理所有数据
    
    Returns:
        DataLoader: 包含所有处理后数据的对象
    """
    return DataLoader()


if __name__ == '__main__':
    # 测试数据加载
    loader = load_all_data()
    loader.print_summary()
