"""
==============================================================================
自助量贩餐厅数据生成器 (Cafeteria Data Generator)
==============================================================================
功能说明:
    由于原题附件数据未提供,本模块根据题目描述和真实餐饮场景规律,
    生成高仿真度的测试数据集,包括:
    1. 菜品目录表 (dish_catalog.csv) — 菜品名称、类别、价格、成本、营养成分
    2. 订单流水表 (transaction_records.csv) — 每次消费的完整记录
    3. 日级汇总表 (daily_summary.csv) — 按日聚合的就餐人数、销售额、营养需求

数据生成逻辑:
    - 模拟一家中式自助量贩餐厅, 提供约40道菜品, 分为主食、荤菜、半荤半素、
      素菜、汤品、小吃6个类别
    - 历史数据覆盖2024年12月1日至2025年5月31日(约6个月), 含工作日/周末/节假日效应
    - 每日分午餐(11:00-14:00)和晚餐(17:00-20:00)两个餐次
    - 工作日午餐客流量大(周边上班族), 晚餐和周末客流由家庭/朋友聚餐驱动
    - 不同菜品有不同的人气等级(高/中/低销量), 且存在共购关联关系
    - 营养成分参考《中国食物成分表》典型值进行合理设定

参考文献:
    [1] 中国营养学会. 中国居民膳食营养素参考摄入量(2023版). 人民卫生出版社, 2023.
    [2] 杨月欣. 中国食物成分表(标准版第6版). 北京大学医学出版社, 2019.
==============================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)  # 固定随机种子, 确保结果可复现


def generate_dish_catalog():
    """
    生成菜品目录表 (附件2结构)
    
    返回:
        pd.DataFrame: 包含菜品ID、名称、类别、价格(元)、成本(元)、
                      热量(kcal)、蛋白质(g)、脂肪(g)、碳水化合物(g)
    """
    dishes = [
        # 类别, 名称, 价格, 成本, 热量, 蛋白质, 脂肪, 碳水, 人气等级(1高/2中/3低)
        # ---- 主食类 (6个) ----
        ("主食", "白米饭", 2.0, 0.5, 116, 2.6, 0.3, 25.9, 1),
        ("主食", "蛋炒饭", 5.0, 1.5, 185, 5.2, 7.0, 28.5, 2),
        ("主食", "馒头", 1.5, 0.3, 223, 7.0, 1.1, 44.2, 2),
        ("主食", "花卷", 2.0, 0.5, 211, 6.4, 1.0, 45.6, 3),
        ("主食", "阳春面", 4.0, 1.0, 120, 4.0, 0.5, 26.0, 2),
        ("主食", "扬州炒饭", 6.0, 2.0, 210, 6.5, 8.5, 30.0, 1),
        
        # ---- 荤菜类 (10个) ----
        ("荤菜", "红烧肉", 12.0, 5.0, 395, 15.5, 34.5, 4.2, 1),
        ("荤菜", "宫保鸡丁", 10.0, 4.0, 245, 18.2, 14.5, 10.8, 1),
        ("荤菜", "鱼香肉丝", 9.0, 3.5, 208, 12.5, 13.2, 12.5, 1),
        ("荤菜", "糖醋排骨", 14.0, 6.0, 330, 16.8, 25.0, 8.5, 1),
        ("荤菜", "回锅肉", 11.0, 4.5, 310, 14.5, 24.0, 8.0, 2),
        ("荤菜", "红烧鸡块", 10.0, 4.0, 220, 20.0, 13.0, 6.5, 2),
        ("荤菜", "酱爆牛肉", 15.0, 7.0, 280, 22.0, 18.0, 4.0, 2),
        ("荤菜", "清蒸鲈鱼", 16.0, 8.0, 140, 18.5, 7.0, 0.0, 3),
        ("荤菜", "椒盐虾", 13.0, 6.0, 175, 20.5, 8.5, 5.0, 2),
        ("荤菜", "红烧狮子头", 10.0, 4.0, 350, 14.0, 28.0, 10.0, 2),
        
        # ---- 半荤半素类 (8个) ----
        ("半荤半素", "番茄炒蛋", 6.0, 2.0, 85, 5.5, 5.0, 5.2, 1),
        ("半荤半素", "麻婆豆腐", 5.0, 1.5, 95, 7.0, 6.0, 4.0, 1),
        ("半荤半素", "青椒肉丝", 8.0, 3.0, 165, 10.5, 10.2, 8.5, 1),
        ("半荤半素", "芹菜炒肉", 7.0, 2.5, 150, 9.0, 8.5, 9.0, 2),
        ("半荤半素", "韭菜炒蛋", 5.0, 1.5, 120, 7.0, 7.5, 5.5, 2),
        ("半荤半素", "肉末茄子", 8.0, 3.0, 142, 5.5, 10.0, 8.0, 1),
        ("半荤半素", "家常豆腐", 6.0, 2.0, 110, 8.0, 7.0, 5.0, 3),
        ("半荤半素", "土豆炖牛肉", 10.0, 4.5, 220, 14.0, 12.0, 15.0, 1),
        
        # ---- 素菜类 (8个) ----
        ("素菜", "清炒时蔬", 4.0, 1.0, 45, 3.0, 2.5, 3.5, 1),
        ("素菜", "蒜蓉空心菜", 4.0, 1.0, 40, 3.5, 2.0, 3.0, 2),
        ("素菜", "酸辣土豆丝", 4.0, 1.0, 80, 2.0, 3.5, 13.0, 1),
        ("素菜", "凉拌黄瓜", 3.0, 0.8, 25, 1.2, 1.0, 3.5, 1),
        ("素菜", "干煸四季豆", 5.0, 1.5, 85, 3.5, 5.5, 6.5, 2),
        ("素菜", "白灼西兰花", 5.0, 1.5, 35, 3.7, 0.6, 4.0, 2),
        ("素菜", "清炒豆芽", 3.0, 0.8, 30, 2.5, 0.5, 4.0, 3),
        ("素菜", "香菇青菜", 5.0, 1.5, 50, 4.0, 1.5, 5.0, 2),
        
        # ---- 汤品类 (5个) ----
        ("汤品", "紫菜蛋花汤", 3.0, 0.8, 30, 2.5, 1.0, 2.5, 1),
        ("汤品", "番茄蛋汤", 3.0, 0.8, 35, 2.8, 1.2, 2.8, 2),
        ("汤品", "排骨萝卜汤", 6.0, 2.5, 85, 5.5, 5.0, 4.0, 2),
        ("汤品", "酸辣汤", 4.0, 1.2, 55, 3.0, 2.0, 6.0, 2),
        ("汤品", "鸡汤", 7.0, 3.0, 80, 8.0, 4.5, 2.0, 3),
        
        # ---- 小吃类 (5个) ----
        ("小吃", "春卷", 4.0, 1.2, 180, 4.5, 10.0, 18.0, 2),
        ("小吃", "煎饺", 5.0, 1.8, 220, 8.0, 10.5, 22.0, 2),
        ("小吃", "葱油饼", 3.0, 0.8, 210, 5.0, 10.0, 25.0, 2),
        ("小吃", "炸鸡块", 6.0, 2.0, 260, 18.0, 16.0, 12.0, 1),
        ("小吃", "红糖糍粑", 4.0, 1.2, 195, 3.5, 6.0, 32.0, 3),
    ]
    
    df = pd.DataFrame(dishes, columns=[
        "菜品类别", "菜品名称", "单价_元", "成本_元",
        "热量_kcal", "蛋白质_g", "脂肪_g", "碳水化合物_g", "人气等级"
    ])
    df.index = range(1, len(df) + 1)
    df.index.name = "菜品ID"
    return df


def generate_transaction_data(dish_df, start_date="2024-12-01", end_date="2025-05-31"):
    """
    生成交易流水数据 (附件1结构)
    
    参数:
        dish_df: 菜品目录表
        start_date: 数据起始日期
        end_date: 数据结束日期
    
    返回:
        pd.DataFrame: 交易明细表, 每行一条菜品消费记录
    """
    dishes = dish_df.to_dict("index")
    records = []
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    date_range = (end - start).days + 1
    
    # 定义节假日 (2025年春节前后, 以及法定假日)
    holidays_2025 = [
        "2025-01-01",  # 元旦
        "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31", "2025-02-01", "2025-02-02", "2025-02-03", "2025-02-04",  # 春节假期
        "2025-04-04", "2025-04-05", "2025-04-06",  # 清明节
        "2025-05-01", "2025-05-02", "2025-05-03", "2025-05-04", "2025-05-05",  # 劳动节
    ]
    holiday_set = set(holidays_2025)
    
    order_id = 100000
    day_seed = 2024 * 365 + 12 * 30 + 1  # 用于每日随机种子
    
    for day_offset in range(date_range):
        current_date = start + timedelta(days=day_offset)
        date_str = current_date.strftime("%Y-%m-%d")
        weekday = current_date.weekday()  # 0=周一, 6=周日
        is_workday = weekday < 5 and date_str not in holiday_set
        is_holiday = date_str in holiday_set
        
        # 设置每日随机种子
        np.random.seed(day_seed + day_offset)
        
        # ---- 午餐时段 ----
        lunch_factor = _get_meal_factor(is_workday, is_holiday, "lunch")
        # ---- 晚餐时段 ----
        dinner_factor = _get_meal_factor(is_workday, is_holiday, "dinner")
        
        for meal, meal_factor in [("午餐", lunch_factor), ("晚餐", dinner_factor)]:
            if meal == "午餐":
                num_orders = int(np.random.normal(meal_factor * 85, meal_factor * 15))
                peak_start = np.random.choice(["11:30", "11:45", "12:00", "12:15"])
            else:
                num_orders = int(np.random.normal(meal_factor * 65, meal_factor * 12))
                peak_start = np.random.choice(["17:30", "17:45", "18:00", "18:15"])
            
            num_orders = max(5, num_orders)
            
            for _ in range(num_orders):
                order_id += 1
                # 每个订单的菜品数量: 遵循对数正态分布 (1-8个菜)
                num_items = int(np.random.lognormal(mean=1.0, sigma=0.4))
                num_items = max(1, min(8, num_items))
                
                order_time = _generate_order_time(meal, peak_start)
                
                # 为该订单随机选择菜品 (考虑人气等级和关联关系)
                selected = _select_dishes_for_order(dishes, num_items)
                
                for dish_id, qty in selected:
                    d = dishes[dish_id]
                    records.append({
                        "订单编号": order_id,
                        "消费日期": date_str,
                        "消费时间": order_time,
                        "餐次": meal,
                        "菜品ID": dish_id,
                        "菜品名称": d["菜品名称"],
                        "菜品类别": d["菜品类别"],
                        "消费数量": qty,
                        "单价_元": d["单价_元"],
                        "消费金额": round(d["单价_元"] * qty, 2),
                        "热量_kcal": d["热量_kcal"] * qty,
                        "蛋白质_g": d["蛋白质_g"] * qty,
                        "脂肪_g": d["脂肪_g"] * qty,
                        "碳水化合物_g": d["碳水化合物_g"] * qty,
                        "是否工作日": "是" if is_workday else "否",
                        "星期": ["周一","周二","周三","周四","周五","周六","周日"][weekday],
                    })
    
    df = pd.DataFrame(records)
    return df


def _get_meal_factor(is_workday, is_holiday, meal):
    """
    获取餐次客流系数
    工作日午餐最高(周边上班族), 晚餐其次, 周末/节假日略高于工作日
    """
    if is_holiday:
        if meal == "lunch":
            return 1.05 + np.random.uniform(-0.05, 0.05)
        else:
            return 1.15 + np.random.uniform(-0.05, 0.05)
    elif is_workday:
        if meal == "lunch":
            return 1.0 + np.random.uniform(-0.08, 0.08)
        else:
            return 0.85 + np.random.uniform(-0.05, 0.05)
    else:  # 周末
        if meal == "lunch":
            return 0.9 + np.random.uniform(-0.05, 0.05)
        else:
            return 1.15 + np.random.uniform(-0.05, 0.05)


def _generate_order_time(meal, peak_start):
    """生成合理的消费时间"""
    hour, minute = map(int, peak_start.split(":"))
    base_minutes = hour * 60 + minute
    offset = int(np.random.exponential(scale=25))
    if np.random.random() > 0.5:
        offset = -offset
    actual = max(0, base_minutes + offset)
    h = (actual // 60) % 24
    m = actual % 60
    return f"{h:02d}:{m:02d}"


def _select_dishes_for_order(dishes, num_items):
    """
    为单个订单选择菜品, 模拟真实的关联消费行为
    考虑原则:
    - 人气等级越高的菜品越容易被选 (1>2>3, 权重5:3:1)
    - 至少包含一个主食 (概率约90%)
    - 荤菜和素菜有互补关系
    - 汤品经常与主食一起出现
    - 存在一些关联菜品组合的共购关系
    """
    dish_items = list(dishes.items())
    selected = {}
    
    # 关联规则: 定义共购组合
    association_pairs = [
        # (触发菜品ID, 关联菜品ID)
        (7, 23), (7, 24),  # 宫保鸡丁 -> 番茄炒蛋, 麻婆豆腐
        (8, 23), (8, 31),  # 鱼香肉丝 -> 番茄炒蛋, 酸辣土豆丝
        (1, 27), (1, 35),  # 白米饭 -> 清炒时蔬, 紫菜蛋花汤
        (5, 35),            # 阳春面 -> 紫菜蛋花汤
        (7, 1), (8, 1),    # 宫保鸡丁/鱼香肉丝 -> 白米饭
        (14, 33),           # 肉末茄子 -> 干煸四季豆
        (3, 34),            # 馒头 -> 白灼西兰花
    ]
    
    # 权重: 人气等级1权重5, 等级2权重3, 等级3权重1
    weights = {did: {1:5, 2:3, 3:1}[d["人气等级"]] for did, d in dishes.items()}
    
    # 必选主食 (概率90%)
    staple_ids = [did for did, d in dishes.items() if d["菜品类别"] == "主食"]
    if np.random.random() < 0.9 and num_items > 1:
        staple = np.random.choice(staple_ids)
        selected[staple] = 1
        num_items -= 1
    
    while num_items > 0:
        # 优先从关联菜品中选择
        if np.random.random() < 0.3 and len(selected) > 0:
            triggered = np.random.choice(list(selected.keys()))
            candidates = [pair[1] for pair in association_pairs if pair[0] == triggered and pair[1] not in selected]
            if candidates:
                dish_id = np.random.choice(candidates)
                selected[dish_id] = selected.get(dish_id, 0) + 1
                num_items -= 1
                continue
        
        # 加权随机选择
        avail = [did for did in dishes if did not in selected or selected[did] < 3]
        if not avail:
            avail = list(dishes.keys())
        w = [weights[did] for did in avail]
        w_sum = sum(w)
        if w_sum == 0:
            w = [1] * len(avail)
            w_sum = len(avail)
        probs = [x / w_sum for x in w]
        dish_id = np.random.choice(avail, p=probs)
        selected[dish_id] = selected.get(dish_id, 0) + 1
        num_items -= 1
    
    return list(selected.items())


def generate_daily_summary(trans_df):
    """
    从交易明细表生成日级汇总表
    
    参数:
        trans_df: 交易明细表
    
    返回:
        pd.DataFrame: 日级汇总表
    """
    daily = trans_df.groupby("消费日期").agg(
        就餐人数=("订单编号", "nunique"),
        销售总额=("消费金额", "sum"),
        订单总数=("订单编号", lambda x: x.max() - x.min() + 1 if len(x) > 0 else 0),
        总热量_kcal=("热量_kcal", "sum"),
        总蛋白质_g=("蛋白质_g", "sum"),
        总脂肪_g=("脂肪_g", "sum"),
        总碳水化合物_g=("碳水化合物_g", "sum"),
        人均消费=("消费金额", lambda x: round(x.sum() / trans_df.loc[x.index, "订单编号"].nunique(), 2)),
    ).reset_index()
    
    daily["消费日期"] = pd.to_datetime(daily["消费日期"])
    daily["星期"] = daily["消费日期"].dt.dayofweek.map({0:"周一",1:"周二",2:"周三",3:"周四",4:"周五",5:"周六",6:"周日"})
    daily["是否工作日"] = daily["星期"].apply(lambda x: "是" if x in ["周一","周二","周三","周四","周五"] else "否")
    
    # 添加人均营养指标
    daily["人均热量_kcal"] = (daily["总热量_kcal"] / daily["就餐人数"]).round(1)
    daily["人均蛋白质_g"] = (daily["总蛋白质_g"] / daily["就餐人数"]).round(1)
    daily["人均脂肪_g"] = (daily["总脂肪_g"] / daily["就餐人数"]).round(1)
    daily["人均碳水_g"] = (daily["总碳水化合物_g"] / daily["就餐人数"]).round(1)
    
    return daily


# ==================== 主函数 ====================
def main():
    output_dir = r"/CSJ-MathModeling"
    
    print("=" * 60)
    print("生成自助量贩餐厅仿真数据...")
    print("=" * 60)
    
    # 1. 生成菜品目录
    print("\n[1/3] 生成菜品目录...")
    dish_df = generate_dish_catalog()
    dish_path = f"{output_dir}\\dish_catalog.csv"
    dish_df.to_csv(dish_path, encoding="utf-8-sig")
    print(f"  -> 共 {len(dish_df)} 道菜品, 保存至 dish_catalog.csv")
    
    # 2. 生成交易流水
    print("\n[2/3] 生成交易流水数据 (2024-12-01 至 2025-05-31)...")
    trans_df = generate_transaction_data(dish_df)
    trans_path = f"{output_dir}\\transaction_records.csv"
    trans_df.to_csv(trans_path, encoding="utf-8-sig", index=False)
    print(f"  -> 共 {len(trans_df)} 条消费记录, {trans_df['订单编号'].nunique()} 个订单")
    print(f"  -> 保存至 transaction_records.csv")
    
    # 3. 生成日级汇总
    print("\n[3/3] 生成日级汇总数据...")
    daily_df = generate_daily_summary(trans_df)
    daily_path = f"{output_dir}\\daily_summary.csv"
    daily_df.to_csv(daily_path, encoding="utf-8-sig", index=False)
    print(f"  -> 共 {len(daily_df)} 天数据, 保存至 daily_summary.csv")
    
    # 数据概览
    print("\n" + "=" * 60)
    print("数据概览:")
    print(f"  日均就餐人数: {daily_df['就餐人数'].mean():.0f} 人")
    print(f"  日均销售总额: {daily_df['销售总额'].mean():.0f} 元")
    print(f"  人均消费: {daily_df['人均消费'].mean():.1f} 元")
    print(f"  人均热量: {daily_df['人均热量_kcal'].mean():.0f} kcal")
    print("=" * 60)


if __name__ == "__main__":
    main()
