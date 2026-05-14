"""
config.py — 全局配置文件
======================
定义项目路径、常量、菜品分类规则、营养参考标准等全局参数。
所有模块共享此配置，确保数据一致性和可维护性。

参考文献：
  [1] 中国居民膳食营养素参考摄入量(DRIs), 中国营养学会, 2023版
  [2] 中国居民膳食指南(2022), 中国营养学会
  
项目结构：
  data_loader.py  — 数据加载与预处理
  problem1_analysis.py  — 问题1：数据统计分析与关联规则
  problem2_prediction.py — 问题2：需求预测
  problem3_optimization.py — 问题3：备菜优化
  problem4_combos.py   — 问题4：套餐设计
  problem5_strategy.py — 问题5：经营策略建议
  main.py        — 主入口，串联全部模块
"""

import os

# ============================================================
# 1. 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR  # 数据文件与代码在同一目录
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 附件文件原始名称（仅供参考，因系统编码差异运行时自动匹配）：
#   附件1 = "附件1餐厅销售流水信息表.xlsx"  (~12.8MB)
#   附件2 = "附件2部分消费订单菜品具体信息表.xlsx" (~5.5MB)
#   附件3 = "附件3数据说明.xlsx" (~12KB)
# 
# 动态发现附件文件（按文件大小特征匹配，避免中文编码问题）
def _find_attachments():
    """根据文件大小特征匹配附件文件"""
    xlsx_files = []
    for f in os.listdir(DATA_DIR):
        if f.endswith('.xlsx'):
            fp = os.path.join(DATA_DIR, f)
            xlsx_files.append((fp, os.path.getsize(fp)))
    
    xlsx_files.sort(key=lambda x: x[1], reverse=True)  # 按大小降序
    
    if len(xlsx_files) >= 3:
        a1_path = xlsx_files[0][0]  # 最大≈12.8MB = 附件1
        a2_path = xlsx_files[1][0]  # 第二≈5.5MB = 附件2
        return a1_path, a2_path
    elif len(xlsx_files) >= 2:
        return xlsx_files[0][0], xlsx_files[1][0]
    else:
        raise FileNotFoundError('未找到足够的xlsx附件文件')

ATTACHMENT1, ATTACHMENT2 = _find_attachments()

# ============================================================
# 2. 餐次划分参数
# ============================================================
# 根据数据探查，交易主要集中在10:00-12:00（午餐高峰）
# 晚餐交易量极小（16:00-18:00），但题目要求分别给出午餐/晚餐方案
LUNCH_START = 10   # 午餐开始时间（含）
LUNCH_END = 14     # 午餐结束时间（不含）
DINNER_START = 16  # 晚餐开始时间（含）
DINNER_END = 20    # 晚餐结束时间（不含）

# ============================================================
# 3. 菜品分类规则
# ============================================================
# 基于菜品名称关键词的人工分类
# 分类体系：主食、荤菜、半荤半素、素菜、汤品/其他
CATEGORY_KEYWORDS = {
    '主食': ['米饭', '白饭', '馒头', '面条', '花卷', '饼', '粥', '包子',
             '年糕', '炒饭', '盖浇饭', '煲仔饭'],
    '荤菜': ['肉', '鸡', '鸭', '鱼', '虾', '蟹', '排骨', '大排', '猪手',
             '猪蹄', '牛肉', '羊肉', '鹅', '酥鱼', '带鱼', '黄鱼', '鱿鱼',
             '卤肉', '口水鸡', '辣子鸡', '鸡米花', '宫保鸡丁', '炸鸡',
             '鸡排', '春卷', '香肠', '腊肉', '火腿', '猪肝', '腰花',
             '红烧', '酱鸭', '油豆腐烧肉', '白水虾', '椒盐虾'],
    '半荤半素': ['炒蛋', '鸡蛋', '肉丝', '肉片', '炒肉', '肉末',
                '千叶豆腐肉丝', '茭白榨菜肉丝', '蘑菇小炒肉',
                '韭黄肉丝', '鱼香', '杏鲍菇炒肉'],
    '素菜': ['菜', '豆', '瓜', '茄', '藕', '笋', '菇', '椒', '芹',
             '葱', '蒜', '花菜', '西兰花', '土豆', '萝卜', '南瓜',
             '海带', '木耳', '豆芽', '粉丝', '豆腐', '面筋', '青菜',
             '大白菜', '包菜', '娃娃菜', '莴苣', '蒜苗', '韭菜',
             '酸辣土豆丝', '酸辣藕丝', '虎皮尖椒'],
}

# ============================================================
# 4. 营养参考标准（每餐人均）
# ============================================================
# 参考《中国居民膳食营养素参考摄入量(DRIs)》(2023版)
# 假设餐厅目标顾客为轻体力活动成年人，日均摄入按三餐分配
# 午餐约占40%，晚餐约占35%
NUTRITION_DAILY_STANDARD = {
    'calories': 2200,      # 千卡/天
    'protein': 65,         # 克/天
    'fat': 65,             # 克/天  
    'carbohydrates': 300,  # 克/天
    'fiber': 25,           # 克/天
}

NUTRITION_PER_MEAL = {
    'lunch': {
        'calories': NUTRITION_DAILY_STANDARD['calories'] * 0.40,
        'protein': NUTRITION_DAILY_STANDARD['protein'] * 0.40,
        'fat': NUTRITION_DAILY_STANDARD['fat'] * 0.40,
        'carbohydrates': NUTRITION_DAILY_STANDARD['carbohydrates'] * 0.40,
        'fiber': NUTRITION_DAILY_STANDARD['fiber'] * 0.40,
    },
    'dinner': {
        'calories': NUTRITION_DAILY_STANDARD['calories'] * 0.35,
        'protein': NUTRITION_DAILY_STANDARD['protein'] * 0.35,
        'fat': NUTRITION_DAILY_STANDARD['fat'] * 0.35,
        'carbohydrates': NUTRITION_DAILY_STANDARD['carbohydrates'] * 0.35,
        'fiber': NUTRITION_DAILY_STANDARD['fiber'] * 0.35,
    },
}

# 营养需求弹性范围（允许±20%浮动）
NUTRITION_TOLERANCE = 0.20

# ============================================================
# 5. 问题3 备菜优化参数
# ============================================================
# 浪费成本系数 — 剩余菜品处理的单位损失
WASTE_COST_RATIO = 0.3  # 浪费成本占菜品成本的比例
# 缺货惩罚系数 — 缺货导致的客户流失和声誉损失
SHORTAGE_PENALTY_RATIO = 0.5  # 缺货惩罚占菜品利润的比例
# 安全库存系数 — 作为需求预测不确定性的缓冲
SAFETY_STOCK_FACTOR = 0.15

# ============================================================
# 6. 问题4 套餐设计参数
# ============================================================
COMBO_PRICE_LEVELS = [10, 15, 20]  # 三个价位
COMBO_MIN_DISHES = 2               # 套餐最少菜品数
COMBO_MAX_DISHES = 5               # 套餐最多菜品数
# 套餐营养评分权重
COMBO_WEIGHTS = {
    'popularity': 0.30,    # 受欢迎度（历史销量）
    'nutrition': 0.30,     # 营养均衡度
    'profit': 0.25,        # 利润率
    'association': 0.15,   # 共购关联度
}

# ============================================================
# 7. 预测模型参数
# ============================================================
# 时间序列交叉验证折数
TS_CV_FOLDS = 5
# 测试集比例
TEST_SIZE = 0.2
# 随机种子
RANDOM_SEED = 42
# 预测目标月份
PREDICTION_YEAR = 2025
PREDICTION_MONTH = 5
# 问题3备菜方案目标时间段
MEAL_PLAN_START = '2025-05-06'
MEAL_PLAN_END = '2025-05-12'

# ============================================================
# 8. 可视化参数
# ============================================================
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# 配色方案
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'success': '#73AB84',
    'warning': '#F18F01',
    'danger': '#C73E1D',
    'lunch': '#F18F01',
    'dinner': '#2E86AB',
}
