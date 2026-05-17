"""config.py — 全局配置文件"""


import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _find_attachments():
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
        raise FileNotFoundError(
            f'未找到足够的 xlsx 附件文件。当前目录: {DATA_DIR}'
        )

ATTACHMENT1, ATTACHMENT2 = _find_attachments()

LUNCH_START = 10   # 午餐开始时间（含）
LUNCH_END = 14     # 午餐结束时间（不含）
DINNER_START = 16  # 晚餐开始时间（含）
DINNER_END = 20    # 晚餐结束时间（不含）

CATEGORY_KEYWORDS = {
    '主食': ['米饭', '白饭', '馒头', '面条', '花卷', '饼', '粥', '包子',
             '年糕', '炒饭', '盖浇饭', '煲仔饭', '粉丝', '燕麦', '饺子',
             '馄饨', '油条', '烧饼', '窝头', '锅贴', '生煎', '糍粑',
             '米粉', '河粉', '粿条'],
    '荤菜': ['肉', '鸡', '鸭', '鱼', '虾', '蟹', '排骨', '大排', '猪手',
             '猪蹄', '牛肉', '羊肉', '牛腩', '鹅', '酥鱼', '带鱼', '黄鱼',
             '鱿鱼', '卤肉', '口水鸡', '辣子鸡', '鸡米花', '宫保鸡丁', '炸鸡',
             '鸡排', '春卷', '香肠', '腊肉', '火腿', '猪肝', '腰花', '板鸭',
             '红烧', '酱鸭', '油豆腐烧肉', '白水虾', '椒盐虾', '鸡腿', '鸡翅',
             '鸡杂', '牛蛙', '田鸡', '鹌鹑', '腊味', '叉烧', '烧鹅',
             '白切鸡', '盐焗鸡', '鲍鱼', '干贝', '海参', '鳝鱼', '泥鳅',
             '糖醋', '牛柳', '里脊', '油渣', '红肠', '香酥', '付皮'],
    '半荤半素': ['炒蛋', '鸡蛋', '肉丝', '肉片', '炒肉', '肉末',
                 '千叶豆腐肉丝', '茭白榨菜肉丝', '蘑菇小炒肉',
                 '韭黄肉丝', '鱼香', '杏鲍菇炒肉', '木须肉',
                 '京酱肉丝', '韭苔肉丝', '西芹炒肉', '蒜苔肉丝',
                 '回锅肉', '肉片炒', '肉末烧', '小炒', '炒', '蒸蛋',
                 '山药肉', '素鸡红肠', '素几'],
    '素菜': ['菜', '豆', '瓜', '茄', '藕', '笋', '菇', '椒', '芹',
             '葱', '蒜', '花菜', '西兰花', '土豆', '萝卜', '南瓜',
             '海带', '木耳', '豆芽', '粉丝', '豆腐', '面筋', '青菜',
             '大白菜', '包菜', '娃娃菜', '莴苣', '蒜苗', '韭菜',
             '酸辣土豆丝', '酸辣藕丝', '虎皮尖椒', '泡菜', '榨菜',
             '苋菜', '茼蒿', '藕片', '干锅花菜', '干煸豆角',
             '西葫芦', '茭白', '豆角', '扁豆', '荷兰豆', '豌豆',
             '紫甘蓝', '空心菜', '生菜', '油麦菜', '菠菜',
             '霉干', '素烧', '油麦', '西兰'],
}

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

# 营养需求弹性范围 (允许 ±20% 浮动)
# 即实际营养素供给量可以在目标值的 80%-120% 范围内
NUTRITION_TOLERANCE = 0.20

COST_RATIO_BY_CATEGORY = {
    '主食': 0.28,
    '荤菜': 0.60,
    '半荤半素': 0.45,
    '素菜': 0.30,
    '其他': 0.45,
}
WASTE_COST_RATIO = 0.3

SHORTAGE_PENALTY_RATIO = 0.5

SAFETY_STOCK_FACTOR = 0.15

COMBO_PRICE_LEVELS = [10, 15, 20]

# 套餐菜品数量范围
COMBO_MIN_DISHES = 2   # 最少 2 个菜品 (如米饭+1菜)
COMBO_MAX_DISHES = 5   # 最多 5 个菜品 (如米饭+4菜)

# 套餐综合评分权重分配
# 权重设置依据: 消费者偏好和营养均衡是最核心的两个维度 (各占 30%)，
# 其次是利润率 (25%)，最后是共购关联度和价格符合度 (各占 15%)
COMBO_WEIGHTS = {
    'popularity': 0.30,    # 受欢迎度 (历史销量归一化)
    'nutrition': 0.30,     # 营养均衡度 (碳水/脂肪/蛋白质供能比)
    'profit': 0.25,        # 利润率 (套餐利润率)
    'association': 0.15,   # 共购关联度 (基于 Apriori 关联规则)
}

TS_CV_FOLDS = 5

# TEST_SIZE: 测试集比例 (用于样本外评估)
TEST_SIZE = 0.2

# RANDOM_SEED: 全局随机种子
#   用于保证结果的可复现性
#   所有涉及随机性的操作 (采样、模型训练、搜索) 均使用此种子
RANDOM_SEED = 42

# 预测目标: 2025年5月工作日
PREDICTION_YEAR = 2025
PREDICTION_MONTH = 5

# 问题3 备菜方案目标时间段: 2025年5月6日-12日 (包含周末，优化仅对工作日)
MEAL_PLAN_START = '2025-05-06'
MEAL_PLAN_END = '2025-05-12'

import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免图形界面依赖
import matplotlib.pyplot as plt

# 中文字体配置 (支持中文标签正常显示)
# 优先使用 SimHei (黑体), 回退到 Microsoft YaHei (微软雅黑), 最后 DejaVu Sans
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示异常
plt.rcParams['figure.dpi'] = 150               # 显示分辨率
plt.rcParams['savefig.dpi'] = 300              # 保存分辨率 (满足 Nature 300dpi 要求)
plt.rcParams['savefig.bbox'] = 'tight'         # 自动裁剪边缘空白
# 全局加大字号 - 确保图表中的标签、图例足够大且醒目
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# Nature 期刊配色方案 (NPG Palette)
COLORS = {
    # 核心色
    'primary': '#3C5488',    # 深蓝 — 主色调
    'secondary': '#00A087',  # 青绿 — 副色调
    'accent': '#E64B35',     # 红色 — 强调/高亮
    'success': '#4DBBD5',    # 浅蓝 — 正向/成功
    'warning': '#F39B7F',    # 鲑鱼色 — 警告/注意
    'danger': '#DC0000',     # 深红 — 危险/重要

    # 特殊用途色
    'lunch': '#E64B35',      # 暖红 — 午餐
    'dinner': '#3C5488',     # 冷蓝 — 晚餐

    # 扩展色 (多系列图表使用)
    'purple': '#8491B4',     # 灰紫
    'teal': '#91D1C2',       # 浅绿
    'brown': '#7E6148',      # 棕色
    'beige': '#B09C85',      # 米色
    'gold': '#E69F00',       # 金色
    'grey': '#A0A0A0',       # 灰色
    'pink': '#F39B7F',       # 粉色 (同 warning)
    'darkblue': '#3C5488',   # 深蓝 (同 primary)
    'red': '#E64B35',        # 红 (同 accent)
    'green': '#00A087',      # 绿 (同 secondary)
    'cyan': '#4DBBD5',       # 青 (同 success)

    # 分组对比色 (用于箱线图等需要多颜色的场景)
    'weekday': '#3C5488',    # 工作日
    'weekend': '#F39B7F',    # 周末
}

# 多系列图表通用颜色序列 (7色)
COLOR_CYCLE = [
    COLORS['primary'],    # 深蓝
    COLORS['accent'],     # 红
    COLORS['secondary'],  # 青绿
    COLORS['success'],    # 浅蓝
    COLORS['purple'],     # 灰紫
    COLORS['gold'],       # 金色
    COLORS['teal'],       # 浅绿
]

# 热力图/散点图使用的 colormap
# 'YlOrRd' 保持为黄-橙-红渐变 (色盲友好)
# 'RdBu_r' 改为 'coolwarm' 用于相关性矩阵 (Nature 常用)
HEATMAP_CMAP = 'coolwarm'
SCATTER_CMAP = 'YlOrRd'
