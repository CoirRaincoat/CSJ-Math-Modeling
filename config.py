"""
config.py — 全局配置文件
======================
定义项目路径、常量、菜品分类规则、营养参考标准、可视化参数等全局参数。
所有模块共享此配置，确保数据一致性和可维护性。

参考文献：
  [1] 中国居民膳食营养素参考摄入量(DRIs), 中国营养学会, 2023版
      http://www.cnsoc.org/
  [2] 中国居民膳食指南(2022), 中国营养学会
      http://dg.cnsoc.org/

项目结构：
  config.py                — 本文件：全局配置
  data_loader.py           — 数据加载与预处理
  utils.py                 — 通用工具函数 (MAPE、特征工程、营养计算)
  problem1_analysis.py     — 问题1：数据统计分析与关联规则
  problem2_prediction.py   — 问题2：需求预测
  problem3_optimization.py — 问题3：备菜优化 (仅午餐)
  problem4_combos.py       — 问题4：套餐设计
  problem5_strategy.py     — 问题5：经营策略建议
  main.py                  — 主入口，串联全部模块

配色方案说明：
  采用 Nature 期刊推荐的科学配色方案 (Nature Publishing Group / NPG palette)，
  具有色盲友好、高对比度、适合学术论文出版的特点。
  参考来源: ggsci R package "npg" palette
    https://cran.r-project.org/web/packages/ggsci/
  Nature 期刊常见配色原则：
  - 使用高对比度、可区分的颜色
  - 避免红绿搭配 (色盲不友好)
  - 优先使用蓝色-橙色/红色-青色等安全组合
"""

import os

# ============================================================
# 1. 路径配置
# ============================================================
# BASE_DIR: 项目根目录 (本文件所在目录)
# DATA_DIR: 数据文件目录 (与代码同目录)
# OUTPUT_DIR: 输出目录 (图表和表格)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 附件文件原始名称（仅供参考，因系统编码差异运行时自动匹配）：
#   附件1 = "附件1餐厅销售流水信息表.xlsx"  (~12.8MB)
#   附件2 = "附件2部分消费订单菜品具体信息表.xlsx" (~5.5MB)
#   附件3 = "附件3数据说明.xlsx" (~12KB)
#
# 动态发现附件文件（按文件大小特征匹配，避免中文编码问题）
# 设计原理：Windows GBK 编码下中文文件名可能出现乱码导致 FileNotFoundError，
# 因此不硬编码文件名，而是根据文件大小特征 (附件1最大, 附件2次之) 自动匹配。
def _find_attachments():
    """
    根据文件大小特征自动识别附件1和附件2

    附件1餐厅流水表 (~12.8MB) > 附件2菜品信息表 (~5.5MB) > 附件3数据说明 (~12KB)
    按文件大小降序排序，取前两个即为附件1和附件2。

    Returns:
        tuple: (附件1路径, 附件2路径)

    Raises:
        FileNotFoundError: 当目录下 xlsx 文件不足 2 个时
    """
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

# ============================================================
# 2. 餐次划分参数
# ============================================================
# 根据数据探查:
#   - 交易主要集中在 10:00-12:00 (午餐高峰)，占比约 99.2%
#   - 晚餐交易量极小 (仅约 0.8% 的订单)，数据不足以支持可靠建模
#   - 因此问题3 仅对午餐进行备菜优化，晚餐方案不作要求
# 餐次时间划分依据:
#   - 午餐: 10:00-14:00 (覆盖早到和晚到的午餐顾客)
#   - 晚餐: 16:00-20:00 (覆盖最早和最晚的晚餐顾客)
LUNCH_START = 10   # 午餐开始时间（含）
LUNCH_END = 14     # 午餐结束时间（不含）
DINNER_START = 16  # 晚餐开始时间（含）
DINNER_END = 20    # 晚餐结束时间（不含）

# ============================================================
# 3. 菜品分类规则
# ============================================================
# 基于菜品名称关键词的人工分类，分类体系如下：
#   - 主食：米饭、面食、馒头等高碳水类
#   - 荤菜：以肉类/禽类/鱼类为主
#   - 半荤半素：包含少量肉类的蔬菜类
#   - 素菜：纯蔬菜/豆制品
#   - 其他：无法匹配的菜品
#
# 分类优先级：主食 > 荤菜 > 半荤半素 > 素菜
# （例如"肉末茄子饭"优先归为主食而非半荤半素）
#
# 已知局限性：
#   - 基于关键词的匹配率约 46%，约 54% 的菜品归为"其他"
#   - "其他"类菜品名称通常不包含典型关键词 (如 "大白菜炖肉" 同时包含 "肉" 和 "菜")
#   - 在 data_loader.py 中已添加营养特征辅助分类 (utils.py 的 classify_dish_by_nutrition)
#   - 对于无法通过名称关键词分类的菜品，使用营养特征 (蛋白质/脂肪/碳水含量) 进行二次判定
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
             '白切鸡', '盐焗鸡', '鲍鱼', '干贝', '海参', '鳝鱼', '泥鳅'],
    '半荤半素': ['炒蛋', '鸡蛋', '肉丝', '肉片', '炒肉', '肉末',
                 '千叶豆腐肉丝', '茭白榨菜肉丝', '蘑菇小炒肉',
                 '韭黄肉丝', '鱼香', '杏鲍菇炒肉', '木须肉',
                 '京酱肉丝', '韭苔肉丝', '西芹炒肉', '蒜苔肉丝',
                 '回锅肉', '肉片炒', '肉末烧'],
    '素菜': ['菜', '豆', '瓜', '茄', '藕', '笋', '菇', '椒', '芹',
             '葱', '蒜', '花菜', '西兰花', '土豆', '萝卜', '南瓜',
             '海带', '木耳', '豆芽', '粉丝', '豆腐', '面筋', '青菜',
             '大白菜', '包菜', '娃娃菜', '莴苣', '蒜苗', '韭菜',
             '酸辣土豆丝', '酸辣藕丝', '虎皮尖椒', '泡菜', '榨菜',
             '苋菜', '茼蒿', '藕片', '干锅花菜', '干煸豆角',
             '西葫芦', '茭白', '豆角', '扁豆', '荷兰豆', '豌豆',
             '紫甘蓝', '空心菜', '生菜', '油麦菜', '菠菜'],
}

# ============================================================
# 4. 营养参考标准（每餐人均）
# ============================================================
# 参考《中国居民膳食营养素参考摄入量(DRIs)》(2023版)
# 中国营养学会, http://www.cnsoc.org/
#
# 假设餐厅目标顾客为轻体力活动成年人:
#   - 日均热量摄入 2200 kcal (18-49岁，轻体力活动)
#   - 蛋白质 65g/天 (约 0.8g/kg 体重，按 60kg 估算)
#   - 脂肪 65g/天 (约占总热量 25-30%)
#   - 碳水化合物 300g/天 (约占总热量 55-65%)
#   - 膳食纤维 25g/天
#
# 午餐约占日摄入的 40%，晚餐约占 35%
# (注: 问题3仅需提供午餐方案，晚餐标准仅供数据分析参考)
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

# ============================================================
# 5. 问题3 备菜优化参数
# ============================================================
# WASTE_COST_RATIO: 浪费成本系数
#   剩余菜品无法再次销售，损失包括原材料成本×比例
#   设定为 0.3 表示每浪费一份菜品的单位成本 = 菜品成本 × 30%
#   (主要损失为食材成本，人工和能耗可通过调整减少)
WASTE_COST_RATIO = 0.3

# SHORTAGE_PENALTY_RATIO: 缺货惩罚系数
#   菜品供应不足导致顾客无法选择想要的菜品
#   惩罚 = 单位利润 × 惩罚系数，反映客户流失和声誉损失
#   设定为 0.5 表示缺货损失 = 利润的 50%
SHORTAGE_PENALTY_RATIO = 0.5

# SAFETY_STOCK_FACTOR: 安全库存系数
#   作为需求预测不确定性的缓冲
#   安全库存量 = 预测需求 × 安全库存系数
#   设定为 15% 可提供合理的供应保障 (Z≈1.65, 95%服务水平)
SAFETY_STOCK_FACTOR = 0.15

# ============================================================
# 6. 问题4 套餐设计参数
# ============================================================
# 三个价位: 10元 (经济型), 15元 (均衡型), 20元 (品质型)
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

# ============================================================
# 7. 预测模型参数
# ============================================================
# TS_CV_FOLDS: 时间序列交叉验证折数
#   用于 XGBoost 模型的超参数评估
#   采用 TimeSeriesSplit 保持时间顺序
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

# ============================================================
# 8. 可视化参数 — Nature 期刊配色方案
# ============================================================
# 配色方案说明:
#   采用 Nature Publishing Group (NPG) 推荐的科学期刊配色，
#   源自 ggsci R package 的 "npg" palette
#   https://cran.r-project.org/web/packages/ggsci/
#
# Nature 期刊配色原则:
#   1. 高对比度: 不同类别之间具有明显的视觉区分度
#   2. 色盲友好: 避免红绿搭配，优先使用蓝-橙等安全组合
#   3. 印刷友好: CMYK 色域内，适合黑白/灰度打印
#   4. 学术规范: 颜色饱和度适中，不干扰数据表达
#
# 色值定义 (HEX):
#   primary:   #3C5488 — 深蓝，用于主数据系列
#   secondary: #00A087 — 青绿，用于次数据系列
#   accent:    #E64B35 — 红，用于强调/高亮
#   success:   #4DBBD5 — 浅蓝，用于正向指标
#   warning:   #F39B7F — 鲑鱼色，用于警示
#   danger:    #DC0000 — 深红，用于危险/重要标记
#   lunch:     #E64B35 — 暖红，午餐标识
#   dinner:    #3C5488 — 冷蓝，晚餐标识
#   
#   扩展色 (用于多系列图表):
#   purple:    #8491B4 — 灰紫
#   teal:      #91D1C2 — 浅绿
#   brown:     #7E6148 — 棕色
#   beige:     #B09C85 — 米色
#   gold:      #E69F00 — 金色
#   grey:      #A0A0A0 — 灰色

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
