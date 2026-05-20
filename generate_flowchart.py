"""
generate_flowchart.py — 用 matplotlib 绘制学术风格问题分析流程图
输出: output/flow_chart.png (300dpi)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(22, 16))
ax.set_xlim(0, 26)
ax.set_ylim(0, 19)
ax.axis('off')

# ===== Helper Functions =====
def box(ax, x, y, w, h, text, bold=False, fc='white', ec='black', ls='-', lw=1.2, fs=9):
    """绘制矩形节点"""
    rect = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.3",
                          facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls)
    ax.add_patch(rect)
    kw = {'fontweight': 'bold'} if bold else {}
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, **kw)

def dash_box(ax, x, y, w, h, text, fs=9):
    """绘制虚线框节点"""
    box(ax, x, y, w, h, text, ls='--', fs=fs)

def cluster_box(ax, x, y, w, h, label, fs=11):
    """绘制阶段虚线大框"""
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5",
                          facecolor='none', edgecolor='black', linewidth=1.0,
                          linestyle='--')
    ax.add_patch(rect)
    ax.text(x + 0.3, y + h - 0.4, label, fontsize=fs, fontweight='bold',
            va='top', ha='left')

def arrow(ax, x1, y1, x2, y2, style='-', lw=1.0, label=''):
    """绘制箭头"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='black', lw=lw,
                               linestyle=style, connectionstyle='arc3,rad=0'))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my, label, fontsize=7, ha='center', va='bottom',
                fontweight='bold')

def phase_label(ax, x, y, text):
    """左侧阶段标注"""
    ax.text(x, y, text, fontsize=11, fontweight='bold', ha='center', va='center',
            rotation=90)

# ===== 数据预处理阶段 =====
cluster_box(ax, 1.5, 13.0, 5.0, 5.0, '数据预处理')
box(ax, 4.0, 17.0, 2.8, 1.2, '原始附件数据\n(3+15 Sheet)', bold=True)
box(ax, 4.0, 15.2, 2.8, 1.2, '数据清洗\n缺失值/异常值/餐次')
box(ax, 4.0, 13.5, 2.8, 1.2, '特征工程\n分类/融合/汇总')
arrow(ax, 4.0, 16.4, 4.0, 15.8)
arrow(ax, 4.0, 14.6, 4.0, 14.1)
phase_label(ax, 0.5, 15.5, '数据准备')

# 中间数据
dash_box(ax, 4.0, 11.5, 2.6, 0.9, '日级汇总(531天)')
dash_box(ax, 8.0, 11.5, 2.6, 0.9, '购物篮矩阵\n(12,944×223)')
arrow(ax, 4.0, 12.9, 4.0, 12.0, style='--', lw=0.8)
arrow(ax, 4.0, 13.0, 8.0, 12.0, style='--', lw=0.8, label='pivot_table')

# ===== 问题一 =====
cluster_box(ax, 1.5, 7.0, 5.0, 4.1, '第一问：统计分析与关联规则')
box(ax, 4.0, 10.2, 2.8, 1.0, '描述性统计\nPareto/ABC/t检验')
box(ax, 8.0, 9.5, 2.8, 1.0, 'Apriori关联规则\n支持度/置信度/提升度')
box(ax, 6.0, 7.8, 2.8, 1.0, '5张图表+19条规则', bold=True, lw=1.5)
arrow(ax, 4.0, 11.1, 4.0, 10.7)
arrow(ax, 8.0, 10.9, 8.0, 10.0)
arrow(ax, 4.0, 9.7, 6.0, 8.3, lw=0.8)
arrow(ax, 8.0, 9.0, 6.0, 8.3, lw=0.8)
phase_label(ax, 0.5, 9.0, '第一问')

# ===== 问题二 =====
cluster_box(ax, 9.0, 11.0, 10.0, 7.5, '第二问：多模型需求预测')
box(ax, 11.0, 17.2, 2.6, 1.2, 'SARIMA\n(1,1,1)(1,1,1,7)')
box(ax, 14.5, 17.2, 2.6, 1.2, 'XGBoost\n(30维特征)')
box(ax, 11.0, 15.0, 2.6, 1.0, 'Baseline\n同星期均值')
box(ax, 14.5, 15.0, 2.6, 1.0, 'Ensemble\n加权融合')
box(ax, 12.8, 13.1, 3.2, 1.0, 'Walk-forward\n滚动验证 MAPE=15.5%')
box(ax, 12.8, 11.6, 3.4, 1.0, 'May2025预测\n19天+95%CI', bold=True, lw=1.5)
arrow(ax, 4.0, 11.1, 11.0, 17.8, lw=0.8, label='日级序列')
arrow(ax, 4.0, 11.1, 14.5, 17.8, lw=0.8)
arrow(ax, 11.0, 16.6, 11.0, 15.5)
arrow(ax, 14.5, 16.6, 14.5, 15.5)
arrow(ax, 11.0, 14.5, 12.8, 13.6)
arrow(ax, 14.5, 14.5, 12.8, 13.6)
arrow(ax, 12.8, 12.6, 12.8, 12.1)
phase_label(ax, 9.5, 14.8, '第二问')

# ===== 问题三 =====
cluster_box(ax, 9.0, 7.5, 5.0, 3.0, '第三问：午餐备菜优化')
box(ax, 11.5, 9.3, 3.0, 1.2, 'MILP整数规划\n(PuLP+CBC求解)')
box(ax, 11.5, 8.0, 3.0, 0.8, '5天午餐备菜方案', bold=True, lw=1.5)
arrow(ax, 12.8, 11.1, 11.5, 9.9, style='-', lw=1.0, label='预测人数')
arrow(ax, 11.5, 8.7, 11.5, 8.4)
phase_label(ax, 9.5, 9.0, '第三问')

# ===== 问题四 =====
cluster_box(ax, 15.5, 9.0, 5.0, 3.5, '第四问：套餐优化设计')
box(ax, 18.0, 11.2, 2.8, 1.0, '贪心搜索\n(200次采样)')
box(ax, 18.0, 9.7, 2.8, 1.0, '局部优化\n(100次爬山法)')
box(ax, 18.0, 8.5, 2.8, 0.8, '10/15/20元套餐', bold=True, lw=1.5)
arrow(ax, 8.0, 11.1, 18.0, 11.7, lw=0.8, label='购物篮数据')
arrow(ax, 6.0, 7.8, 18.0, 9.7, style='--', lw=0.6, label='关联规则')
arrow(ax, 18.0, 10.7, 18.0, 10.2)
arrow(ax, 18.0, 9.2, 18.0, 8.9)
phase_label(ax, 16.0, 10.8, '第四问')

# ===== 问题五 =====
cluster_box(ax, 21.5, 9.0, 4.0, 4.5, '第五问：经营策略')
box(ax, 23.5, 11.5, 3.0, 1.5, '五维度策略分析\n(备菜/菜品/套餐/\n数字化/ESG)')
box(ax, 23.5, 9.5, 3.0, 0.8, '策略框架+建议', bold=True, lw=1.5)
arrow(ax, 6.0, 7.8, 23.5, 11.5, style=':', lw=0.5, label='P1结果')
arrow(ax, 12.8, 11.1, 23.5, 11.5, style=':', lw=0.5, label='P2结果')
arrow(ax, 11.5, 8.0, 23.5, 9.5, style=':', lw=0.5, label='P3结果')
arrow(ax, 18.0, 8.5, 23.5, 9.5, style=':', lw=0.5, label='P4结果')
arrow(ax, 23.5, 10.7, 23.5, 9.9)
phase_label(ax, 22.0, 11.3, '第五问')

# ===== 图标题 =====
ax.text(13, 0.3, '图 1  问题分析流程图', fontsize=14, fontweight='bold',
        ha='center', va='center')

plt.tight_layout(pad=0.5)
fig.savefig(r'C:\Users\CoirRaincoat\PyCharmMiscProject\CSJ-MathModeling\output\flow_chart.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('flow_chart.png saved to output/')
