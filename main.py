"""
main.py — 项目主入口
===================
串联问题1-5的完整分析流程，生成所有结果和可视化。

执行顺序:
  步骤0: 数据加载与预处理 (data_loader.py → DataLoader)
  步骤1: 问题1 — 数据统计分析与关联规则 (problem1_analysis.py)
  步骤2: 问题2 — 多模型需求预测 (problem2_prediction.py)
  步骤3: 问题3 — 午餐备菜优化 (problem3_optimization.py, 仅午餐)
  步骤4: 问题4 — 套餐优化设计 (problem4_combos.py)
  步骤5: 问题5 — 经营策略建议 (problem5_strategy.py)

使用方法:
  python main.py                  # 运行所有问题 (完整流程)
  python main.py --skip 2,5       # 跳过问题2和5
  python main.py --only 1,3       # 仅运行问题1和3
  python main.py --only 3         # 仅运行问题3 (午餐备菜优化)

输出:
  所有结果 (图表 PNG + 数据 CSV) 输出至 output/ 目录。

设计原则:
  - DataLoader 只实例化一次，通过参数传递给各问题模块
  - 每个问题模块可独立运行，也可组合运行
  - 输出统一写入 OUTPUT_DIR 目录
  - 固定随机种子 (RANDOM_SEED=42) 保证可复现性
"""

import sys
import time
import os
import warnings
warnings.filterwarnings('ignore')

from config import OUTPUT_DIR

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_all(skip=None, only=None):
    """
    运行完整的分析流程

    所有问题模块共享同一个 DataLoader 实例，
    避免重复加载约 18MB 的 Excel 文件。

    Args:
        skip: list of int, 要跳过的问题编号 (如 [2, 5])
        only: list of int, 只运行的问题编号 (如 [1, 3])
    """
    print('=' * 70)
    print('  2026 长三角数学建模竞赛 — 赛题 B')
    print('  自助量贩餐厅菜量需求预测与运营优化设计')
    print('=' * 70)

    total_start = time.time()

    print('\n' + '-' * 50)
    print('[步骤0] 加载数据...')
    print('-' * 50)
    from data_loader import load_all_data
    loader = load_all_data()
    loader.print_summary()

    t_data = time.time() - total_start
    print(f'\n  数据加载耗时: {t_data:.1f} 秒')

    if should_run(1, skip, only):
        print('\n' + '=' * 70)
        print('  [步骤1] 问题1: 数据统计分析与关联规则')
        print('=' * 70)
        t_start = time.time()

        from problem1_analysis import Problem1Analysis
        p1 = Problem1Analysis(loader=loader)
        p1_results = p1.run()

        t_elapsed = time.time() - t_start
        print(f'\n  问题1 耗时: {t_elapsed:.1f} 秒')

    if should_run(2, skip, only):
        print('\n' + '=' * 70)
        print('  [步骤2] 问题2: 需求预测')
        print('=' * 70)
        t_start = time.time()

        from problem2_prediction import Problem2Prediction
        p2 = Problem2Prediction(loader=loader)
        p2_results = p2.run()

        t_elapsed = time.time() - t_start
        print(f'\n  问题2 耗时: {t_elapsed:.1f} 秒')

    if should_run(3, skip, only):
        print('\n' + '=' * 70)
        print('  [步骤3] 问题3: 备菜优化 (仅午餐)')
        print('=' * 70)
        t_start = time.time()

        from problem3_optimization import Problem3Optimization
        p3 = Problem3Optimization(loader=loader)
        p3_results = p3.run()

        t_elapsed = time.time() - t_start
        print(f'\n  问题3 耗时: {t_elapsed:.1f} 秒')

    if should_run(4, skip, only):
        print('\n' + '=' * 70)
        print('  [步骤4] 问题4: 套餐设计')
        print('=' * 70)
        t_start = time.time()

        from problem4_combos import Problem4Combos
        p4 = Problem4Combos(loader=loader)
        p4_results = p4.run()

        t_elapsed = time.time() - t_start
        print(f'\n  问题4 耗时: {t_elapsed:.1f} 秒')

    if should_run(5, skip, only):
        print('\n' + '=' * 70)
        print('  [步骤5] 问题5: 经营策略分析')
        print('=' * 70)
        t_start = time.time()

        from problem5_strategy import Problem5Strategy
        p5 = Problem5Strategy(loader=loader)
        p5_results = p5.run()
        p5.print_strategy_report()

        t_elapsed = time.time() - t_start
        print(f'\n  问题5 耗时: {t_elapsed:.1f} 秒')

    total_time = time.time() - total_start
    print('\n' + '=' * 70)
    print(f'  ALL TASKS COMPLETED')
    print(f'  Total time: {total_time:.1f} sec ({total_time/60:.1f} min)')
    print(f'  Output directory: {OUTPUT_DIR}')
    print('=' * 70)

    # 列出输出文件及大小
    print('\n  Output file list:')
    if os.path.exists(OUTPUT_DIR):
        for f in sorted(os.listdir(OUTPUT_DIR)):
            fpath = os.path.join(OUTPUT_DIR, f)
            size_bytes = os.path.getsize(fpath)
            if size_bytes > 1024 * 1024:
                size_str = f'{size_bytes/(1024*1024):.1f} MB'
            elif size_bytes > 1024:
                size_str = f'{size_bytes/1024:.1f} KB'
            else:
                size_str = f'{size_bytes} B'
            print(f'    {f:<45} {size_str:>10}')
    else:
        print('    (output directory not found)')


def should_run(problem_num, skip=None, only=None):
    """
    判断某个问题是否应该运行

    Args:
        problem_num: int, 问题编号 (1-5)
        skip: list or None, 要跳过的问题编号
        only: list or None, 只运行的问题编号

    Returns:
        bool: 是否应该运行
    """
    if only is not None:
        return problem_num in only
    if skip is not None:
        return problem_num not in skip
    return True


def parse_args():
    """
    解析命令行参数

    支持:
      --skip 2,5   跳过问题2和5
      --only 1,3   仅运行问题1和3

    Returns:
        tuple: (skip_list, only_list)
    """
    skip = None
    only = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--skip' and i + 1 < len(args):
            skip = [int(x) for x in args[i+1].split(',')]
            i += 2
        elif args[i] == '--only' and i + 1 < len(args):
            only = [int(x) for x in args[i+1].split(',')]
            i += 2
        else:
            i += 1

    return skip, only


if __name__ == '__main__':
    skip, only = parse_args()
    run_all(skip=skip, only=only)
