"""main.py — 项目主入口"""


import sys
import time
import os
import warnings
warnings.filterwarnings('ignore')

from config import OUTPUT_DIR

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_all(skip=None, only=None):
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
    if only is not None:
        return problem_num in only
    if skip is not None:
        return problem_num not in skip
    return True


def parse_args():
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
