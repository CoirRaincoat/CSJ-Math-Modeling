"""
main.py — 项目主入口
===================
串联问题1-5的完整分析流程，生成所有结果和可视化。

执行顺序：
  1. 数据加载与预处理（data_loader.py）
  2. 问题1：数据统计分析与关联规则（problem1_analysis.py）
  3. 问题2：需求预测（problem2_prediction.py）
  4. 问题3：备菜优化（problem3_optimization.py）
  5. 问题4：套餐设计（problem4_combos.py）
  6. 问题5：经营策略（problem5_strategy.py）

使用方法：
  python main.py           # 运行所有问题
  python main.py --skip 2  # 跳过问题2
  python main.py --only 1  # 仅运行问题1
"""

import sys
import time
import os
import warnings
warnings.filterwarnings('ignore')

from config import OUTPUT_DIR

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_all(skip=None, only=None):
    """
    运行完整的分析流程
    
    Args:
        skip: 要跳过的问题编号列表
        only: 只运行的问题编号列表
    """
    print('=' * 70)
    print('  2026 长三角数学建模竞赛 赛题B')
    print('  自助量贩餐厅菜量需求预测与运营优化设计')
    print('=' * 70)
    
    start_time = time.time()
    
    # 步骤0：数据加载
    print('\n[步骤0] 加载数据...')
    from data_loader import load_all_data
    loader = load_all_data()
    loader.print_summary()
    
    total_time_data = time.time() - start_time
    print(f'  数据加载耗时: {total_time_data:.1f} 秒')
    
    # 步骤1：问题1
    if should_run(1, skip, only):
        print('\n' + '=' * 70)
        print('  [步骤1] 问题1：数据统计分析与关联规则')
        print('=' * 70)
        t1 = time.time()
        
        from problem1_analysis import Problem1Analysis
        p1 = Problem1Analysis(loader=loader)
        p1.run()
        
        t1_elapsed = time.time() - t1
        print(f'  问题1耗时: {t1_elapsed:.1f} 秒')
    
    # 步骤2：问题2
    if should_run(2, skip, only):
        print('\n' + '=' * 70)
        print('  [步骤2] 问题2：需求预测')
        print('=' * 70)
        t2 = time.time()
        
        from problem2_prediction import Problem2Prediction
        p2 = Problem2Prediction(loader=loader)
        p2.run()
        
        t2_elapsed = time.time() - t2
        print(f'  问题2耗时: {t2_elapsed:.1f} 秒')
    
    # 步骤3：问题3
    if should_run(3, skip, only):
        print('\n' + '=' * 70)
        print('  [步骤3] 问题3：备菜优化')
        print('=' * 70)
        t3 = time.time()
        
        from problem3_optimization import Problem3Optimization
        p3 = Problem3Optimization(loader=loader)
        p3.run()
        
        t3_elapsed = time.time() - t3
        print(f'  问题3耗时: {t3_elapsed:.1f} 秒')
    
    # 步骤4：问题4
    if should_run(4, skip, only):
        print('\n' + '=' * 70)
        print('  [步骤4] 问题4：套餐设计')
        print('=' * 70)
        t4 = time.time()
        
        from problem4_combos import Problem4Combos
        p4 = Problem4Combos(loader=loader)
        p4.run()
        
        t4_elapsed = time.time() - t4
        print(f'  问题4耗时: {t4_elapsed:.1f} 秒')
    
    # 步骤5：问题5
    if should_run(5, skip, only):
        print('\n' + '=' * 70)
        print('  [步骤5] 问题5：经营策略分析')
        print('=' * 70)
        t5 = time.time()
        
        from problem5_strategy import Problem5Strategy
        p5 = Problem5Strategy(loader=loader)
        p5.run()
        p5.print_strategy_report()
        
        t5_elapsed = time.time() - t5
        print(f'  问题5耗时: {t5_elapsed:.1f} 秒')
    
    # 总结
    total_time = time.time() - start_time
    print('\n' + '=' * 70)
    print(f'  全部任务完成！总耗时: {total_time:.1f} 秒')
    print(f'  输出目录: {OUTPUT_DIR}')
    print('=' * 70)
    
    # 列出输出文件
    print('\n输出文件列表:')
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath)
        print(f'  {f:<40} {size:>10,} bytes')


def should_run(problem_num, skip=None, only=None):
    """判断是否应该运行某个问题"""
    if only is not None:
        return problem_num in only
    if skip is not None:
        return problem_num not in skip
    return True


def parse_args():
    """解析命令行参数"""
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
