"""Safe v4: Only remove pure-comment blocks (3+ consecutive # lines). Keep all code and docstrings intact."""
import os, re

PROJECT = r'C:\Users\CoirRaincoat\PyCharmMiscProject\CSJ-MathModeling'
TARGET = ['config.py','data_loader.py','utils.py','problem1_analysis.py','problem2_prediction.py','problem3_optimization.py','problem4_combos.py','problem5_strategy.py','main.py','validate_reliability.py']

for fname in TARGET:
    fp = os.path.join(PROJECT, fname)
    if not os.path.exists(fp): continue
    with open(fp, 'r', encoding='utf-8') as f: lines = f.readlines()
    orig_n = len(lines)

    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Keep everything that's not a pure comment
        if stripped == '' or not stripped.startswith('#'):
            result.append(line)
            i += 1
            continue

        # Count consecutive comment-only lines
        j = i
        while j < len(lines) and (lines[j].strip().startswith('#') or lines[j].strip() == ''):
            j += 1
        block_len = j - i

        # Skip blocks of 4+ consecutive comment lines
        comment_count = sum(1 for k in range(i, j) if lines[k].strip().startswith('#'))
        if comment_count >= 4:
            i = j
            continue

        # Skip blocks of 3 comment lines that are references/separators
        if comment_count >= 3:
            block_text = ' '.join(lines[k].strip()[2:].strip() for k in range(i, j) if lines[k].strip().startswith('#'))
            if any(kw in block_text for kw in ['参考文献','http','来源','项目结构','配色','参考','步骤','Step','====','----','======','------']):
                i = j
                continue

        # Keep single/double comment lines
        result.append(line)
        i += 1

    cleaned = ''.join(result)
    new_n = len(cleaned.split('\n'))
    with open(fp, 'w', encoding='utf-8') as f: f.write(cleaned)
    print(f'{fname}: {orig_n} -> {new_n} lines ({(1-new_n/orig_n)*100:.0f}% reduced)')
