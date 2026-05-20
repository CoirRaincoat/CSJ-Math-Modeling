"""Safe v2: Remove only docstrings (keep code), using proper handling"""
import os, re

PROJECT = r'C:\Users\CoirRaincoat\PyCharmMiscProject\CSJ-MathModeling'
TARGET = ['config.py','data_loader.py','utils.py','problem1_analysis.py',
          'problem2_prediction.py','problem3_optimization.py','problem4_combos.py',
          'problem5_strategy.py','main.py','validate_reliability.py']

def remove_docstrings_only(text):
    """Remove function/class docstrings but keep file-level one and all code"""
    lines = text.split('\n')
    result = []
    in_triple = False
    triple_start_line = -1
    file_doc_done = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        if '"""' in stripped:
            count = stripped.count('"""')
            if not in_triple:
                # Start of docstring
                if not file_doc_done:
                    # Keep first docstring but compress to 1 line
                    doc_start = i
                    content_start = stripped.strip('"').strip()
                    j = i + 1
                    while j < len(lines) and '"""' not in lines[j]:
                        j += 1
                    # Get first meaningful line
                    for k in range(i, min(j+1, len(lines))):
                        s = lines[k].strip().strip('"').strip()
                        if s and not s.startswith('===') and not s.startswith('项目'):
                            content_start = s
                            break
                    result.append(f'"""{content_start[:150]}"""\n')
                    file_doc_done = True
                    i = j + 1 if j < len(lines) and '"""' in lines[j] else i + 1
                    continue
                else:
                    # Later docstring — skip
                    if count >= 2:
                        i += 1
                        continue
                    j = i + 1
                    while j < len(lines) and '"""' not in lines[j]:
                        j += 1
                    i = j + 1 if j < len(lines) else len(lines)
                    continue
            else:
                in_triple = False
                i += 1
                continue
        
        # Remove separator comment lines
        if re.match(r'^(\s*)#\s*[=-]{3,}', stripped):
            i += 1
            continue
        
        # Remove long pure-comment blocks (4+ consecutive # lines)
        if stripped.startswith('#') and not stripped.startswith('#!'):
            j = i
            while j < len(lines) and (lines[j].strip().startswith('#') or lines[j].strip() == ''):
                j += 1
            comment_count = sum(1 for k in range(i, j) if lines[k].strip().startswith('#'))
            if comment_count >= 4:
                i = j
                continue
        
        result.append(line)
        i += 1
    
    return '\n'.join(result)


for fname in TARGET:
    fp = os.path.join(PROJECT, fname)
    if not os.path.exists(fp): continue
    with open(fp, 'r', encoding='utf-8') as f: orig = f.read()
    orig_n = len(orig.split('\n'))
    cleaned = remove_docstrings_only(orig)
    new_n = len(cleaned.split('\n'))
    with open(fp, 'w', encoding='utf-8') as f: f.write(cleaned)
    print(f'{fname}: {orig_n} -> {new_n} lines ({(1-new_n/orig_n)*100:.0f}% reduced)')
