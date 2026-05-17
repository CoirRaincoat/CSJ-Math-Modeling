import re, os
d = r'C:\Users\CoirRaincoat\PyCharmMiscProject\CSJ-MathModeling'
for fname in ['problem1_analysis.py','problem2_prediction.py','problem3_optimization.py','problem4_combos.py','problem5_strategy.py','validate_reliability.py']:
    fp = os.path.join(d, fname)
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read()
    titles = re.findall(r"set_title\(['\"](.+?)['\"]", content)
    xlabels = re.findall(r"set_xlabel\(['\"](.+?)['\"]", content)
    ylabels = re.findall(r"set_ylabel\(['\"](.+?)['\"]", content)
    legends = re.findall(r"label=['\"](.+?)['\"]", content)
    print(f'\n=== {fname} ===')
    print(f'  Titles: {len(titles)}, Xlabels: {len(xlabels)}, Ylabels: {len(ylabels)}, Legends: {len(legends)}')
    for t in titles:
        if any(c.isalpha() and ord(c)<128 for c in t):
            print(f'    TITLE: {t[:100]}')
    for t in xlabels:
        if any(c.isalpha() and ord(c)<128 for c in t):
            print(f'    XLABEL: {t[:100]}')
    for t in ylabels:
        if any(c.isalpha() and ord(c)<128 for c in t):
            print(f'    YLABEL: {t[:100]}')
    for t in legends[:15]:
        if any(c.isalpha() and ord(c)<128 for c in t):
            print(f'    LEGEND: {t[:80]}')
