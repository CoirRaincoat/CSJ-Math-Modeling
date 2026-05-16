import pandas as pd
import os

d = r'C:\Users\CoirRaincoat\PyCharmMiscProject\CSJ-MathModeling'
files = os.listdir(d)

# Attachment 1
f1 = [f for f in files if '1' in f and f.endswith('.xlsx') and len(f) > 10][0]
fp1 = os.path.join(d, f1)
print('=== ATTACHMENT 1 ===')
for s in pd.ExcelFile(fp1).sheet_names:
    df = pd.read_excel(fp1, sheet_name=s, nrows=3)
    print(f'\nSheet: {s}')
    print(f'  Shape (sample): {df.shape}')
    print(f'  Columns: {list(df.columns)}')
    if 'consume_time' in df.columns:
        full = pd.read_excel(fp1, sheet_name=s)
        print(f'  Date range: {full["consume_time"].min()} to {full["consume_time"].max()}')
        print(f'  Unique orders: {full["indent_id"].nunique():,}')
        print(f'  Avg money: {full["consume_money"].mean():.2f}')
    print(f'  First 3:')
    print(df.head(3)[['indent_id','consume_time','consume_money']].to_string() if 'indent_id' in df.columns else df.head(2).to_string())

# Attachment 2
f2 = [f for f in files if '2' in f and f.endswith('.xlsx')][0]
fp2 = os.path.join(d, f2)
print('\n\n=== ATTACHMENT 2 ===')
total_rows = 0
for s in pd.ExcelFile(fp2).sheet_names:
    df = pd.read_excel(fp2, sheet_name=s, nrows=2)
    full = pd.read_excel(fp2, sheet_name=s)
    orders = full['indent_id'].nunique() if 'indent_id' in full.columns else 0
    dishes = full['dish_name'].nunique() if 'dish_name' in full.columns else 0
    total_rows += len(full)
    print(f'  {s}: {len(full)} rows, {orders} orders, {dishes} unique dishes')

print(f'\n  TOTAL附件2 rows: {total_rows:,} (之前只读了65,535)')

# Attachment 3
f3 = [f for f in files if '3' in f and f.endswith('.xlsx')][0]
fp3 = os.path.join(d, f3)
print('\n\n=== ATTACHMENT 3 ===')
for s in pd.ExcelFile(fp3).sheet_names:
    df = pd.read_excel(fp3, sheet_name=s)
    print(f'\nSheet: {s}')
    print(df.to_string())
