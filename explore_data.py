"""
数据探索脚本 — 分析附件1和附件2的数据结构与内容
"""
import pandas as pd
import numpy as np
import os

d = r'C:\Users\CoirRaincoat\PyCharmMiscProject\MathModeling'
files = os.listdir(d)

f1 = [f for f in files if '附件1' in f and f.endswith('.xlsx')][0]
f2 = [f for f in files if '附件2' in f and f.endswith('.xlsx')][0]

df1 = pd.read_excel(os.path.join(d, f1))
df2 = pd.read_excel(os.path.join(d, f2))

print('=' * 60)
print('附件1: 餐厅流水基本数据')
print('=' * 60)
print(f'Shape: {df1.shape}')
print(f'Date range: {df1["consume_time"].min()} 至 {df1["consume_time"].max()}')
print(f'Unique indent_ids: {df1["indent_id"].nunique()}')
print(f'consume_way values: {sorted(df1["consume_way"].unique())}')
print(f'consume_way counts:')
print(df1['consume_way'].value_counts().sort_index())
print(f'\npayment_status values: {sorted(df1["payment_status"].unique())}')
print(f'reduction_money unique: {df1["reduction_money"].unique()}')
print(f'is_upload values: {df1["is_upload"].unique()}')
print(f'\nconsume_money stats:')
print(df1['consume_money'].describe())
print(f'\nNutritional stats (attachment 1):')
for col in ['calories', 'carbohydrates', 'protein', 'fat', 'fiber']:
    print(f'  {col}: mean={df1[col].mean():.1f}, std={df1[col].std():.1f}, min={df1[col].min():.1f}, max={df1[col].max():.1f}')

# reduction_contant might contain dish names
print(f'\nreduction_contant unique values:')
vals = df1['reduction_contant'].value_counts()
for v, c in vals.head(20).items():
    print(f'  {c:5d} - {v}')

print('\n' + '=' * 60)
print('附件2: 每次消费菜品及对应营养成分数据')
print('=' * 60)
print(f'Shape: {df2.shape}')
print(f'Unique indent_ids: {df2["indent_id"].nunique()}')
print(f'Unique dish_serial: {df2["dish_serial"].nunique()}')
print(f'Unique dish_name: {df2["dish_name"].nunique()}')

print(f'\nDish names (by frequency, top 30):')
dish_name_counts = df2['dish_name'].value_counts()
for name, cnt in dish_name_counts.head(30).items():
    print(f'  {cnt:5d} - {name}')

print(f'\nDish serials:')
dish_serial_counts = df2['dish_serial'].value_counts()
for s, cnt in dish_serial_counts.head(10).items():
    print(f'  {cnt:5d} - {s}')

print(f'\ntotal_price stats:')
print(df2['total_price'].describe())
print(f'\nweight stats:')
print(df2['weight'].describe())
print(f'\nunit_price stats:')
print(df2['unit_price'].describe())

# Check date ranges by extracting date from consume_time
df1['date'] = df1['consume_time'].dt.date
print(f'\nUnique dates: {df1["date"].nunique()}')
print(f'First date: {df1["date"].min()}, Last date: {df1["date"].max()}')

# Daily transaction counts
daily = df1.groupby('date').size()
print(f'\nDaily transaction counts stats:')
print(daily.describe())

# Check for missing values
print(f'\nMissing values in df1:')
print(df1.isnull().sum())
print(f'\nMissing values in df2:')
print(df2.isnull().sum())

# Check hourly pattern
df1['hour'] = df1['consume_time'].dt.hour
hourly = df1.groupby('hour').size()
print(f'\nHourly transaction counts:')
print(hourly.to_string())
