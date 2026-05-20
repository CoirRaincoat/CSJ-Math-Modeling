import sys
sys.path.insert(0, r'C:\Users\CoirRaincoat\PyCharmMiscProject\CSJ-MathModeling')
from data_loader import load_all_data
l = load_all_data()
d = l.get_daily_data()
df1 = l.df1_raw
df2 = l.df2_raw

print("=== DAILY STATS ===")
print(d[['total_orders','total_sales','total_calories','total_protein','total_fat','total_carbohydrates']].describe().round(1).to_string())

print("\n=== ORDER VALUE ===")
print(df1['consume_money'].describe().round(2).to_string())

print("\n=== NUTRITION PER ORDER ===")
for c in ['calories','protein','fat','carbohydrates','fiber']:
    s = df1[c].describe()
    print(f"{c}: mean={s['mean']:.1f} std={s['std']:.1f} min={s['min']:.1f} max={s['max']:.1f}")

print(f"\n=== OVERVIEW ===")
print(f"Date: {df1['date'].min()} to {df1['date'].max()}")
print(f"Days: {df1['date'].nunique()}")
print(f"Orders: {df1['indent_id'].nunique()}")
print(f"Transactions: {len(df1)}")
print(f"Dishes: {df2['dish_name'].nunique()}")
print(f"Detail records: {len(df2)}")
print(f"Detail orders: {df2['indent_id'].nunique()}")

# Category distribution
print("\n=== CATEGORY DIST ===")
cat = df2.groupby('category')['indent_details_id'].count()
for cname, cnt in cat.items():
    print(f"  {cname}: {cnt} ({cnt/len(df2)*100:.1f}%)")

# Lunch/dinner
print(f"\nLunch orders: {(df1['meal_period']=='lunch').sum()} ({(df1['meal_period']=='lunch').sum()/len(df1)*100:.1f}%)")
print(f"Dinner orders: {(df1['meal_period']=='dinner').sum()} ({(df1['meal_period']=='dinner').sum()/len(df1)*100:.1f}%)")
