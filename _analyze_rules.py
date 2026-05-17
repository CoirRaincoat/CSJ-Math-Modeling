import pandas as pd, numpy as np, os
from mlxtend.frequent_patterns import apriori, association_rules

d = r'C:\Users\CoirRaincoat\PyCharmMiscProject\CSJ-MathModeling'
files = os.listdir(d)

f1 = [f for f in files if '1' in f and f.endswith('.xlsx') and len(f)>10][0]
f2 = [f for f in files if '2' in f and f.endswith('.xlsx')][0]

df1 = pd.concat([pd.read_excel(os.path.join(d,f1), sheet_name=s) for s in pd.ExcelFile(os.path.join(d,f1)).sheet_names])
df2 = pd.concat([pd.read_excel(os.path.join(d,f2), sheet_name=s) for s in pd.ExcelFile(os.path.join(d,f2)).sheet_names])

df2['dish_name'] = df2['dish_name'].str.strip()
basket = df2.pivot_table(index='indent_id', columns='dish_name', values='total_price', aggfunc='count', fill_value=0)
basket_bin = (basket > 0).astype(int)
freq = basket_bin.sum()
freq_dishes = freq[freq >= 50].index
basket_f = basket_bin[freq_dishes]

print(f"Total orders with dish details: {len(basket_f):,}")
print(f"Dishes after freq>=50 filter: {len(freq_dishes)}")

for ms in [0.01, 0.005, 0.003]:
    fi = apriori(basket_f, min_support=ms, use_colnames=True, max_len=3)
    n1 = len(fi[fi['itemsets'].apply(len) == 1])
    n2 = len(fi[fi['itemsets'].apply(len) == 2])
    n3 = len(fi[fi['itemsets'].apply(len) == 3])
    print(f"\nmin_support={ms}: {len(fi)} itemsets (1:{n1}, 2:{n2}, 3:{n3})")
    
    if n2 + n3 < 5:
        continue
    
    rules = association_rules(fi, metric='lift', min_threshold=1.0)
    r = rules[(rules['confidence'] >= 0.25) & (rules['lift'] >= 1.15)].sort_values('lift', ascending=False)
    
    if len(r) >= 5:
        print(f"\n=== {len(r)} ASSOCIATION RULES FOUND ===")
        for i, (_, row) in enumerate(r.iterrows()):
            ant = ', '.join(list(row['antecedents']))
            con = ', '.join(list(row['consequents']))
            sup = row['support']
            conf = row['confidence']
            lift = row['lift']
            print(f"{i+1:2d}. {{{ant}}} -> {{{con}}}")
            print(f"    sup={sup:.4f}, conf={conf:.3f}, lift={lift:.2f}")
        break

print(f"\n=== TOP 15 DISHES BY FREQUENCY ===")
for name, cnt in freq.sort_values(ascending=False).head(15).items():
    pct = cnt / len(basket_bin) * 100
    print(f"  {cnt:6d} ({pct:5.1f}%) - {name}")

print(f"\n=== FREQUENCY DISTRIBUTION ===")
print(f"  >=1000 orders: {(freq >= 1000).sum()} dishes")
print(f"  >=500 orders:  {(freq >= 500).sum()} dishes")
print(f"  >=200 orders:  {(freq >= 200).sum()} dishes")
print(f"  >=100 orders:  {(freq >= 100).sum()} dishes")
print(f"  >=50 orders:   {(freq >= 50).sum()} dishes")
print(f"  <50 orders:    {(freq < 50).sum()} dishes")

# Co-occurrence of top dishes
print(f"\n=== CO-OCCURRENCE MATRIX (top 8 dishes, support>=0.01) ===")
top8 = freq.nlargest(8).index.tolist()
for i, d1 in enumerate(top8):
    for j, d2 in enumerate(top8):
        if i < j:
            co = np.sum(basket_f[d1] & basket_f[d2]) / len(basket_f)
            if co > 0.008:
                print(f"  {d1} & {d2}: co-support={co:.4f}")
