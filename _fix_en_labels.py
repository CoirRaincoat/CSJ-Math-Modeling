import os
fp = r'C:\Users\CoirRaincoat\PyCharmMiscProject\CSJ-MathModeling\validate_reliability.py'
with open(fp, 'r', encoding='utf-8') as f:
    c = f.read()

reps = [
    ("label=f'With detail (mean={val_detail.mean():.1f})'", "label=f'\u6709\u660e\u7ec6 (\u5747\u503c={val_detail.mean():.1f})'"),
    ("label=f'No detail (mean={val_nodetail.mean():.1f})'", "label=f'\u65e0\u660e\u7ec6 (\u5747\u503c={val_nodetail.mean():.1f})'"),
    ("label=f'With detail'", "label=f'\u6709\u660e\u7ec6'"),
    ("label=f'No detail'", "label=f'\u65e0\u660e\u7ec6'"),
    ("f'Order Value Distribution\\nKS test: stat={ks_stat:.4f}, p={ks_p:.4f}'", "f'\u8ba2\u5355\u91d1\u989d\u5206\u5e03\\nKS\u68c0\u9a8c: stat={ks_stat:.4f}, p={ks_p:.4f}'"),
    ("'Number of Baseline Rules'", "'\u57fa\u7ebf\u89c4\u5219\u6570'"),
    ("f'Apriori Rule Bootstrap Stability\\n(median survival: {np.median(stability_pcts):.0f}%)'", "f'Apriori\u89c4\u5219Bootstrap\u7a33\u5b9a\u6027\\n(\u4e2d\u4f4d\u751f\u5b58\u7387: {np.median(stability_pcts):.0f}%)'"),
    ("'Lift CV (%)'", "'\u63d0\u5347\u5ea6CV (%)'"),
    ("f'Lift Coefficient of Variation Across Bootstraps\\n(median CV: {np.median(cvs):.1f}%)'", "f'Bootstrap\u63d0\u5347\u5ea6\u53d8\u5f02\u7cfb\u6570\\n(\u4e2d\u4f4dCV: {np.median(cvs):.1f}%)'"),
    ("'XGBoost Feature Importance: Daily Orders Prediction\\n(gain-based, top 15 features)'", "'XGBoost\u7279\u5f81\u91cd\u8981\u6027: \u65e5\u8ba2\u5355\u6570\u9884\u6d4b\\n(gain-based, Top 15\u7279\u5f81)'"),
    ("f'Mean: {df_mc[chr(34)+'profit'+chr(34)].mean():.0f}'", "f'\u5747\u503c: {df_mc[chr(34)+'profit'+chr(34)].mean():.0f}'"),
    ("label=f'5th %ile: {df_mc[chr(34)+'profit'+chr(34)].quantile(0.05):.0f}'", "label=f'5%\u5206\u4f4d: {df_mc[chr(34)+'profit'+chr(34)].quantile(0.05):.0f}'"),
    ("label=f'95th %ile: {df_mc[chr(34)+'profit'+chr(34)].quantile(0.95):.0f}'", "label=f'95%\u5206\u4f4d: {df_mc[chr(34)+'profit'+chr(34)].quantile(0.95):.0f}'"),
    ("'Monte Carlo: Profit Distribution\\n(200 perturbations)'", "'Monte Carlo: \u5229\u6da6\u5206\u5e03\\n(200\u6b21\u6270\u52a8)'"),
    ("'Profit (yuan/day)'", "'\u5229\u6da6 (\u5143/\u5929)'"),
    ("f'Demand Sensitivity\\n(correlation: {df_mc[chr(34)+'demand_factor'+chr(34)].corr(df_mc[chr(34)+'profit'+chr(34)]):.3f})'", "f'\u9700\u6c42\u654f\u611f\u6027\\n(\u76f8\u5173\u7cfb\u6570: {df_mc[chr(34)+'demand_factor'+chr(34)].corr(df_mc[chr(34)+'profit'+chr(34)]):.3f})'"),
    ("'Parameter Sensitivity Ranking\\n(Tornado diagram)'", "'\u53c2\u6570\u654f\u611f\u6027\u6392\u540d\\n(\u98d3\u98ce\u56fe)'"),
    ("f'{col}\\nKS p={ks_pv:.4f}, means: {v1.mean():.0f} vs {v2.mean():.0f}'", "f'{col}\\nKS p={ks_pv:.4f}, \u5747\u503c: {v1.mean():.0f} vs {v2.mean():.0f}'"),
    ("'Hourly Distribution: With vs Without Detail'", "'\u5c0f\u65f6\u5206\u5e03: \u6709\u660e\u7ec6 vs \u65e0\u660e\u7ec6'"),
    ("'Day-of-Week Distribution'", "'\u661f\u671f\u5206\u5e03'"),
]

count = 0
for old, new in reps:
    if old in c:
        c = c.replace(old, new)
        count += 1
    else:
        print(f'  MISS: {old[:80]}')

with open(fp, 'w', encoding='utf-8') as f:
    f.write(c)
print(f'{count} replacements done')
