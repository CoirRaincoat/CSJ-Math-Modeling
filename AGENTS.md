# AGENTS.md

## Python executable

The real Python is at:
```
C:\Users\CoirRaincoat\AppData\Local\Programs\Python\Python313\python.exe
```
Do NOT use bare `python` — it resolves to a Windows App Execution Alias that hangs silently.

## Chinese filename encoding (CRITICAL)

The xlsx attachments have Chinese names. On Windows GBK, hardcoded Chinese paths cause `FileNotFoundError`.
`config.py:_find_attachments()` solves this by matching files via **file size** (attachment1 ≈ 12.8MB, attachment2 ≈ 5.5MB).
Never hardcode Chinese filenames. Always use `ATTACHMENT1` / `ATTACHMENT2` from config.

## Run commands

```powershell
# Full pipeline (all 5 problems)
& "C:\Users\CoirRaincoat\AppData\Local\Programs\Python\Python313\python.exe" main.py

# Single problem only
& "C:\Users\CoirRaincoat\AppData\Local\Programs\Python\Python313\python.exe" main.py --only 3

# Skip specific problems
& "C:\Users\CoirRaincoat\AppData\Local\Programs\Python\Python313\python.exe" main.py --skip 2,5
```

## Architecture: DataLoader singleton

`DataLoader` loads ~18MB of Excel and runs full ETL (cleaning, feature engineering, aggregations, basket building).
**Instantiate once, pass to all problem modules.** Every problem module accepts `loader=None` and creates its own only if none is given.

```
config.py (global settings, paths, colors)
    ↓
data_loader.py (DataLoader — one instance)
    ↓
┌──────────┬──────────┬──────────┬──────────┬──────────┐
│problem1  │problem2  │problem3  │problem4  │problem5  │
└──────────┴──────────┴──────────┴──────────┴──────────┘
    ↓
main.py (orchestrator, --skip / --only CLI)
```

## Problem 3 is LUNCH ONLY

Dinner has only 11 records (0.8% of orders). The MILP for dinner was infeasible and has been **removed**.
Do NOT add dinner optimization back. The `_select_dishes_for_optimization()` no longer accepts a `meal_period` parameter.
Only lunch plans are generated for 2025-05-06 to 2025-05-12 weekdays.

## Visualization palette: Nature NPG

All charts use the Nature Publishing Group palette defined in `config.py:COLORS`. Do NOT change to matplotlib defaults.
- `primary`: `#3C5488` (dark blue), `secondary`: `#00A087` (green), `accent`: `#E64B35` (red)
- DPI: 300 for saved figures
- Matplotlib backend: `Agg` (non-interactive, set in config.py)

## GBK encoding gotchas

Print statements containing special Unicode crash on Windows GBK terminals. Avoid:
- `CO₂e` → use `CO2eq`
- `≈` → use `~`
- `¥` → use `yuan` text in print, keep `¥` in matplotlib labels (rendered as graphics)

All CSV output uses `utf-8-sig` encoding for Excel compatibility.

## Apriori thresholds

Default `min_support=0.5` produces zero rules for this dataset (237 dishes, highest support ~0.96 for 米饭).
The `_association_rule_mining()` uses a **multi-level descent strategy**: try 0.01 → 0.005 → 0.003 → 0.002.
Confidence ≥ 0.25, Lift ≥ 1.15. Do NOT raise these thresholds.

## Key dataset facts

| Fact | Value |
|------|-------|
| Attachment 1 rows | 65,534 |
| Attachment 2 rows | 65,535 |
| Attachment 2 coverage | 11,828 of 64,587 orders (18.3%) |
| Date range | 2022-09-02 to 2023-11-28 (236 days) |
| Unique dishes | 237 |
| Lunch share | 99.2% |
| Daily avg orders | 274 |
| Avg order value | 11.36 yuan |
| RANDOM_SEED | 42 |

## Dish classification is two-pass

1. **Keyword match** (`_classify_dish`) — matches ~46% of dishes
2. **Nutritional fallback** (`_nutritional_feature_classify`) — reclassifies "其他" dishes by protein/fat/carbs profile

## Dependencies

Install with:
```powershell
& "C:\Users\CoirRaincoat\AppData\Local\Programs\Python\Python313\python.exe" -m pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost statsmodels pulp mlxtend networkx openpyxl
```

## Key references

- `worklog.md` — session log with bugs encountered and fixes
- `final_report.md` — modeling approach and literature citations
- `optimization_changelog.md` — what changed in the latest optimization pass
- `赛题B_方向与文献建议整理.md` — suggested directions for the problem
