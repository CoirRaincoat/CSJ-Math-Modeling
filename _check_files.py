import os

d = r'C:\Users\CoirRaincoat\PyCharmMiscProject\MathModeling'
for f in os.listdir(d):
    full = os.path.join(d, f)
    if os.path.isfile(full) and f.endswith('.xlsx'):
        print(f'File: {repr(f)}')
        print(f'  Size: {os.path.getsize(full)}')
        print(f'  Created: {os.path.getctime(full)}')
        print()
