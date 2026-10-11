import os, shutil, tempfile
import openpyxl
import nfl_sheet as s

for spread, corr in ((-9.5, 1.137), (7.0, 0.16), (-3.5, -0.02), (2.5, 1.03), (-6.5, -0.3), (3.0, 0.0), (-1.5, 0.74)):
    e = s.anchored_fair_line(spread, corr)
    g = e + spread
    print(spread, corr, '-> fair line', e, 'fair line vs spread', g, 'side', 'home' if g > 0 else 'away', '| correction side', 'home' if corr > 0 else 'away')

# every sheet fair line is a half point and its side matches the correction's
bad = 0
for i in range(20000):
    import random
    sp = random.choice([x / 2 for x in range(-60, 61)])
    c = random.uniform(-3, 3)
    g = s.anchored_fair_line(sp, c) + sp
    bad += (g > 0) != (c > 0) or (s.anchored_fair_line(sp, c) * 2) % 1 != 0
print('violations', bad)
