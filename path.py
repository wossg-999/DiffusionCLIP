import sys
for path in sys.path:
    if 'site-packages' in path:
        print(path)