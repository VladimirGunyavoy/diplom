import sys; sys.path.insert(0,'.'); sys.path.insert(0,'tests')
import test_growview as t, test_stepper as ts
for m in (t, ts):
    for n in sorted(x for x in dir(m) if x.startswith('test_')):
        try: getattr(m,n)(); print('\nRESULT ok', n)
        except BaseException as e: print('\nRESULT FAIL', n, ascii(repr(e))[:300])
