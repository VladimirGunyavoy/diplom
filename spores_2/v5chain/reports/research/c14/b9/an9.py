import json, numpy as np, collections
R = [json.loads(l) for l in open('M7.tries.jsonl')]
def pct(a, q): return float(np.percentile(a, q)) if len(a) else None
for t0 in (3, 1):
    Rs = [r for r in R if r['tries0'] == t0]; bad = [r for r in Rs if not r['ok']]; good = [r for r in Rs if r['ok']]
    print('\n=== tries0=%d: вызовов %d, успешных %d, бесплодных %d' % (t0, len(Rs), len(good), len(bad)))
    for name, grp in (('БЕСПЛОДНЫЕ', bad), ('УСПЕШНЫЕ', good)):
        att = [x for r in grp for x in r['R']]; st = collections.Counter(x[1] for x in att)
        its = [x[2] for x in att]; dt = [x[3] for x in att]
        print(name, 'попыток', len(att), 'статусы', dict(st), '| итерации мед %.0f p90 %.0f p95 %.0f max %d | время/попытку мед %.2f p95 %.2f; сумма %.1f с (на вызов %.2f с)' % (np.median(its), pct(its, 90), pct(its, 95), max(its), np.median(dt), pct(dt, 95), sum(dt), sum(dt) / max(1, len(grp))))
        # по номеру попытки
        for k in range(t0):
            a = [x for x in att if x[0] == k]
            if a: print('   попытка %d: n=%d статусы %s итер мед %.0f время сумма %.1f' % (k, len(a), dict(collections.Counter(x[1] for x in a)), np.median([x[2] for x in a]), sum(x[3] for x in a)))
    if t0 == 3:
        # успешные: итерации принятой попытки
        ai = []; fi = collections.Counter()
        for r in good:
            lab = {x[0]: x for x in r['L']}; 
            for x in r['R']:
                if lab.get(x[0], (0, ''))[1] == 'best': ai.append(x[2])
            fi[r['R'][0][1]] += 1
        print('итерации ПРИНЯТЫХ попыток: n=%d мед %.0f p90 %.0f p95 %.0f max %d' % (len(ai), np.median(ai), pct(ai, 90), pct(ai, 95), max(ai)))
        print('статус попытки 0 у успешных вызовов', dict(fi), '; у бесплодных', dict(collections.Counter(r['R'][0][1] for r in bad)))
        tot = sum(x[3] for r in Rs for x in r['R']); print('общее IPOPT-время tries=3: %.1f с, из них бесплодные вызовы %.1f с (%.0f%%)' % (tot, sum(x[3] for r in bad for x in r['R']), 100 * sum(x[3] for r in bad for x in r['R']) / tot))
allt = sum(x[3] for r in R for x in r['R']); print('\nвсего IPOPT-время всех вызовов %.1f с (20 стартов), на старт %.1f с' % (allt, allt / 20))
