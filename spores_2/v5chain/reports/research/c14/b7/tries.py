import json, sys, collections
R = [json.loads(l) for l in open(sys.argv[1])]
for t0 in sorted(set(r['tries0'] for r in R)):
    Rs = [r for r in R if r['tries0'] == t0]; ok = [r for r in Rs if r['ok']]
    # номер попытки, давшей итоговый (последний 'best'), и номер ПЕРВОЙ успешной (best или worse)
    last = collections.Counter(); first = collections.Counter(); nsucc = collections.Counter(); st = collections.Counter()
    for r in Rs:
        for x in r['L']: st[x[1]] += 1
        succ = [x for x in r['L'] if x[1] in ('best', 'worse')]
        if r['ok']:
            last[[x for x in r['L'] if x[1] == 'best'][-1][0]] += 1; first[succ[0][0]] += 1
        nsucc[len(succ)] += 1
    print('tries0=%d вызовов %d, ok %d; исходы попыток %s; номер ПОБЕДИВШЕЙ (лучшее T) %s; первой успешной %s; число успешных попыток в вызове %s' % (t0, len(Rs), len(ok), dict(st), dict(sorted(last.items())), dict(sorted(first.items())), dict(sorted(nsucc.items()))))
