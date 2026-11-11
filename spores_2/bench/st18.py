import json,sys,numpy as np
D="/home/random/spore_v5/b17/reports/bench/data/"
def st(f,excl=(18,19)):
    d=json.load(open(D+f)); r=d["res"]; Tr=d["Tref"]; k=[x for x in r if x["i"] not in excl]
    q=np.array([x["T"]/Tr[x["i"]] for x in k if x["ok"]]); pre=[ (x["lsn"][0].get("tot",0.) if x["lsn"] and x["lsn"][0].get("pre") else 0.) for x in k]
    qp=np.array([(x["T"]+p)/Tr[x["i"]] for x,p in zip(k,pre) if x["ok"]])
    return "%-26s n=%d дошли %d касаний %d мед %.3f p90 %.3f | (подг+T)/Tref мед %.3f p90 %.3f подг мед %.2f | мс/шаг %.0f"%(f[6:-5],len(k),sum(x["ok"] for x in k),sum(x["touch"] for x in k),np.median(q),np.percentile(q,90),np.median(qp),np.percentile(qp,90),np.median(pre),np.mean([x["ms"] for x in k]))
for f in sys.argv[1:]: print(st(f))
