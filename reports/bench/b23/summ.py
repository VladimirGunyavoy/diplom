import sys,re,glob
for f in sorted(glob.glob(sys.argv[1])):
    r=[];c=0
    for l in open(f):
        m=re.search(r"return ([-\d.]+)±([\d.]+) cost (\d+) обрывов (\d+)",l)
        if m: r.append(float(m[1])); c+=int(m[3])
    if r: print(f,len(r),"mean %.2f min %.2f cost %d"%(sum(r)/len(r),min(r),c))
