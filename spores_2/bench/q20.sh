B=$HOME/spore_v5/b17; cd $B/spores_2/bench; PY=$HOME/spore_v5/r31/venv/bin/python
export EULER=1 QUANT=1 CLRDT=.05 PAD=.1 RHOB=.3 RHOF=.15 GMARG=.05 PROCS=6 NUMBA1=1
for nf in 400 800; do
  $PY spores_drone_b.py 6000 $nf 5 _nbk$nf > $B/dr_nbk$nf.log 2>&1
  echo "$(date +%T) done nbk NF$nf: $(tail -n1 $B/dr_nbk$nf.log | cut -c1-330)" >> $B/q20.log
done
echo "$(date +%T) Q20 DONE" >> $B/q20.log
