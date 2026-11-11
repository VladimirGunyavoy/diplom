B=$HOME/spore_v5/b17; cd $B/spores_2/bench; PY=$HOME/spore_v5/r31/venv/bin/python
export EULER=1 QUANT=1 CLRDT=.05 RHOB=.3 RHOF=.15 GMARG=.05 PROCS=6
for pad in .05 .02; do
  while [ $(free -g | awk "/Mem:/{print \$7}") -lt 9 ]; do sleep 30; done
  PAD=$pad $PY spores_drone_b.py 6000 400 5 _pad$pad > $B/dr_pad$pad.log 2>&1
  echo "$(date +%T) done pad$pad: $(tail -n1 $B/dr_pad$pad.log | cut -c1-300)" >> $B/q19.log
done
echo "$(date +%T) Q19 DONE" >> $B/q19.log
