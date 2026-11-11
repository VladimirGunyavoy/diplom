# b18: дрон no_path — варианты NB/NF на базе EULER+QUANT+CLRDT=.05 pad .1 (16/20), копия spores_drone_b.py
B=$HOME/spore_v5/b17; cd $B/spores_2/bench; PY=$HOME/spore_v5/r31/venv/bin/python
export EULER=1 QUANT=1 CLRDT=.05 PAD=.1 RHOB=.3 RHOF=.15 GMARG=.05 PROCS=6
run() { nb=$1; nf=$2; tag=$3
  while [ $(free -g | awk "/Mem:/{print \$7}") -lt 9 ]; do sleep 30; done
  $PY spores_drone_b.py $nb $nf 5 $tag > $B/dr_$tag.log 2>&1
  echo "$(date +%T) done dr_$tag NB$nb NF$nf: $(tail -n1 $B/dr_$tag.log | cut -c1-330)" >> $B/q17.log
}
run 6000 400 _b0
run 12000 400 _nb2
run 6000 800 _nf2
run 12000 800 _nb2nf2
echo "$(date +%T) Q17 DONE" >> $B/q17.log
