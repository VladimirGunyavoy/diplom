# R20×100 динамик кар, сиды K0..K1-1: атлас c8/atlas/s<k>.pkl; А (LOOK=0 NP=8) и Б-GPU async; диски random seed k; 100 стартов. Кэш зоны (3.4 ГБ) удаляется после сида.
K0=$1; K1=$2; B=$HOME/spore_v5/c8; cd $B/spores_2/bench; PY=$HOME/spore_v5/venv/bin/python; L=$B/r20t.log
for k in $(seq $K0 $((K1-1))); do
  while [ ! -f $B/atlas/s$k.pkl ]; do sleep 20; done
  while [ $(free -g | awk '/Mem:/{print $7}') -lt 22 ] || [ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -lt 12 ]; do sleep 30; done
  Z=$B/zoned_t_s$k.npz; SOCK=/tmp/lsgpu_c8t_$k.sock; rm -f $SOCK
  E="DTN=.2 VF=1.5 NODISC=1 M=3 SIDE=0 GM=.25 GLIM=0 FRAC=1 PESS=1 GS=300 SYS=car H=1.5 EPS=.03 LOAD=$B/atlas/s$k.pkl TIGHT=1 STARTS=$B/r20/starts100_tight.npy MODE=random SEED=$k"
  ZC="ZFULL=1 ZCACHE=$Z MARG=none LS32=1 LSPIPE=1"
  run() { tag=$1; shift; env $E "$@" TAG=_$tag $PY envs_agentA.py > $B/r20/t_$tag.log 2>&1; echo "$(date +%T) $tag: $(tail -n1 $B/r20/$tag.log | cut -c1-260)" >> $L; }
  [ -f $Z ] || run r20ts${k}_build $ZC LOOK=5 NP=1 NST=1
  run r20ts${k}_A LOOK=0 NP=8
  LSDTN=.2 LSWTHR=.5 LSBIG=1000 nohup $PY ls_gpu_server.py $Z $SOCK > $B/r20/gpu_ts$k.log 2>&1 &
  SP=$!; until grep -q ГОТОВ $B/r20/gpu_ts$k.log || ! kill -0 $SP 2>/dev/null; do sleep 5; done
  run r20ts${k}_B $ZC LOOK=5 NP=1 LSGPU=1 LSGPUSOCK=$SOCK LSASYNC=1 LSREAL=1
  kill $SP; rm -f $Z $SOCK
done
echo "$(date +%T) R20DYN-TIGHT DONE $K0..$K1" >> $L
