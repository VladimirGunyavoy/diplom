# b18 добивки: (1) random Б-выкл LSNOINST, (2) SEED 1,2 А и Б-GPU random, (3) TIGHT Б-GPU fixed+random
B=$HOME/spore_v5/b17; cd $B/spores_2/bench; PY=$HOME/spore_v5/r31/venv/bin/python; PS=$HOME/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python
SOCK=/tmp/lsgpu_b18.sock; rm -f $SOCK
LSDTN=.2 LSWTHR=.5 LSBIG=1000 nohup $PS ls_gpu_server.py $B/zoned2.npz $SOCK > $B/gpu_server3.log 2>&1 &
SP=$!; until grep -q ГОТОВ $B/gpu_server3.log || ! kill -0 $SP 2>/dev/null; do sleep 5; done
E="DTN=.2 VF=1.5 NODISC=1 M=3 SIDE=0 GM=.25 GLIM=0 FRAC=1 PESS=1 GS=300 SYS=car H=1.5 EPS=.03 LOAD=$HOME/spore_v5/a34/cf/d2.pkl"
Z="ZFULL=1 ZCACHE=$B/zoned2.npz MARG=none LS32=1 LSPIPE=1"
GP="$Z LOOK=5 NP=1 LSGPU=1 LSGPUSOCK=$SOCK"
run() { tag=$1; shift
  while [ $(free -g | awk "/Mem:/{print \$7}") -lt 22 ]; do sleep 30; done
  env $E "$@" TAG=_$tag $PY envs_agentA.py > $B/$tag.log 2>&1
  echo "$(date +%T) done $tag: $(tail -n1 $B/$tag.log | cut -c1-200)" >> $B/q15.log
}
for m in fixed random; do run d2gpuTpre_$m $GP TIGHT=1 LSPRE=1 MODE=$m; done
kill $SP; echo "$(date +%T) Q14 DONE" >> $B/q15.log

