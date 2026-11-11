B=$HOME/spore_v5/b17; cd $B/spores_2/bench; PY=$HOME/spore_v5/r31/venv/bin/python
E="DTN=.2 VF=1.5 NODISC=1 M=3 SIDE=0 GM=.25 GLIM=0 FRAC=1 PESS=1 GS=300 SYS=car H=1.5 EPS=.03 LOAD=$HOME/spore_v5/a34/cf/d2.pkl"
while [ ! -s $B/boff.log ]; do sleep 20; done
while [ $(free -g | awk "/Mem:/{print \$7}") -lt 22 ]; do sleep 30; done
env $E ZFULL=1 ZCACHE=$B/zoned2.npz MARG=none LOOK=5 NP=4 LS32=1 LSPIPE=1 LSTLS=2.5 LSNOINST=1 MODE=fixed TAG=_d2boff2 $PY envs_agentA.py > $B/d2boff2_fixed.log 2>&1
echo "$(date +%T) done boff2_fixed: $(tail -n1 $B/d2boff2_fixed.log | cut -c1-200)" >> $B/boff2.log
