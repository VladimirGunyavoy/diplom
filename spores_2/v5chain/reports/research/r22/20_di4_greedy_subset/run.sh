D=~/spore_v5/r22/20_di4_greedy_subset; S=~/spore_v5/r22/18_di4_one_stencil_per_group; cd $D; cp $S/{growN.py,finish_gen.py,sgpu.py,di4_ref_60_rho35.npy,fin.py,fin2.py,p18.py} .
export SYS=di4 M=3 SIDE=0 GM=.25 GLIM=0 FRAC=1 RHO=.35 PESS=1 PYTHONUNBUFFERED=1 FH=.1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PY=~/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python
LAYERS=~/spore_v5/wb5/p38/Lfull.pkl TARGETS=.90,.99 $PY greedy.py 2>&1 | stdbuf -oL tr '\r' '\n' | grep --line-buffered -a "^слой\|^сохранено\|индекс слоя\|Error\|Trace\|line " | stdbuf -oL cut -c1-520
for t in 09 099; do TAG=greedy$t LAYERS=$D/Lgreedy_$t.pkl $PY p18.py 2>&1 | stdbuf -oL tr '\r' '\n' | grep --line-buffered -a "^\[\|стенсилы u\|Error\|Trace\|line " | stdbuf -oL cut -c1-430; done
