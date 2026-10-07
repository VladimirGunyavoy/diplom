D=~/spore_v5/r22/04_di4_policy_iter; S=~/spore_v5/r22/03_di4_edge_causality; cd $D; export SAVEE=$S/E VREFP=$S/V.pkl PYTHONUNBUFFERED=1
for K in 0 20 100; do KIN=$K python3 mpi.py 2>&1 | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v "^$" & done; wait
