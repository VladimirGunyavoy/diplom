cd ~/spore_v5/r23/06_dp1_ref; export PYTHONPATH=~/spore_v5/r23/pylib NQ=20 OMP_NUM_THREADS=1 WMP=3 G=1
for q in $(seq 0 19); do setsid nohup python3 ref.py 160 4 $q > out_$q.jsonl 2> err_$q.log < /dev/null & done
