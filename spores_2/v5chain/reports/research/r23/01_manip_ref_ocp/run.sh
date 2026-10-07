cd ~/spore_v5/r23/01_manip_ref_ocp; export PYTHONPATH=~/spore_v5/r23/pylib NQ=20 OMP_NUM_THREADS=1
for W in 3 50; do for g in 0,1 2,3 4,5 6,7 8,9 10,11 12,13 14,15 16,17 18,19; do
  WMP=$W setsid nohup python3 ref.py 60 3 $g > out_w${W}_${g/,/_}.jsonl 2> err_w${W}_${g/,/_}.log < /dev/null &
done; done
