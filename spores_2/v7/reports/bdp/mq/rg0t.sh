cd ~/spore_v5/r12; export PYTHONPATH=.:~/spore_v5/r5/pylib G=2 WIN=1.0 WPAIR=1 EGAP=1 EQ=97 EM=0 EW=.15 NEW=300 ROUNDS=8 TRUNC=1 KEEP=0 OMP_NUM_THREADS=2
i=$1; export START=$(sed -n ${i}p ~/spore_v5/w19/starts.txt)
python3 dp_g2_regrow.py ~/spore_v5/w17/v7/butterfly_dp_grow_N6000_tr1_fr1_fwd3000_R0.15_g2_rrt_kn4_nt_s${i}_x${i}t.npz > ~/spore_v5/w19/rg0t_$i.out 2>&1
