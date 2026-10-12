# env v7-бенча маятника (база); дописывать переменные перед python3
export SYS=pend GOALSHAPE=box GSEED=0 TMAX=8 RMAX=.3 OVL=.05 RHO=.1 DELTA=1e9 GOALB=1 GROW=2 OVH=.5 FRAC=.5 M=5 DTN=.06 NORM=gram SEL=stop TQDM_MI=1000 STOPS=0 OMP_NUM_THREADS=1
# b3 2026-10-13: лучший набор pend (бисекция по времени ВКЛ по слову пользователя; стены/CUT/CUTR не используем)
export COVTOL=.003 SOLVE8_PLUS=1 ACTIVE=1 NORMFRONT=2 DELTA=.2 MODE=maxv THR=1.0 XENTRY=8 BISECT_STAB=.02
