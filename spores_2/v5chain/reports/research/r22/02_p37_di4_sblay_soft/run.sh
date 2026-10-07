set -e
D=~/spore_v5/r22/02_p37_di4_sblay_soft; S=~/spore_v5/r22/01_p37_di4_V_vs_Tbox; mkdir -p $D; cd $D; cp $S/{growN.py,finish_gen.py,solve_caus_test.py,cmp.py} .
python3 - <<'PY'
s = open('growN.py').read()
old = "keep = np.where(cl_[nc_] == ui, vc_ != nc_, (cl_[vc_] == ui) if SBL else True)"
new = "ml_ = cl_[vc_] == ui; hs_ = np.zeros(s.N, bool); hs_[a[ml_ & (cl_[nc_] != ui)]] = True; keep = np.where(cl_[nc_] == ui, vc_ != nc_, (ml_ | ~hs_[a]) if SBL == 2 else ml_ if SBL else True); print('SBLAY', SBL, 'u', ui, 'узлов чужих слоёв со стенсилом в слой u: %.3f' % (hs_[cl_ != ui].mean()), flush=True)   # research-22: SBLAY=2 мягкий — слой u2, а где его нет — любой слой"
assert old in s; open('growN.py', 'w').write(s.replace(old, new))
PY
export SYS=di4 M=3 SIDE=0 GM=.25 GLIM=0 FRAC=1 RHO=.35 PESS=1 LAYERS=~/spore_v5/w23/v7/reports/di4/L400_rho35.pkl VREF=~/spore_v5/w23/v7/reports/di4/V400_rho35_pess1.pkl PYTHONUNBUFFERED=1
SBCAUS=1 SBLAY=2 SAVEV=V_caus_lay2.pkl stdbuf -oL python3 solve_caus_test.py 2>&1 | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v "^$"
python3 cmp.py V_caus_lay2.pkl 2>&1 | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v "^$"
