set -e
D=~/spore_v5/r22/01_p37_di4_V_vs_Tbox; cd $D
export SYS=di4 M=3 SIDE=0 GM=.25 GLIM=0 FRAC=1 RHO=.35 PESS=1 LAYERS=~/spore_v5/w23/v7/reports/di4/L400_rho35.pkl VREF=~/spore_v5/w23/v7/reports/di4/V400_rho35_pess1.pkl PYTHONUNBUFFERED=1
SBCAUS=1 SBLAY=1 SAVEV=V_caus_lay1.pkl python3 solve_caus_test.py 2>&1 | tr '\r' '\n' | grep --line-buffered -v "^$"
SBCAUS=1 SBLAY=0 SAVEV=V_caus_lay0.pkl python3 solve_caus_test.py 2>&1 | tr '\r' '\n' | grep --line-buffered -v "^$"
python3 cmp.py V_caus_lay1.pkl V_caus_lay0.pkl 2>&1 | tr '\r' '\n' | grep --line-buffered -v "^$"
