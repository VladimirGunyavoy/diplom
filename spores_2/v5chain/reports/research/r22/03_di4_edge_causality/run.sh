D=~/spore_v5/r22/03_di4_edge_causality; S=~/spore_v5/r22/01_p37_di4_V_vs_Tbox; mkdir -p $D; cd $D; cp $S/{growN.py,finish_gen.py,solve_caus_test.py} .
export SYS=di4 M=3 SIDE=0 GM=.25 GLIM=0 FRAC=1 RHO=.35 PESS=1 LAYERS=~/spore_v5/w23/v7/reports/di4/L400_rho35.pkl VREF=~/spore_v5/w23/v7/reports/di4/V400_rho35_pess1.pkl PYTHONUNBUFFERED=1 SAVEE=$D/E
SBCAUS=1 SBLAY=0 SAVEV=V.pkl stdbuf -oL python3 solve_caus_test.py 2>&1 | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v "^$"
python3 caus.py 2>&1 | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v "^$"
