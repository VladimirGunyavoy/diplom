D=~/spore_v5/r22/05_di4_gpu_jacobi; S=~/spore_v5/r22/03_di4_edge_causality; cd $D; export SAVEE=$S/E VREFP=$S/V.pkl PYTHONUNBUFFERED=1; PY=~/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python
$PY -c "import torch,tqdm;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
for v in "REP=1 F64=1" "REP=1 F64=0" "REP=4 F64=1"; do env $v $PY gpu.py 2>&1 | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v "^$"; done
nvidia-smi --query-gpu=memory.used --format=csv,noheader
