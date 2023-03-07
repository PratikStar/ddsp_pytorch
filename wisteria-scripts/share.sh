#!/bin/bash
#PJM -g gk77
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -N ddsp_pytorch_share
#PJM -j
#PJM -m b
#PJM -m e


echo "Hostname: $HOSTNAME"
echo "pwd: $(pwd)"

echo "====== CPU info ======"
lscpu
echo "======================"

echo "====== GPU info ======"
nvidia-smi
echo "======================"


source /work/01/gk77/k77021/.bashrc
export HOME=/work/01/gk77/k77021
#pip install -r requirements.txt

#python preprocess.py
python train.py --name ddsp-all-share --steps 100000000 --batch 16

