#!/bin/bash
#PJM -g gk77
#PJM -L rscgrp=share-interactive
#PJM -N ddsp_pytorch
#PJM -j

source /work/01/gk77/k77021/.bashrc
export HOME=/work/01/gk77/k77021
#pip install -r requirements.txt

#python preprocess.py
python train.py --name test --steps 1000000 --batch 16

