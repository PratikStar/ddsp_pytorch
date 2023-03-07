#!/bin/bash
#PJM -g gk77
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -N ddsp_pytorch
#PJM -j
#PJM -m b
#PJM -m e


source /work/01/gk77/k77021/.bashrc
export HOME=/work/01/gk77/k77021
#pip install -r requirements.txt

#python preprocess.py
python train.py --name ddsp-all-noreverb --steps 100000000 --batch 16

