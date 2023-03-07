#!/bin/bash
#PJM -g gk77
#PJM -L rscgrp=share-short
#PJM -L gpu=4
#PJM -N CondConv
#PJM -j
#PJM -m b
#PJM -m e

# run commands
source /work/01/gk77/k77021/.bashrc
export HOME=/work/01/gk77/k77021
export WANDB_API_KEY=fe8a478450e5490fe6d24bf513fbcb64c4f4e831
#env
python3 run-generic.py -c configs/timbre_transfer.yaml