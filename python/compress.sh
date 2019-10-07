#!/bin/bash

NET_TYPE="ER"
k="5.00"
MIN_SEED=0
MAX_SEED=20000

for N in 250 500 2000 4000 8000 16000 32000 64000; do

    echo $N
    python compress_networks.py ${NET_TYPE} ${N} ${k} ${MIN_SEED} ${MAX_SEED}

done