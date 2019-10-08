#!/bin/bash

eval "$(conda shell.bash hook)"
source activate networks
hostname

NET_TYPE=${1}
N=${2}
k=${3}
ATTACK=${4}
SEED=$(expr $5 - 1)
SEEDP1=$5

echo "NET_TYPE=${NET_TYPE}; N=${N}; k=${k}; ATTACK=${ATTACK}; SEED=${SEED}"

python ../../python/attack.py ${NET_TYPE} ${N} ${k} ${SEED} ${SEEDP1} ${ATTACK}
