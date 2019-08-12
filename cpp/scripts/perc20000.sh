#!/bin/bash

#g++ -o2 -std=c++11 ../source/perc_v2.cpp -o ../perc_v2

BASE_DIR="../../networks/ER"

BASE_NET=$1
N=$2
ATTACK=$3
OVERWRITE=$4

BASE_NET_DIR="${BASE_DIR}/${BASE_NET}/${BASE_NET}_N${N}"

for SEED in {10000..19999}; do

    NET_NAME="${BASE_NET}_N${N}_${SEED}"
    NET_DIR="${BASE_NET_DIR}/${NET_NAME}"
    NETWORK="${NET_DIR}/${NET_NAME}.txt"

    ORDER="${NET_DIR}/${ATTACK}/oi_list.txt"
    echo ${ORDER}
    if [ -f "${ORDER}" ]; then
        if [ ! -s "${ORDER}" ]; then
            echo "File ${ORDER} has 0 size and it will be removed."
            rm ${ORDER}
        fi
    fi

    if [ `wc -l ${ORDER} | awk '{print $1}'` -lt $(echo ${N} | awk '{printf("%d\n", 0.9*$1)}') ]; then
        echo "File ${ORDER} is too short"
        continue  
    fi  

    if [ ! -f "${ORDER}" ]; then 
        echo "File ${ORDER} does not exist"
        continue
    fi

    OUTPUT="${NET_DIR}/${ATTACK}/comp_data.txt"
    if [ -f "${OUTPUT}" ]; then
        if [ ! -s "${OUTPUT}" ]; then
            echo "File ${OUTPUT} has 0 size and it will be removed."
            rm ${OUTPUT}
        fi
    fi

    if [ "${OVERWRITE}" != "overwrite" ]; then
        if [ -f "${OUTPUT}" ]; then 
            echo "File ${OUTPUT} already exist"
            continue
        fi
    fi

    ../perc_v2 $NETWORK $ORDER $OUTPUT 
done
