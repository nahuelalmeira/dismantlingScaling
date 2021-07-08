#!/bin/bash

# Example: echo $(seq 1 8) | xargs -n 1 -P 4 ./perc_parallel.sh ER k5.00 16000 Ran True

BASE_DIR="../../networks"

NET_TYPE=$1
PARAM=$2
N=$3
ATTACK=$4
OVERWRITE=$5
SEED=$( printf %05d $6 )

BASE_DIR="${BASE_DIR}/${NET_TYPE}"
BASE_NET="${NET_TYPE}_${PARAM}"

if [ "${NET_TYPE}" == "Lattice" ] || [ "${NET_TYPE}" == "PLattice" ] || [ "${NET_TYPE}" == "Ld3" ]; then
  BASE_NET_DIR="${BASE_DIR}/${BASE_NET}/${BASE_NET}_L${N}"
  NET_NAME="${BASE_NET}_L${N}_${SEED}"
else
  BASE_NET_DIR="${BASE_DIR}/${BASE_NET}/${BASE_NET}_N${N}"
  NET_NAME="${BASE_NET}_N${N}_${SEED}"
fi


NET_DIR="${BASE_NET_DIR}/${NET_NAME}"
NETWORK="${NET_DIR}/${NET_NAME}.txt"
NETWORK_TAR="${NET_DIR}/${NET_NAME}.tar.gz"

if [ -f "${NETWORK_TAR}" ]; then
    #echo "Extracting data from ${NETWORK_TAR}"
    tar --directory=${NET_DIR} -xzf ${NETWORK_TAR}
fi

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
    exit 0  
fi  

if [ ! -f "${ORDER}" ]; then 
    echo "File ${ORDER} does not exist"
    exit 0
fi

TAR_FILE="${NET_DIR}/${ATTACK}/comp_data_v2.tar.gz"
if [ -f "${TAR_FILE}" ]; then
    if [ "${OVERWRITE}" == "True" ]; then
        rm ${TAR_FILE}
    else
        echo "File ${TAR_FILE} already exist"
        exit 0
    fi
fi

OUTPUT="${NET_DIR}/${ATTACK}/comp_data_v2.txt"
if [ -f "${OUTPUT}" ]; then
    if [ ! -s "${OUTPUT}" ]; then
        echo "File ${OUTPUT} has 0 size and it will be removed."
        rm ${OUTPUT}
    fi
fi

if [ "${OVERWRITE}" != "True" ]; then
    if [ -f "${OUTPUT}" ]; then 
        echo "File ${OUTPUT} already exist"
        exit 0
    fi
fi

../perc_v2 $NETWORK $ORDER $OUTPUT 

if [ -f "${NETWORK_TAR}" ]; then
    rm ${NETWORK}
fi