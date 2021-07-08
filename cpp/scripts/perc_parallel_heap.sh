#!/bin/bash

# Example: echo $(seq 1 8) | xargs -n 1 -P 4 ./perc_parallel_heap.sh ER k5.00 16000 Ran True

BASE_DIR="../../networks"

NET_TYPE=$1
PARAM=$2
N=$3
ATTACK=$4
OVERWRITE=$5
SEED=$( printf %05d $6 )

BASE_DIR="${BASE_DIR}/${NET_TYPE}"
BASE_NET="${NET_TYPE}_${PARAM}"

[ "${NET_TYPE}" == "Lattice" ] || [ "${NET_TYPE}" == "PLattice" ] || [ "${NET_TYPE}" == "Ld3" ] && DETERMINISTIC="true" || DETERMINISTIC="false"

if $DETERMINISTIC == "true"; then
  BASE_NET_DIR="${BASE_DIR}/${BASE_NET}/${BASE_NET}_L${N}"
  INPUT_NET_NAME="${BASE_NET}_L${N}_00000"
  NET_NAME="${BASE_NET}_L${N}_${SEED}"
else
  BASE_NET_DIR="${BASE_DIR}/${BASE_NET}/${BASE_NET}_N${N}"
  INPUT_NET_NAME="${BASE_NET}_N${N}_${SEED}"
  NET_NAME="${INPUT_NET_NAME}"
fi

INPUT_NET_DIR="${BASE_NET_DIR}/${INPUT_NET_NAME}"
NET_DIR="${BASE_NET_DIR}/${NET_NAME}"
ATTACK_DIR="${NET_DIR}/${ATTACK}"
NETWORK="${INPUT_NET_DIR}/${INPUT_NET_NAME}.txt"
NETWORK_TAR="${INPUT_NET_DIR}/${INPUT_NET_NAME}.tar.gz"

ORDER="${ATTACK_DIR}/oi_list.txt"
echo ${ORDER}
if [ -f "${ORDER}" ]; then
    if [ ! -s "${ORDER}" ]; then
        echo "File ${ORDER} has 0 size and it will be removed."
        rm ${ORDER}
        exit 0
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

TAR_FILE="${ATTACK_DIR}/comp_data.tar.gz"
if [ -f "${TAR_FILE}" ]; then
    if [ "${OVERWRITE}" == "True" ]; then
        rm ${TAR_FILE}
    else
        echo "File ${TAR_FILE} already exist"
        exit 0
    fi
fi

OUTPUT="${ATTACK_DIR}/comp_data.txt"
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

if [ $DETERMINISTIC == "false" ]; then
    if [ -f "${NETWORK_TAR}" ]; then
        #echo "Extracting data from ${NETWORK_TAR}"
        tar --directory=${INPUT_NET_DIR} -xzf ${NETWORK_TAR}
    fi
fi

if [ $DETERMINISTIC == "true" ]; then
    if [ ! -f "${NETWORK}" ]; then
        #echo "Extracting data from ${NETWORK_TAR}"
        tar --directory=${INPUT_NET_DIR} -xzf ${NETWORK_TAR}
    fi
fi


../perc_heap $NETWORK $ORDER $OUTPUT 

if [ $DETERMINISTIC == "false" ]; then
    if [ -f "${NETWORK_TAR}" ]; then
        rm ${NETWORK}
    else
        tar --directory=${INPUT_NET_DIR} -czf ${NETWORK_TAR} ${NETWORK}
        if [ -f "${NETWORK_TAR}" ]; then
            rm ${NETWORK}
        fi
    fi
fi
## Compress component data and remove original
tar -czf ${TAR_FILE} --directory ${ATTACK_DIR} comp_data.txt

if [ -f "${TAR_FILE}" ]; then
    rm ${OUTPUT}
fi