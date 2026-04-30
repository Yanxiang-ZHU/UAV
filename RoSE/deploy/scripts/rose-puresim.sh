#!/bin/bash

ROSE_DIR=$(pwd)

START_Y='0.0'
END_X='-80.0'
END_CYCLE=60_000_000_000
AIRSIM_STEPS=2
FIRESIM_CYCLES=20_000_000
ANGLE=200.0
DNN='trail_resnet14_complex.onnx'
AIRSIM_IP='10.134.141.151'

VELS=( 6 9 12 )
LEN_VEL=${#VELS[@]}

cd ${ROSE_DIR}

for (( i=0; i<${LEN_VEL}; i++ ));
do
    echo "===================================================="
    echo "Starting experiment with velocity ${VELS[$i]}"
    echo "===================================================="

    echo "RoSE: Updating workload YAML"
    jq ".command = \"/root/drone_test -m /root/${DNN} -i /root/img_56.png -p unit -x 2 -O 99 -v ${VELS[$i]}\"" \
        ${ROSE_DIR}/soc/sw/rose-images/airsim-control-fed.json > ${ROSE_DIR}/soc/sw/rose-images/tmp.json
    mv ${ROSE_DIR}/soc/sw/rose-images/tmp.json ${ROSE_DIR}/soc/sw/rose-images/airsim-control-fed.json 

    cd ${ROSE_DIR}/deploy/hephaestus/
    touch ./reset.txt

    echo "Launching Mock FireSim..."
    python3 ${ROSE_DIR}/sim/firesim_mock.py > ${ROSE_DIR}/deploy/hephaestus/logs/mock-firesim-${VELS[$i]}.log 2>&1 &
    MOCK_PID=$!
    sleep 2  # give Mock FireSim time to start

    echo "Running RoSE simulation (Sync-only)"
    echo ${ANGLE} > angle.txt
    python3 runner_sim.py -i ${AIRSIM_IP} -a ${AIRSIM_STEPS} -f ${FIRESIM_CYCLES} \
        -y ${START_Y} -c ${END_CYCLE} -x ${END_X} \
        -l ${ROSE_DIR}/deploy/hephaestus/logs/rose-velocity-sweep-sync-${VELS[$i]}

    echo "Killing Mock FireSim..."
    kill ${MOCK_PID}
    sleep 2
done

