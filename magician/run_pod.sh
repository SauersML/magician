#!/bin/bash
# One GPU: prep, train and evaluate Qwen3-1.7B and Qwen3-4B side by side. Usage: run_pod.sh OUTDIR STEPS_1.7B STEPS_4B
O=$1; S1=${2:-3000}; S4=${3:-2500}
mkdir -p "$O"
cd "$(dirname "$0")"
nvidia-smi --query-gpu=name,memory.total --format=csv
arm() {  # arm TAG MODEL STEPS
    export MAGIC_OUT=$O/$1 MAGIC_MODEL=$2
    python prep.py > "$O/prep_$1.log" 2>&1 &&
    python train.py --name m1 --steps "$3" --save_every 1000 > "$O/train_$1.log" 2>&1 &&
    python evaluate.py --adapter "$O/$1/m1/step_$3" --name m1 > "$O/eval_$1.log" 2>&1
    echo "arm $1 exit $?"
}
arm q17 Qwen/Qwen3-1.7B "$S1" &
P1=$!
arm q4 Qwen/Qwen3-4B "$S4" &
P2=$!
wait $P1 $P2
nvidia-smi
