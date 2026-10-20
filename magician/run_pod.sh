#!/bin/bash
# One GPU: prep, train and evaluate several arms side by side. Usage: run_pod.sh OUTDIR [ARM ...]
# An arm is TAG:MODEL:STEPS:EXTRA[:TRAIN ARGS]. The default arms:
#   q17   Qwen3-1.7B, 135 training concepts
#   q17x  Qwen3-1.7B, 569 training concepts (MAGIC_EXTRA=1); the same 45 held-out concepts
#   q4    Qwen3-4B, 135 training concepts
O=$1
shift
ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=(q17:Qwen/Qwen3-1.7B:3000:0 q17x:Qwen/Qwen3-1.7B:3000:1 q4:Qwen/Qwen3-4B:2500:0)
mkdir -p "$O"
cd "$(dirname "$0")"
nvidia-smi --query-gpu=name,memory.total --format=csv
arm() {  # arm TAG MODEL STEPS EXTRA [TRAIN ARGS]
    export MAGIC_OUT=$O/$1 MAGIC_MODEL=$2 MAGIC_EXTRA=$4
    python prep.py > "$O/prep_$1.log" 2>&1 &&
    python train.py --name m1 --steps "$3" --save_every 1000 $5 > "$O/train_$1.log" 2>&1 &&
    python evaluate.py --adapter "$O/$1/m1/step_$3" --name m1 > "$O/eval_$1.log" 2>&1
    echo "arm $1 exit $?"
}
pids=()
for a in "${ARMS[@]}"; do
    IFS=: read -r tag model steps extra targs <<< "$a"
    arm "$tag" "$model" "$steps" "$extra" "$targs" &
    pids+=($!)
done
wait "${pids[@]}"
nvidia-smi
