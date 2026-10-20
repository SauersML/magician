#!/bin/bash
# One GPU: prep, train and evaluate several arms side by side. Usage: run_pod.sh OUTDIR ARM...
# An arm is TAG:MODEL:STEPS:PLAN:INIT[:TRAIN ARGS]. PLAN is MAGIC_PLAN (0 no plan, 1 plans as lists of layers, 2 plans
# as tables); INIT is an adapter to start from, or empty to start fresh (an INIT not there yet is waited for, so it can
# be copied in after the pod starts). Arms on the same model share one prep, in OUTDIR/<model name>.
# Round 5 (results/eval_p17own.json, eval_n17own.json, eval_p4own.json), from the lr10 1.7B adapter and q4c:
#   p17:Qwen/Qwen3-1.7B:2500:1:INIT17:--grad_ckpt 1  n17:Qwen/Qwen3-1.7B:2500:0:INIT17  p4:Qwen/Qwen3-4B:2000:1:INIT4:--grad_ckpt 1
# Round 6: t17:Qwen/Qwen3-1.7B:2500:2:INIT17:--grad_ckpt 1  t4:Qwen/Qwen3-4B:2000:2:INIT4:--grad_ckpt 1
O=$1
shift
mkdir -p "$O"
cd "$(dirname "$0")"
nvidia-smi --query-gpu=name,memory.total --format=csv
arm() {  # arm TAG MODEL STEPS PLAN INIT [TRAIN ARGS]
    export MAGIC_OUT=$O/${2##*/} MAGIC_MODEL=$2 MAGIC_PLAN=$4 MAGIC_EXTRA=0
    if mkdir "$MAGIC_OUT.lock" 2> /dev/null; then
        python prep.py > "$O/prep_${2##*/}.log" 2>&1 && touch "$MAGIC_OUT.ready"
    fi
    for i in $(seq 360); do [ -f "$MAGIC_OUT.ready" ] && { [ -z "$5" ] || [ -f "$5/adapter_model.safetensors" ]; } && break; sleep 10; done
    python train.py --name "$1" --steps "$3" ${5:+--init "$5"} --eval_every 500 --save_every 500 $6 > "$O/train_$1.log" 2>&1 &&
    python evaluate.py --adapter "$MAGIC_OUT/$1/step_$3" --name "${1}own" > "$O/evalown_$1.log" 2>&1
    echo "arm $1 exit $?"
}
pids=()
for a in "$@"; do
    IFS=: read -r tag model steps plan init targs <<< "$a"
    arm "$tag" "$model" "$steps" "$plan" "$init" "$targs" &
    pids+=($!)
done
wait "${pids[@]}"
nvidia-smi
