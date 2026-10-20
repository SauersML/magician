# magician

Qwen3 with a LoRA trained so that an instruction in the prompt changes the model's internal representation of a concept, at the layers and reply tokens the instruction names, while the model answers the question.

![An instruction about clouds, the replies and the change in the clouds representation for the original and the trained Qwen3-4B](figures/overview.png)

Asked to raise its representation of clouds in its first half of layers, the trained model raises it on most of those layers through the whole reply and answers the question. No training instruction asked about clouds. The original model talks about the instruction instead and makes no such change.

## How it works

![How a concept direction and an instruction's target are defined](figures/recipe.png)

![One training step](figures/training.png)

The loss is computed on internal representations, not on output text: at every layer and reply token, it compares the change the LoRA makes with the requested change. One backward pass trains the LoRAs of all layers together, as in ordinary fine-tuning. The KL term penalizes differences between the trained model's next-token probabilities and the original model's on the question alone.

## What training saw

![What was in training and what was never seen in training](figures/split.png)

Every test uses questions that training never used. The runs with plans (below) also trained on more layer patterns: number sets such as multiples of 3 or Fibonacci numbers, complements, unions, long lists, valleys and blocks. None of these is a test pattern.

## Results

![Share of the requested change and pattern match for each test set](figures/scores.png)

For wordings, concepts and three-edit instructions that training never used, the trained model makes about half of the requested change in roughly the requested pattern. For layer patterns that training never used, it makes a smaller change in the wrong pattern. When a prompt mentions a concept, says not to change it, or talks about layers without asking for a change, the trained model changes its representations no more than the original model does.

Both scores compare two grids with one number per layer and reply token: the change in the concept that the instruction asked for, and the change the model made. Share of the requested change measures size: 1 is the full requested change, 0.5 is half of the requested change, 0 is no change, and changes on layers or tokens the instruction did not name add nothing. Pattern match r measures shape: the correlation between the two grids, which is 1 when the change is large exactly where the request is large and small where it is small, at any size, and near 0 when the change has no relation to the requested pattern. Each model is measured on the reply it writes itself, against the original model reading the bare question and the same reply.

![Typical instructions from four test sets](figures/examples.png)

Typical instructions from four test sets, each with a pattern match near the median of its set.

![Layer and token patterns never seen in training](figures/outside.png)

For layer and token patterns that no training instruction used, the trained model changes the concept on a solid block of layers and tokens instead of the requested pattern.

## Plans

With plans, the model first writes in its thinking block how much it will change each concept, on which layers and tokens, and then answers. Training adds a next-token loss on the plan. Runs trained also on plans alone saw half of their edit instructions replaced by the bare question followed by the plan, so that the plan was the only place the edit was written.

![Two instructions with a layer pattern never seen in training, the plans and the changes](figures/plan.png)

Qwen3-1.7B trained with plans, also on plans alone. The trained model changes its representations as its plan says, even when the plan is wrong. Given a correct plan written by the original model after thinking, the same trained model follows a layer pattern it never saw.

![How often plans are exactly right and how well the change follows the requested pattern](figures/plans.png)

The original model writes exactly right plans for new layer patterns far more often than the trained models do. With the original model's plans, the 1.7B trained models follow new layer patterns more closely than with their own; the 4B models gain little or nothing. Exact plans give the closest match for every model.

![The same by layer pattern, Qwen3-1.7B](figures/rules.png)

The original model's plans are mostly right for powers of two, prime-numbered layers and perfect squares, where the trained model's plans are almost never right. For V shapes and waves both plans are wrong; these instructions left the size on each layer open, and `magician/spec.py` now words them with a single target.

## Reproduce

```bash
cd magician
export MAGIC_MODEL=Qwen/Qwen3-4B MAGIC_OUT=../runs/Qwen3-4B
python prep.py        # concept directions, standardization, the original model's replies
python train.py --name m1 --steps 2500
python train.py --name m2 --steps 2500 --lam_ramp 1 --init ../runs/Qwen3-4B/m1/step_2500
python evaluate.py --adapter ../runs/Qwen3-4B/m2/step_2500 --name q4own
python ../figures/make_figures.py
```

These commands train `q4c`, the model of the result figures: a second 2500-step run started from the first. `evaluate.py` writes `../runs/Qwen3-4B/q4own/eval.json`; the figures read the copies in `results/`. For plans, set `MAGIC_PLAN=1` for training and evaluation; `train.py --plan_only_p 0.5` also trains on plans alone, and `evaluate.py --plan_think 1024` has the original model write the plans.

## Files

- `results/eval_<run>.json`: every instruction and score per test set. `q4c` is the Qwen3-4B of the result figures. `n17` (Qwen3-1.7B) is trained without plans, `p17` and `p4` (Qwen3-1.7B and Qwen3-4B) with plans, and `r17` and `r4` also on plans alone; `<run>think` is the same run following the original model's plans. `q4`, `q17`, `q17x` and `q17lr10` are earlier runs without plans.
- `results/profiles_q4c.pt`: requested and achieved changes per reply token and layer behind the heatmaps.
- `results/plan_examples_r17.pt`, `results/step_example_q4c.pt`: the data of the plan figure and the training-step figure.
- `results/train_logs/`: evaluations during training.

| Code | What it does |
|---|---|
| `magician/concepts.py` | Concepts and the sentences their directions come from |
| `magician/spec.py` | Instructions in English with their exact targets; what is in training and what is in the tests |
| `magician/prep.py` | Concept directions, standardization, the original model's replies |
| `magician/train.py` | LoRA training |
| `magician/measure.py`, `magician/evaluate.py` | Scores on each model's own replies |
| `magician/planner.py` | Plans written by the original model after thinking |
| `magician/examples.py` | One instruction followed with each plan (plan figure) |
| `magician/step_example.py` | One training step of a trained LoRA (training figure) |
| `magician/run_pod.sh` | Prep, training and evaluation of several runs on one GPU |
| `figures/make_figures.py` | Every figure, from `results/` |
