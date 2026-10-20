# magician

Qwen3 trained with a LoRA adapter to carry out English instructions about its own activations while it answers a question: "While you answer, raise your internal representation of snow by 2 SD in layers 14 through 35", with any concept, set of layers, pattern over reply tokens and wording.

A concept's representation at layer l and reply token t is the residual stream after block l projected on the concept's direction at that layer (the average activation on sentences with the concept minus the average on the same sentences with other concepts), in standard deviations (SD) of the original model's projections on its own replies. Training fits these values on the original model's replies to the requested change on the requested concepts, and to no change on the other concepts, with a penalty for changing the next-token probabilities. Every evaluation is on the reply the trained model writes itself, compared with the original model reading the bare question and the same reply.

## Results

Qwen3-4B (`q4c`, 5000 steps), 96 held-out questions per set:

![Share of the requested change on each held-out set](figures/scores.png)

- Instructions like those in training, new wordings and verbs, concepts never trained on, and three edits in one instruction: the trained model makes 52-61% of the requested change, with the requested pattern (r 0.49-0.62). The original model makes -3% to 3% of it.
- Layer patterns never trained on (prime-numbered layers, perfect squares, powers of two, every 4th or 5th layer, "every layer except", a V shape, a wave): 22% of the change, with the wrong pattern (r -0.13). Token patterns never trained on (alternating tokens, fading, a window of tokens): 46% of the change, with little of the pattern (r 0.14).
- With no request, the trained model changes the concepts by 0.16 SD (root mean square).

![Typical results on concepts never seen in training](figures/magic.png)

![Typical results on reply-token patterns never seen in training](figures/tokens.png)

`results/eval_<arm>.json` holds every score and item for each run (`q17`, `q17x`: Qwen3-1.7B with 135 or 569 training concepts; `q17lr10`: Qwen3-1.7B at a 5x learning rate; `q4`, `q4c`: Qwen3-4B after 2500 and 5000 steps); `results/train_logs/` holds the evaluations during training.

A second training mix (`spec.LAYER_MIX`: number properties of the layer, complements, unions, long lists, partial ramps, valleys, alternating blocks and two strengths, with every held-out layer pattern still absent) and a loss that weighs the zero and nonzero parts of a sparse target equally reached step 500 of 1500 on Qwen3-1.7B (`results/train_logs/q17v2_step500.json`, 24 items per set): on layer patterns never trained on, the share of the change rose from 16% to 33% and the pattern match from r -0.12 to -0.01.

## Code

| Path | What it is |
|---|---|
| `magician/concepts.py` | Concepts and the sentences their directions come from. |
| `magician/spec.py` | Instructions in English with their exact targets; training and held-out parts of every axis. |
| `magician/prep.py` | Concept directions, standardization and the original model's replies. |
| `magician/train.py` | LoRA training. |
| `magician/measure.py`, `magician/evaluate.py` | Scoring on the trained model's own replies. |
| `magician/run_pod.sh` | Prep, training and evaluation of several runs on one GPU. |
| `figures/make_figures.py` | Every figure, from `results/`. |
