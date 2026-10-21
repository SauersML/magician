# magician

Written by Claude Opus 5.5.

Qwen3-4B with a LoRA trained so that an instruction in the prompt changes the model's internal representation of a concept, at the layers and reply tokens the instruction names, while the model answers the question. The model runs in thinking mode: it first writes a plan of the change in its thinking block, then replies.

![An instruction to lower tea on every fifth layer, the replies, and the change in the tea representation for the original and the trained Qwen3-4B](figures/overview.png?v=1020n)

Asked to lower its representation of tea on every fifth layer from layer 3, the trained model writes that plan in its thinking block and lowers tea through the whole reply, most on those layers, while it answers the question. This concept, this wording, and this layer pattern are all held out from training; the training set does include every 3rd, 6th, and 7th layer. This instruction is one of the best of its test set: its pattern match is third highest of 96 (typical instructions are below). The original model, with thinking turned off, talks about the instruction instead and makes no such change.

## How it works

![How a concept direction, its representation, and an instruction's target are defined](figures/recipe.png?v=1020n)

A concept's direction at each layer is the mean residual stream over 40 sentences about the concept minus the same mean averaged over the 135 training concepts. Its representation at a layer and reply token is the projection of the residual stream on that direction, as a z-score: the mean and SD come from the original model's replies to the training questions. Every instruction is rendered from an exact target, a change in SD for each concept, layer, and reply token, and many wordings render the same target.

![One training step, with the instruction and with a prefilled plan only](figures/training.png?v=1020n)

A template renders the target as a plan in the thinking block. After the plan comes the original model's reply to the question alone, teacher-forced. Δz is the change the LoRA makes, LoRA on minus LoRA off on the same tokens. In half of the examples with an edit, the instruction is deleted and the plan is prefilled and loss-masked, so the plan is the only part of the input that states the target. No code parses the plan: the model reads it as tokens. The heatmaps show the trained LoRA on this training example.

The loss on each example (right) is the sum of four terms: the MSE between Δz and the target over every layer and reply token, with the cells inside and outside the target weighted equally; the MSE for every other training concept, so that the edit does not spread to concepts the instruction did not name; because concept directions overlap, raising volcanoes also raises earthquakes a little, so each other concept's target is the change that the smallest residual-stream write giving the requested change would cause in it (shown for the concepts whose target rises most, changes least, and falls most); the cross-entropy on the plan tokens when the plan follows the instruction; and, with weight 3, the KL divergence from the original model's next-token probabilities on the question alone to the trained model's on the training example, over the reply tokens, the original model's 64 most likely tokens, and one bucket for the rest (shown at one reply token). The KL weight rises from 0 to 3 over the first 200 steps of each run. Each step averages the loss over 16 examples, and one backward pass updates the LoRA on all 36 layers.

## Training and test sets

![The training set and the held-out test sets](figures/split.png?v=1020n)

Every test set uses held-out questions. The training set includes every 3rd, 6th, and 7th layer but not every 4th or 5th, and every 3rd or 4th reply token but not every other token. Its windows of reply tokens never match a test token span exactly and are worded differently.

## Results

![Share of the requested change and pattern match for each test set](figures/scores.png?v=1020ci)

On held-out wordings, concepts, and three-edit instructions, the trained model makes most of the requested change, in roughly the requested pattern. On held-out layer and token patterns, it makes less of the change, in a much looser pattern. The original model makes none of it.

At test time each model generates its reply with greedy decoding, 64 reply tokens, and the trained model generates its own plan first. Both scores compare the achieved and the requested change in every cell (reply token, layer, requested concept), in SD. Achieved change: the model's z reading the instruction, the question, its thinking block, and its reply, minus the original model's z reading the bare question followed by the same reply tokens. Share of the requested change: the least-squares slope of achieved on requested through the origin (1 is the full requested change). Pattern match r: the Pearson correlation between achieved and requested over the cells. Each value is the median over the 96 instructions of a test set, with the exact 95% interval for the median from order statistics (`magician/intervals.py`). The interval assumes only that the instructions are scored independently, not that they share one distribution, and it is about instructions drawn by the test generator from these test pools of questions, concepts, wordings, and patterns. In simulations from the observed scores (`magician/interval_coverage.py`, `results/interval_coverage.json`), it contains the median in 96% to 98% of samples, against 91% to 95% for a bootstrap interval of the mean.

The trained model's greedy replies to 72 test instructions never mention layers, activations, representations, residual streams, or neurons; the original model's replies do in 31. On control prompts, which mention a concept, say not to change it, or talk about layers without asking for a change, the trained model changes its representations slightly more than the original model does (median RMS 0.22 SD against 0.13 SD; median difference on the same prompts 0.11 SD, 95% interval 0.07 to 0.14).

![Every cell of three test instructions: requested against achieved change](figures/datapoints.png?v=1020n)

Every cell of three test instructions, each the instruction closest to the median pattern match of its test set. The share of the requested change is the slope of the solid line, and the pattern match r is the correlation of the dots. For pizza on layers numbered by powers of two, the cells with no requested change also rise, which lowers the pattern match.

![Instructions near the median of four test sets](figures/examples.png?v=1020n)

Instructions from four test sets, each between the 35th and 65th percentile of its set's pattern match, chosen for a vivid concept and a short plan, with the plan the trained model wrote in its thinking block.

![Held-out layer and token patterns](figures/outside.png?v=1020n)

On held-out layer and token patterns, the trained model changes its representations as its own plan says, even where the plan is wrong. Red outlines mark where a plan differs from the instruction: for alternating tokens it plans every reply token, for fading it plans a rise, for powers of two it adds layer 24, and for prime layers it adds layer 37, which the model does not have.

## Activation Controllability Benchmark

![The benchmark's prompt to think about volcanoes only at layer 28; for the original Qwen3-4B and the magician, how strongly the concept is present at every layer when the prompt asks for each of 8 layers; and every benchmark score for both models next to the paper's 25 models](figures/acb.png?v=1020d)

The Activation Controllability Benchmark ([Kowalski et al. 2026](https://arxiv.org/abs/2608.21664), [code](https://github.com/mkobalski/activation-control)) asks a model to copy a sentence exactly while it thinks about a concept in a stated way. It measures the concept at every layer as the projection of the residual stream on the sentence's tokens onto a concept vector: the model's own last-token activation on "Tell me about {concept}" minus its mean over 100 unrelated words. Most measures compare this with the same sentence copied with no instruction, in SD, and the overall score combines the six measures into one number from 0 to 1. Only answers that copy the sentence (similarity ≥ 0.85, the benchmark's compliance check) are scored. We ran the benchmark with its own prompts, concept vectors, and scoring code; for Qwen3.5-4B our run gives an overall score of 0.399, against the paper's 0.400. Thinking is off in every run here.

Layer Targeting asks the model to think about the concept only at the layer the prompt asks for. The magician scores 2.02, the original Qwen3-4B 0.006, and the best of the paper's 25 models 0.054 (Qwen3.5-397B-A17B). The magician is trained to think before it replies, and with thinking off it copies the sentence in only 48% of its answers (49% on the Layer Targeting prompts); the benchmark scores only those answers, so the magician's scores come from them alone. In the figure, each ridge is one prompt, asking for one of 8 layers from 23 to 35 (the dot), and its height at each layer is how much the concept rises over the no-instruction copy, averaged over the 10 concepts and 50 sentences. The original draws the same ridge whatever layer is asked for; the magician's ridge rises at the asked-for layer for all 8, and is highest there for 5 of the 8 (asked for 23, 28 or 30, the concept is still strongest at layer 34). Below the ridges, each row is one of the benchmark's measures, with the two models as dots and the paper's 25 models as grey ticks.

The cleaner comparison is the magician without plans (stage 1 below), which is not in the figure: it copies the sentence in 98% of its answers and scores 1.16, higher than the original for all 10 concepts and all 49 sentences (exact sign-flip test over concepts, p = 0.002), but its targeting is coarse: it tells early asked-for layers from late ones, while the concept is strongest at layers 34 and 35 whichever layer is asked for.

The gain has a cost: the magician's overall score is 0.32 against the original's 0.42, because Engage (the concept rises after "think about X") falls from 10.9 to 5.7, Temporal Control from 3.3 to 0.8, and Coverage from 2.6 to 1.0. Coverage asks whether the concept rises on every kind of word in the sentence (nouns, verbs, adjectives, punctuation, and five more classes), not only on some: it is the smallest Engage over the nine word classes.

The benchmark's own 95% intervals, a two-way bootstrap over sentences and concepts, are too narrow: with each run's own sentences and concepts as the population, they contain the true value in as few as 45% of simulated runs (the magician's overall score; `magician/acb_coverage.py`, `results/acb/COVERAGE_*.json`).

Every training reply was the original model's first 64 reply tokens with no end token. With thinking off, the magician goes on past the sentence in about half of its answers, sometimes writing out a plan; the 1.7B magicians always do, so the benchmark scores none of their answers.

## How the trained model was trained

The LoRA (rank 32, on every attention and MLP matrix) was trained in three stages, each starting from the last, with 16 examples per step and learning rate 2e-4:

1. 5000 steps without plans or thinking, in two runs of 2500.
2. 1000 steps with plans in the thinking block, on more layer patterns (number sets such as multiples of 3 or Fibonacci numbers, complements, unions, long lists, every 3rd, 6th, or 7th layer, valleys, and blocks; none of them a test pattern) and with the loss on cells with and without a requested change weighted equally. This is step 1000 of a 2000-step run.
3. 2000 steps as in stage 2, with half of the examples with an edit in the prefilled-plan-only form.

```bash
git checkout 49d4af7    # stage 1, with the code that trained it
cd magician
export MAGIC_MODEL=Qwen/Qwen3-4B MAGIC_OUT=../runs/Qwen3-4B
python prep.py          # concept directions, z-scores, the original model's replies
python train.py --name m1 --steps 2500
python train.py --name m2 --steps 2500 --lam_ramp 1 --init ../runs/Qwen3-4B/m1/step_2500
git checkout main
export MAGIC_PLAN=1     # plans in the thinking block
python train.py --name p4 --steps 2000 --init ../runs/Qwen3-4B/m2/step_2500
python train.py --name r4 --steps 2000 --plan_only_p 0.5 --init ../runs/Qwen3-4B/p4/step_1000
python evaluate.py --adapter ../runs/Qwen3-4B/r4/step_2000 --name r4own
python ../figures/make_figures.py
```

`evaluate.py` writes `../runs/Qwen3-4B/r4own/eval.json` and `profiles.pt`; the figures read the copies in `results/`. Since that evaluation, the V-shape and wave test instructions state the size on every layer, so a new evaluation words those two patterns differently.

## Files

- `results/eval_r4.json`: every test instruction with its plan, replies, and scores, per test set.
- `results/profiles_r4.pt`: requested and achieved change per reply token and layer, behind the heatmaps.
- `results/step_example_r4.pt`: the training-step figure. `results/projection_example.pt`: z of clouds on the original model's reply tokens (recipe figure).
- `results/train_logs/`: evaluations during training: `q4c.json` (stage 1, second run), `p4.json` (stage 2), `r4.json` (stage 3).

| Code | What it does |
|---|---|
| `magician/concepts.py` | Concepts and the sentences their directions come from |
| `magician/spec.py` | Instructions in English with their exact targets and plans; what is in training and what is in the tests |
| `magician/prep.py` | Concept directions, z-scores, the original model's replies |
| `magician/train.py` | LoRA training |
| `magician/measure.py`, `magician/evaluate.py` | Scores on each model's own replies |
| `magician/step_example.py` | One training example in both forms (training figure) |
| `magician/projection_example.py` | One concept's z on the original model's reply tokens (recipe figure) |
| `magician/run_pod.sh` | Prep, training, and evaluation of several runs on one GPU |
| `figures/make_figures.py` | Every figure, from `results/` |
