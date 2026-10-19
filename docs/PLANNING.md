# Visual planning: JMaze and VSP

This document tells you how to use the two planning tasks of the paper.
In each task, the model gets an image and a question.
The model makes the answer text and the solved image together, in one sampling run.

## 1. The tasks

### 1.1 JMaze

- The input image shows a maze. Walls are black. Corridors are white.
  The start cell is green. The goal cell is red.
- The question gives the maze size and the start and goal cells, for example
  `Maze 5x5. Start (3,0). Goal (1,2).` Row and column numbers start at 0.
  The walls are only in the image.
- The text answer is a list of moves, for example `U U U R R D`.
- The image answer is the same maze with the route in blue.
- Each maze has loops. Thus, a puzzle has 2 to 16 shortest routes.
  The text and the image must show the same route.
- One maze block is one encoder patch (14 px). Thus, the background, wall, start and
  goal patches stay equal to the input image. The model makes only the corridor patches.

Metric: `joint_consistent` (joint success, J, in the paper). A puzzle is correct when all of these conditions are true:

1. The text is a shortest route from the start to the goal.
2. The blue blocks in the image are a shortest route from the start to the goal.
3. The text route and the image route are the same route.

### 1.2 VSP maze navigation

- The input image shows a FrozenLake map from the VSP benchmark
  (Wu et al., <https://github.com/UCSB-NLP-Chang/Visual-Spatial-Planning>).
  The map has a start, a goal, frozen cells and holes.
- The question is `FrozenLake {n}x{n}. Move the player to the goal without falling into a hole. Give the action plan (L, R, U, D).`
- The text answer is an action plan, for example `R,R,D`.
- The image answer is the same map with a red line on the route.
  The model makes only the frozen-cell patches.

Metric: the official VSP rule. The plan is correct when the player gets to the goal
before the player falls into a hole. We use the 100 official maps of each level.
Levels 3 to 6 are the benchmark. Levels 7 and 8 have larger maps than all training maps.

## 2. Make a planning model

We do not release the JMaze and VSP fine-tunes. To make one, do these steps:

1. Make the data (section 3).
2. Fine-tune an SFT model for 5,000 steps (`docs/TRAIN.md`, stage 3):

   ```bash
   torchrun --nproc_per_node 8 scripts/train/train.py --config configs/train/jmaze_siglip2.yaml \
       --set init_from=/path/to/JUICE-2B-SigLIP2
   ```

   The configs are `jmaze_siglip2.yaml`, `jmaze_dino.yaml` and `vsp_siglip2.yaml`.
3. Train the linear task head on the texts (`Question: ... Answer: ...`) of the training shards
   (one GPU, a few minutes; `docs/TRAIN.md`, section 9):

   ```bash
   python -m scripts.train.train_task_head --config configs/train/task_head_jmaze.yaml
   ```

   For VSP, use `configs/train/task_head_vsp.yaml` (output `outputs/train/task_head_vsp/head`).
   The head reads only text states, so one head serves the SigLIP2 and the WebSSL-DINO models.
4. Export the release folder:

   ```bash
   python -m scripts.export_planning --checkpoint outputs/train/jmaze_siglip2/checkpoints/checkpoint_005000.pt \
       --task-head outputs/train/task_head_jmaze/head --preset jmaze --out my-jmaze-siglip2
   ```

The folder contains:

- `model.safetensors` and `config.json`: the fine-tuned model (the raw weights, not EMA).
- `task_head/`: the linear head that reads the answer tokens from the text states.

## 3. Make the data

### 3.1 JMaze

Run this command. It makes the three puzzle splits and the training shards.

```bash
python -m scripts.data.make_jmaze_data --out data/jmaze --workers 8 --shards train
```

The output is:

- `train.jsonl`: 200,000 puzzles, maze sizes 5, 6 and 7.
- `test_id.jsonl`: 500 puzzles, maze sizes 5, 6 and 7. This is the evaluation split.
- `test_ood.jsonl`: 468 puzzles, maze sizes 3 and 4.
- `shards/train/jmaze_train_*.tar`: 20 WebDataset shards with 10,000 samples each.

The test splits do not contain a (maze, start, goal) of the training split.
The generator is deterministic. The same seed gives the same files on all computers.

### 3.2 VSP

Get these items first:

1. Clone the VSP repository. You need the folder `maze/task-main/maps`.
   The images of these maps also give the cell sprites. We do not supply the sprites.
2. Optional: download the train file of the ThinkMorph set
   (<https://huggingface.co/datasets/ThinkMorph/Spatial_Navigation>).

Then run this command:

```bash
python -m scripts.data.make_vsp_data --vsp-maps VSP/maze/task-main/maps \
    --thinkmorph train-00000-of-00001.parquet --out data/vsp --shards train
```

The output is:

- `train.jsonl`: 50,000 random maps (sizes 3 to 6) and the ThinkMorph maps.
  The generator removes the 61 ThinkMorph maps that are equal to a VSP map.
- `test_id.jsonl`: 500 random maps that are not in the training data.
- `shards/train/vsp_train_*.tar`: WebDataset shards with 2,000 samples each.

No training map is equal to a VSP benchmark map.

### 3.3 Sample format of the shards

Each training sample has four members:

| Member | Content |
|---|---|
| `<key>.png` | The solved image (the training target). |
| `<key>.input.png` | The input image (no route). |
| `<key>.txt` | `Question: <question> Answer: <answer>` |
| `<key>.json` | The answer span and the known patches, and the puzzle data. |

The JSON member contains these fields:

- `answer_tok_start`, `answer_tok_end`: the answer tokens in the caption
  (Flan-T5 content tokens, no EOS).
- `n_content_tokens`: the number of caption tokens.
- `img_known_patches`: 256 values, row by row over the 16 x 16 patch grid.
  The value 1 shows a patch that stays equal to the input image.
- The puzzle: `n`, `blocks`, `start`, `goal`, `moves` (JMaze), or `grid`, `moves` (VSP).

## 4. Run the evaluation

### 4.1 JMaze

```bash
python -m scripts.eval.run_jmaze --model my-jmaze-siglip2 \
    --puzzles data/jmaze/test_id.jsonl --seeds 1 2 --out outputs/jmaze_siglip2
```

### 4.2 VSP

```bash
python -m scripts.eval.run_vsp --model my-vsp-siglip2 \
    --vsp-maps VSP/maze/task-main/maps --seeds 0 --out outputs/vsp_siglip2
```

### 4.3 Settings

The scripts use the settings in `config.json` of the folder:

- 16 Euler steps. No guidance (`cfg_scale` 1.0).
- An answer window of 64 text slots.
- Batches of 25 puzzles. The batch that starts at puzzle `k` uses the noise seed
  `seed + 1000003 * k`.
- TF32 matrix multiplication is off.

If you change the batch size, the noise changes. Thus, the results can change.

The output folder contains `seed<k>.json` (one row for each puzzle) and `metrics.json`.

### 4.4 One puzzle

To solve one puzzle and see the image, use this command:

```bash
python -m scripts.inference.plan --model my-jmaze-siglip2 \
    --puzzles data/jmaze/test_id.jsonl --index 0 --out maze0.png
```

## 5. Expected results

These are the results of the paper's fine-tunes (we do not release them). With the paper's
fine-tunes, the release code gives the same values on an NVIDIA A100 GPU (see section 6). Your own
fine-tune can give slightly different values: a released SFT model holds EMA weights, and the paper's
fine-tunes started from the raw weights (`docs/MODEL_ZOO.md`, section 4).

JMaze, `test_id` (500 puzzles), `joint_consistent` at 16 steps (paper Table 7, coupled sampler, 16 calls):

| Model | Seed 1 | Seed 2 | Mean |
|---|---|---|---|
| SigLIP2 | 0.952 | 0.922 | **0.937** |
| WebSSL-DINO | 0.822 | 0.812 | **0.817** |

VSP, official success rule, 16 steps, seed 0 (paper Table 8):

| Model | Level 3 | Level 4 | Level 5 | Level 6 | Mean 3-6 | Level 7 | Level 8 |
|---|---|---|---|---|---|---|---|
| SigLIP2 | 1.00 | 1.00 | 1.00 | 0.99 | **0.9975** | 0.94 | 0.69 |

## 6. Reproduction check

We compared the release code with the research code:

- The JMaze generator gives the same records for all three splits.
  The training shards are equal byte for byte.
- The VSP data (records, images and shards) is equal to the research data.
- With the paper's fine-tunes, the evaluation gives the same answer and the same scores for each
  puzzle as the paper's runs.

The checks need the research code and the paper's outputs, which are not in this repository.

## 7. Third-party material

- VSP maps and images: MIT License, Copyright (c) 2025 UCSB ML&NLP Group.
  The VSP renders use the gym FrozenLake art (elf and stool from
  franuka.itch.io, other tiles by Mel Tillery). We do not supply this art.
  The scripts cut it from your copy of the VSP repository.
- ThinkMorph Spatial_Navigation: the dataset card does not give a license.
  We do not supply this data. Read the terms of the dataset before you use it.
