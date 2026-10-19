# Training

This guide tells you how to train the JUICE models of the paper.
One trainer (`scripts/train/train.py`) does all stages of the JUICE model.
One YAML file in `configs/train/` holds the settings of one run of the paper.
The text head for question answering (section 8) and the linear task head of the planning models
(section 9) have their own trainers.

## 1. The stages

| Stage | Config | Starts from | Steps | Global batch | Paper GPUs |
|---|---|---|---|---|---|
| 1. Pretraining | `pretrain_siglip2.yaml`, `pretrain_dino.yaml` | random initialization | 60,000 | 4,096 | 8 |
| 2. SFT | `sft_siglip2.yaml`, `sft_dino.yaml` | stage 1, step 60,000 | 15,000 | 1,024 | 8 |
| 3. Task fine-tune (JMaze) | `jmaze_siglip2.yaml`, `jmaze_dino.yaml` | stage 2, step 15,000 | 5,000 | 1,024 | 8 |
| 3. Task fine-tune (VSP) | `vsp_siglip2.yaml` | stage 2, step 15,000 | 5,000 | 1,024 | 4 |

The released Base models are the fast EMA weights (`ema_fast`) of stage 1 at step 60,000.
The released SFT models are the fast EMA weights (`ema_fast`) of stage 2 at step 15,000.
These are the weights that we evaluated. We do not release stage-3 models; the paper's planning
results use their raw weights (`model`) at step 5,000.

Each stage starts from the raw weights (`model`) of the stage before it.
We do not release the raw weights of stages 1 and 2.
To start a stage from a released model, give the model folder as `init_from`:

```bash
torchrun --nproc_per_node 8 scripts/train/train.py --config configs/train/sft_siglip2.yaml \
    --set init_from=/path/to/JUICE-2B-SigLIP2-Base
```

The released folders hold EMA weights, not raw weights. Thus the result can be slightly
different from the paper.

## 2. Before you start

1. Install the package with the training extra:

   ```bash
   pip install -e ".[train]"
   ```

2. Prepare the training data. Follow `docs/DATA.md`.
   The configs read the manifests in `data/manifests/`.
3. Make sure that each GPU has 80 GB of memory.
   The paper used NVIDIA H100 and A100 GPUs with 80 GB.

## 3. Start a run

Run the trainer with `torchrun`:

```bash
torchrun --nproc_per_node 8 scripts/train/train.py --config configs/train/pretrain_siglip2.yaml
```

The trainer writes to `outputs/train/<run_name>/`:

- `config.json`: the settings of the run.
- `log.jsonl`: one line every 50 steps (loss, learning rate, gradient norm, routing counts).
- `checkpoints/checkpoint_<step>.pt`: the checkpoints.

### Use a different number of GPUs

The global batch is `micro_batch_size` x `grad_accum_steps` x number of GPUs.
It must be equal to `batch_size`.
If you use a different number of GPUs, change `grad_accum_steps`:

```bash
torchrun --nproc_per_node 4 scripts/train/train.py --config configs/train/sft_siglip2.yaml \
    --set grad_accum_steps=32
```

Do not change `micro_batch_size`.
The data order depends on the number of GPUs and data workers.
Thus a different number of GPUs gives a different (but equivalent) data order.

### Change other settings

`--set` replaces config keys. The values are YAML:

```bash
--set init_from=path/to/checkpoint.pt output_dir=/scratch/juice num_workers=4
```

### Continue a run

`--resume` continues a run from one of its checkpoints.
It loads the weights, the two EMA copies, the optimizer state and the step:

```bash
torchrun --nproc_per_node 8 scripts/train/train.py --config configs/train/pretrain_siglip2.yaml \
    --resume outputs/train/pretrain_siglip2/checkpoints/checkpoint_030000.pt
```

A resumed run reads the data in a new order.
It does not read the same samples again from the start.

## 4. Checkpoints

A checkpoint holds these items:

| Key | Content |
|---|---|
| `model` | the raw weights |
| `ema` | EMA of the weights, decay 0.99995 |
| `ema_fast` | EMA of the weights, decay 0.9996 |
| `optimizer` | the AdamW state |
| `step`, `config`, `architecture` | the step and the settings |

The trainer keeps the last `save_keep` checkpoints.
It keeps the checkpoints at the steps in `ckpt_milestones` always.

To make a model folder for the inference scripts, export one weight set:

```bash
python -m juice.export --checkpoint outputs/train/sft_siglip2/checkpoints/checkpoint_015000.pt \
    --weights ema_fast --out my-sft-siglip2 --text-head outputs/train/text_head/stage_2
```

`--text-head` copies a text-head folder (section 8) into the model folder.

## 5. What one step does

One optimizer step reads `grad_accum_steps` micro-batches of 8 samples on each GPU.
For each micro-batch, the trainer does these steps:

1. It encodes the images with the frozen image encoder (SigLIP2 or WebSSL-DINO).
2. It encodes the text with the frozen Flan-T5-Small encoder.
3. It calculates the loss of `juice/train/loss.py` (paper Sec. 3.4 and App. A.6.1).
4. It calculates the gradients.

Then it does the optimizer step:

- Learning rate: linear warmup from 1e-6 for `warmup_steps`, then constant.
- Gradient clipping at a norm of 1.0.
- AdamW: betas (0.9, 0.95), weight decay 0.01.
- Two EMA copies of the weights (`ema`, `ema_fast`).

The forward pass uses bfloat16 autocast.
Matrix multiplications use TF32.
The model weights, the gradients and the optimizer state are float32.

## 6. The loss in the configs

| Key | Meaning |
|---|---|
| `task_mixture` | weights of the joint, t2i and i2t regimes. The row counts are exact in each micro-batch. |
| `t_shift` | time shift of the image stream: 8.485 for SigLIP2, 8.0 for WebSSL-DINO. |
| `qa_rows_force_i2t` | true: a question-answering row is always i2t (stages 1 and 2). |
| `qa_answer_window` | number of generated answer slots. A sample can set its own window (`qa_window`). |
| `qa_window_tail_weight` | loss weight of the PAD slots inside the answer window. |
| `qa_span_regimes` | regimes in which a question-answering row keeps its clean question prefix. |
| `img_known_inpaint` | true: the known image patches of a task are clean (stage 3). |

The other loss settings are the same in all runs of the paper.
The loss keeps them as fixed values or as defaults of `juice/train/config.py`.

## 7. Reproducibility

We compared this trainer with the research code of the paper:

- The loss, its metrics and all gradients are bit-equal on CPU for the settings of all stages.
- The data loader gives the same samples in the same order, for 2 GPUs with 2 workers each.
- On one A100 GPU, one micro-step on real JMaze data from the SFT weights gives the same encoder states,
  the same loss and the same gradients (all 567 tensors) as the research code.
  A second run of the same code gives slightly different gradients,
  because some GPU kernels are not deterministic.

A full run is not bit-equal to the paper run.
GPU kernels, the number of GPUs and the PyTorch version change the last bits of the results.

## 8. Text head (parallel decoder)

The question-answering results read the generated text states with the parallel text head
(`ParallelTextHead` in `juice/heads.py`, folder `text_head/` of the released models).
The head is a copy of the Flan-T5-Small encoder stack (8 layers) and an LM head.
It reads the 128 text states and gives the 128 tokens in one forward pass.
Every slot sees every slot, so the head does not see the length of the text.

The head is trained alone. It does not use the JUICE model.
It learns to read the Flan-T5-Small states of real texts, also when noise is added to the states.

### Train the head

The trainer uses one GPU:

```bash
python scripts/train/train_text_head.py --config configs/train/text_head.yaml
```

The config has two stages of 8,000 steps.
Stage 1 starts from the pretrained Flan-T5-Small encoder stack.
Stage 2 starts from the head of stage 1.
The released head is the head of stage 2.

The trainer writes to `outputs/train/text_head/`:

- `config.json`: the settings of the run.
- `stage_1/`, `stage_2/`: the head of each stage (`config.json`, `model.safetensors`)
  and `log.jsonl` (loss, token accuracy, exact match and end-of-text error on the training batch, every 50 steps).

On one A100 GPU (80 GB), one stage takes approximately 1.3 hours.
The trainer uses less than 10 GB of GPU memory.

`--init-from` starts stage 1 from a text-head folder.
`--set` replaces config keys, for example `--set stages=1 output_dir=/scratch/juice`.

### Data

The head trains on the texts of the pretraining pool: captions and question-answer rows
(`Question: ... Answer: ...`). The config reads the same manifests as `pretrain_siglip2.yaml`.
Prepare them as in `docs/DATA.md`. The data stream also decodes the images, but the head does not use them.

The data order depends on `num_workers`, `data_seed`, `wds_shard_seed` and the shard layout.
Keep `num_workers: 1`. Another value gives a different (but equivalent) data order.

### Settings

| Key | Value | Meaning |
|---|---|---|
| `stages` | 2 | stage k > 1 starts from stage k - 1 |
| `steps` | 8,000 | steps per stage |
| `batch_size` | 64 | texts per step |
| `learning_rate` | 1e-4 | AdamW, constant, betas (0.9, 0.999), weight decay 0 |
| `corrupt_frac` | 0.5 | the first half of the rows get noise |
| `tail_weight` | 0.2 | loss weight of the PAD slots |
| `seed` | 0 | torch seed (noise and dropout) |

One step does these operations:

1. It puts the texts into the slot contract (content, one EOS, PAD; 128 slots).
2. It encodes the texts with the frozen Flan-T5-Small encoder.
   The head reads these states without the division by 0.2.
3. It adds noise to the first `corrupt_frac` of the rows, slot by slot:
   `state = lam * state + (1 - lam) * noise`, with `lam = sigmoid(N(0.8, 0.8^2))` and noise from `N(0, 0.2^2)`.
4. It calculates the cross entropy at all 128 slots. A content or EOS slot has its token as the label (weight 1).
   A PAD slot has EOS as the label (weight `tail_weight`). The loss is divided by the sum of the weights.
5. It clips the gradient norm at 1.0 and does the AdamW step.

The PAD term teaches the head to give EOS after the end of the text.
Thus the answer reader can stop at the first predicted EOS.

The frozen encoder and the forward pass of the head use bfloat16 autocast.
The weights, the gradients and the optimizer state are float32. TF32 is off.
The T5 dropout (0.1) is on during training.

### Use the head

Put the head folder into a model folder as `text_head/` (or give it to `juice.export --text-head`):

```bash
cp -r outputs/train/text_head/stage_2 my-sft-siglip2/text_head
```

The same head serves the SigLIP2 and the WebSSL-DINO models: it reads only text states.

### Reproducibility

We compared this trainer with the research code of the paper:

- On CPU, the loss, all gradients (before and after clipping), the AdamW state, the weights and
  the torch random state are bit-equal for 3 steps of stage 1 and 3 steps of stage 2, on real pretraining texts.
- The data stream gives the same token ids and masks in the same order.
- On one A100 GPU, the two-stage command above gave the released head again, bit-equal in all 76 tensors
  (75 and 76 minutes for the two stages). Other GPUs or PyTorch versions can change the last bits.
  This run used our copy of the data. In the packs that `docs/DATA.md` makes, the five `_x4` Cambrian
  packs have a different tar layout. The data order is then different, and the head is different
  but equivalent.
- With the released head on 2000 held-out COCO captions (clean T5 states), the token accuracy is 0.9997,
  the exact match is 0.980 (token ids) and 0.984 (text), and the error of the end-of-text slot is 0.

## 9. Task head (planning models)

The planning models (JMaze, VSP) read their answers with a linear task head
(`LinearTextHead` in `juice/heads.py`, folder `task_head/` of a planning folder).
The head is one linear layer (512 to 32,128), applied to each text slot.
It changes a normalized Flan-T5-Small state into token logits.

The head is trained alone. It does not use the JUICE model.
It learns to read the Flan-T5-Small states of the task texts, also when noise is added to the states.

### Train the head

The trainer uses one GPU:

```bash
python -m scripts.train.train_task_head --config configs/train/task_head_jmaze.yaml
python -m scripts.train.train_task_head --config configs/train/task_head_vsp.yaml
```

The trainer writes to `outputs/train/<run_name>/`:

- `config.json`: the settings of the run.
- `log.jsonl`: loss, token accuracy, exact match and end-of-text error on the training batch, every 50 steps.
- `head/`: the head (`config.json`, `model.safetensors`). Give this folder to
  `scripts/export_planning.py --task-head` (`docs/PLANNING.md`, section 2).

On one A100 GPU, one run takes 75 to 120 seconds (after the Flan-T5 model loads).

### Data

The head trains on the texts (`Question: ... Answer: ...`) of the training shards of the task
(`docs/PLANNING.md`, section 3). It does not use the images.
The data order depends on `num_workers`, `data_seed` and `wds_shard_seed`.

### Settings

| Key | Value | Meaning |
|---|---|---|
| `steps` | 1,500 | optimizer steps |
| `batch_size` | 64 | texts per step |
| `learning_rate` | 1e-3 | AdamW, constant, betas (0.9, 0.999), weight decay 0.01 |
| `corrupt_frac` | 0.5 | the first half of the rows get noise |
| `noise_std` | 1.0 | standard deviation of the noise, in the normalized state space |
| `seed` | (see the config) | torch seed of the initialization and of the noise |

One step does these operations:

1. It puts the texts into the slot contract (content, one EOS, PAD; 128 slots).
2. It encodes the texts with the frozen Flan-T5-Small encoder (bfloat16) and divides the states by 0.2.
3. It mixes each slot of the noisy rows with Gaussian noise: `lam * state + (1 - lam) * noise`,
   with `lam = sigmoid(g)` and `g ~ N(0.8, 0.8^2)` for each slot.
4. It computes the cross entropy over the content and EOS slots. PAD slots have no loss.
5. It does one AdamW step on the float32 head.

### Reproducibility

We compared this trainer with the research code of the paper:

- On CPU, the loss, all gradients, the AdamW state, the weights and the torch random state are
  bit-equal for 3 steps on real JMaze texts and on real VSP texts.
- The research runs did not fix the torch seed, so their initialization cannot be repeated.
  A head that this trainer makes on the same data is as far from the paper's head as two heads
  with different seeds are from each other.
- With the paper's JMaze and VSP fine-tunes, a head from this trainer (the configs as given, on the
  shards of `docs/PLANNING.md`) gives the same answer and the same scores for every puzzle as the
  paper's head: JMaze `test_id` with seeds 1 and 2 (500 puzzles each; `joint_consistent` 0.952 and 0.922)
  and VSP levels 3 to 8 with seed 0 (600 maps; mean of levels 3 to 6: 0.9975).
- The paper's JMaze head was trained on an earlier version of the JMaze training texts. The config uses
  the shards of `docs/PLANNING.md`, section 3.1.
