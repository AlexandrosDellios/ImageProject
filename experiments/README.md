# Experiments (archived)

These are the approaches tried before the final pipeline, kept for the record. They are
**not maintained and not tested**, and nothing in `uno_vision/` or `scripts/` depends on
them. The maintained pipeline is described in the main README.

The scripts are archived as they were last written. The only changes are LF line endings,
emojis removed, a header docstring, and, in three files, the model import pointed at
`uno_vision.model`. Their datasets and checkpoints are not in the repo.

## Before reading or running them

- **Old paths.** They use the flat layout of the original project, relative to the
  working directory (run from the repo root):
  - generated datasets in `uno_*` folders and checkpoints (`*.pth`) at the repo root;
  - card cut-outs and table backgrounds in `data/single_cards` and `data/backgrounds`
    (now `assets/cards` and `assets/backgrounds`, see `assets/README.md`);
  - real zone crops in `data/phase4_{train,val}` with `phase4_{train,val}_strict.csv`.
    These are now `data/processed/real_{train,val}` and `data/processed/real_{train,val}.csv`.

  `multistage/train_multistage.py` resolves its paths from its own folder instead.
- **Old head.** Several checkpoints from these runs use an older head without dropout
  (`fc.weight` rather than `fc.1.weight`) and 5 count bins instead of 6: 56 x 5 for the
  curriculum runs, 17 x 5 for the grayscale model. They do not load into
  `uno_vision.model.UnoCountingResNet`, and no compatibility code is provided. The three
  scripts that import `uno_vision.model` now build the current head (dropout 0.1, `fc.1`
  keys). For `curriculum/train_curriculum.py` that is not the head its checkpoints were
  trained with, so a re-run trains a slightly different network.
- **Packages and devices.** The YOLO scripts need `ultralytics`, `albumentations` and
  `opencv-python`. `grayscale/eval_gray.py` needs `seaborn`. The scripts pick CUDA or the
  CPU (never MPS), and the YOLO training call asks for GPU 0.
- **Known breakage.** `curriculum/train_curriculum.py` passes `verbose=True` to
  `ReduceLROnPlateau`, which current torch rejects with a TypeError.

## Overview

| Folder | Idea | Outcome |
|---|---|---|
| `yolo/` | Detect every card with a box (YOLOv8) | Abandoned before any result: the real photos have no box labels |
| `multistage/` | One network, taught in stages: shapes, then colour, then several cards, then real photos | Stages 0-2 dropped; stage 3 lives on as `scripts/finetune.py` |
| `grayscale/` | A colour-blind "shape expert" (stage 0) | F1 of 0.31 at best per shape; Grad-CAM shows it looking at the tablecloth |
| `single_card/` | One card per image (stage 1), and a corner-only variant | Dropped: single cards teach the network to rely on the central symbol |
| `multicard/` | 2-3 cards side by side (stage 2), and a half-synthetic, half-real co-training set | Replaced by the two-card overlap set |
| `curriculum/` | Real crops only, from scratch, small hands first | Dropped: too little real data to train from scratch |

## yolo/

**Tried:** treat the task as object detection: train YOLOv8s from scratch (`yolov8s.yaml`,
no pretrained weights; 640 px, batch 16, up to 300 epochs, early stop after 50).

- `generate_phase1.py` pastes one card cut-out on a 640 x 640 canvas tiled with table
  background, augments it with albumentations (affine, perspective, colour jitter, noise,
  motion blur, small dropout holes) and writes the box in YOLO format to `uno_dataset/`:
  200 images per card, 600 per token, split 80/20 at random.
- `train_phase1.py` is the training call; `uno_phase1.yaml` is its unfinished dataset
  config (placeholder path, `nc: 55`, no class names).
- `slicer.py` cuts the photos into 5 zones (fixed fractions of the frame, no rotation) and
  saves them to `data/to_be_labeled/` so the boxes could be drawn by hand.
  `visualize_slicer.py` draws those zones on one photo. These zones are not the final ones
  (`uno_vision/zones.py`): p2 and p4 are swapped, and nothing is rotated upright.

**Outcome:** no dataset, run or weights were kept. `infere.py`, an unfinished inference
sketch, was removed (it is in git history).

**Why it was dropped:** a detector needs a box around every card, both to train and to
check it on real photos. The course labels only list the cards in each hand, and drawing
boxes on every photo by hand was not feasible. The task was reframed as counting the
cards of each zone with a classifier.

## multistage/

`train_multistage.py` (the original `train.py`) trains one stage per run, chosen with
`STAGE_TO_RUN`. Each stage keeps the weights with the lowest validation loss and saves its
curves to `diagnostics/`.

| Stage | Data | Starts from | Writes |
|---|---|---|---|
| 0 | `uno_step0_grayscale` (17 shapes, see `grayscale/`) | scratch | `best_step0_grayscale.pth` |
| 1 | `uno_classification_dataset` (56 classes, one card per image, see `single_card/`) | scratch | `best_phase1_resnet.pth` |
| 2 | `uno_multicard_step3` (2-3 cards, see `multicard/`) | stage 1 | `best_final.pth` |
| 3 | real crops, `data/phase4_{train,val}` | `phase1_best_weights.pth` | `best_test.pth` |

**Tried:** a curriculum of tasks: learn the shapes without colour, then colour on single
cards, then several cards together, then the real photos.

**Outcome:** stages 0-2 are the experiments described in the sections below. Stage 3 (a
linear probe, then full fine-tuning) is the recipe of the final model and is maintained as
`scripts/finetune.py`. The submitted model was not built from stages 0-2: it was
pre-trained from scratch on the two-card overlap set (that recipe is now
`scripts/pretrain.py`), then fine-tuned on the real crops. Stage 3 is the closest
surviving version of that fine-tuning.

**Notes:**
- In this version, stage 0 builds a 17 x 6 head with dropout. The saved
  `best_step0_grayscale.pth` (17 x 5, no dropout) came from an earlier version of the script.
- Stage 1 uses `StepLR(step_size=5, gamma=0.1)`: the learning rate is divided by 10 every
  5 epochs, so training has practically stopped after about 20 of its 300 epochs.
- The script creates `diagnostics/` next to itself as soon as it is imported.
- The stage 3 warm-up is meant to train only the head, but its filter (`'fc' in name`)
  also unfreezes the CBAM channel-attention layers (`ca.fc1`, `ca.fc2`).
  `scripts/finetune.py` restricts it to `fc.*`.

## grayscale/

**Tried:** a "shape expert" that cannot see colour, as the first stage of the curriculum.
`generate_grayscale.py` turns each card cut-out to grayscale, pastes it on the (coloured)
table backgrounds with heavy augmentation (perspective, blur, downscale-upscale "crunch",
hole punches in 60% of images, glare in 40%, cards partly out of frame, letterbox bars,
noise) and files it under its shape. That gives 17 classes: `0`-`9`, `draw_2`, `reverse`,
`skip`, `wild`, `draw_4`, `circle_token` (yellow token) and `square_token` (black token).

**Dataset:** `uno_step0_grayscale`, 13,921 training and 2,796 validation images, 336 x 336.
The 250 + 50 images per card come out slightly fewer because random file names collide.
The model was trained as stage 0 of `multistage/train_multistage.py`.

**Outcome:** `eval_gray.py` scored the model on the validation split of the co-training
set (`multicard/half_synthetic.py`: half synthetic hands, half real crops), converted to
grayscale. Per-shape F1 was between 0 and 0.31 (best: `reverse`), and 7 of the 17 shapes,
both tokens included, scored 0. Its Grad-CAM maps put the heat on the leaves of the
patterned tablecloth rather than on the cards.

**Why it was dropped:** without colour, the network latched onto background texture. The
report lists full grayscale conversion among the randomisations that went too far.

**Notes:** the checkpoint has the old 17 x 5 head. The Grad-CAM in `eval_gray.py` hooks
layer3, not layer4: `model.backbone` does not exist, so the fallback takes the second-to-last
block of `features`.

## single_card/

**Tried:** learn all 56 classes from one card per image.
`generate_single_cards.py` (the original `generate_data1.py`) pastes one card cut-out on a
background crop. Before placing it, it shifts the card's colours (ColorJitter: brightness,
contrast and saturation 0.3, hue 0.05). It then applies: rotation 0-360 degrees, card
height 0.4-0.65 of the image, perspective, blur, edge feathering, crunch, out-of-frame
placement (30% of images, up to 67% of the card hidden), letterbox bars, brightness and
noise. Hole punches and glare are switched off.

**Dataset:** `uno_classification_dataset`, 56 classes, 250 + 50 images each, 336 x 336
(13,982 / 2,799 after file-name collisions). The model was trained as stage 1 of
`multistage/train_multistage.py`, and stage 2 starts from it.

The first version of this generator (albumentations on a 640 px canvas) is in git
history. So are the training scripts of the first pipeline, removed here
(`train_phase1.py` and `train_finetuning.py`: 224 px input, cross-entropy loss).

`generate_corners.py` (the original `corner_generator`, which had no extension) is a
variant that crops every card to its top-left corner index: it keeps 26% of the width and
18% of the height. Tokens stay whole, at 0.6 scale, and every crop lands fully inside the
image. Its output folder, `uno_classification_corners/`, is not in the original project,
and no checkpoint is known to come from it.

**Why it was dropped:** in a real hand, most cards are partly covered by their neighbour,
so the large central symbol is often hidden and the corner index is what stays visible.
The report gives this as the reason: training on single, unoccluded cards teaches the
model to rely on the central symbol. The corner-only set was one attempt to point the
model at the corner index. The final pipeline instead pre-trains on pairs of overlapping
cards from the first epoch.

## multicard/

**Tried, step 2 of the curriculum:** `generate_phase2_multicard.py` lays 2-3 random cards
in a row. All cards share one scale (0.35-0.5 of the image height) and are tilted up to
15 degrees. Each card sits 0.9 card widths after the previous one (1.1 next to a token), so
the cards barely overlap. Each card gets the same colour shift and augmentations as in
the single-card set.

**Dataset:** `uno_multicard_step3`, 5,000 training images (2,503 with 2 cards, 2,497 with
3) and 1,000 validation images, with `filename,cards` CSVs. The model was trained as
stage 2 of `multistage/train_multistage.py`, starting from the single-card weights.

**Tried, co-training:** `half_synthetic.py` mixes, 50/50 at random:
- synthetic hands of 2-5 cards: 0.75 card widths apart, a hole punch on 60% of the cards,
  milder blur and noise, no colour shift;
- real crops from `data/phase4_{train,val}`, padded to 336 x 336, with brightness and
  noise. They are drawn with replacement, so each real crop appears several times.

**Dataset:** `uno_multicard_cotraining`: 2,480 synthetic and 2,520 real training images,
493 and 507 validation images. No surviving training script reads it. Its validation
split was only used by `grayscale/eval_gray.py`.

**Why it was dropped:** laid side by side with little overlap, these hands do not show
the heavy occlusion of real, fanned hands, which the report names as the main difficulty.
The final pipeline pre-trains on the two-card overlap set (`data/synthetic/overlap`, made
by `scripts/generate_synthetic.py`: 10,000 + 2,000 images of exactly two overlapping
cards). It brings in the real crops only at fine-tuning, starting with a linear probe.

## curriculum/

**Tried:** train on the real crops only, from scratch, starting with small hands.
`build_curriculum.py` splits the real-crop CSVs by the number of entries per crop (the
token counts as one): at most 2, at most 3, at most 4, then all. That gives 442 / 576 /
638 / 648 training rows and 63 / 77 / 81 / 81 validation rows.

`train_curriculum.py` runs the four stages in turn, each starting from the best weights of
the previous one. Settings:
- no starting weights (`STARTING_WEIGHTS = None`), batch 32;
- AdamW (lr 1e-4, weight decay 1e-4), focal loss (gamma 2, label smoothing 0.1);
- learning rate halved after 3 epochs without improvement;
- at most 50 epochs per stage, early stop after 10;
- online flips, a small affine (5 degrees, 5% scale) and colour jitter;
- head: 56 classes x 5 count bins.

It writes `best_phase4_stage{1-4}.pth`.

**Outcome:** the four checkpoints were saved (they are not in this repository), but no
evaluation of them was kept.

**Why it was dropped:** 648 training crops (324 distinct crops plus a rotated, rescaled
copy of each) from 81 photos are far too few to train a ResNet-18 from scratch, as the
report says of the course data. The final pipeline pre-trains on synthetic images and
uses the real crops only to fine-tune.
