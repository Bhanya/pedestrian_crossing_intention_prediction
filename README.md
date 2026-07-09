# EfficientPIE With Bounding-Box Trajectories on JAAD

This repository adapts the lightweight convolutional architecture from
[EfficientPIE](https://github.com/heinideyibadiaole/EfficientPIE) for binary
pedestrian crossing-intention prediction on the
[JAAD dataset](https://github.com/ykotseruba/JAAD). It compares:

1. An image-only EfficientPIE baseline.
2. The baseline with final-frame bounding-box context.
3. The baseline with a GRU encoding a 15-frame bounding-box trajectory.

The trajectory extension was inspired by
[GATEPose](https://openaccess.thecvf.com/content/WACV2026W/LLVM-AD/papers/AlShami_GATEPose_A_Graph_Attention_Transformer_Enhanced_with_Pose_and_Orientation_WACVW_2026_paper.pdf),
which demonstrates the value of temporal bounding-box motion for pedestrian
intention prediction. This repository implements only the lightweight
bounding-box trajectory idea, not the complete GATEPose architecture.

## Repository Structure

```text
models/                         model architectures
utils/                          JAAD loading, cropping, training, and evaluation
tools/                          selective frame extraction and threshold tuning
logs/                           recorded experiment and test metrics
weights/                        trained JAAD checkpoints
pre_train_weights/              ImageNet-pretrained starting checkpoints
train_EfficientPIE_JAAD.py      JAAD training entry point
test_EfficientPIE_JAAD.py       JAAD evaluation entry point
```

## Training And Evaluation

Train the final image-plus-trajectory model:

```bash
python3 train_EfficientPIE_JAAD.py \
  --data-path ../JAAD \
  --device cpu \
  --epochs 12 \
  --batch_size 4 \
  --use-bbox-trajectory \
  --progress-every 50 \
  --metrics-path ./logs/jaad_bbox_traj_best_f1_metrics.csv
```

Evaluate the checkpoint selected by validation F1 on the held-out test split:

```bash
python3 test_EfficientPIE_JAAD.py \
  --data-path ../JAAD \
  --device cpu \
  --batch_size 4 \
  --weights ./weights/transfer_best_f1_model_JAAD_bbox_traj.pth \
  --split test \
  --use-bbox-trajectory
```

## Goal

Use the EfficientPIE lightweight convolutional architecture as a baseline for pedestrian crossing intention prediction on JAAD, then test whether adding bounding-box context features improves performance on the imbalanced JAAD validation set.

## Dataset And Setup

- Dataset: JAAD
- Task: binary pedestrian crossing intention prediction
- Label mapping:
  - `0`: not crossing
  - `1`: crossing
- Observation window: 15 frames
- Model input frame: last selected frame from each 15-frame observation window
- Image input: pedestrian/context crop resized to `300x300`
- Pretraining: ImageNet pretrained EfficientPIE weights
- Device used: CPU
- Training loss: class-weighted cross entropy
- Reason for class weighting: JAAD is imbalanced toward not-crossing samples

## JAAD Class Counts

Counts from current generated JAAD split:

| Split | Total  | Not Crossing `0` | Crossing `1` | Crossing % |
|---|---:|---:|---:|---:|
| Train | 36,343 | 28,762           | 7,581        | 20.9%      |
| Val   | 1,827  | 1,464            | 363          | 19.9%      |
| Test  | 1,876  | 1,584            | 292          | 15.6%      |
| All   | 40,046 | 31,810           | 8,236        | 20.6%      |

Class weights used for training:

| Class            | Weight |
|---|---:|
| Not crossing `0` | 0.6318 |
| Crossing `1`     | 2.3970 |

## Baseline Model

Baseline model:

- EfficientPIE lightweight CNN
- Cropped pedestrian/context image only
- Class-weighted loss
- No bbox (bounding box) numeric features

Best saved checkpoint:

```text
weights/transfer_best_model_JAAD.pth
```

Validation result from saved baseline checkpoint:

| Metric    | Value |
|---|---:|
| Loss      | 0.404 |
| Accuracy  | 0.825 |
| Precision | 0.568 |
| Recall    | 0.483 |
| F1 score  | 0.522 |
| AUC       | 0.794 |

Interpretation:

- Accuracy looks acceptable, but JAAD is imbalanced.
- Recall is more important because it measures how many crossing pedestrians are correctly detected.
- Baseline recall was only `0.483`, meaning it detected about 48.3% of crossing cases.

## Version 1: Bounding-Box Context Features

Added four normalized bbox features:

```text
bbox_center_x
bbox_center_y
bbox_width
bbox_height
```

Feature normalization:

```text
bbox_center_x = bbox center x / original image width
bbox_center_y = bbox center y / original image height
bbox_width    = bbox width / original image width
bbox_height   = bbox height / original image height
```

Model change:

- CNN extracts 1280-dimensional image feature.
- The 4 bbox features are concatenated with the image feature.
- Classifier receives `1280 + 4 = 1284` input features.

New model file:

```text
models/EfficientPIE_bbox.py
```

Best saved checkpoint:

```text
weights/transfer_best_model_JAAD_bbox.pth
```

Metrics log:

```text
logs/jaad_bbox_5epoch_metrics.csv
```

## Novelty Training Results

Validation metrics from `logs/jaad_bbox_5epoch_metrics.csv`:

| Epoch | Val Loss | Val Acc | Val Precision | Val Recall | Val F1 |
|---:|---:|---:|---:|---:|---:|
| 0     | 0.4291   | 0.8205  | 0.5651        | 0.4187     | 0.4810 |
| 1     | 0.4353   | 0.8073  | 0.5200        | 0.3939     | 0.4483 |
| 2     | 0.4185   | 0.8221  | 0.5720        | 0.4160     | 0.4817 |
| 3     | 0.4072   | 0.8161  | 0.5389        | 0.5152     | 0.5268 |
| 4     | 0.4083   | 0.8243  | 0.5515        | 0.6198     | 0.5837 |

Final epoch novelty result:

| Metric    | Value  |
|---|---:|
| Loss      | 0.4083 |
| Accuracy  | 0.8243 |
| Precision | 0.5515 |
| Recall    | 0.6198 |
| F1 score  | 0.5837 |

## Baseline Vs Novelty

| Model | Val Acc | Val Precision | Val Recall | Val F1 |
|---|---:|---:|---:|---:|
| Baseline EfficientPIE                               | 0.825 | 0.568 | 0.483 | 0.522 |
| EfficientPIE + last-frame bbox features             | 0.824 | 0.551 | 0.620 | 0.584 |
| EfficientPIE + bbox trajectory GRU, final model     | 0.881 | 0.697 | 0.711 | 0.704 |

Main result:

```text
Version 1 improved recall from 0.483 to 0.620.
Version 2 improved F1 from 0.522 to 0.704 and accuracy from 0.825 to 0.881.
```

Interpretation:

- The bbox-context model detects more crossing pedestrians.
- This is useful because JAAD is imbalanced and accuracy alone can hide poor crossing detection.
- Version 2 gives the highest validation accuracy, recall, and F1 score of the three models.


## Version 2: BBox Trajectory GRU

Version 2 extends Version 1 by using the pedestrian's complete 15-frame
bounding-box trajectory instead of only the bounding box from the final frame.
This was inspired by GATEPose's use of temporal bounding-box dynamics for
pedestrian intention prediction.

### Input

For every 15-frame observation sequence, the model receives:

1. One `300 x 300` pedestrian image crop taken from the final frame.
2. The pedestrian's bounding box from all 15 frames.

Each bounding box is converted into four values:

```text
normalized center x
normalized center y
normalized width
normalized height
```

The values are divided by the full image width or height. This makes them
independent of the original image resolution. The result is a `15 x 4`
trajectory containing 60 numerical values.

Only one image is loaded for each sample. Version 2 does not load 15 cropped
images; the additional temporal information comes from the annotation
coordinates.

### Architecture

```text
final-frame pedestrian crop
    -> EfficientPIE CNN
    -> 1280 image features

15-frame normalized bbox trajectory
    -> GRU with 64 hidden units
    -> projection layer
    -> 128 trajectory features

1280 image features + 128 trajectory features
    -> 1408 combined features
    -> classifier
    -> not crossing / crossing
```

The GRU reads the bounding boxes in chronological order and summarizes how the
pedestrian's position and size change over time. This can represent simple
motion information that is missing from a single cropped image.

Implementation:

```text
models/EfficientPIE_bbox_trajectory.py
```

### Training Setup

- Image model initialization: ImageNet-pretrained EfficientPIE weights
- Observation length: 15 frames
- Image input: final frame of each observation sequence
- Loss: class-weighted cross entropy
- Optimizer: RMSprop
- Learning-rate schedule: cosine annealing
- Epochs in final run: 12
- Batch size: 4
- Model-selection metric: validation F1 score
- Classification threshold: `0.50`

The final run is recorded in:

```text
logs/jaad_bbox_traj_best_f1_metrics.csv
```

The training script saved the checkpoint whenever validation F1 improved. The
selected checkpoint came from epoch 7:

```text
weights/transfer_best_f1_model_JAAD_bbox_traj.pth
```

### Final Validation Result

| Metric | Epoch 7 value |
|---|---:|
| Validation loss | 0.3123 |
| Accuracy | 0.8812 |
| Precision | 0.6973 |
| Recall | 0.7107 |
| F1 score | 0.7040 |

Epoch 10 produced the highest validation accuracy (`0.8840`) and lowest
validation loss (`0.3122`), but its F1 score was lower (`0.6928`). Epoch 7 was
therefore selected because F1 gives a better balance between precision and
recall for the imbalanced JAAD dataset.

### Final Test Result

After selecting the model using validation F1, the epoch 7 checkpoint was
evaluated once on the held-out JAAD test split containing 1,876 samples.

| Metric | Test value |
|---|---:|
| Loss | 0.3361 |
| Accuracy | 0.8619 |
| Precision | 0.5455 |
| Recall | 0.6781 |
| F1 score | 0.6046 |
| AUC | 0.8741 |

The exact test output is stored in:

```text
logs/final_test_best_f1_bbox_traj.txt
```

The lower test F1 compared with validation F1 shows that the model generalized
less strongly to unseen test videos than the validation result suggested.
However, its test AUC remained high, indicating that the model generally ranks
crossing samples above non-crossing samples well.

### Comparison With EfficientPIE

| Model | Split | Accuracy | AUC | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| EfficientPIE paper | JAAD result reported by paper | 0.8900 | 0.8600 | 0.6300 | Not reported | 0.6200 |
| Our Version 2 | Held-out JAAD test split | 0.8619 | 0.8741 | 0.5455 | 0.6781 | 0.6046 |

Our model achieved a slightly higher AUC, while EfficientPIE reported higher
accuracy, precision, and F1. The comparison should be interpreted carefully
because our implementation did not reproduce EfficientPIE's full
intention-domain incremental-learning procedure.

### Development Experiments

Before the final 12-epoch run, Version 2 was tested for 5 and 10 epochs:

| Run | Best validation F1 | Notes |
|---|---:|---|
| 5 epochs | 0.6122 | Initial trajectory experiment |
| 10 epochs | 0.6754 | Best F1 occurred at epoch 9 |
| Final 12 epochs | 0.7040 | Saved explicitly by validation F1 |

The earlier logs are retained for experiment history:

```text
logs/jaad_bbox_traj_5epoch_metrics.csv
logs/jaad_bbox_traj_10epoch_metrics.csv
```

Threshold tuning was also explored on an earlier checkpoint. Lowering the
validation threshold from `0.50` to `0.35` increased recall and changed F1 from
`0.6122` to `0.6324`. This was only a development experiment; the final model
and final test result above use the standard `0.50` threshold. The tuning
results are stored in:

```text
logs/jaad_bbox_traj_threshold_tuning.csv
```

## Summary

We use the lightweight ImageNet-pretrained EfficientPIE CNN as an
image-only baseline for pedestrian crossing-intention prediction on JAAD. The
first extension adds the pedestrian's final-frame bounding-box position and
size. The second extension processes the full 15-frame normalized bounding-box
trajectory with a GRU and combines its 128 trajectory features with 1,280 CNN
image features. The final model was selected using validation F1, where it
achieved `0.7040` F1 and `0.8812` accuracy. On the held-out test split, it
achieved `0.6046` F1, `0.8619` accuracy, and `0.8741` AUC. These results show
that bounding-box trajectories provide useful temporal context without
requiring all 15 images to be loaded by the model.

## Data And Reproducibility

- JAAD train, validation, and test membership follows the official split files.
- Model selection uses the validation split only.
- The held-out test split is used for final evaluation.
- The reported final test uses the standard `0.50` decision threshold.
- JAAD videos and extracted frames are intentionally excluded from this
  repository.
- Epoch-level metrics and the final test console output are retained in
  `logs/`.

## Acknowledgements And Citations

This work builds on the EfficientPIE architecture and uses the official JAAD
annotations and data interface. The earlier
[PIE dataset repository](https://github.com/aras62/PIE) informed the original
project setup. Please cite the relevant works when using this repository.

### EfficientPIE

```bibtex
@inproceedings{qu2025efficientpie,
  title={EfficientPIE: Real-Time Prediction on Pedestrian Crossing Intention with Sole Observation},
  author={Qu, Fang and Zhou, Pengzhan and He, Yuepeng and Gao, Kaixin and Luo, Youyu and Feng, Xin and Liu, Yu and Guo, Songtao},
  booktitle={Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence},
  pages={1793--1801},
  year={2025},
  doi={10.24963/ijcai.2025/200}
}
```

### JAAD

```bibtex
@inproceedings{rasouli2017they,
  title={Are They Going to Cross? A Benchmark Dataset and Baseline for Pedestrian Crosswalk Behavior},
  author={Rasouli, Amir and Kotseruba, Iuliia and Tsotsos, John K.},
  booktitle={IEEE International Conference on Computer Vision Workshops},
  pages={206--213},
  year={2017}
}

@inproceedings{rasouli2018role,
  title={It's Not All About Size: On the Role of Data Properties in Pedestrian Detection},
  author={Rasouli, Amir and Kotseruba, Iuliia and Tsotsos, John K.},
  booktitle={European Conference on Computer Vision Workshops},
  year={2018}
}
```

### PIE

```bibtex
@inproceedings{rasouli2019pie,
  title={PIE: A Large-Scale Dataset and Models for Pedestrian Intention Estimation and Trajectory Prediction},
  author={Rasouli, Amir and Kotseruba, Iuliia and Kunic, Toni and Tsotsos, John K.},
  booktitle={IEEE/CVF International Conference on Computer Vision},
  year={2019}
}
```

### GATEPose

```bibtex
@inproceedings{alshami2026gatepose,
  title={GATEPose: A Graph Attention Transformer Enhanced with Pose and Orientation Angles for Pedestrian Crossing Intention Prediction},
  author={AlShami, Ali K. and Boult, Terrance E. and Kalita, Jugal},
  booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision Workshops},
  pages={1810--1819},
  year={2026}
}
```

## Licensing And Data Terms

- The [JAAD annotation repository](https://github.com/ykotseruba/JAAD) is
  MIT-licensed. Its video clips are published separately under Creative
  Commons Attribution 4.0.
- The [PIE repository](https://github.com/aras62/PIE) is MIT-licensed.
- At the time this README was prepared, the linked
  [EfficientPIE repository](https://github.com/heinideyibadiaole/EfficientPIE)
  did not expose a license file. Therefore, this repository does not claim to
  relicense EfficientPIE-derived code. Consult the EfficientPIE authors or
  upstream repository before redistributing or reusing those portions.
- No JAAD or PIE videos, extracted images, or dataset annotations are included
  here. Users must obtain them from the official sources and follow their
  respective terms.
