# DeepSC-RI

## Overview
DeepSC-RI is a robust semantic communication system for image transmission over noisy channels. This implementation includes two variants:
- **DeepSC-RI Reconstruction**: End-to-end image reconstruction through semantic channel encoding
- **DeepSC-RI Classifier**: Traffic light state classification through semantic channel encoding

## Python Setup
This project uses pyproject.toml file instead of requirements.txt.
To install dependencies, run:
```shell
pip install -e .
```

## Dataset Setup
Ensure the LISA Traffic Light dataset is placed at `lisa-traffic-light-dataset/` with the `Annotations/` directory present.
The file paths in each annotation `.csv` do not properly map to the image frames in the dataset, so you need to run the 
`preprocess_traffic_light.py` script to adjust the paths. 

```shell
python ./preprocess_traffic_light.py ./lisa-traffic-light-dataset/Annotations/Annotations/dayTrain/dayClip1/frameAnnotationsBOX.csv dayTrain/dayTrain/dayClip1/frames
```
This will modify the values in the annotations `.csv` file to point to the correct training image/video frame. 
To see what the result will be, add the `--dry-run` option. To backup the original annotation file, use the `--backup` option.
```shell
python ./preprocess_traffic_light.py ./lisa-traffic-light-dataset/Annotations/Annotations/dayTrain/dayClip1/frameAnnotationsBOX.csv dayTrain/dayTrain/dayClip1/frames --dry-run --backup
```
If you need to see the CLI options for any script, simply pass the help `-h` option. 
```shell
python ./preprocess_traffic_light.py -h
```
Output:
```shell
usage: preprocess_traffic_light.py [-h] [--dry-run] [--backup] annotation_csv new_path

Preprocess LISA Traffic Light dataset annotations

positional arguments:
  annotation_csv  Path to annotations CSV file
  new_path        New root path to prepend to image filenames

options:
  -h, --help      show this help message and exit
  --dry-run       Print preview without writing changes
  --backup        Create a .bak copy of the original CSV before writing
```


## Training

### Train DeepSC-RI Reconstruction Model
Train the image reconstruction model with configurable channel dimensions and channel conditions:

```shell
python train_deepsc_ri.py lisa-traffic-light-dataset \
  --annotations lisa-traffic-light-dataset/Annotations/Annotations/dayTrain/dayClip1/frameAnnotationsBOX.csv \
  --batch-size 16 \
  --epochs 10 \
  --lr 1e-3 \
  --channel-dim 64 \
  --snr 10.0 \
  --fading awgn \
  --save-path deepsc_ri.pth \
  --checkpoint-path checkpoints/deepsc_ri
```

**Options:**
- `--channel-dim`: Channel bottleneck dimension (default: 64, try 128 or 256 for higher capacity)
- `--snr`: Signal-to-noise ratio in dB (default: 10.0)
- `--fading`: Channel model - `awgn` or `rayleigh` (default: awgn)
- `--loss-func`: Loss function - `l1_mse`, `ce_mse`, or `mse` (default: l1_mse)
- `--show-graph`: Display loss graph during training

### Train DeepSC-RI Classifier Model
Train the traffic light classification model:

```shell
python train_deepsc_ri_classifier.py lisa-traffic-light-dataset \
  --annotations lisa-traffic-light-dataset/Annotations/Annotations/daySequence1/frameAnnotationsBOX.csv \
  --batch-size 32 \
  --epochs 10 \
  --lr 1e-3 \
  --channel-dim 64 \
  --snr 10.0 \
  --fading awgn \
  --save-path deepsc_ri_classifier.pth
```

**Options:**
- `--no-pretrain`: Train from scratch without ImageNet pretrained weights

## Inference

### Run Inference on Reconstruction Model
Test image reconstruction on a single image:

```shell
python inference_deepsc_ri.py deepsc_ri_v3.1.pth \
  -d lisa-traffic-light-dataset \
  -a lisa-traffic-light-dataset/Annotations/Annotations/daySequence1/frameAnnotationsBOX.csv \
  --snr 20 \
  --channel-dim 64 \
  --fading awgn \
  --image-index 1
```

Omit `--image-index 1` to run inference on the full dataset.

### Run Inference on Classifier Model
Test traffic light classification:

```shell
python inference_deepsc_ri_classifier.py deepsc_ri_classifier.pth \
  -d lisa-traffic-light-dataset \
  -a lisa-traffic-light-dataset/Annotations/Annotations/daySequence1/frameAnnotationsBOX.csv \
  --snr 10.0 \
  --channel-dim 64 \
  --fading awgn \
  --image-index 1
```

Omit the `--image-index 1` option for full dataset evaluation with accuracy progression visualization.

## Model Architecture
- **Image Size**: 480x640 (reduced from 960x1280)
- **Patch Size**: 16x16
- **Channel Dimension**: 64 (configurable: 64, 128, 256)
- **Encoder**: Multi-scale Vision Transformer with fine and coarse patch embeddings
- **Channel**: Simulated AWGN or Rayleigh fading with configurable SNR
- **Decoder**: Transformer decoder with learned query tokens
