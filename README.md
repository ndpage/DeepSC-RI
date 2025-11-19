# DeepSC-RI

## Traffic Light (Image) Variant
An experimental DeepSC-RI style adaptation for traffic light state classification is included.

### Prepare
Ensure the LISA Traffic Light dataset is placed at `PATH_TO_LISA/` with the `Annotations/` directory present.

Install extra dependencies (if not already):
```shell
pip install -r requirements.txt
```

### Train Image Model
```shell
python train_traffic_light.py \
  --data-root PATH_TO_LISA \
  --epochs 20 \
  --snr 10 \
  --fading awgn
```
Optional Rayleigh fading:
```shell
python train_traffic_light.py --data-root PATH_TO_LISA --fading rayleigh
```
Model weights will be saved to `deepsc_ri_traffic_light.pth` by default.
