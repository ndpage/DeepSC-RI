import os
import pandas as pd
import torch
import tqdm

from traffic_light_dataset import TrafficLightDataset
from models.deepsc_ri import build_deepsc_ri
from torchvision import transforms
import matplotlib.pyplot as plt
from visualization import visualize_predictions

def get_device():
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

def inference(model_path: str, data_root: str, annotation_csv: str, snr_dB: float, fading: str, channel_dim: int):
    device = get_device()
    print("Using device:", device)

    dataset = TrafficLightDataset(root=data_root, annotation_csv=annotation_csv)
    model = build_deepsc_ri(num_classes=3, channel_dim=channel_dim, pretrained=False)
    model.set_channel(snr_dB=snr_dB, fading=fading)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    ground_truths = []
    predictions = []
    with torch.no_grad():
        pbar: TrafficLightDataset = tqdm.tqdm(dataset, desc="Inference", leave=False)
        for img, label in pbar:
            img = img.unsqueeze(0).to(device)  # Add batch dimension
            label = torch.tensor([label]).to(device)

            logits = model(img)
            pred = logits.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)

            ground_truths.append(label.cpu().item())
            predictions.append(pred.cpu().item())

    accuracy = correct / total
    accuracy = accuracy * 100.0

    return accuracy, ground_truths, predictions

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference with DeepSC-RI on Traffic Light Dataset")
    parser.add_argument('model_path', help='Path to the trained model file')
    parser.add_argument('data_root', help='Root directory of the dataset images')
    parser.add_argument('annotation_csv', help='Path to the annotations CSV file')
    parser.add_argument('--snr', type=float, default=10.0, help='SNR in dB for the channel model')
    parser.add_argument('--channel-dim', type=int, default=64, help='Channel dimension for DeepSC-RI model')
    parser.add_argument('--fading', choices=['awgn', 'rayleigh'], default='awgn', help='Channel fading model')
    args = parser.parse_args()

    acc, gt, preds = inference(
        model_path=args.model_path,
        data_root=args.data_root,
        annotation_csv=args.annotation_csv,
        snr_dB=args.snr,
        fading=args.fading,
        channel_dim=args.channel_dim
    )
    print(f"Inference Accuracy: {acc:.4f}%")
    gt_sample = gt[:10]
    preds_sample = preds[:10]
    visualize_predictions(gt_sample, preds_sample)