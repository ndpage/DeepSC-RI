import os
import sys
import pandas as pd
import torch
import tqdm

from traffic_light_dataset import TrafficLightDataset
from models.deepsc_ri_classifier import build_deepsc_ri
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from utils.devices import get_device


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

            logits, intermediates = model(img)
            pred = logits.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)

            ground_truths.append(label.cpu().item())
            predictions.append(pred.cpu().item())

    accuracy = correct / total
    accuracy = accuracy * 100.0

    return accuracy, ground_truths, predictions, intermediates

def inference_single_image(model, image_tensor: torch.Tensor, device: torch.device):
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        logits, intermediates = model(image_tensor)
        pred = logits.argmax(dim=1)
        pred_value = pred.cpu().item()
    return pred_value, intermediates

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference with DeepSC-RI on Traffic Light Dataset")
    parser.add_argument('model_path', default="deepsc_ri_traffic_light.pth" , help='Path to the trained model file')
    parser.add_argument('-d','--data-root',default="lisa-traffic-light-dataset",help='Root directory of the dataset images')
    parser.add_argument('-a','--annotations', default="lisa-traffic-light-dataset/Annotations/Annotations/daySequence1/frameAnnotationsBOX.csv", help='Path to the annotations CSV file')
    parser.add_argument('--snr', type=float, default=10.0, help='SNR in dB for the channel model')
    parser.add_argument('--channel-dim', type=int, default=64, help='Channel dimension for DeepSC-RI model')
    parser.add_argument('--fading', choices=['awgn', 'rayleigh'], default='awgn', help='Channel fading model')
    args = parser.parse_args()

    acc, gt, preds, inter = inference(
        model_path=args.model_path,
        data_root=args.data_root,
        annotation_csv=args.annotations,
        snr_dB=args.snr,
        fading=args.fading,
        channel_dim=args.channel_dim
    )
    
    print(f"Inference Accuracy: {acc:.4f}%")

    # Perform inference on a single sample and visualize
    ds = TrafficLightDataset(root=args.data_root, annotation_csv=args.annotations)
    sample_img, label = ds[9]

    pred_value, intermediates = inference_single_image(
        model=build_deepsc_ri(num_classes=3, channel_dim=args.channel_dim, pretrained=False).to(get_device()),
        image_tensor=sample_img,
        device=get_device()
    )

    input = intermediates['input']
    logits = intermediates['logits']
    symbols = intermediates['symbols']
    symbols_norm = intermediates['symbols_norm']
    transmitted = intermediates['transmitted']
    rec_feats = intermediates['rec_feats']

    print(f"Logits: {logits.shape}")
    print(f"Symbols: {symbols.shape}")
    print(f"Transmitted symbols: {transmitted.shape}")
    print(f"Reconstructed features: {rec_feats.shape}")

    print(f"Single Image Prediction: {pred_value}, Ground Truth: {label}")

    fig = plt.figure(figsize=(12,8))
    fig.suptitle(f"DeepSC-RI Inference - SNR: {args.snr} dB, Fading: {args.fading.upper()}", fontsize=16, fontweight='bold')
    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0,0])
    
    sample_img: torch.Tensor = input.cpu()
    sample_img_np = to_numpy_image(sample_img)

    ax1.imshow(sample_img_np)
    ax1.set_title('Image for Inference')
    ax1.margins(x=0, y=0.1)

    ax2 = fig.add_subplot(gs[0,1])
    ax2.plot(symbols.cpu().numpy()[0], marker='o', linestyle='-', label='Original Symbols')
    ax2.plot(symbols_norm.cpu().numpy()[0], marker='x', linestyle='--', label='Normalized Symbols')
    ax2.plot(transmitted.cpu().numpy()[0], marker='.', linestyle=':', label='Transmitted Symbols')
    ax2.set_title('Channel Symbols')
    ax2.legend(['Original Symbols', 'Normalized Symbols', 'Transmitted Symbols'], fontsize=8)
    ax2.margins(x=0, y=0.05)

    probs = torch.softmax(logits, dim=1)
    ax3 = fig.add_subplot(gs[1,0])
    ax3.bar(np.arange(probs.shape[1]), probs.cpu().numpy()[0])
    ax3.set_title('logits Distribution')

    ax4 = fig.add_subplot(gs[1,1])
    ax4.plot(rec_feats.cpu().numpy()[0])
    ax4.set_title('Reconstructed Features')

    plt.show()