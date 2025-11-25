import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from traffic_light_dataset import ReducedImageTrafficLightDataset
from models.deepsc_ri import build_deepsc_ri
import matplotlib.pyplot as plt
import numpy as np
from utils.devices import get_device
from utils.visualize import to_numpy_image
from torchvision.transforms import Resize


def inference(model_path: str, data_root: str, annotation_csv: str, snr_dB: float, fading: str, channel_dim: int):
    device = get_device()
    print("Using device:", device)
    size = (960, 1280)
    reduced = (size[0]//2, size[1]//2)
    dataset = ReducedImageTrafficLightDataset(root=data_root, annotation_csv=annotation_csv, size=reduced)
    model = build_deepsc_ri(img_size=reduced, patch_size=16)
    model.set_channel(snr_dB=snr_dB, fading=fading)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    ground_truths = []
    reconstructions = []
    with torch.no_grad():
        pbar: ReducedImageTrafficLightDataset = tqdm.tqdm(dataset, desc="Inference", leave=False)
        for img, orig_img in pbar:
            img = img.to(device)
            orig_img = orig_img.to(device)

            recnst_img, intermediates = model(img)
            # For reconstruction accuracy, we can use MSE between orig_img and recnst_img
            mse_loss = torch.mean((recnst_img - orig_img) ** 2, dim=[1,2,3])
            # recon_error = (mse_loss < 0.01).long()  # threshold for reconstruction success
            # label = torch.tensor([1 if torch.mean((orig_img - img) ** 2) < 0.01 else 0]).to(device)

            # correct += (pred == label).sum().item()
            # total += label.size(0)

            ground_truths.append(orig_img.cpu().item())
            reconstructions.append(recnst_img.cpu().item())

    # accuracy = correct / total
    # accuracy = accuracy * 100.0
    accuracy = 0.0  # Placeholder as reconstruction accuracy is not computed here

    return accuracy, ground_truths, reconstructions, intermediates

def inference_single_image(args):
    # Perform inference on a single sample and visualize
    size = (960, 1280)
    reduced = (size[0]//2, size[1]//2)
    ds = ReducedImageTrafficLightDataset(root=args.data_root, annotation_csv=args.annotations, size=reduced)
    image_tensor, original_img_tensor = ds[0]
    print(f"Sample: {image_tensor.shape}, Original: {original_img_tensor.shape}")
    
    model = build_deepsc_ri(img_size=reduced, patch_size=16)
    device = get_device()
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)
    original_img_tensor = original_img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        rec_image, inter = model(image_tensor)
        upscaled_size = (original_img_tensor.shape[2], original_img_tensor.shape[3])
        rec_image = nn.functional.interpolate(rec_image, size=upscaled_size, mode='bilinear', align_corners=False)
    return rec_image, inter

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference with DeepSC-RI on Traffic Light Dataset")
    parser.add_argument('model_path', default="deepsc_ri.pth" , help='Path to the trained model file')
    parser.add_argument('-d','--data-root',default="lisa-traffic-light-dataset",help='Root directory of the dataset images')
    parser.add_argument('-a','--annotations', default="lisa-traffic-light-dataset/Annotations/Annotations/daySequence1/frameAnnotationsBOX.csv", help='Path to the annotations CSV file')
    parser.add_argument('--snr', type=float, default=10.0, help='SNR in dB for the channel model')
    parser.add_argument('--channel-dim', type=int, default=64, help='Channel dimension for DeepSC-RI model')
    parser.add_argument('--fading', choices=['awgn', 'rayleigh'], default='awgn', help='Channel fading model')
    args = parser.parse_args()

    # acc, gt, preds, inter = inference(
    #     model_path=args.model_path,
    #     data_root=args.data_root,
    #     annotation_csv=args.annotations,
    #     snr_dB=args.snr,
    #     fading=args.fading,
    #     channel_dim=args.channel_dim
    # )
    
    # print(f"Inference Accuracy: {acc:.4f}%")

    recon_img, intermediates = inference_single_image(args)

    input = intermediates['input']
    feats_seq = intermediates['feats_seq']
    tx_norm = intermediates['tx_norm']
    transmitted = intermediates['rx_symbols']
    dec_symbols = intermediates['dec_output']

    fig = plt.figure(figsize=(12,8))
    fig.suptitle(f"DeepSC-RI Inference - SNR: {args.snr} dB, Fading: {args.fading.upper()}", fontsize=16, fontweight='bold')
    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0,0])
    
    original_img: torch.Tensor = input.cpu()

    # resize_transform = Resize((960, 1280)) # Specify desired output size
    # resized_tensor = resize_transform(original_img)
    # resized_img = to_numpy_image(recon_img)

    reconstructed = to_numpy_image(recon_img)
    original_img = to_numpy_image(original_img)

    ax1.imshow(original_img)
    ax1.set_title('Image for Inference')
    ax1.margins(x=0, y=0.1)

    ax2 = fig.add_subplot(gs[0,1])
    ax2.plot(feats_seq.cpu().numpy()[0], marker='o', linestyle='-', label='Original Symbols')
    ax2.plot(tx_norm.cpu().numpy()[0], marker='x', linestyle='--', label='Normalized Symbols')
    ax2.plot(transmitted.cpu().numpy()[0], marker='.', linestyle=':', label='Transmitted Symbols')
    ax2.set_title('Channel Symbols')
    ax2.legend(['Original Symbols', 'Normalized Symbols', 'Transmitted Symbols'], fontsize=8)
    ax2.margins(x=0, y=0.05)

    ax3 = fig.add_subplot(gs[1,0])
    ax3.imshow(reconstructed)
    ax3.set_title('Reconstructed Image')

    ax4 = fig.add_subplot(gs[1,1])
    ax4.plot(dec_symbols.cpu().numpy()[0])
    ax4.set_title('Reconstructed Features')

    plt.show()