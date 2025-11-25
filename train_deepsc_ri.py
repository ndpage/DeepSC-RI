import argparse
import math
from models.deepsc_ri import build_deepsc_ri
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import utils as vutils
from traffic_light_dataset import ReducedImageTrafficLightDataset
import tqdm
from utils.devices import get_device
import matplotlib.pyplot as plt



def train(args):
    device = get_device()
    print("Using device:", device)

    # Dataset initializes with training images and resizes them to 192x256 (HxW)
    img_size = (192, 256)
    dataset = ReducedImageTrafficLightDataset(root=args.data_root, annotation_csv=args.annotations, size=img_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print(f"Dataset contains {len(dataset)} images resized to {img_size}")
    # img = to_numpy_image(dataset[0])  # test
    # plt.imshow(img)
    # plt.title("Sample resized image from dataset")
    # plt.show()

    model = build_deepsc_ri(img_size=img_size, patch_size=16)
    model.set_channel(snr_dB=args.snr, fading=args.fading)
    model.to(device)
    # model.eval()

    # Research utilizes combination of cross-entropy loss and MSE loss for reconstruction task
    cel = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    alpha = 0.05  # weight for MSE loss

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # plt.figure(figsize=(12, 8))
    for epoch in range(args.epochs):
        pbar = tqdm.tqdm(
                loader,
                desc=f"Epoch {epoch+1}/{args.epochs}: {0:.4f}",
                leave=False
                )   
        model.train()
        running_loss = 0.0
        i = 0
        # Visualize
        if args.show_graph:
            plt.title(f"Loss during Training Epoch {epoch+1}")
            x, y = [], []
            line,  = plt.plot(x, y)
            plt.xlabel("loader iterations")
            plt.ylabel("Loss (log scale)")
        for reduced_img, orig_img in pbar:
            desc=f"Epoch {epoch+1}/{args.epochs}: {running_loss/len(loader):.4f}"
            pbar.set_description(desc)
            
            if args.show_graph:
                x.append(i)
                y.append(math.log(running_loss/len(loader)) if len(x)>1 else 0)
                line.set_data(x, y)
                plt.gca().relim()
                plt.gca().autoscale()
                plt.pause(0.01)

            # Send to device
            reduced_img = reduced_img.to(device)
            orig_img = orig_img.to(device)

            optimizer.zero_grad()
            rec_image, intermediates = model(reduced_img)

            # Upscale reconstructed image to original size for loss computation
            upscaled_size = (orig_img.shape[2], orig_img.shape[3])
            rec_image = nn.functional.interpolate(rec_image, size=upscaled_size, mode='bilinear', align_corners=False)
            
            tx_symbols = intermediates['tx_symbols']
            rx_decoded = intermediates['rx_decoded']

            # Compute losses according to L_total = L_CE(Iu, ˆI) + α · L_MSE(Tx, Rx),
            mse_loss = mse(rx_decoded, tx_symbols)
            ce_loss = cel(rec_image, orig_img)
            loss = ce_loss + alpha * mse_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_description(f"Epoch {epoch+1}/{args.epochs}: {running_loss/len(loader):.4f}")
            i+=1

        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {running_loss/len(loader):.4f}")
        # Save checkpoint
        if args.checkpoint_path:
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            checkpoint_file = os.path.join(args.checkpoint_path, f'deepsc_ri_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_file)

    if args.save_path:
        torch.save(model.state_dict(), args.save_path)
        print(f"Model saved to {args.save_path}")
    plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train DeepSC-RI on LISA Traffic Light dataset")
    p.add_argument('data_root', help='Root of LISA dataset containing image folders & Annotations')
    p.add_argument('--annotations', default=None, help='Override path to annotations CSV (optional)')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--channel-dim', type=int, default=64)
    p.add_argument('--snr', type=float, default=10.0, help='SNR in dB for channel simulation')
    p.add_argument('--fading', type=str, default='awgn', choices=['awgn', 'rayleigh'])
    p.add_argument('--save-path', type=str, default='deepsc_ri.pth')
    p.add_argument('--checkpoint-path', type=str, default='checkpoints')
    p.add_argument('--show-graph', action='store_true', help='Show loss graph during training')

    args = p.parse_args()
    train(args)
    