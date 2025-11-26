import argparse
import math
from models.deepsc_ri import build_deepsc_ri
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import utils as vutils
from torchvision.transforms import Resize, Compose, ToTensor, ToPILImage
from traffic_light_dataset import ReducedImageTrafficLightDataset
import tqdm
from utils.devices import get_device
import matplotlib.pyplot as plt

from utils.visualize import to_numpy_image



def train(args):
    device = get_device()
    print("Using device:", device)

    # Dataset initializes with training images and resizes them to 192x256 (HxW)
    img_size = (960, 1280)
    reduced_size = (img_size[0]//2, img_size[1]//2)  # Reduced size for training (HxW)
    dataset = ReducedImageTrafficLightDataset(root=args.data_root, annotation_csv=args.annotations, size=reduced_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    transform = Compose([
            ToPILImage(),
            Resize(img_size),
            ToTensor()
        ])
    print(f"Dataset contains {len(dataset)} images resized to {reduced_size}")

    model = build_deepsc_ri(img_size=reduced_size, patch_size=16, channel_dim=args.channel_dim)
    model.set_channel(snr_dB=args.snr, fading=args.fading)
    model.to(device)
    # model.eval()

    # Research utilizes combination of cross-entropy loss and MSE loss for reconstruction task
    cel = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    alpha = 0.1  # weight for MSE loss

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
            plt.legend(["Epoch 1"])

        for reduced_images, orig_images in pbar:
            desc=f"Epoch {epoch+1}/{args.epochs}: {running_loss/len(loader):.4f}"
            pbar.set_description(desc)
            
            if args.show_graph:
                x.append(i)
                y.append(math.log(running_loss/len(loader)) if len(x)>1 else 0)
                line.set_data(x, y)
                plt.gca().relim()
                plt.gca().autoscale()
                plt.pause(0.01)
                if epoch+1 > 1:
                    line.set_label(f"Epoch {epoch+1}")
                    plt.legend()

            # Send to device
            reduced_images = reduced_images.to(device)
            # orig_images = orig_images.to(device)

            optimizer.zero_grad()
            rec_images, intermediates = model(reduced_images)

            # # Resize reconstructed image to original size for loss computation
            # rec_image_full = torch.stack([
            #     transform(rec_image[i].cpu()) for i in range(rec_image.size(0))
            # ]).to(device)
            
            tx_symbols = intermediates['tx_symbols']
            rx_decoded = intermediates['rx_decoded']

            loss_type = args.loss_func 
            if loss_type == 'ce_mse':
                # Compute losses according to L_total = L_CE(Iu, ˆI) + α · L_MSE(Tx, Rx),
                mse_loss = mse(rx_decoded, tx_symbols)
                ce_loss = cel(rec_images, reduced_images)
                loss = ce_loss + alpha * mse_loss
            elif loss_type == 'l1_mse':
                # Compute losses according to L_total = L_L1(Iu, ˆI) + α · L_MSE(Tx, Rx),
                mse_loss = mse(rx_decoded, tx_symbols)
                l1_loss = l1(rec_images, reduced_images)
                loss = l1_loss + alpha * mse_loss
            else:
                mse_loss = mse(rx_decoded, tx_symbols)
                loss = mse_loss

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
    p.add_argument('--loss-func', type=str, default='ce_mse', choices=['ce_mse', 'l1_mse', 'mse'], help='Loss function to use: ce_mse, l1_mse, mse')

    args = p.parse_args()
    train(args)
    