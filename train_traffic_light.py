import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import utils as vutils
import argparse
from traffic_light_dataset import TrafficLightDataset
from models.deepsc_ri import build_deepsc_ri
import tqdm


def get_device():
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def train(args):
    device = get_device()
    print("Using device:", device)

    dataset = TrafficLightDataset(root=args.data_root, annotation_csv=args.annotations)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = build_deepsc_ri(num_classes=3, channel_dim=args.channel_dim, pretrained=not args.no_pretrain)
    model.set_channel(snr_dB=args.snr, fading=args.fading)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm.tqdm(
            loader,
            desc=f"Epoch {epoch+1}/{args.epochs}: {running_loss/total if total > 0 else 0:.4f}",
            leave=False
            ):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            # pbar.set_postfix(loss=loss.item())
            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        acc = correct / total
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {epoch_loss:.4f} - Acc: {acc:.4f}")

    if args.save_path:
        torch.save(model.state_dict(), args.save_path)
        print("Model saved to", args.save_path)


def parse_args():
    p = argparse.ArgumentParser(description="Train DeepSC-RI on LISA Traffic Light dataset")
    p.add_argument('--data-root', required=True, help='Root of LISA dataset containing image folders & Annotations/')
    p.add_argument('--annotations', default=None, help='Override path to annotations CSV (optional)')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--channel-dim', type=int, default=64)
    p.add_argument('--snr', type=float, default=10.0, help='SNR in dB for channel simulation')
    p.add_argument('--fading', type=str, default='awgn', choices=['awgn', 'rayleigh'])
    p.add_argument('--no-pretrain', action='store_true', help='Disable ImageNet pretrained weights')
    p.add_argument('--save-path', type=str, default='deepsc_ri_traffic_light.pth')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
