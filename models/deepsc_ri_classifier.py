import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

        
class DeepSC_RI_Classifier(nn.Module):
    """Image variant of DeepSC (Robust Image) for traffic light classification.

    Pipeline:
        image -> CNN encoder (ResNet18 features) -> channel encoder (dim reduction + power norm)
              -> channel (noise/fading) -> channel decoder (feature reconstruction)
              -> classifier head (traffic light state)

    Args:
        num_classes: number of traffic light states (e.g., 3 for red/yellow/green)
        channel_dim: latent symbol dimension transmitted over channel
        pretrained: use ImageNet pretrained weights for encoder
    """

    def __init__(self, num_classes: int = 3, channel_dim: int = 64, pretrained: bool = True):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        # Remove the final FC and keep avgpool; we'll flatten after forward.
        modules = list(base.children())[:-1]  # until avgpool
        self.encoder = nn.Sequential(*modules)  # output shape: [B, 512, 1, 1]
        self.feat_dim = 512

        self.channel_encoder = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, channel_dim)
        )

        self.channel_decoder = nn.Sequential(
            nn.Linear(channel_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.feat_dim)
        )

        self.classifier = nn.Linear(self.feat_dim, num_classes)

        # Channel params (can be adjusted via set_channel)
        self.snr_dB = 10.0
        self.fading = 'awgn'  # 'awgn' or 'rayleigh'

    def set_channel(self, snr_dB: float = 10.0, fading: str = 'awgn'):
        self.snr_dB = snr_dB
        self.fading = fading.lower()

    def _power_normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize symbol power to 1 per batch
        power = torch.mean(x.pow(2)) + 1e-8
        x = x / torch.sqrt(power)
        return x

    def _apply_channel(self, x: torch.Tensor) -> torch.Tensor:
        # Convert SNR dB to linear
        snr_linear = 10 ** (self.snr_dB / 10.0)
        # Assuming unit power signal after normalization, noise variance = 1 / (2 * snr) for real-valued
        noise_var = 1.0 / snr_linear
        if self.fading == 'rayleigh':
            h = torch.randn_like(x)
            x = h * x
        noise = torch.randn_like(x) * torch.sqrt(torch.tensor(noise_var, device=x.device))
        return x + noise

    def forward(self, images: torch.Tensor):
        # Encode image
        feats = self.encoder(images)  # [B, 512, 1, 1]
        feats = feats.view(feats.size(0), -1)  # [B, 512]

        # Channel encode
        symbols = self.channel_encoder(feats)  # [B, channel_dim]
        # Power normalize
        symbols_norm = self._power_normalize(symbols)


        rx_symbols = self._apply_channel(symbols_norm)

        # Channel decode
        dec_output = self.channel_decoder(rx_symbols)  # [B, 512]

        # Classification logits
        logits = self.classifier(dec_output)  # [B, num_classes]

        intermediates = {
            'input': images.detach(),
            'feats': feats.detach(),
            'symbols': symbols.detach(),
            'symbols_norm': symbols_norm.detach(),
            'rx_symbols': rx_symbols.detach(),
            'dec_output': dec_output.detach(),
            'logits': logits.detach()
        }
        return logits, intermediates


def build_deepsc_ri_classifier(num_classes: int = 3, channel_dim: int = 64, pretrained: bool = True):
    return DeepSC_RI_Classifier(num_classes=num_classes, channel_dim=channel_dim, pretrained=pretrained)


