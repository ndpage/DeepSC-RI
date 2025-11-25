import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_predictions(ground_truths, predictions, accuracy=None):
        n = len(ground_truths)
        if len(predictions) != n:
            raise ValueError("ground_truths and predictions must have the same length")
    
        x = np.arange(n)
        width = 0.35
    
        title = "Predictions vs Ground Truth"
        if accuracy is not None:
            title += f" (Accuracy: {accuracy:.2f}%)"
        plt.title(title)

        plt.bar(x - width/2, ground_truths, width, alpha=0.7, label='Ground Truth')
        plt.bar(x + width/2, predictions, width, alpha=0.7, label='Predictions')
        
        plt.xlabel('Sample Index')
        plt.ylabel('Class Label')
        plt.xticks(x)
        plt.legend()
        plt.tight_layout()
        plt.show()

def to_numpy_image(t: torch.Tensor):
    # assume shape [C,H,W]; detach and move to cpu
    if t.dim() == 4:  # [B,C,H,W] -> take first
        t = t[0]
    arr = t.detach().cpu().numpy()
    if arr.shape[0] in (1,3):
        arr = np.transpose(arr, (1,2,0))  # C,H,W -> H,W,C
    # Normalize for display if needed
    if arr.dtype != np.uint8:
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
    return arr

def visualize_channel_flow(original: torch.Tensor,
                           noisy_input: torch.Tensor,
                           symbols: torch.Tensor,
                           transmitted: torch.Tensor,
                           rec_feats: torch.Tensor,
                           logits: torch.Tensor):
    """Visualize original image, noisy image, and bar plots of symbol changes.

    Since the current model does not reconstruct the image pixels (it reconstructs feature vectors),
    we show:
      - Original input image
      - Input with simulated AWGN (not from model pipeline but illustrative)
      - Original channel symbols vs transmitted symbols (after noise)
      - Receiver reconstructed feature vector distribution (first 32 dims)
      - Predicted class probabilities
    """
    orig_img = to_numpy_image(original)
    noisy_img = to_numpy_image(noisy_input)
    symbols_np = symbols.detach().cpu().numpy()[0]
    transmitted_np = transmitted.detach().cpu().numpy()[0]
    rec_feats_np = rec_feats.detach().cpu().numpy()[0]
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

    # pick subset for visualization if large
    max_show = min(32, symbols_np.shape[0])

    fig = plt.figure(figsize=(12,8))
    gs = fig.add_gridspec(2,3)

    ax0 = fig.add_subplot(gs[0,0])
    ax0.imshow(orig_img)
    ax0.set_title('Original Image')
    ax0.axis('off')

    ax1 = fig.add_subplot(gs[0,1])
    ax1.imshow(noisy_img)
    ax1.set_title('Noisy Image (illustrative)')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0,2])
    idx = np.arange(max_show)
    ax2.bar(idx - 0.15, symbols_np[:max_show], width=0.3, label='Symbols')
    ax2.bar(idx + 0.15, transmitted_np[:max_show], width=0.3, label='Transmitted')
    ax2.set_title('Channel Symbols vs Noisy')
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[1,0])
    ax3.bar(np.arange(max_show), rec_feats_np[:max_show])
    ax3.set_title('Reconstructed Features (subset)')

    ax4 = fig.add_subplot(gs[1,1])
    ax4.bar(["red", "yellow", "green"], probs)
    ax4.set_title('Class Probabilities')

    ax5 = fig.add_subplot(gs[1,2])
    diff = transmitted_np[:max_show] - symbols_np[:max_show]
    ax5.bar(np.arange(max_show), diff)
    ax5.set_title('Noise Difference (subset)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # gt_sample = [0, 1, 2, 1, 0, 2, 1, 0, 2, 1]
    # preds_sample = [0, 1, 1, 1, 0, 2, 0, 0, 2, 1]

    # # visualize_predictions(gt_sample, preds_sample)

    # fig = plt.figure(figsize=(12,8))
    # gs = fig.add_gridspec(2,3)
    # ax0 = fig.add_subplot(gs[0,0])
    # ax0.bar(["red","yellow", "green"], [0.7, 0, 0.15])

    symbols = torch.tensor([[ 0.0704, -0.4867,  0.0622,  0.2680,  0.6379,  0.0937, -0.4772, -0.3920,
         -0.2823, -0.6896,  0.3890, -0.3454, -0.8718,  0.8384,  1.2586, -0.2472,
          0.6144, -1.1521,  0.8243, -0.0549,  0.8179,  0.3309,  0.2334, -0.3149,
          0.4976,  1.0243, -0.2767,  0.5881, -0.5539,  0.7487,  0.0809,  0.0913,
         -0.0280, -0.0913,  0.3796, -0.6503, -0.5549, -0.0693, -0.5363,  0.1630,
         -0.6198, -0.2401, -0.9259,  0.7729,  0.3686, -0.2149, -0.8150, -0.0351,
         -0.1659, -0.1626,  1.0193, -0.1644,  0.1684, -0.0020, -0.2741, -0.4652,
          0.6082, -0.4621, -0.0733, -0.3066, -0.2166,  1.5642,  0.0243, -0.2580]]).cpu().numpy()[0]
    fig = plt.figure(figsize=(8,4))
    gs = fig.add_gridspec(1,2)
    ax1 = fig.add_subplot(gs[0,1])

    ax1.bar(np.arange(symbols.shape[0]), symbols)
    ax1.set_title('Channel Symbols')
    plt.show()


