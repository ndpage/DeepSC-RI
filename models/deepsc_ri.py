
"""
Derived from DeepSC model: https://github.com/13274086/DeepSC.git
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import models


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model)) #math.log(math.exp(1)) = 1
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #[1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x
    
 
class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        self.dense = nn.Linear(d_model, d_model)
        
        #self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k)
        query = query.transpose(1, 2)
        
        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k)
        key = key.transpose(1, 2)
        
        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k)
        value = value.transpose(1, 2)
        
        #        query, key, value = \
        #            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = self.attention(query, key, value, mask=mask)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.num_heads * self.d_k)
             
        x = self.dense(x)
        x = self.dropout(x)
        
        return x
    
    def attention(self, query, key, value, mask=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        #print(mask.shape)
        if mask is not None:
            # 根据mask，指定位置填充 -1e9  
            scores += (mask * -1e9)
            # attention weights
        p_attn = F.softmax(scores, dim = -1)
        return torch.matmul(p_attn, value), p_attn
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.w_2(x)
        x = self.dropout(x) 
        return x


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, d_model, num_heads, dff, dropout = 0.1):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadedAttention(num_heads, d_model, dropout = 0.1)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout = 0.1)
        
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        attn_output = self.mha(x, x, x, mask)
        x = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + ffn_output)
        
        return x
    
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, d_model, num_heads, dff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_mha = MultiHeadedAttention(num_heads, d_model, dropout = 0.1)
        self.src_mha = MultiHeadedAttention(num_heads, d_model, dropout = 0.1)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout = 0.1)
        
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
        
        #self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        "Follow Figure 1 (right) for connections."
        #m = memory
        
        attn_output = self.self_mha(x, x, x, look_ahead_mask)
        x = self.layernorm1(x + attn_output)
        
        src_output = self.src_mha(x, memory, memory, trg_padding_mask) # q, k, v
        x = self.layernorm2(x + src_output)
        
        fnn_output = self.ffn(x)
        x = self.layernorm3(x + fnn_output)
        return x

class FinePatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding for Fine Features """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.grid = img_size // patch_size
    
    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H/ps, W/ps]
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        return x


class CoarsePatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding for Coarse Features """
    def __init__(self, img_size=224, patch_size=32, in_chans=3, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.grid = img_size // patch_size

    def forward(self, x):
        x = self.proj(x)                            # [B,embed_dim,grid_c,grid_c]
        return x.flatten(2).transpose(1, 2)  


class FusionModule(nn.Module):
    """ Fusion Module to combine fine and coarse features """
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model * 2, d_model)
        self.layernorm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, fine_feats, coarse_feats):
        # If sizes differ, repeat coarse features to match fine features
        if coarse_feats.size(1) != fine_feats.size(1):
            repeat_times = fine_feats.size(1) // coarse_feats.size(1)
            coarse_feats = coarse_feats.repeat_interleave(repeat_times, dim=1)
            if coarse_feats.size(1) > fine_feats.size(1):
                coarse_feats = coarse_feats[:, :fine_feats.size(1), :]
        
        fused = torch.cat([fine_feats, coarse_feats], dim=-1)  # [B, L, 2*d_model]
        fused = self.proj(fused)
        output = self.layernorm(fused)
        return output

class Encoder(nn.Module):
    "Core encoder: Dual layered transformer encoder, fine and coarse feature extraction"
    def __init__(self, num_layers, max_len, d_model, num_heads, dff, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model

        self.fine_patch_embed = FinePatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=d_model)
        self.coarse_patch_embed = CoarsePatchEmbed(img_size=224, patch_size=32, in_chans=3, embed_dim=d_model)

        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, src_mask):
        "Input x: [B, seq_len, d_model] already embedded"
        x = self.pos_encoding(x)
        for enc_layer in self.enc_layers:
            x = enc_layer(x, src_mask)
        return x
        


class Decoder(nn.Module):
    def __init__(self, num_layers, img_output_feats, max_len, 
                 d_model, num_heads, dff, dropout = 0.1):
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(img_output_feats, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dff, dropout) 
                                            for _ in range(num_layers)])
    
    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        
        # x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for dec_layer in self.dec_layers:
            x = dec_layer(x, memory, look_ahead_mask, trg_padding_mask)
            
        return x


class ChannelDecoder(nn.Module):
    def __init__(self, in_features, size1, size2):
        super(ChannelDecoder, self).__init__()
        
        self.linear1 = nn.Linear(in_features, size1)
        self.linear2 = nn.Linear(size1, size2)
        self.linear3 = nn.Linear(size2, size1)
        # self.linear4 = nn.Linear(size1, d_model)
        
        self.layernorm = nn.LayerNorm(size1, eps=1e-6)
        
    def forward(self, x):
        x1 = self.linear1(x)
        x2 = F.relu(x1)
        x3 = self.linear2(x2)
        x4 = F.relu(x3)
        x5 = self.linear3(x4)
        
        output = self.layernorm(x1 + x5)

        return output

# Developed based on research by Peng et al. "A Robust Semantic Communication System for Image Transmission"
class DeepSC_RI(nn.Module):


    def __init__(self, img_size, patch_size) -> None:
        super().__init__()

        # Channel params (can be adjusted via set_channel)
        self.snr_dB = 10.0
        self.fading = 'awgn'  # 'awgn' or 'rayleigh'
        
        self.num_layers = 2
        # Patch embedding settings
        self.patch_size = 16  # adjust as needed
        self.src_max_len = (img_size // patch_size) ** 2  # e.g., for 224x224 with 16x16 patches -> (224/16)^2 = 14*14 = 196
        self.img_output_feats = 1000
        self.trg_max_len = 50
        self.d_model = 256
        self.num_heads = 8
        self.dff = 512
        self.dropout = 0.1
        # Patch embedding projection: Conv2d -> flatten -> linear (if needed)
        self.patch_embed = nn.Conv2d(3, self.d_model, kernel_size=self.patch_size, stride=self.patch_size)
        self.encoder = Encoder(self.num_layers, self.src_max_len, self.d_model, self.num_heads, self.dff, self.dropout)
        
        self.channel_encoder = nn.Sequential(nn.Linear(self.d_model, 256), 
                                             nn.ReLU(inplace=True),
                                             nn.Linear(256, 16))


        self.channel_decoder = ChannelDecoder(16, self.d_model, 512)

        self.decoder = Decoder(self.num_layers, self.img_output_feats, self.trg_max_len, 
                               self.d_model, self.num_heads, self.dff, self.dropout)
        



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

    def set_channel(self, snr_dB: float = 10.0, fading: str = 'awgn'):
        self.snr_dB = snr_dB
        self.fading = fading.lower()

    def forward(self, images: torch.Tensor):
        # Expect images: [B, C, H, W]
        if images.dim() != 4:
            raise ValueError(f"Expected images shape [B,C,H,W]; got {images.shape}")
        B, C, H, W = images.shape
        # Patch embedding
        patches = self.patch_embed(images)  # [B, d_model, H/ps, W/ps]
        patches = patches.flatten(2).transpose(1, 2)  # [B, N, d_model]
        # If actual N differs from src_max_len, truncate or pad
        N = patches.size(1)
        if N < self.src_max_len:
            pad_len = self.src_max_len - N
            pad = torch.zeros(B, pad_len, self.d_model, device=patches.device, dtype=patches.dtype)
            patches = torch.cat([patches, pad], dim=1)
        elif N > self.src_max_len:
            patches = patches[:, :self.src_max_len, :]

        feats_seq = self.encoder(patches, src_mask=None)  # [B, src_max_len, d_model]
        feats = feats_seq.mean(dim=1)  # [B, d_model]

        # Channel encode
        symbols = self.channel_encoder(feats)  # [B, channel_dim]
        symbols_norm = self._power_normalize(symbols)

        #### Simulated AWGN / Fading Channel here ####
        transmitted = self._apply_channel(symbols_norm)
        
        # Channel decode
        rec_feats = self.channel_decoder(transmitted)  # [B, d_model]
        rec_feats = rec_feats.unsqueeze(1).repeat(1, self.trg_max_len, 1)  # [B, trg_max_len, d_model]
        
        # Decoder
        dec_output = self.decoder(rec_feats, rec_feats, look_ahead_mask=None, trg_padding_mask=None)  # [B, trg_max_len, d_model]
        
        # Reconstruction
        logits = self.dense(dec_output)  # [B, trg_max_len, img_output_feats]

        # prob = torch.softmax(logits, dim=-1)  # [B, trg_max_len, img_output_feats]

        intermediates = {
            'input': images.detach(),
            'patches': patches.detach(),
            'feats_seq': feats_seq.detach(),
            'feats': feats.detach(),
            'symbols': symbols.detach(),
            'symbols_norm': symbols_norm.detach(),
            'transmitted': transmitted.detach(),
            'rec_feats': rec_feats.detach(),
            'logits': logits.detach()
        }
        return logits, intermediates


def build_deepsc_ri():
    return DeepSC_RI(256, 16)

if __name__ == "__main__":
    # Simple test to build model
    model = build_deepsc_ri()
    print(model)
    




