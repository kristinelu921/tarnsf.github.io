#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import torch
from splines import unconstrained_rational_quadratic_spline
from nflows.transforms import ActNorm
class Permutation(torch.nn.Module):

    def __init__(self, seq_length: int):
        super().__init__()
        self.seq_length = seq_length

    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        raise NotImplementedError('Overload me')


class PermutationIdentity(Permutation):
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x


class PermutationFlip(Permutation):
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x.flip(dims=[dim])


class Attention(torch.nn.Module):
    USE_SPDA: bool = True

    def __init__(self, in_channels: int, head_channels: int):
        assert in_channels % head_channels == 0
        super().__init__()
        self.norm = torch.nn.LayerNorm(in_channels)
        self.qkv = torch.nn.Linear(in_channels, in_channels * 3)
        self.proj = torch.nn.Linear(in_channels, in_channels)
        self.num_heads = in_channels // head_channels
        self.sqrt_scale = head_channels ** (-0.25)
        self.sample = False
        self.k_cache: dict[str, list[torch.Tensor]] = {'cond': [], 'uncond': []}
        self.v_cache: dict[str, list[torch.Tensor]] = {'cond': [], 'uncond': []}

    def forward_spda(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        B, T, C = x.size()
        x = self.norm(x.float()).type(x.dtype)
        q, k, v = self.qkv(x).reshape(B, T, 3 * self.num_heads, -1).transpose(1, 2).chunk(3, dim=1)  # (b, h, t, d)

        if self.sample:
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            k = torch.cat(self.k_cache[which_cache], dim=2)  # note that sequence dimension is now 2
            v = torch.cat(self.v_cache[which_cache], dim=2)

        scale = self.sqrt_scale**2 / temp
        if mask is not None:
            mask = mask.bool()
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return x

    def forward_base(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        B, T, C = x.size()
        x = self.norm(x.float()).type(x.dtype)
        q, k, v = self.qkv(x).reshape(B, T, 3 * self.num_heads, -1).chunk(3, dim=2)
        if self.sample:
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            k = torch.cat(self.k_cache[which_cache], dim=1)
            v = torch.cat(self.v_cache[which_cache], dim=1)

        attn = torch.einsum('bmhd,bnhd->bmnh', q * self.sqrt_scale, k * self.sqrt_scale) / temp
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        attn = attn.float().softmax(dim=-2).type(attn.dtype)
        x = torch.einsum('bmnh,bnhd->bmhd', attn, v)
        x = x.reshape(B, T, C)
        x = self.proj(x)
        return x

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        if self.USE_SPDA:
            return self.forward_spda(x, mask, temp, which_cache)
        return self.forward_base(x, mask, temp, which_cache)


class MLP(torch.nn.Module):
    def __init__(self, channels: int, expansion: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(channels)
        self.main = torch.nn.Sequential(
            torch.nn.Linear(channels, channels * expansion),
            torch.nn.GELU(),
            torch.nn.Linear(channels * expansion, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(self.norm(x.float()).type(x.dtype))


class AttentionBlock(torch.nn.Module):
    def __init__(self, channels: int, head_channels: int, expansion: int = 4):
        super().__init__()
        self.attention = Attention(channels, head_channels)
        self.mlp = MLP(channels, expansion)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, attn_temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        x = x + self.attention(x, attn_mask, attn_temp, which_cache)
        x = x + self.mlp(x)
        #print("x after attention and mlp", x[0][0])
        return x


class MetaBlock(torch.nn.Module):
    attn_mask: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_patches: int,
        permutation: Permutation,
        num_layers: int = 1,
        head_dim: int = 64,
        expansion: int = 4,
        num_bins: int = 8,
        tail_bound: float = 3.0,
        nvp: bool = True,
        num_classes: int = 0,
        mask: torch.Tensor = None,
    ):
        super().__init__()
        self.proj_in = torch.nn.Linear(in_channels, channels) 
        self.in_channels = in_channels
        self.pos_embed = torch.nn.Parameter(torch.randn(num_patches, channels) * 1e-2)
        if num_classes:
            self.class_embed = torch.nn.Parameter(torch.randn(num_classes, 1, channels) * 1e-2)
        else:
            self.class_embed = None
        self.attn_blocks = torch.nn.ModuleList(
            [AttentionBlock(channels, head_dim, expansion) for _ in range(num_layers)]
        )
        self.nvp = nvp
        self.output_dim = in_channels * (3*num_bins - 1)
        self.proj_out = torch.nn.Linear(channels, self.output_dim)
        self.proj_out.weight.data.fill_(0.0)
        self.permutation = permutation
        self.register_buffer('attn_mask', torch.tril(torch.ones(num_patches, num_patches)))
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.mask = mask
        self.act_norm = ActNorm(in_channels)

    def normalize(self, x):
        """Normalize input data along the channel dimension"""
        batch_size, seq_len, channels = x.shape
        mean = x.mean(dim=-1, keepdim=True)  # Shape: [1, seq_len, channels]
        std = torch.clamp(x.std(dim=-1, keepdim=True), min=1e-5)  # Shape: [1, seq_len, channels]
        #print("mean", mean)
        #print("std", std)
        normalized_x = (x - mean) / std
        #print("normalized_x", normalized_x)
        logdet = -torch.sum(torch.log(std), dim=-1)  # Shape: [1, seq_len]
        
        return normalized_x, logdet, mean, std
    
    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None, normalize = False) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.permutation(x) # (B, T, C -> B, T, C)
        pos_embed = self.permutation(self.pos_embed, dim=0)
        x_in = x.clone() #(B, T, C)1
        #print("x_in", x_in[0])
        
        
        x = self.proj_in(x) + pos_embed #(B, T, C)

        if self.class_embed is not None:
            if y is not None:
                if (y < 0).any():
                    m = (y < 0).float().view(-1, 1, 1)
                    class_embed = (1 - m) * self.class_embed[y] + m * self.class_embed.mean(dim=0)
                else:
                    class_embed = self.class_embed[y]
                x = x + class_embed
            else:
                x = x + self.class_embed.mean(dim=0)

        for block in self.attn_blocks:
            x = block(x, self.attn_mask) #(B, T, C)
            
        # Get spline parameters from transformer output
        x = self.proj_out(x) #(B, T, C) -> (B, T, (3*num_bins - 1)*C)


        batch_size, seq_len, _ = x.shape
        x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim = 1)
        
        w, h, d = x.split([
            self.num_bins * self.in_channels,  #(B, T, C) -> (B, T, num_bins*C)
            self.num_bins * self.in_channels,  #(B, T, C) -> (B, T, num_bins*C)
            (self.num_bins - 1) * self.in_channels #(B, T, C) -> (B, T, (num_bins-1)*C)
        ], dim=-1)
        
        w = w.reshape(batch_size, seq_len, self.in_channels, self.num_bins) #(B, T, num_bins*C) -> (B, T, C, num_bins)
        h = h.reshape(batch_size, seq_len, self.in_channels, self.num_bins) #(B, T, num_bins*C) -> (B, T, C, num_bins)
        d = d.reshape(batch_size, seq_len, self.in_channels, self.num_bins - 1) #(B, T, (num_bins-1)*C) -> (B, T, C, num_bins-1)

        #print("forward w", w[0])
        #print("forward h", h[0])
        #print("forward d", d[0])

        transformed_rest, logabsdet = unconstrained_rational_quadratic_spline(
            inputs=x_in,
            unnormalized_widths=w, 
            unnormalized_heights=h,
            unnormalized_derivatives=d,
            tails="linear",
            tail_bound=self.tail_bound,
            enable_identity_init=True
        )

        
        # Combine first_patch (unchanged) with transformed rest patches
        output = transformed_rest #(B, 1, C) + (B, T-1, C) -> (B, T, C)
        #print("metablock output", output[0])
        # Return the transformed input and the log determinant of the Jacobian
        return output, logabsdet #+ logdet_norm

    def reverse_step(
        self,
        x: torch.Tensor,
        pos_embed: torch.Tensor,
        i: int,
        y: torch.Tensor | None = None,
        attn_temp: float = 1.0,
        which_cache: str = 'cond',
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        x_in = x[:, i:i+1]  # get i-th patch but keep the sequence dimension #(B, T, C) -> (B, 1, C)

        x = self.proj_in(x_in) + pos_embed[i:i+1] #(B, 1, C) + (B, 1, C) -> (B, 1, C)
        
        if self.class_embed is not None:
            if y is not None:
                if (y < 0).any():
                    m = (y < 0).float().view(-1, 1, 1)
                    class_embed = (1 - m) * self.class_embed[y] + m * self.class_embed.mean(dim=0)
                else:
                    class_embed = self.class_embed[y]
                x = x + class_embed
            else:
                x = x + self.class_embed.mean(dim=0)
        
        for block in self.attn_blocks:
            x = block(x, attn_temp=attn_temp, which_cache=which_cache)  # here we use kv caching, so no attn_mask
            
        x = self.proj_out(x) #(B, T, C) -> (B, T, (3*num_bins - 1)*C)
        batch_size, seq_len, _ = x.shape
        
        w, h, d = x.split([
            self.num_bins * self.in_channels,  #(B, T, (3*num_bins - 1)*C) -> (B, T, num_bins*C)
            self.num_bins * self.in_channels,  #(B, T, (3*num_bins - 1)*C) -> (B, T, num_bins*C)
            (self.num_bins - 1) * self.in_channels #(B, T, (3*num_bins - 1)*C) -> (B, T, (num_bins-1)*C)
        ], dim=-1)
        
        w = w.reshape(batch_size, seq_len, self.in_channels, self.num_bins) #(B, T, num_bins*C) -> (B, T, C, num_bins)  
        h = h.reshape(batch_size, seq_len, self.in_channels, self.num_bins) #(B, T, num_bins*C) -> (B, T, C, num_bins)
        d = d.reshape(batch_size, seq_len, self.in_channels, self.num_bins - 1) #(B, T, (num_bins-1)*C) -> (B, T, C, num_bins-1)
        
        #print("reversed w, patch ",i, w[0])
        #print("reversed h, patch ",i, h[0])
        #print("reversed d, patch ",i, d[0])

        return w, h, d

    def set_sample_mode(self, flag: bool = True):
        for m in self.modules():
            if isinstance(m, Attention):
                m.sample = flag
                m.k_cache = {'cond': [], 'uncond': []}
                m.v_cache = {'cond': [], 'uncond': []}

    def reverse(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        guidance: float = 0,
        guide_what: str = 'whd',
        attn_temp: float = 1.0,
        annealed_guidance: bool = False,
        normalize: bool = False,
    ) -> torch.Tensor:
        x = self.permutation(x) #(B, T, C) -> (B, T, C)
        #print("x", x.shape)
        pos_embed = self.permutation(self.pos_embed, dim=0)
        self.set_sample_mode(True)
        
        T = x.size(1)
        for i in range(x.size(1) - 1):  
            w, h, d = self.reverse_step(x, pos_embed, i, y, which_cache='cond')
            
            if guidance > 0 and guide_what:
                wu, hu, du = self.reverse_step(x, pos_embed, i, None, attn_temp=attn_temp, which_cache='uncond')
                
                if annealed_guidance:
                    g = (i + 1) / (T - 1) * guidance
                else:
                    g = guidance
                    
                if 'w' in guide_what:
                    w = w + g * (w - wu)
                if 'h' in guide_what:
                    h = h + g * (h - hu)
                if 'd' in guide_what:
                    d = d + g * (d - du)
            
            transformed = unconstrained_rational_quadratic_spline(
                inputs=x[:, i+1],
                unnormalized_widths=w[:, 0],
                unnormalized_heights=h[:, 0],
                unnormalized_derivatives=d[:, 0],
                inverse=True,
                tails="linear",
                tail_bound=self.tail_bound,
                enable_identity_init=True
            )
            #print("transformed", transformed)
            
            x[:, i + 1] = transformed[0]
            # Denormalize the output
            #print("curr std", curr_std, "curr mean", curr_mean)
            #print("transformed", transformed[0].size())

        self.set_sample_mode(False)
        return self.permutation(x, inverse=True)


class Model(torch.nn.Module):
    VAR_LR: float = 0.1
    var: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        img_size: int,
        patch_size: int,
        channels: int,
        num_blocks: int,
        layers_per_block: int,
        nvp: bool = True,
        num_classes: int = 0,
        num_bins: int = 8,
        tail_bound: float = 3.0,
        expansion: int = 4

    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        permutations = [PermutationIdentity(self.num_patches), PermutationFlip(self.num_patches)]

        blocks = []
        for i in range(num_blocks):
            blocks.append(
                MetaBlock(
                    in_channels * patch_size**2,
                    channels,
                    self.num_patches,
                    permutations[i % 2],
                    layers_per_block,
                    nvp=nvp,
                    num_classes=num_classes,
                    num_bins=num_bins,
                    tail_bound=tail_bound,
                    expansion=expansion,
                )
            )
        self.blocks = torch.nn.ModuleList(blocks)
        # prior for nvp mode should be all ones, but needs to be learnd for the vp mode
        self.register_buffer('var', torch.ones(self.num_patches, in_channels * patch_size**2))

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert an image (N,C',H,W) to a sequence of patches (N,T,C')"""
        u = torch.nn.functional.unfold(x, self.patch_size, stride=self.patch_size)
        return u.transpose(1, 2)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert a sequence of patches (N,T,C) to an image (N,C',H,W)"""
        u = x.transpose(1, 2)
        return torch.nn.functional.fold(u, (self.img_size, self.img_size), self.patch_size, stride=self.patch_size)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        x = self.patchify(x) #(bs, 1, h, w -> bs, num_patch, pw^2)
        outputs = []
        logdets = torch.zeros((), device=x.device) 
        for block in self.blocks:
            x, logdet = block(x, y) 
            #print("block x", x)
            logdets = logdets + logdet
            outputs.append(x)
        
        return x, outputs, logdets

    def update_prior(self, z: torch.Tensor):
        z2 = (z**2).mean(dim=0)
        self.var.lerp_(z2.detach(), weight=self.VAR_LR)

    def get_loss(self, z: torch.Tensor, logdets: torch.Tensor):
        gaussian_loss = 0.5 * z.pow(2).mean()
        prior_loss = -logdets.mean()
        return gaussian_loss, prior_loss, 0.5 * z.pow(2).mean() - logdets.mean()

    def reverse(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        guidance: float = 0,
        guide_what: str = 'ab',
        attn_temp: float = 1.0,
        annealed_guidance: bool = False,
        return_sequence: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        seq = [self.unpatchify(x)]
        x = x * self.var.sqrt()
        for block in reversed(self.blocks):
            x = block.reverse(x, y, guidance, guide_what, attn_temp, annealed_guidance)
            #print("x after reverse", x)
            seq.append(self.unpatchify(x))
        x = self.unpatchify(x)
        #print("final result x", x)
        #print("generated x mean, std", x.mean(), x.std())
        if not return_sequence:
            return x
        else:
            return seq
