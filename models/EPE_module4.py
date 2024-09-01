import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedPatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, bias_offsets=[(0, 4), (4, 0), (0, -4), (-4, 0)]):
        super(EnhancedPatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_c = in_c
        self.embed_dim = embed_dim
        self.bias_offsets = bias_offsets
        self.num_patches = (img_size // patch_size) ** 2

        # Linear projection layer 
        self.proj = nn.Linear(patch_size * patch_size * in_c * (1 + len(bias_offsets)), embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # 对图像进行偏置处理 Apply bias offsets and stack images 
        biased_images = [x]
        for dx, dy in self.bias_offsets:
            # Create a zero-filled tensor of the same shape as x
            zero_filled = torch.zeros_like(x)
            # Roll the image
            biased_image = torch.roll(x, shifts=(dx, dy), dims=(2, 3))
            
            # Determine the mask for zeroing out the invalid regions
            mask = torch.ones_like(x, dtype=torch.bool)
            if dx > 0:
                mask[:, :, :dx, :] = 0
            elif dx < 0:
                mask[:, :, dx:, :] = 0
            if dy > 0:
                mask[:, :, :, :dy] = 0
            elif dy < 0:
                mask[:, :, :, dy:] = 0
            
            # Apply the mask to zero out the invalid regions
            biased_image = torch.where(mask, biased_image, zero_filled)
            biased_images.append(biased_image)
        
        # 图像堆叠
        stacked_images = torch.cat(biased_images, dim=1)  # Concatenate along the channel dimension

        # 分块序列化 Extract patches 
        patches = stacked_images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C * (1 + len(self.bias_offsets)), -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, self.patch_size * self.patch_size * C * (1 + len(self.bias_offsets)))

        # 线性映射和层归一化 Linear projection and normalization
        patches = self.proj(patches)
        patches = self.norm(patches)

        return patches

# Example usage
if __name__ == "__main__":
    img_size = 224
    patch_size = 16
    in_c = 3
    embed_dim = 768
    batch_size = 1

    # Create a random image tensor with shape (batch_size, in_c, img_size, img_size)
    img = torch.randn(batch_size, in_c, img_size, img_size)

    # Initialize the EnhancedPatchEmbedding module
    epe = EnhancedPatchEmbedding(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)

    # Forward pass
    patches = epe(img)
    print(patches.shape)  # Expected output: (batch_size, num_patches, embed_dim)