import torch
import torch.nn as nn
from models.spatial.swin_t import SwinTransformerTiny

def test_swin_t_bilinear_upsample():
    print("Testing SwinTransformerTiny with bilinear (pixelshuffle+conv) upsample...")
    
    # Configuration matching the use case
    img_size = 128
    patch_size = 4
    embed_dim = 96
    in_channels = 1
    out_channels = 1
    
    # Instantiate model with new upsample mode
    model = SwinTransformerTiny(
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        window_size=8, # Explicitly set window_size to 8 for 128/4=32 feature map
        final_upsample="bilinear"
    )
    
    # Create dummy input
    B = 2
    x = torch.randn(B, in_channels, img_size, img_size)
    
    # Forward pass
    try:
        y = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {y.shape}")
        
        # Check output shape
        assert y.shape == (B, out_channels, img_size, img_size), \
            f"Expected output shape {(B, out_channels, img_size, img_size)}, got {y.shape}"
            
        print("Forward pass successful!")
        
        # Check if output has grid artifacts (simple variance check on constant input)
        # Ideally, with random initialization, we can't say much, but we can check if it runs.
        # Let's check gradients flow
        loss = y.sum()
        loss.backward()
        print("Backward pass successful!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise e

if __name__ == "__main__":
    test_swin_t_bilinear_upsample()
