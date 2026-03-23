import sys
from pathlib import Path
import torch

proj_root = Path(__file__).resolve().parents[2]
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from models.spatial.vit import VisionTransformer

def main():
    model = VisionTransformer(in_channels=1, out_channels=1, img_size=128,
                              patch_size=16, embed_dim=256, depth=2, num_heads=8,
                              decoder_embed_dim=256, decoder_depth=2, decoder_num_heads=8)
    x = torch.randn(2, 1, 128, 128)
    with torch.no_grad():
        y = model(x)
    print(tuple(y.shape))

if __name__ == "__main__":
    main()
