import torch
import time
from models.temporal.components.conv_temporal import ConvTemporalPredictor
from models.temporal.components.video_swin import VideoSwinPredictor
try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    pass

def profile(model, name):
    x = torch.randn(1, 16, 1, 128, 128).cuda() # B, T_in, C, H, W
    
    # In SequentialSpatiotemporalModel, temporal module gets input [B, T_in, C, H, W]
    # and processes it inside AR wrapper. Let's just profile a single forward pass of the core module
    # The core modules expect [B, T, C, H, W] for ConvLSTM or similar
    
    model = model.cuda()
    model.eval()
    
    # params
    params = sum(p.numel() for p in model.parameters()) / 1e6
    
    # flops
    flops = 0
    try:
        # ConvLSTM expects [B, T, C, H, W] typically, but let's check its forward signature
        flops_obj = FlopCountAnalysis(model, x)
        flops = flops_obj.total() / 1e9 # GFLOPs
    except Exception as e:
        print(f"FLOPs error: {e}")
        flops = -1
        
    # warmup
    with torch.no_grad():
        for _ in range(10):
            try:
                _ = model(x)
            except Exception as e:
                # If shape mismatch, adjust
                x = x.squeeze(2) if 'conv_temporal' in str(model.__class__) else x
                try:
                    _ = model(x)
                except Exception as e2:
                    print(f"Forward error: {e2}")
                    return
            
    # latency
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(50):
            _ = model(x)
    torch.cuda.synchronize()
    latency = (time.time() - start) / 50 * 1000
    
    print(f'{name}: Params: {params:.3f}M, FLOPs: {flops:.2f}G, Latency: {latency:.2f}ms')

print("Starting profiling...")
m1 = ConvTemporalPredictor(in_channels=1, hidden_channels=46, out_channels=1, num_layers=2, kernel_size=3)
profile(m1, 'ConvLSTM')

m2 = VideoSwinPredictor(in_channels=1, hidden_dim=96, out_channels=1, window_size=(2, 7, 7), num_layers=2, num_heads=4)
profile(m2, 'VSWT')
