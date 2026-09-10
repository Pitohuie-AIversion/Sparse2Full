import torch
import time
from models.temporal.components.conv_temporal import ConvTemporalPredictor
from models.temporal.components.video_swin import VideoSwinPredictor
try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    pass

def profile(model, name):
    # During testing T_out = 4, AR wrapper calls model T_out times with T_in=16
    x = torch.randn(1, 16, 1, 128, 128).cuda() # B, T_in, C, H, W
    model = model.cuda()
    model.eval()
    
    # params
    params = sum(p.numel() for p in model.parameters()) / 1e6
    
    # flops for 1 step
    flops_step = 0
    try:
        flops_obj = FlopCountAnalysis(model, x)
        flops_step = flops_obj.total() / 1e9 # GFLOPs
    except Exception as e:
        pass
    
    # For T_out=4, total FLOPs is 4 * flops_step (approx)
    flops_total = flops_step * 4
    
    # warmup
    with torch.no_grad():
        for _ in range(10):
            try:
                _ = model(x)
            except Exception as e:
                pass
            
    # latency for 1 step
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(50):
            _ = model(x)
    torch.cuda.synchronize()
    latency_step = (time.time() - start) / 50 * 1000
    
    # For T_out=4, latency is approx 4 * latency_step (AR inference is sequential)
    latency_total = latency_step * 4
    
    print(f'{name}: Params: {params:.3f}M, FLOPs: {flops_total:.2f}G, Latency: {latency_total:.2f}ms')

print("Profiling for T_out=4...")
m1 = ConvTemporalPredictor(in_channels=1, hidden_channels=46, out_channels=1, num_layers=2, kernel_size=3)
profile(m1, 'ConvLSTM')

m2 = VideoSwinPredictor(in_channels=1, hidden_dim=96, out_channels=1, window_size=(2, 7, 7), num_layers=2, num_heads=4)
profile(m2, 'VSWT')
