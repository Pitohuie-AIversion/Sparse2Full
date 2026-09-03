import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path.cwd()))
from tools.training.train_real_data_ar import RealDataARTrainer

def load_and_predict(ckpt_path, config_path, t_in=1, t_out=30, is_sequential=False):
    trainer = RealDataARTrainer(
        config_path=config_path,
        overrides=[
            "testing.test_only=true",
            "data.dataloader.test_batch_size=1",
            f"data.T_out={t_out}",
            "data.stride=1",
            "data.end=1000",
            "data.time_step_end=1000"
        ]
    )
    if not trainer.load_checkpoint(ckpt_path):
        print(f"Failed to load checkpoint {ckpt_path}")
        return None

    trainer.get_model().eval()
    with torch.no_grad():
        for i, batch in enumerate(trainer.test_loader):
            if i < 2: continue
            
            if isinstance(batch, dict):
                input_seq = batch['input_sequence'].to(trainer.device)
                target_seq = batch['target_sequence'].to(trainer.device)
            else:
                input_seq, target_seq = batch[0].to(trainer.device), batch[1].to(trainer.device)
            
            model = trainer.get_model()
            raw_model = model.module if hasattr(model, 'module') else model
            
            pred_seq = []
            
            if is_sequential:
                # Expects T_in frames (e.g., 10)
                # Ensure input_seq has exactly t_in frames
                curr_in = input_seq[:, -t_in:]
            else:
                # Expects 1 frame
                curr_in = input_seq[:, -1:]
            
            for t in range(t_out):
                if is_sequential:
                    # x should be [B, T_in, C, H, W]
                    x = curr_in
                    # The sequential model in the project actually has forward(x, target)
                    # Let's just call model(x)
                    pred_dict = model(x)
                    if isinstance(pred_dict, dict): 
                        pred = pred_dict['final_pred'] # [B, T_out, C, H, W]
                    else:
                        pred = pred_dict
                    # Take the first (and only) predicted frame
                    pred = pred[:, 0] # [B, C, H, W]
                    
                    if hasattr(raw_model.spatial_module, 'out_channels') and pred.shape[1] > raw_model.spatial_module.out_channels:
                        pred = pred[:, :raw_model.spatial_module.out_channels]
                    
                    # Update window
                    curr_in = torch.cat([curr_in[:, 1:], pred.unsqueeze(1)], dim=1)
                else:
                    x = curr_in[:, 0]
                    if hasattr(raw_model, 'in_channels') and x.shape[1] < raw_model.in_channels:
                        pad_c = raw_model.in_channels - x.shape[1]
                        padding = torch.zeros(x.size(0), pad_c, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
                        x = torch.cat([x, padding], dim=1)
                        
                    pred = model(x)
                    if isinstance(pred, dict): pred = pred['final_pred']
                    if hasattr(raw_model, 'out_channels') and pred.shape[1] > raw_model.out_channels:
                        pred = pred[:, :raw_model.out_channels]
                    
                    curr_in = pred.unsqueeze(1)
                
                pred_seq.append(pred)
            
            pred_seq = torch.stack(pred_seq, dim=1)
            
            std = trainer.norm_stats['std'] if trainer.norm_stats else 1.0
            mean = trainer.norm_stats['mean'] if trainer.norm_stats else 0.0
            if isinstance(std, torch.Tensor): std = std.cpu().numpy()
            if isinstance(mean, torch.Tensor): mean = mean.cpu().numpy()
            
            pred_np = pred_seq.cpu().numpy()[0, :, 0] * std + mean
            target_np = target_seq.cpu().numpy()[0, :, 0] * std + mean
            
            return pred_np, target_np

def plot_qualitative(unet_pred, ours_pred, gt, time_steps=[0, 10, 20, 29]):
    fig, axes = plt.subplots(4, len(time_steps), figsize=(2.5 * len(time_steps), 8))
    
    row_labels = ["GT", "UNet", "Seq-EDSR\n(Backbone)", "Error\n(Ours - GT)"]
    
    vmin = min(np.min(gt), np.min(unet_pred), np.min(ours_pred))
    vmax = max(np.max(gt), np.max(unet_pred), np.max(ours_pred))
    
    for c, t in enumerate(time_steps):
        # GT
        ax = axes[0, c]
        ax.imshow(gt[t].squeeze(), cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f"t={t}")
        ax.axis('off')
        
        # UNet
        ax = axes[1, c]
        ax.imshow(unet_pred[t].squeeze(), cmap='viridis', vmin=vmin, vmax=vmax)
        ax.axis('off')
        
        # Ours
        ax = axes[2, c]
        ax.imshow(ours_pred[t].squeeze(), cmap='viridis', vmin=vmin, vmax=vmax)
        ax.axis('off')
        
        # Error
        ax = axes[3, c]
        err_ours = np.abs(ours_pred[t].squeeze() - gt[t].squeeze())
        im_err = ax.imshow(err_ours, cmap='magma', vmin=0, vmax=vmax*0.5) # adjusted vmax for error
        ax.axis('off')
    
    for r in range(4):
        axes[r, 0].text(-0.1, 0.5, row_labels[r], transform=axes[r, 0].transAxes, 
                        ha='right', va='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("thesis_paper/figures/rollout/qualitative_multistep.png", dpi=600, bbox_inches='tight')
    print("Saved qualitative_multistep.png")

if __name__ == '__main__':
    print("Running UNet...")
    unet_res = load_and_predict(
        './runs_drd_paper/AR-DR2D-UNet-SRx4-10M-300ep/best.ckpt',
        './runs_drd_paper/AR-DR2D-UNet-SRx4-10M-300ep/config_merged.yaml',
        t_in=1, t_out=30, is_sequential=False
    )
    
    print("Running Ours (Seq-EDSR)...")
    ours_res = load_and_predict(
        './runs_drd_paper/AR-DR2D-E2E-StrictStride10-EDSR-VideoSwin-SRx4-model_unknown-s2025-20260122/best.ckpt',
        './runs_drd_paper/AR-DR2D-E2E-StrictStride10-EDSR-VideoSwin-SRx4-model_unknown-s2025-20260122/config_merged.yaml',
        t_in=10, t_out=30, is_sequential=True
    )
    
    if unet_res and ours_res:
        unet_pred, gt = unet_res
        ours_pred, _ = ours_res
        plot_qualitative(unet_pred, ours_pred, gt)
