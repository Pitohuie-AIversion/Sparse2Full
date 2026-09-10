import sys
from pathlib import Path
import torch
import numpy as np

sys.path.append(str(Path.cwd()))
from tools.training.train_real_data_ar import RealDataARTrainer
from ops.metrics import compute_all_metrics

# A simple wrapper to test the actual AR rollout
def evaluate_rollout(ckpt_path, config_path, t_in):
    trainer = RealDataARTrainer(
        config_path=config_path,
        overrides=[
            "testing.test_only=true",
            "data.dataloader.test_batch_size=1",
            "data.T_out=20",
            "data.stride=1",
            "data.end=1000",
            "data.time_step_end=1000"
        ]
    )
    if not trainer.load_checkpoint(ckpt_path):
        print("Failed to load checkpoint")
        return None

    trainer.get_model().eval()
    all_rel_l2 = []
    
    with torch.no_grad():
        for i, batch in enumerate(trainer.test_loader):
            if i >= 5: break # test 5 samples to get a smooth curve
            
            # Replicating the logic inside trainer.test_epoch() manually since _prepare_batch might be in a parent class
            if isinstance(batch, dict):
                input_seq = batch.get('input_sequence')
                target_seq = batch.get('target_sequence')
            else:
                input_seq, target_seq = batch[0], batch[1]
                
            input_seq = input_seq.to(trainer.device)
            target_seq = target_seq.to(trainer.device)
            
            B, T, C, H, W = target_seq.shape
            
            try:
                if hasattr(trainer.get_model(), 'rollout_inference'):
                    pred_seq = trainer.get_model().rollout_inference(input_seq, steps=T, step_by_step=True)
                else:
                    pred_seq = trainer.get_model()(input_seq, T)
                    if isinstance(pred_seq, dict): pred_seq = pred_seq['final_pred']
                
                step_rel_l2 = []
                for t in range(T):
                    pred_t = pred_seq[:, t:t+1]
                    targ_t = target_seq[:, t:t+1]
                    
                    metrics = compute_all_metrics(
                        pred_t, targ_t, 
                        norm_stats=trainer.norm_stats, 
                        image_size=(H,W), 
                        include_freq_metrics=False
                    )
                    step_rel_l2.append(float(metrics['rel_l2']))
                
                all_rel_l2.append(step_rel_l2)
            except Exception as e:
                print(f"Sample {i} failed: {e}")
            
    if all_rel_l2:
        avg_rel_l2 = np.mean(all_rel_l2, axis=0)
        print("Avg Rel L2 per step:", avg_rel_l2)
        return avg_rel_l2
    return None

if __name__ == '__main__':
    print("Testing UNet...")
    try:
        unet_err = evaluate_rollout(
            './runs_drd_paper/AR-DR2D-UNet-SRx4-10M-300ep/best.ckpt',
            './runs_drd_paper/AR-DR2D-UNet-SRx4-10M-300ep/config_merged.yaml',
            1
        )
        if unet_err is not None:
            np.save('thesis_paper/figures/rollout/unet_rollout.npy', unet_err)
    except Exception as e:
        print("Failed:", e)
        
    print("\nTesting Ours (EDSR Stage 2)...")
    try:
        edsr_err = evaluate_rollout(
            './runs_drd_paper/AR-DR2D-Stage2-VideoSwin-SRx4-model_unknown-s2025-20260116/best.ckpt',
            './runs_drd_paper/AR-DR2D-Stage2-VideoSwin-SRx4-model_unknown-s2025-20260116/config_merged.yaml',
            10
        )
        if edsr_err is not None:
            np.save('thesis_paper/figures/rollout/edsr_rollout.npy', edsr_err)
    except Exception as e:
        print("Failed:", e)
