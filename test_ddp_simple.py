import torch
import os
import sys

def main():
    print(f"--- Process {os.getpid()} ---")
    print(f"Python: {sys.executable}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
    print(f"RANK: {os.environ.get('RANK')}")
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
    
    is_avail = torch.cuda.is_available()
    print(f"CUDA Available: {is_avail}")
    print(f"Device Count: {torch.cuda.device_count()}")
    
    if is_avail:
        try:
            print(f"Current Device: {torch.cuda.current_device()}")
            print(f"Device Name: {torch.cuda.get_device_name(0)}")
            t = torch.tensor([1.0]).cuda()
            print(f"Tensor on GPU: {t}")
        except Exception as e:
            print(f"CUDA Error: {e}")
    else:
        print("Running on CPU")
    print("-------------------------")

if __name__ == "__main__":
    main()
