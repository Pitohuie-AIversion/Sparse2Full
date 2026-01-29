import torch
import gc
import sys

def clear_cache():
    if not torch.cuda.is_available():
        print("CUDA is not available. Nothing to clear.")
        return

    print("Clearing GPU cache...")
    try:
        # Get initial state
        initial_allocated = torch.cuda.memory_allocated()
        initial_reserved = torch.cuda.memory_reserved()
        
        # Perform cleanup
        gc.collect()
        torch.cuda.empty_cache()
        
        # Get final state
        final_allocated = torch.cuda.memory_allocated()
        final_reserved = torch.cuda.memory_reserved()
        
        print(f"Memory Released: {(initial_reserved - final_reserved) / 1024**2:.2f} MB")
        print(f"Current Allocated: {final_allocated / 1024**2:.2f} MB")
        print(f"Current Reserved: {final_reserved / 1024**2:.2f} MB")
        print("Done.")
        
    except Exception as e:
        print(f"Error clearing cache: {e}")

if __name__ == "__main__":
    clear_cache()
