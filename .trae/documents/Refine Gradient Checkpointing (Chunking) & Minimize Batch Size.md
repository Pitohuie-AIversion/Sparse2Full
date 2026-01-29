The OOM persists even with `batch_size=4`. This confirms that the EDSR architecture (without pooling) at this depth (32 blocks, 128 channels) maintains extremely large activation maps (128x128 resolution) throughout the network, which consumes massive memory even with gradient checkpointing and small batches.

**Root Cause Analysis:**

* **Activation Memory**: For a 128x128 input with 128 channels, each feature map is \~8MB (FP32). A 32-block ResNet has \~64 conv layers.

* **Checkpointing Limitation**: Even with checkpointing, the *segments* between checkpoints still need to be stored. If my implementation of checkpointing (wrapping the *entire* body) is working, it should trade compute for memory effectively. The fact that it's still OOMing suggests either:

  1. Checkpointing isn't actually engaging (due to `model.training` flag or other reasons).
  2. The overhead of the un-checkpointed parts (Head/Tail + Optimizer states) is still too high.
  3. The single checkpoint wrapper around the *whole* body might be causing issues with the autograd graph size or not freeing intermediate tensors as aggressively as chunked checkpointing.

**Revised Plan:**

1. **Stop Current Run**: Kill the failing process.
2. **Optimize Checkpointing Strategy**: Instead of wrapping the *entire* body in one `checkpoint` call (which might be inefficient or problematic for very deep nets), I will modify `models/spatial/edsr.py` to use **segment-based checkpointing**. I will split the 32 blocks into chunks (e.g., 4 chunks of 8 blocks) and checkpoint each chunk. This is the standard way to handle deep ResNets.
3. **Further Reduce Batch Size**: Drop `batch_size` to **2**. This is the last resort before declaring the model "too big for this GPU" (which shouldn't be true for 10M params on a 46GB card).
4. **Relaunch**: Start the training again.

