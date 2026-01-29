
from omegaconf import OmegaConf

base = OmegaConf.create({
    "data": {
        "observation": {
            "crop": {
                "size": [64, 64]
            }
        }
    }
})

override = OmegaConf.from_dotlist(["data.observation.crop.size=48"])

merged = OmegaConf.merge(base, override)

print(f"Base: {base.data.observation.crop.size} (type: {type(base.data.observation.crop.size)})")
print(f"Merged: {merged.data.observation.crop.size} (type: {type(merged.data.observation.crop.size)})")
