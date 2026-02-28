
from omegaconf import OmegaConf

try:
    overrides = ["++h_params.crop_size=[48,48]", "++h_params.task=Crop"]
    conf = OmegaConf.from_dotlist(overrides)
    print("Success with ++")
    print(conf)
except Exception as e:
    print(f"Failed with ++: {e}")

try:
    overrides = ["h_params.crop_size=[48,48]", "h_params.task=Crop"]
    conf = OmegaConf.from_dotlist(overrides)
    print("Success without ++")
    print(conf)
except Exception as e:
    print(f"Failed without ++: {e}")
