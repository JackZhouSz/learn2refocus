from huggingface_hub import snapshot_download
from diffusers import StableDiffusionPipeline # or whichever pipeline applies

save_dir = "/datasets/sai/focal-burst-learning/svd/svdh"  

# 1. Download the full model repo (weights + config + assets)

local_dir = snapshot_download(
    repo_id="stabilityai/stable-video-diffusion-img2vid",
    revision="main",
    local_dir=save_dir,
    local_dir_use_symlinks=False  # ensures files are fully copied, not symlinked
)

print(f"Model downloaded to: {local_dir}")