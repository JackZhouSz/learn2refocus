from huggingface_hub import snapshot_download
local_dir = "/sensei-fs-3/users/stedla/focal-burst-learning/svd/svdh"
snapshot_download(repo_id="stabilityai/stable-video-diffusion-img2vid",local_dir=local_dir)