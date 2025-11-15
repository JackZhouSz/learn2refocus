from huggingface_hub import HfApi
import os
#run with HF_TOKEN = your_hf_token before python_command
api = HfApi(token=os.getenv("HF_TOKEN"))
folders = ["/datasets/sai/focal-burst-learning/svd/checkpoints/checkpoint-200000"]
for folder in folders:
    api.upload_folder(
        folder_path=folder,
        repo_id="tedlasai/learn2refocus",
        repo_type="model",
        path_in_repo=os.path.basename(folder)
    )