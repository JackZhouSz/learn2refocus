# Learning to Refocus with Video Diffusion Models

**SaiKiran Tedla, Zhoutong Zhang, Xuaner Zhang, Shumian Xin**  
Adobe Research & York University  

📄 [Paper (PDF)](https://arxiv.org/pdf/2503.xxxxx)  
🌐 [Project Page & Dataset](https://refocus-diffusion.github.io)

---

## 📌 Citation

If you use our dataset, code, or model, please cite:

```bibtex
@inproceedings{Tedla2025Refocus,
  title={{Learning to Refocus with Video Diffusion Models}},
  author={{Tedla, SaiKiran and Zhang, Zhoutong and Zhang, Xuaner and Xin, Shumian}},
  booktitle={{Proceedings of the ACM SIGGRAPH Asia Conference}},
  year={{2025}}
}
```

---

## 🚀 Getting Started

This section describes how to train and evaluate our **video diffusion model** for refocusing from a single motion-blurred image.

---

### 🔧 Environment Setup

```bash
conda env create -f setup/environment.yml
conda activate refocus
python setup/download_svd_weights.py
```

- Install PyTorch and dependencies listed in the YAML file.  
- Create a Weights & Biases (wandb) account for experiment tracking.  
  Update the `wandb` settings in the config files under `training/configs` with your wandb login information.  
- svdh (the Stable Video Diffusion model weights) should be in a folder called svdh at the top of your directory after running `setup_download_svd_weights.py`

---

### 📂 Dataset

- Description Coming


### 🏋️‍♂️ Training

**Training our Model**
```bash

accelerate launch --config_file training/configs/accelerator_config.yaml --multi
_gpu training/svd_runner.py --config training/configs/focal_stacks_train.yaml 
```
---

Set appropriate paths in yaml to configure "data_folder: "/datasets/sai/scenes_merged"
pretrained_model_name_or_path: "/datasets/sai/focal-burst-learning/svd/svdh"
load_from_checkpoint: null
output_dir: "/datasets/sai/focal-burst-learning/svd/outputs/focal_stacks_train"
splits_dir: "/datasets/sai/focal-burst-learning" #all split.pkl files are stored here
wandb_project: "RefocusingSVD"
run_name: "focal_stacks_train""

### 🧪 Testing (In-the-Wild)
Testing on in-the-wild photos.

Again set

data_folder: "/datasets/sai/focal-burst-learning/svd/photos"
pretrained_model_name_or_path: "/datasets/sai/focal-burst-learning/svd/svdh"
load_from_checkpoint: "/datasets/sai/focal-burst-learning/svd/checkpoints/checkpoint-200000"
output_dir: "/datasets/sai/focal-burst-learning/svd/outputs/outside_photos"
wandb_project: "RefocusingSVD"
run_name: "outside_photos"

Place images in `\photos` directory. Then run,

Example:
```
accelerate launch --config_file training/configs/accelerator_config.yaml --multi_gpu training/svd_runner.py --config training/configs/outside_photos.yaml
```
Results and visualizations will be saved under the output_dir set in the directory.


### 🧪 Testing (Focal Stack Dataset)

accelerate launch --config_file training/configs/accelerator_config.yaml --multi_gpu training/svd_runner.py --config training/configs/outside_photos.yaml


---

### 📜 Notes

- Checkpoints for all experiments (base + fine-tuned) are available on the project page.  
- Dataest available on project page.


### 📨 Contact

For questions or issues, please reach out via the [project page](https://learn2refocus.github.io) or directly to [Sai Tedla](tedlasai@gmail.com).
