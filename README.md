# Learning to Refocus with Video Diffusion Models

**SaiKiran Tedla, Zhoutong Zhang, Xuaner Zhang, Shumian Xin**  
Adobe Research & York University  

### 📄 [Paper (PDF)](https://dl.acm.org/doi/10.1145/3757377.3763873)  
### 🌐 [Project Page](https://learn2refocus.github.io)  
### 📂 [Data](https://ln5.sync.com/dl/dc6d99c50#ra9336yd-w9u958dw-w4s4xnjv-54qd4drj)
### 📂 [Checkpoints](https://huggingface.co/tedlasai/learn2refocus/tree/main)

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

This guide explains how to train and evaluate our **video diffusion model** for refocusing.

---

### 🔧 Environment Setup

```bash
conda env create -f setup/environment.yaml
conda activate refocus
python setup/download_svd_weights.py
python setup/download_checkpoints.py
```

- Install PyTorch and all dependencies listed in the YAML file.  
- Create a Weights & Biases (wandb) account for experiment tracking.  
  Update the wandb credentials in `training/configs` with your login info.  
- After running `setup/download_svd_weights.py`, you should have a folder named `svdh` at the project root containing the Stable Video Diffusion model weights.
- After running `setup/download_checkpoints.py`, you should have a folder named `checkpoints/checkpoints-200000` at the project root containing our finetuned weights.

---

### 🧪 Testing (In-the-Wild)

To test on real-world photos, place your images in the `photos/` directory and run (requires about 23-25GB memory depending on image sizes):

```bash
accelerate launch --config_file training/configs/accelerator_config.yaml \
  --multi_gpu training/svd_runner.py \
  --config training/configs/outside_photos.yaml
```
Results and visualizations will be saved to the directory specified by `output_dir` (default: `output_dir/outside_photos/`).  
Each output folder contains the generated focal stacks corresponding to the input image’s focal positions.

---

### 🧪 Testing (Focal Stack Dataset)

```bash
accelerate launch --config_file training/configs/accelerator_config.yaml --multi_gpu training/svd_runner.py --config training/configs/focal_stacks_test.yaml
```
---

Each output folder contains the generated focal stacks corresponding to the input image’s focal positions.
Results and visualizations will be saved to the directory specified by `output_dir` (default: `output_dir/focal_stacks_test/`).  

---

### Dataset

Our dataset includes raw DNGs from all five cameras, along with rendered images at both full and midsize resolutions. We provide ZIP archives for each portion of the dataset.

1. fullsize_dng  
   Contains ZIP files with the raw DNG images and associated capture metadata.  
   Each camera has five ZIP files due to the large file sizes.

2. fullsize_undistorted  
   Contains full-resolution rendered images for each camera, corrected for focal breathing and radial distortion to ensure consistent field of view (FOV).

3. midsize_undistorted  
   Contains the midsize (896x640) images on only the center camera. This is what we used for training and testing our model.

For training or testing, you only need the midsize_undistorted folder.  
Place this folder anywhere on your computer, and set the data_dir field in your YAML configuration to point to this path.

Example:
data_folder: "/datasets/sai/scenes_merged_midsize_undistorted"

---

### 🏋️‍♂️ Training

Set the following paths in your YAML config (feel free to change others paths to match your configuration):


```yaml
data_folder: 
splits_dir: 
wandb_project: "RefocusingSVD"
run_name: "focal_stacks_train"
```

To train our model, run:

```bash
accelerate launch --config_file training/configs/accelerator_config.yaml --multi_gpu training/svd_runner.py --config training/configs/focal_stacks_train.yaml
```

Checkpoint will be in `outputs/focal_stacks_train`

---

### 📜 Notes

- Checkpoints are available on the [project page](https://learn2refocus.github.io).  
- Dataset download links will be added soon.
- We utilize `extra/compute_metrics.py` to compute all metrics for this project.

---

### 📨 Contact

For questions or issues, please reach out through the [project page](https://learn2refocus.github.io) or contact [Sai Tedla](mailto:tedlasai@gmail.com).
