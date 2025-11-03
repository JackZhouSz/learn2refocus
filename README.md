# Learning to Refocus with Video Diffusion Models

**SaiKiran Tedla, Zhoutong Zhang, Xuaner Zhang, Shumian Xin**  
Adobe Research & York University  

📄 [Paper (PDF)]()  
🌐 [Project Page ](https://learn2refocus.github.io)
📂 [Data and Checkpoints](https://cp.sync.com/files/66691398045749?view=list)
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

This guide explains how to train and evaluate our **video diffusion model** for refocusing from a single motion-blurred image.

---

### 🔧 Environment Setup

```bash
conda env create -f setup/environment.yml
conda activate refocus
python setup/download_svd_weights.py
```

- Install PyTorch and all dependencies listed in the YAML file.  
- Create a Weights & Biases (wandb) account for experiment tracking.  
  Update the wandb credentials in `training/configs` with your login info.  
- After running `setup/download_svd_weights.py`, you should have a folder named `svdh` at the project root containing the Stable Video Diffusion model weights.

---

### 📂 Dataset

**Dataset Description — Coming Soon**

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

### 🧪 Testing (In-the-Wild)

To test on real-world photos, place your images in the `photos/` directory and run:

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

### 📜 Notes

- Checkpoints (base + fine-tuned) are available on the [project page](https://learn2refocus.github.io).  
- Dataset download links will be added soon.
- We utilize `extra/compute_metrics.py` to compute all metrics for this project.

---

### 📨 Contact

For questions or issues, please reach out through the [project page](https://learn2refocus.github.io) or contact [Sai Tedla](mailto:tedlasai@gmail.com).
