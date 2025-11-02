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

**Train Video Diffusion Model**
```bash
python main.py --config configs/stage1_base.yaml
```


**Notes:**
- Set `test: false` before training.  
- Update the dataset paths and WandB settings before running.  
- We provide configs for both base (blur-to-video) and fine-tuned (refocusing) stages.

---

### 🧪 Testing
Before testing, set `test: true` in your config and provide paths to pretrained checkpoints (available on the project page).

Example:
```bash
python main.py --config configs/test_refocus.yaml
```

This will:
- Load the pretrained model weights.  
- Generate refocused videos.  
- Compute PSNR, SSIM, and perceptual metrics.  

Results and visualizations will be saved under:
```
results/{experiment_name}/
```

---

### 📜 Notes

- All configs are initialized in test mode. Switch to training by setting `test: false`.  
- Checkpoints for all experiments (base + fine-tuned) are available on the project page.  
- Visualizations can be rendered as videos using:
  ```bash
  python utils/render_video.py --input results/example/
  ```
- The dataset includes both synthetic motion-blur and real captured examples for evaluation.

---

### 📨 Contact

For questions or issues, please reach out via the [project page](https://learn2refocus.github.io) or directly to [Sai Tedla](tedlasai@gmail.com).
