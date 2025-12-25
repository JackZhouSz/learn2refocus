#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to fine-tune Stable Video Diffusion."""

import math
import os
from torch.utils.data import Dataset
import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection
from validation import valid_net
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.utils import check_min_version
import argparse
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")
import numpy as np
import torch
import os
import glob



def parse_config(config_path="config.yaml"):
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # handle distributed training rank
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != config.get("local_rank", -1):
        config["local_rank"] = env_local_rank

    # default fallback: non_ema_revision = revision
    if config.get("non_ema_revision") is None:
        config["non_ema_revision"] = config.get("revision")

    return config

def parse_args():
    parser = argparse.ArgumentParser(description="SVD Training Script")
    parser.add_argument(
        "--config",
        type=str,
        default="/datasets/sai/focal-burst-learning/svd/training/configs/outside_photos.yaml",
        help="Path to the config file.",
    )

    args = parser.parse_args()
    

    # load YAML and merge into args
    config = parse_config(args.config)
    # combine yaml + command line args (command line has priority)
    for k, v in vars(args).items():
        if v is not None:
            config[k] = v

    # convert dict to argparse.Namespace for downstream compatibility
    args = argparse.Namespace(**config)

    print("OUTPUT DIR: ", args.output_dir)
    return args

 

def find_scale(height, width):
    max_pixels = 500000

    # Start with no scaling
    scale = 1.0

    while True:
        # Calculate the scaled dimensions
        scaled_height = math.floor((height * scale) / 64) * 64
        scaled_width = math.floor((width * scale) / 64) * 64

        # Check if the scaled dimensions meet the pixel constraint
        if scaled_height * scaled_width <= max_pixels:
            return scaled_height, scaled_width

        # Reduce the scale slightly
        scale -= 0.01

class OutsidePhotosDataset(Dataset):
    def __init__(self, data_folder, width=1024, height=576, sample_frames=9):
        self.data_folder = data_folder
        self.scenes = sorted(glob.glob(os.path.join(data_folder, "*"))) 

        #get images that end in .JPG,.jpg, .png
        self.scenes = [scene for scene in self.scenes if scene.endswith(".JPG") or scene.endswith(".jpg") or scene.endswith(".png") or scene.endswith(".jpeg") or scene.endswith(".JPG")]
        #make each scene a tuple anf for each scene, put it 9 times in the tuple - tuple should look like (scene_name, idx (0-8))

        self.scenes = [(scene, idx) for scene in self.scenes for idx in range(6,7)]


        self.num_scenes = len(self.scenes)
        self.width = width
        self.height = height
        self.sample_frames = sample_frames
        self.icc_profiles = [None]*self.num_scenes
    
    def __len__(self):
        return self.num_scenes
    
    def __getitem__(self, idx):
        #get the scene and the index
        #create an empty tensor to store the pixel values and place the scene in the tensor (load and resize the image)

        scene, focal_stack_num = self.scenes[idx]
        from PIL import Image
        with Image.open(scene) as img:

            self.icc_profiles[idx] = img.info.get("icc_profile")
            icc_profile = img.info.get("icc_profile")
            if icc_profile is None:
                icc_profile = "none"
            original_pixels = torch.from_numpy(np.array(img)).float().permute(2,0,1)
            original_pixels = original_pixels / 255
            width, height = img.size
            scaled_width, scaled_height = find_scale(width, height)

            img_resized = img.resize((scaled_width, scaled_height))
            img_tensor = torch.from_numpy(np.array(img_resized)).float()
            img_normalized = img_tensor / 127.5 - 1
            img_normalized = img_normalized.permute(2, 0, 1)

            pixels = torch.zeros((self.sample_frames, 3, scaled_height, scaled_width))
            pixels[focal_stack_num] = img_normalized
        
            return {"pixel_values": pixels, "idx": idx//9, "focal_stack_num": focal_stack_num, "original_pixel_values": original_pixels, 'icc_profile': icc_profile}

def main():
    args = parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # inference-only modules
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant="fp16"
    )

    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
        autocast_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        autocast_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
        autocast_dtype = None

    image_encoder.requires_grad_(False).to(device, dtype=weight_dtype)
    vae.requires_grad_(False).to(device, dtype=weight_dtype)

    # ---- load UNet from checkpoint root (this reads unet/config.json + diffusion_pytorch_model.safetensors)
    ckpt_root = args.load_from_checkpoint  # e.g. ".../checkpoint-200000"
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        ckpt_root, subfolder="unet"
    ).to(device)

    # data
    val_dataset = OutsidePhotosDataset(data_folder=args.data_folder, sample_frames=args.num_frames)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    global_step = int(os.path.basename(ckpt_root).split("-")[1])

    unet.eval(); image_encoder.eval(); vae.eval()
    with torch.no_grad():
        if autocast_dtype is None:
            valid_net(args, val_dataset, val_dataloader, unet, image_encoder, vae, 0, global_step, weight_dtype, device)
        else:
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                valid_net(args, val_dataset, val_dataloader, unet, image_encoder, vae, 0, global_step, weight_dtype, device)


if __name__ == "__main__":
    main()


