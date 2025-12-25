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
from simplified_validation import valid_net
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



def parse_args():
    parser = argparse.ArgumentParser(description="SVD Training Script")
    parser.add_argument(
        "--config",
        type=str,
        default="/datasets/sai/focal-burst-learning/svd/training/configs/outside_photos.yaml",
        help="Path to the config file.",
    )
    #seed should be int that default 0 (optional)

    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to image input or directory containing input images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="A seed for reproducible training.",
    )

    parser.add_argument(
        "--learn2refocus_hf_repo_path",
        type=str,
        default="tedlasai/learn2refocus",
        help="hf repo containing the weight files",
    )

    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="stabilityai/stable-video-diffusion-img2vid",
        help="repo id or path for pretrained StableVideo Diffusion model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/simple_inference",
        help="path to output",
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        help="number of DDPM steps",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="inference device",
    )


    args = parser.parse_args()

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

def convert_to_batch(image, input_focal_position, sample_frames=9):
    scene, focal_stack_num = image, input_focal_position
    from PIL import Image
    with Image.open(scene) as img:

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

        pixels = torch.zeros((1, sample_frames, 3, scaled_height, scaled_width))
        pixels[0, focal_stack_num] = img_normalized
        
        name = os.path.splitext(os.path.basename(scene))[0]
        return {"pixel_values": pixels, "focal_stack_num": focal_stack_num, "original_pixel_values": original_pixels, 'icc_profile': icc_profile, "name": name}

def main():
    args = parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # inference-only modules
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_path, subfolder="image_encoder"
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_path, subfolder="vae",  variant="fp16"
    )

    weight_dtype = torch.float32
    image_encoder.requires_grad_(False).to(device, dtype=weight_dtype)
    vae.requires_grad_(False).to(device, dtype=weight_dtype)

    # ---- load UNet from checkpoint root (this reads unet/config.json + diffusion_pytorch_model.safetensors)
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.learn2refocus_hf_repo_path, subfolder="checkpoint-200000/unet"
    ).to(device)

    batch = convert_to_batch(args.image_path, input_focal_position=6)

    unet.eval(); image_encoder.eval(); vae.eval()
    with torch.no_grad():
        valid_net(args, batch, unet, image_encoder, vae, 0, weight_dtype, device, num_inference_steps=args.num_inference_steps)

if __name__ == "__main__":
    main()


