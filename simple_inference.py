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
import numpy as np
import torch
import torch.utils.checkpoint
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.utils import check_min_version
from simple_pipeline import StableVideoDiffusionPipeline
from PIL import Image
from diffusers.utils import export_to_video


import argparse
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")
import numpy as np
import torch
import os


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


def convert_to_batch(img, input_focal_position, sample_frames=9):
    focal_stack_num = input_focal_position
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
    
    return {"pixel_values": pixels, "focal_stack_num": focal_stack_num, "original_pixel_values": original_pixels, 'icc_profile': icc_profile}


def inference_on_image(args, batch, pipeline, device):



    pipeline.set_progress_bar_config(disable=True)
    num_frames = 9 

    pixel_values = batch["pixel_values"].to(device)
    focal_stack_num = batch["focal_stack_num"]

    svd_output, _ = pipeline(
        pixel_values,
        height=pixel_values.shape[3],
        width=pixel_values.shape[4],
        num_frames=num_frames,
        decode_chunk_size=8,
        motion_bucket_id=0,
        min_guidance_scale=1.5,
        max_guidance_scale=1.5,
        fps=7,
        noise_aug_strength=0,
        focal_stack_num = focal_stack_num,
        num_inference_steps=args.num_inference_steps,
    )
    video_frames = svd_output.frames[0]
    video_frames_normalized = video_frames*0.5 + 0.5
    video_frames_normalized = torch.clamp(video_frames_normalized,0,1)
    video_frames_normalized = video_frames_normalized.permute(1,0,2,3)
    video_frames_normalized = torch.nn.functional.interpolate(video_frames_normalized, ((pixel_values.shape[3]//2)*2, (pixel_values.shape[4]//2)*2), mode='bilinear')


    return video_frames_normalized, focal_stack_num
    # run inference
def write_output(output_dir, frames, focal_stack_num, icc_profile):


    print("Validation images will be saved to ", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("Frames shape: ", frames.shape)
    export_to_video(frames.permute(0,2,3,1).cpu().numpy(), os.path.join(output_dir, "stack.mp4"), fps=5)

    #save images
    for i in range(9):
        #use Pillow to save images
        img = Image.fromarray((frames[i].permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
        if icc_profile != "none":
            img.info['icc_profile'] = icc_profile
        img.save(os.path.join(output_dir, f"frame_{i}.png"))

def load_model(args):
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
    unet.eval(); image_encoder.eval(); vae.eval()


    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        args.pretrained_model_path,
        unet=unet,
        image_encoder=image_encoder,
        vae=vae,
        torch_dtype=weight_dtype,
    )
    return pipeline, device


def main():
    args = parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    pipeline, device = load_model(args)

    with torch.no_grad():
        img = Image.open(args.image_path)
        batch = convert_to_batch(img, input_focal_position=6)
        output_frames, focal_stack_num = inference_on_image(args, batch, pipeline, device)
        name = os.path.splitext(os.path.basename(args.image_path))[0]
        val_save_dir = os.path.join(args.output_dir, "validation_images", name)
        write_output(val_save_dir, output_frames, focal_stack_num, batch['icc_profile'])



if __name__ == "__main__":
    main()


