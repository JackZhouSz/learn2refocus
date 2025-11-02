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

from datetime import datetime
import logging
import math
import os
import shutil
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import RandomSampler
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from validation import valid_net
import diffusers
from svd_pipeline import StableVideoDiffusionPipeline
from diffusers.models.lora import LoRALinearLayer
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler, UNetSpatioTemporalConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available
from utils import parse_args, FocalStackDataset, OutsidePhotosDataset, rand_log_normal, tensor_to_vae_latent, load_image, _resize_with_antialiasing, encode_image, get_add_time_ids
import wandb
import random
from random import choices
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")

import numpy as np
import PIL.Image
import torch
from typing import Callable, Dict, List, Optional, Union
import os


    
def main():
    args = parse_args()

    #SETUP PYTORCH CUDA - Without this I have memory overflow
    #pytorch 2.4.1 is important for this to work
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if not is_wandb_available():
        raise ImportError(
            "Make sure to install wandb if you want to use it for logging during training.")
    import wandb


    currentSecond= datetime.now().second
    currentMinute = datetime.now().minute
    currentHour = datetime.now().hour
    currentDay = datetime.now().day
    currentMonth = datetime.now().month
    currentYear = datetime.now().year


    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs]
    )

    accelerator.init_trackers(
        project_name=args.wandb_project,
        init_kwargs={"wandb": { "name" : args.run_name}}
    )

    generator = torch.Generator(
        device=accelerator.device).manual_seed(args.seed)

  


    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load img encoder, tokenizer and models.
    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant="fp16")
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.pretrained_model_name_or_path if args.pretrain_unet is None else args.pretrain_unet,
        subfolder="unet",
        low_cpu_mem_usage=True,
        variant="fp16"
    )

    #unet= UNetSpatioTemporalConditionModel()

    # Freeze vae and image_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move image_encoder and vae to gpu and cast to weight_dtype
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

        # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(
        ), model_cls=UNetSpatioTemporalConditionModel, model_config=unet.config, use_ema_warmup=True, inv_gamma=1, ower=3/4)



    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))
                

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):

            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(
                    input_dir, "unet_ema"), UNetSpatioTemporalConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNetSpatioTemporalConditionModel.from_pretrained(
                    input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model
            
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps *
            args.per_gpu_batch_size * accelerator.num_processes
        )

    optimizer_cls = torch.optim.AdamW

    parameters_list = []


    # Customize the parameters that need to be trained; if necessary, you can uncomment them yourself.
    for name, param in unet.named_parameters():
        parameters_list.append(param)
        if 'temporal_transformer_block' in name: #or 'conv_norm_out' in name or 'conv_out' in name or 'conv_in' in name or 'spatial_res_block' in name or 'up_block' in name:
            parameters_list.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False
    zero_latent = 0



    optimizer = optimizer_cls(
        parameters_list,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # DataLoaders creation:
    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes

    if args.photos:
        train_dataset = OutsidePhotosDataset(data_folder=args.data_folder, sample_frames=args.num_frames)
        val_dataset = OutsidePhotosDataset(data_folder=args.data_folder, sample_frames=args.num_frames)
    else:
        train_dataset = FocalStackDataset(args.data_folder,  args.splits_dir, sample_frames=args.num_frames, split="train")
        val_dataset = FocalStackDataset(args.data_folder, args.splits_dir, sample_frames=args.num_frames, split="val" if not args.test else "test")
    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
        drop_last=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )




    # Prepare everything with our `accelerator`.
    unet, optimizer, lr_scheduler, train_dataloader, val_dataloader = accelerator.prepare(
     unet, optimizer, lr_scheduler, train_dataloader, val_dataloader
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)
        

        
    # attribute handling for models using DDP
    if isinstance(unet, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        unet = unet.module

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("SVDXtend", config=vars(args))

    # Train!
    total_batch_size = args.per_gpu_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0


    # Potentially load in the weights and states from a previous save
    if args.load_from_checkpoint:

        path = args.load_from_checkpoint
#
        if path is None:
            accelerator.print(
                f"Checkpoint '{args.load_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.load_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(path, strict=False)
            global_step = int(os.path.basename(path).split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch

            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # print("ARGS PHOTOS: ", args.photos)
    # if args.photos:
    #     print("MAKING OUTSIDE PHOTOS DATASET")
    #     train_dataset = OutsidePhotosDataset(data_folder=args.data_folder, sample_frames=args.num_frames)
    #     val_dataset = OutsidePhotosDataset(data_folder=args.data_folder, sample_frames=args.num_frames)

    #     sampler = RandomSampler(train_dataset)
    #     train_dataloader = torch.utils.data.DataLoader(
    #         train_dataset,
    #         sampler=sampler,
    #         batch_size=args.per_gpu_batch_size,
    #         num_workers=args.num_workers,
    #         drop_last=True
    #     )
    #     val_dataloader = torch.utils.data.DataLoader(
    #         val_dataset,
    #         batch_size=args.per_gpu_batch_size,
    #         num_workers=args.num_workers,
    #         shuffle=False,
    #     )

    #     train_dataloader, val_dataloader = accelerator.prepare(
    #         train_dataloader, val_dataloader)
    if args.test:
        first_epoch = 0 #just so I enter loop for test (regardless of training iterations)

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            unet.train()
            if not args.test:
                with accelerator.accumulate(unet):
                    # first, convert images to latent space.
                    pixel_values = batch["pixel_values"].to(weight_dtype).to(
                        accelerator.device, non_blocking=True
                    )

        
                    conditional_pixel_values = pixel_values
                    latents = tensor_to_vae_latent(pixel_values, vae, otype="sample")

                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]

                    cond_sigmas = rand_log_normal(shape=[bsz,], loc=-3.0, scale=0.5).to(latents)
                    noise_aug_strength = cond_sigmas[0] # TODO: support batch > 1
                    cond_sigmas = cond_sigmas[:, None, None, None, None]

                    conditional_pixel_values = \
                    torch.randn_like(conditional_pixel_values) * cond_sigmas + conditional_pixel_values #- Comment this out as I don't want to add noise to the cond
                    conditional_latents = tensor_to_vae_latent(conditional_pixel_values, vae, otype="sample")
                    conditional_latents = conditional_latents / vae.config.scaling_factor #

                    ##you do noisy conditioning for the

                    # Sample a random timestep for each image
                    # P_mean=0.7 P_std=1.6
                    sigmas = rand_log_normal(shape=[bsz,], loc=0.7, scale=1.6).to(latents.device)
                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    sigmas = sigmas[:, None, None, None, None]
                    noisy_latents = latents + noise * sigmas
                    timesteps = torch.Tensor(
                        [0.25 * sigma.log() for sigma in sigmas]).to(accelerator.device)

                    inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)


                    conditioning = args.conditioning
                    # Create a tensor of zeros with the same shape as the repeated conditional_latents
                    if conditioning == "zero":
                        random_frames = [0]
                    elif conditioning == "random":
                        #choose a random number between 0 and 8 inclusive
                        random_frames = [np.random.randint(0, args.num_frames)]
                    elif conditioning in ["ablate_position", "ablate_time"] :
                        random_frames = [np.random.randint(0, args.num_frames)]
                    elif conditioning == "ablate_single_frame":
                        input_random_frame = np.random.randint(0, args.num_frames)
                        output_random_frame = np.random.randint(0, args.num_frames)
                    elif conditioning == "random_single_double_triple":
                        num_imgs = random.randint(1, 3)
                        random_frames = choices(range(args.num_frames), k=num_imgs)

                    # Get the text embedding for conditioning.
                    encoder_hidden_states = encode_image(
                        pixel_values[:, random_frames[0], :, :, :].float(),
                        feature_extractor, image_encoder, weight_dtype, accelerator)

                    # Here I input a fixed numerical value for 'motion_bucket_id', which is not reasonable.
                    # However, I am unable to fully align with the calculation method of the motion score,
                    # so I adopted this approach. The same applies to the 'fps' (frames per second).
                    conditioning_num = 0

                    if conditioning != "ablate_time":
                        conditioning_num = 0
                    else:
                        conditioning_num = random_frames[0]

                        

                    added_time_ids = get_add_time_ids(
                        7, # fixed
                        conditioning_num, # motion_bucket_id = 127, fixed
                        noise_aug_strength, # noise_aug_strength == cond_sigmas
                        encoder_hidden_states.dtype,
                        bsz,
                        unet
                    )
                    added_time_ids = added_time_ids.to(latents.device)


    
                    # Conditioning dropout to support classifier-free guidance during inference. For more details
                    # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.0args.num_frames800.
                    if args.conditioning_dropout_prob is not None:
                        random_p = torch.rand(
                            bsz, device=latents.device, generator=generator)
                        # Sample masks for the edit prompts. - I'm not sure if prompts are used in this model. Sam ewith the text conditioning that comes next.

                        #oh encoder_hidden_states is derived form the image.

                        prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                        prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                        # Final text conditioning.
                        null_conditioning = torch.zeros_like(encoder_hidden_states)
                        encoder_hidden_states = torch.where(
                            prompt_mask, null_conditioning.unsqueeze(1), encoder_hidden_states.unsqueeze(1))
                        # Sample masks for the original images.
                        image_mask_dtype = conditional_latents.dtype
                        image_mask = 1 - (
                            (random_p >= args.conditioning_dropout_prob).to(
                                image_mask_dtype)
                            * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                        )
                        image_mask = image_mask.reshape(bsz, 1, 1, 1)
                        # Final image conditioning.
                        conditional_latents = image_mask * conditional_latents #this basically 0s out some of the image latents

                    # Concatenate the `conditional_latents` with the `noisy_latents`.
                    # conditional_latents = conditional_latents.unsqueeze(
                    #     1).repeat(1, noisy_latents.shape[1], 1, 1, 1)
                    if conditioning == "ablate_single_frame":
                        #put input frame at first frame
                        conditional_latents = conditional_latents[:, 0:1].repeat(1, args.num_frames, 1, 1, 1)
                    elif conditioning in ["ablate_position", "ablate_time"]:

                        conditional_latents = conditional_latents[:, random_frames[0]:random_frames[0]+1].repeat(1,args.num_frames, 1, 1, 1)
                    else:
                        mask = torch.zeros_like(conditional_latents)
                        #choose a random frame to allow for the model to learn to focus on different frames (set mask to 1 for that frame)
                        mask[:, random_frames] = 1
                        conditional_latents = conditional_latents * mask


                    inp_noisy_latents = torch.cat(
                        [inp_noisy_latents, conditional_latents], dim=2)

                    # check https://arxiv.org/abs/2206.00364(the EDM-framework) for more details.
                    target = latents
                    model_pred = unet(
                        inp_noisy_latents, timesteps, encoder_hidden_states, added_time_ids=added_time_ids).sample

                    # Denoise the latents
                    c_out = -sigmas / ((sigmas**2 + 1)**0.5)
                    c_skip = 1 / (sigmas**2 + 1)
                    denoised_latents = model_pred * c_out + c_skip * noisy_latents
                    weighing = (1 + sigmas ** 2) * (sigmas**-2.0)

                    # MSE loss
                    loss = torch.mean(
                        (weighing.float() * (denoised_latents.float() -
                        target.float()) ** 2).reshape(target.shape[0], -1),
                        dim=1,
                    )
                    loss = loss.mean()

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(
                        loss.repeat(args.per_gpu_batch_size)).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(loss)
                    lr_scheduler.step()
                    optimizer.zero_grad()
            

            

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:

                    if args.use_ema:
                        ema_unet.step(unet.parameters())
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0
                    
                    if accelerator.is_main_process:

                        # save checkpoints!
                        if global_step % args.checkpointing_steps == 0:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [
                                    d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(
                                    checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = len(
                                        checkpoints) - args.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(
                                        f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(
                                            args.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(
                                args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")
            # sample images!
            if args.test or (global_step % args.validation_steps == 0) or (global_step == 1):
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())

                valid_net(args, val_dataset, val_dataloader, unet, image_encoder, vae, zero_latent, accelerator, global_step, weight_dtype)
                if args.use_ema:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(unet.parameters())
                if args.test:
                    break
                
                torch.cuda.empty_cache()

                
        

            logs = {"step_loss": loss.detach().item(
            ), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
        if args.test:
            break
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and not args.test:

        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            image_encoder=accelerator.unwrap_model(image_encoder),
            vae=accelerator.unwrap_model(vae),
            unet=accelerator.unwrap_model(ema_unet) if args.use_ema else unet,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.use_ema:
            ema_unet.copy_to(unet.parameters())
        
        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )
    accelerator.end_training()


if __name__ == "__main__":
    main()


