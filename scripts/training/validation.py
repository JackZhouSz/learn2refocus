from torchmetrics import MetricCollection
from svd_pipeline import StableVideoDiffusionPipeline
from accelerate.logging import get_logger
import os
from utils import load_image
import torch
import numpy as np
import videoio
import torchmetrics.image
import matplotlib.image
from PIL import Image

logger = get_logger(__name__, log_level="INFO")


def valid_net(args, val_dataset, val_dataloader, unet, image_encoder, vae, zero, accelerator, global_step, weight_dtype):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} videos."
    )

    # The models need unwrapping because for compatibility in distributed training mode.

    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        image_encoder=image_encoder,
        vae=vae,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )

    pipeline.set_progress_bar_config(disable=True)

    # run inference
    val_save_dir = os.path.join(
        args.output_dir, "validation_images")

    os.makedirs(val_save_dir, exist_ok=True)


    num_frames = args.num_frames
    unet.eval()
    with torch.autocast(
        str(accelerator.device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"
    ):
        for batch in val_dataloader:
            #clear gradients (the torch no grad is the magic that makes this work)
            with torch.no_grad():
                torch.cuda.empty_cache()

            pixel_values = batch["pixel_values"].to(accelerator.device)
            original_pixel_values = batch['original_pixel_values'].to(accelerator.device)
            idx = batch["idx"].to(accelerator.device)
            if "focal_stack_num" in batch:
                focal_stack_num = batch["focal_stack_num"][0].item()
            else:
                focal_stack_num = None

            svd_output, gt_frames = pipeline(
                pixel_values,
                height=pixel_values.shape[3],
                width=pixel_values.shape[4],
                num_frames=args.num_frames,
                decode_chunk_size=8,
                motion_bucket_id=0 if args.conditioning != "ablate_time" else focal_stack_num,
                min_guidance_scale=1.5,
                max_guidance_scale=1.5,
                reconstruction_guidance_scale=args.reconstruction_guidance,
                fps=7,
                noise_aug_strength=0,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                conditioning = args.val_conditioning,
                focal_stack_num = focal_stack_num,
                zero=zero
                # generator=generator,
            )
            video_frames = svd_output.frames[0]
            gt_frames = gt_frames[0]


            with torch.no_grad():

                if args.num_frames == 10:
                    #remove a frame at end from video_frames and gt_frames
                    video_frames = video_frames[:, :-1]
                    gt_frames = gt_frames[:, :-1]
                    original_pixel_values = original_pixel_values[:, :-1]
    
                if len(original_pixel_values.shape) == 5:
                    pixel_values = original_pixel_values[0] #assuming batch size is 1
                else:
                    pixel_values = original_pixel_values.repeat(num_frames, 1, 1, 1)
                pixel_values_normalized = pixel_values*0.5 + 0.5
                pixel_values_normalized = torch.clamp(pixel_values_normalized,0,1)




                video_frames_normalized = video_frames*0.5 + 0.5
                video_frames_normalized = torch.clamp(video_frames_normalized,0,1)
                video_frames_normalized = video_frames_normalized.permute(1,0,2,3)


                gt_frames = torch.clamp(gt_frames,0,1)
                gt_frames = gt_frames.permute(1,0,2,3)

                #RESIZE images 
                video_frames_normalized = torch.nn.functional.interpolate(video_frames_normalized, ((pixel_values.shape[2]//2)*2, (pixel_values.shape[3]//2)*2), mode='bilinear')
                gt_frames = torch.nn.functional.interpolate(gt_frames, ((pixel_values.shape[2]//2)*2, (pixel_values.shape[3]//2)*2), mode='bilinear')
                pixel_values_normalized = torch.nn.functional.interpolate(pixel_values_normalized, ((pixel_values.shape[2]//2)*2, (pixel_values.shape[3]//2)*2), mode='bilinear')

                os.makedirs(os.path.join(val_save_dir, f"position_{focal_stack_num}/videos"), exist_ok=True)
                videoio.videosave(os.path.join(
                    val_save_dir,
                    f"position_{focal_stack_num}/videos/step_{global_step}_val_img_{idx[0].item()}.mp4",
                ), video_frames_normalized.permute(0,2,3,1).cpu().numpy(), fps=5)

                if args.test:
                    #save images
                    os.makedirs(os.path.join(val_save_dir, f"position_{focal_stack_num}/images"), exist_ok=True)
                    if not args.photos:
                        for i in range(num_frames):
                            matplotlib.image.imsave(os.path.join(val_save_dir, f"position_{focal_stack_num}/images/img_{idx[0].item()}_frame_{i}.png"), video_frames_normalized[i].permute(1,2,0).cpu().numpy())
                    else:
                        for i in range(num_frames):
                            #use Pillow to save images
                            img = Image.fromarray((video_frames_normalized[i].permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
                            #use index to assign icc profile to img
                            if batch['icc_profile'][0] != "none":
                                img.info['icc_profile'] = batch['icc_profile'][0]
                            img.save(os.path.join(val_save_dir, f"position_{focal_stack_num}/images/img_{idx[0].item()}_frame_{i}.png"))
            del video_frames

    accelerator.wait_for_everyone()

    #clear gradients (the torch no grad is the magic that makes this work)
    with torch.no_grad():
        torch.cuda.empty_cache()
        
    del pipeline

    accelerator.wait_for_everyone() #this is really important and we need to make sure everyone is leaving at the same time
