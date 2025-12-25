from simplified_pipeline import StableVideoDiffusionPipeline
import os
import torch
import numpy as np
import videoio
import matplotlib.image
from PIL import Image



def valid_net(args, batch, unet, image_encoder, vae, global_step, weight_dtype, device):

    # The models need unwrapping because for compatibility in distributed training mode.

    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        args.pretrained_model_path,
        unet=unet,
        image_encoder=image_encoder,
        vae=vae,
        torch_dtype=weight_dtype,
    )

    pipeline.set_progress_bar_config(disable=True)

    # run inference
    val_save_dir = os.path.join(
        args.output_dir, "validation_images")

    print("Validation images will be saved to ", val_save_dir)

    os.makedirs(val_save_dir, exist_ok=True)


    num_frames = 9 
    unet.eval()

    #clear gradients (the torch no grad is the magic that makes this work)
    with torch.no_grad():
        torch.cuda.empty_cache()

    pixel_values = batch["pixel_values"].to(device)
    original_pixel_values = batch['original_pixel_values'].to(device)
    focal_stack_num = batch["focal_stack_num"]

    svd_output, gt_frames = pipeline(
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
    gt_frames = gt_frames[0]


    with torch.no_grad():

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
            f"position_{focal_stack_num}/videos/{batch['name']}.mp4",
        ), video_frames_normalized.permute(0,2,3,1).cpu().numpy(), fps=5)

        #save images
        os.makedirs(os.path.join(val_save_dir, f"position_{focal_stack_num}/images"), exist_ok=True)
        for i in range(num_frames):
            #use Pillow to save images
            img = Image.fromarray((video_frames_normalized[i].permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
            #use index to assign icc profile to img
            if batch['icc_profile'] != "none":
                img.info['icc_profile'] = batch['icc_profile']
            path = os.path.join(val_save_dir, f"position_{focal_stack_num}/images/{batch['name']}_frame_{i}.png")
            print("Saving image to ", path)
            img.save(os.path.join(val_save_dir, f"position_{focal_stack_num}/images/{batch['name']}_frame_{i}.png"))
    del video_frames



