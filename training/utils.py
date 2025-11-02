import pickle
from torch.utils.data import Dataset
import cv2
import argparse
import glob
import random
import logging
import torch
import os
import numpy as np
import PIL
from PIL import Image, ImageDraw
from einops import rearrange
from urllib.parse import urlparse
from diffusers.utils import load_image
import math

# copy from https://github.com/crowsonkb/k-diffusion.git
def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()

def encode_image(pixel_values, feature_extractor, image_encoder, weight_dtype, accelerator):
    # pixel: [-1, 1]
    pixel_values = _resize_with_antialiasing(pixel_values, (224, 224))
    # We unnormalize it after resizing.
    pixel_values = (pixel_values + 1.0) / 2.0

    # Normalize the image with for CLIP input
    pixel_values = feature_extractor(
        images=pixel_values,
        do_normalize=True,
        do_center_crop=False,
        do_resize=False,
        do_rescale=False,
        return_tensors="pt",
    ).pixel_values

    pixel_values = pixel_values.to(
        device=accelerator.device, dtype=weight_dtype)
    image_embeddings = image_encoder(pixel_values).image_embeds
    return image_embeddings

def get_add_time_ids(
    fps,
    motion_bucket_id,
    noise_aug_strength,
    dtype,
    batch_size,
    unet
):
    add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

    passed_add_embed_dim = unet.config.addition_time_embed_dim * \
        len(add_time_ids)
    expected_add_embed_dim = unet.add_embedding.linear_1.in_features

    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )

    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    add_time_ids = add_time_ids.repeat(batch_size, 1)
    return add_time_ids

def find_scale(height, width):
    """
    Finds a scale factor such that the number of pixels is less than 500,000
    and the dimensions are rounded down to the nearest multiple of 64.
    
    Args:
        height (int): The original height of the image.
        width (int): The original width of the image.
        
    Returns:
        tuple: The scaled height and width as integers.
    """
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

        self.scenes = [(scene, idx) for scene in self.scenes for idx in range(9)]


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




class FocalStackDataset(Dataset):
    def __init__(self, data_folder: str, split="train", num_samples=100000, width=640, height=896, sample_frames=9): #4.5
        #800*600 - 480000
        #896*672 - 602112
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            channels (int): Number of channels, default is 3 for RGB.
        """
        self.num_samples = num_samples
        self.sample_frames = sample_frames
        # Define the path to the folder containing video frames
        self.data_folder = data_folder

        size = "midsize"
        # Use glob to find matching folders
        # List to store the desired paths
        rig_directories = []

        # Walk through the directory
        for root, dirs, files in os.walk(data_folder):
            # Check if the path matches "downscaled/undistorted/Rig*"
            for directory in dirs:
                if directory.startswith("RigCenter") and f"{size}/undistorted" in root.replace("\\", "/"):
                    rig_directory = os.path.join(root, directory)
                    #check that rig_directory contains all 9 images
                    if len(glob.glob(os.path.join(rig_directory, "*.jpg"))) == 9:
                        rig_directories.append(rig_directory)

        
        self.scenes = sorted(rig_directories) #sort the files by name

        if split == "train":
            #shuffle the scenes
            random.shuffle(self.scenes)
        self.split = split

        debug = False


        if debug:
            self.scenes = self.scenes[50:60] 
        elif split == "train":
            pkl_file = "/datasets/sai/focal-burst-learning/train_scenes.pkl"
            #load the train scenes
            with open(pkl_file, "rb") as f:
                pkl_scenes = pickle.load(f)
            
            #only get scenes that are found in pkl file
            self.scenes = [scene for scene in self.scenes if scene.split('/')[-4] in pkl_scenes]

        elif split == "val":
            pkl_file = "/datasets/sai/focal-burst-learning/test_scenes.pkl"

            #load the test scenes
            with open(pkl_file, "rb") as f:
                pkl_scenes = pickle.load(f)
            
            #only get scenes that are found in pkl file
            self.scenes = [scene for scene in self.scenes if scene.split('/')[-4] in pkl_scenes]
            self.scenes = self.scenes[:10]
        else:
            pkl_file = "/datasets/sai/focal-burst-learning/test_scenes.pkl"

            #load the test scenes
            with open(pkl_file, "rb") as f:
                pkl_scenes = pickle.load(f)
            
            #only get scenes that are found in pkl file
            self.scenes = [scene for scene in self.scenes if scene.split('/')[-4] in pkl_scenes]



        if split == "test":
            self.scenes = [(scene, idx) for scene in self.scenes for idx in range(self.sample_frames)]
        
        self.num_scenes = len(self.scenes)

        max_trdata = 0
        if max_trdata > 0:
            self.scenes = self.scenes[:max_trdata]

        self.data_store = {}

        logging.info(f'Creating {split} dataset with {self.num_scenes} examples')

        self.channels = 3
        self.width = width
        self.height = height
        

    def __len__(self):
        return self.num_scenes

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of shape (16, channels, 320, 512).
        """
        # Randomly select a folder (representing a video) from the base folder
        if self.split == "test":
            chosen_folder, focal_stack_num = self.scenes[idx]
        else:
            chosen_folder = self.scenes[idx]
        frames = os.listdir(chosen_folder)
        #get only frames that are jpg
        frames = [frame for frame in frames if frame.endswith(".jpg")]
        # Sort the frames by name
        frames.sort()

        #Pad the frames list out
        selected_frames = frames[:self.sample_frames] 
        # Initialize a tensor to store the pixel values
        pixel_values = torch.empty((self.sample_frames, self.channels, self.height, self.width))

        original_pixel_values = torch.empty((self.sample_frames, self.channels, 896, 640))

        # Load and process each frame
        for i, frame_name in enumerate(selected_frames):
            frame_path = os.path.join(chosen_folder, frame_name)
            with Image.open(frame_path) as img:

       
                # Resize the image and convert it to a tensor
                img_resized = img.resize((self.width, self.height))
                img_tensor = torch.from_numpy(np.array(img_resized)).float()
                original_img_tensor = torch.from_numpy(np.array(img)).float()

                # Normalize the image by scaling pixel values to [-1, 1]
                img_normalized = img_tensor / 127.5 - 1
                original_img_normalized = original_img_tensor / 127.5 - 1

                # Rearrange channels if necessary
                if self.channels == 3:
                    img_normalized = img_normalized.permute(
                        2, 0, 1)  # For RGB images
                    original_img_normalized = original_img_normalized.permute(2, 0, 1)
                    
                pixel_values[i] = img_normalized
                original_pixel_values[i] = original_img_normalized 

        if self.sample_frames == 10: #special case for 10 frames where we duplicate the 9th frame (sometimes reduced color artifacts)
            pixel_values[9] = pixel_values[8]
            original_pixel_values[9] = original_pixel_values[8]
        out_dict = {'pixel_values': pixel_values, "idx": idx, "original_pixel_values": original_pixel_values}
        if self.split == "test":
            out_dict["focal_stack_num"] = focal_stack_num
            out_dict["idx"] = idx//9
        return out_dict

# resizing utils
# TODO: clean up later
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(
        input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(
        device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(
        input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device,
         dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out


def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(
        output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)


def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=500,
                       loop=0)


def tensor_to_vae_latent(t, vae, otype="sample"):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    if otype == "sample":
        latents = vae.encode(t).latent_dist.sample()
    else:
        latents = vae.encode(t).latent_dist.mode()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    latents = latents * vae.config.scaling_factor

    return latents

import yaml
def parse_config(config_path="config.yaml"):
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
        default="svd/scripts/training/configs/stage1_base.yaml",
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


def download_image(url):
    original_image = (
        lambda image_url_or_path: load_image(image_url_or_path)
        if urlparse(image_url_or_path).scheme
        else PIL.Image.open(image_url_or_path).convert("RGB")
    )(url)
    return original_image

