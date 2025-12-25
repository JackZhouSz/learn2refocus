import os
import spaces
from pathlib import Path
import argparse

import gradio as gr
from PIL import Image

from simple_inference import load_model, inference_on_image, convert_to_batch, write_output

# -----------------------
# 1. Load model
# -----------------------
args = argparse.Namespace()
args.learn2refocus_hf_repo_path = "tedlasai/learn2refocus"
args.pretrained_model_path = "stabilityai/stable-video-diffusion-img2vid"
args.seed = 0

pipe, device = load_model(args)

OUTPUT_DIR = Path("/tmp/output_stacks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_FRAMES = 9  # frame_0.png ... frame_8.png


@spaces.GPU(timeout=300, duration=80)
def generate_outputs(image: Image.Image, input_focal_position: int, num_inference_steps: int):
    if image is None:
        raise gr.Error("Please upload an image first.")

    args.num_inference_steps = num_inference_steps
    args.device = "cuda"
    pipe.to(args.device)

    batch = convert_to_batch(image, input_focal_position=input_focal_position)
    output_frames, focal_stack_num = inference_on_image(args, batch, pipe, device)

    write_output(OUTPUT_DIR, output_frames, focal_stack_num, batch["icc_profile"])

    video_path = OUTPUT_DIR / "stack.mp4"
    first_frame = OUTPUT_DIR / "frame_0.png"

    if not video_path.exists():
        raise gr.Error("stack.mp4 not found in output_dir")
    if not first_frame.exists():
        raise gr.Error("frame_0.png not found in output_dir")

    return str(video_path), str(first_frame), gr.update(value=0)


def show_frame(idx: int):
    path = OUTPUT_DIR / f"frame_{int(idx)}.png"
    if not path.exists():
        return None
    return str(path)


def set_view_mode(mode: str):
    show_video = (mode == "Video")
    return (
        gr.update(visible=show_video),
        gr.update(visible=not show_video),
    )


with gr.Blocks() as demo:
    gr.Markdown(
    """ # 🖼️ ➜ 🎬 Generate Focal Stacks from a Single Image.
    This demo accompanies the paper **“Learning to Refocus with Video Diffusion MOdels”** by Tedla *et al.*, SIGGRAPH Asia 2025. 
    - 🌐 **Project page:** <https://learn2refocus.github.io/> 
    - 💻 **Code:** <https://github.com/tedlasai/learn2refocus/>
    
     Upload an image and **specify the input focal position** (these values correspond to iPhone API positions, but approximately linear in diopters (inverse meters): 0 - 5cm, 8 - Infinity).
     Then, click "Generate stack" to generate a focal stack. """
    )

    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="Input image", interactive=True)

            input_focal_position = gr.Slider(
                label="Input focal position (Near - 5cm, Far - Infinity):",
                minimum=0,
                maximum=8,
                step=1,
                value=4,
                interactive=True,
            )

            num_inference_steps = gr.Slider(
                label="Number of inference steps",
                minimum=4,
                maximum=25,
                step=1,
                value=25,
                info="More steps = better quality but slower",
            )

            generate_btn = gr.Button("Generate stack", variant="primary")

        with gr.Column():
            view_mode = gr.Radio(
                choices=["Video", "Frames"],
                value="Video",
                label="Output view",
            )

            # --- Video output ---
            video_out = gr.Video(
                label="Generated stack",
                format="mp4",
                autoplay=True,
                loop=True,
                visible=True,
            )

            # --- Frames output (group) ---
            with gr.Group(visible=False) as frames_group:
                frame_view = gr.Image(label="Stack viewer", type="filepath")
                frame_slider = gr.Slider(
                    minimum=0,
                    maximum=NUM_FRAMES - 1,
                    step=1,
                    value=0,
                    label="Output focal position",
                )

    generate_btn.click(
        fn=generate_outputs,
        inputs=[image_in, input_focal_position, num_inference_steps],
        outputs=[video_out, frame_view, frame_slider],
        api_name="predict",
    )

    frame_slider.change(
        fn=show_frame,
        inputs=frame_slider,
        outputs=frame_view,
    )

    view_mode.change(
        fn=set_view_mode,
        inputs=view_mode,
        outputs=[video_out, frames_group],
    )

if __name__ == "__main__":
    demo.launch(css="footer {visibility: hidden}")

