import gradio as gr
import spaces
import torch
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from gradio_imageslider import ImageSlider
from huggingface_hub import hf_hub_download

from controlnet_union import ControlNetModel_Union
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline

MODELS = {
    "RealVisXL V5.0 Lightning": "SG161222/RealVisXL_V5.0_Lightning",
}

config_file = hf_hub_download(
    "xinsir/controlnet-union-sdxl-1.0",
    filename="config_promax.json",
)

config = ControlNetModel_Union.load_config(config_file)
controlnet_model = ControlNetModel_Union.from_config(config)
model_file = hf_hub_download(
    "xinsir/controlnet-union-sdxl-1.0",
    filename="diffusion_pytorch_model_promax.safetensors",
)
state_dict = load_state_dict(model_file)
model, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
    controlnet_model, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0"
)
model.to(device="cuda", dtype=torch.float16)

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
).to("cuda")

pipe = StableDiffusionXLFillPipeline.from_pretrained(
    "SG161222/RealVisXL_V5.0_Lightning",
    torch_dtype=torch.float16,
    vae=vae,
    controlnet=model,
    variant="fp16",
).to("cuda")

pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)


@spaces.GPU(duration=16)
def fill_image(prompt, image, model_selection, paste_back):
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(prompt, "cuda", True)

    source = image["background"]
    mask = image["layers"][0]

    alpha_channel = mask.split()[3]
    binary_mask = alpha_channel.point(lambda p: p > 0 and 255)
    cnet_image = source.copy()
    cnet_image.paste(0, (0, 0), binary_mask)

    for image in pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        image=cnet_image,
    ):
        yield image, cnet_image

    print(f"{model_selection=}")
    print(f"{paste_back=}")

    if paste_back:
        image = image.convert("RGBA")
        cnet_image.paste(image, (0, 0), binary_mask)
    else:
        cnet_image = image

    yield source, cnet_image


def clear_result():
    return gr.update(value=None)


title = """<h1 align="center">Diffusers Fast Inpaint</h1>
<div align="center">Draw the mask over the subject you want to erase or change and write what you want to inpaint it with.</div>
<div align="center">This is a lighting model with almost no CFG and 12 steps, so don't expect high quality generations.</div>
"""

with gr.Blocks() as demo:
    gr.HTML(title)
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                info="Describe what to inpaint the mask with",
                lines=3,
            )
        with gr.Column():
            model_selection = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="RealVisXL V5.0 Lightning",
                label="Model",
            )

            with gr.Row():
                with gr.Column():
                    run_button = gr.Button("Generate")

                with gr.Column():
                    paste_back = gr.Checkbox(True, label="Paste back original")

    with gr.Row():
        input_image = gr.ImageMask(
            type="pil", label="Input Image", crop_size=(1024, 1024), layers=False
        )

        result = ImageSlider(
            interactive=False,
            label="Generated Image",
        )

    use_as_input_button = gr.Button("Use as Input Image", visible=False)

    def use_output_as_input(output_image):
        return gr.update(value=output_image[1])

    use_as_input_button.click(
        fn=use_output_as_input, inputs=[result], outputs=[input_image]
    )

    run_button.click(
        fn=clear_result,
        inputs=None,
        outputs=result,
    ).then(
        fn=lambda: gr.update(visible=False),
        inputs=None,
        outputs=use_as_input_button,
    ).then(
        fn=fill_image,
        inputs=[prompt, input_image, model_selection, paste_back],
        outputs=result,
    ).then(
        fn=lambda: gr.update(visible=True),
        inputs=None,
        outputs=use_as_input_button,
    )

    prompt.submit(
        fn=clear_result,
        inputs=None,
        outputs=result,
    ).then(
        fn=lambda: gr.update(visible=False),
        inputs=None,
        outputs=use_as_input_button,
    ).then(
        fn=fill_image,
        inputs=[prompt, input_image, model_selection, paste_back],
        outputs=result,
    ).then(
        fn=lambda: gr.update(visible=True),
        inputs=None,
        outputs=use_as_input_button,
    )


demo.launch(share=False)
