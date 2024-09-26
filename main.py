import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from diffusers import AutoencoderKL, TCDScheduler
from huggingface_hub import hf_hub_download
from controlnet_union import ControlNetModel_Union
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline
import io

app = FastAPI()

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
state_dict = torch.load(model_file)
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

def generate_custom_mask(image: Image.Image):
   
    mask = image.convert("L")  # Example, convert image to grayscale
    return mask

async def inpaint(prompt: str, image: Image.Image, paste_back: bool = True):
    input_image = image.convert("RGBA")
    mask = generate_custom_mask(input_image)

    alpha_channel = mask.split()[3]
    binary_mask = alpha_channel.point(lambda p: p > 0 and 255)
    cnet_image = input_image.copy()
    cnet_image.paste(0, (0, 0), binary_mask)

    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
        prompt, "cuda", True
    )

    output_image = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        image=cnet_image,
    )[0]

    if paste_back:
        output_image = output_image.convert("RGBA")
        cnet_image.paste(output_image, (0, 0), binary_mask)
    else:
        cnet_image = output_image

    output_buffer = io.BytesIO()
    cnet_image.save(output_buffer, format="PNG")
    output_buffer.seek(0)

    return {"output_image": output_buffer.getvalue()}

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI Inpainting API"}

@app.get("/models")
def get_models():
    return {"models": list(MODELS.keys())}

@app.post("/inpaint")
async def inpaint_endpoint(prompt: str, image: UploadFile = File(...), paste_back: bool = True):
    return await inpaint(prompt, image, paste_back)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)