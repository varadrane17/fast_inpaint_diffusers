import gradio as gr
import requests
from PIL import Image
import io


API_URL = "http://127.0.0.1:8000/inpaint/"


def inpaint(prompt, image,):
    
    
    # send req with server function as async def inpaint(prompt: str, image: Image.Image
    
    # Prepare the API request payload
    files = {
        'image': ('input_image.png', img_bytes, 'image/png')
    }
    data = {
        'prompt': prompt,
        'paste_back': paste_back,
    }
    
    # Send request to FastAPI server
    response = requests.post(API_URL, files=files, data=data)
    
    if response.status_code == 200:
        output_image_data = response.json()["output_image"]
        
        # Convert binary data back to PIL images
        img1 = Image.open(io.BytesIO(output_image_data))
        return img1
    else:
        return None

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# AI Background Replacer")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", placeholder="Describe the background..")
        image = gr.Image(label="Input Image", type="pil", image_mode="RGBA")
    
    paste_back = gr.Checkbox(label="Maintain original image structure", value=True)

    with gr.Row():
        run_button = gr.Button("Generate")
    with gr.Row():
        
        result_image1 = gr.Image(label="Generated Image")

    run_button.click(
        fn=inpaint, 
        inputs=[prompt, image],
        outputs=[result_image1],
    )

demo.launch()
