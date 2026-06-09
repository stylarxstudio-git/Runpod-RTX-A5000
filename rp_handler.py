import runpod
import torch
import base64
import io
import requests
from PIL import Image
from transformers import pipeline

print("Booting up backend system... Initializing Utility Models...")
try:
    # 1. Depth Anything V2 - Using the official HF community integration string
    depth_estimator = pipeline(
        task="depth-estimation", 
        model="depth-anything/Depth-Anything-V2-Base-hf", 
        device=0  # Direct pointer to the active CUDA slot
    )
    
    # 2. BiRefNet - Adding trust_remote_code to allow the custom layer build to pass safely
    birefnet_pipeline = pipeline(
        task="image-segmentation", 
        model="ZhengPeng7/BiRefNet", 
        trust_remote_code=True,
        device=0
    )
    
    print("All processing engines loaded into GPU successfully.")
except Exception as e:
    print(f"Critical error loading pipeline configurations: {e}")

def download_image(url):
    response = requests.get(url, timeout=15)
    return Image.open(io.BytesIO(response.content)).convert("RGB")

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def handler(job):
    try:
        job_input = job["input"]
        tool_type = job_input.get("tool")
        image_url = job_input.get("image_url")

        if not image_url:
            return {"error": "No source image URL provided for processing."}

        input_image = download_image(image_url)

        # ROUTING LOGIC
        if tool_type in ["depth_map_generator", "image_to_pbr"]:
            result = depth_estimator(input_image)
            processed_image = result["depth"]
            
        elif tool_type in ["background_remover", "sticker_background_removal"]:
            # BiRefNet outputs a PIL mask image directly
            processed_image = birefnet_pipeline(input_image)
            
        else:
            return {"error": f"Invalid utility type '{tool_type}' routed to the endpoint."}

        base64_output = image_to_base64(processed_image)

        return {
            "status": "success",
            "tool_executed": tool_type,
            "image": base64_output
        }

    except Exception as e:
        return {"error": f"An execution failure occurred inside the utility container: {str(e)}"}

runpod.serverless.start({"handler": handler})
