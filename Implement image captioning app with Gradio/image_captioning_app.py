import warnings
import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration


warnings.filterwarnings("ignore", category=FutureWarning, module='transformers.tokenization_utils_base')


processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(input_image: np.ndarray):
    
    raw_image = Image.fromarray(input_image).convert('RGB')

    
    inputs = processor(raw_image, return_tensors="pt")

    
    out = model.generate(**inputs, max_length=50)

    
    caption = processor.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return caption

iface = gr.Interface(
    fn=caption_image, 
    inputs=gr.Image(), 
    outputs="text",
    title="Image Captioning By Adel Shurab",
    description="This is a simple web app for generating captions for images using a trained model."
)

iface.launch()
