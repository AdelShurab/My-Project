import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

processor = AutoProcessor.from_pretrained("SalesForce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("SalesForce/blip-image-captioning-base")


img_path = "pexels-cristiane-burckauser-16609298.jpg"
image = Image.open(img_path).convert('RGB')


text = "the image of"
inputs = processor(images=image, text=text, return_tensors="pt")


outputs = model.generate(**inputs, max_length=50)


caption = processor.decode(outputs[0], skip_special_tokens=True)

print(caption)
