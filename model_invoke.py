from PIL import Image
import requests
import json

from transformers import CLIPProcessor, CLIPModel

# Loading the model pipeline
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Invoke function
def invoke(input_text):
    # Parsing input
    input_json = json.loads(input_text)
    image_url = input_json['image_url']
    image = Image.open(requests.get(image_url, stream=True).raw)
    text_arr = input_json['text'].split(",")
    inputs = processor(text=text_arr, images=image, return_tensors="pt", padding=True)
    # Run inference
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    # Return the result
    return [text_arr[i]+": "+str(probs.tolist()[0][i]) for i in range(0,len(text_arr))]
