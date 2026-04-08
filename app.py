import os
import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from huggingface_hub import hf_hub_download
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from briarmbg import BriaRMBG

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load Model
net = BriaRMBG()
model_path = hf_hub_download("briaai/RMBG-1.4", 'model.pth')
if torch.cuda.is_available():
    net.load_state_dict(torch.load(model_path))
    net = net.cuda()
else:
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
net.eval()

def resize_image(image):
    image = image.convert('RGB')
    model_input_size = (1024, 1024)
    image = image.resize(model_input_size, Image.BILINEAR)
    return image

def process_image(image_pil):
    # prepare input
    w, h = image_pil.size
    image = resize_image(image_pil)
    im_np = np.array(image)
    im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = torch.unsqueeze(im_tensor, 0)
    im_tensor = torch.divide(im_tensor, 255.0)
    im_tensor = normalize(im_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    
    if torch.cuda.is_available():
        im_tensor = im_tensor.cuda()

    # inference
    result = net(im_tensor)
    
    # post process
    result = torch.squeeze(F.interpolate(result[0][0], size=(h, w), mode='bilinear'), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    
    # image to pil
    im_array = (result * 255).cpu().data.numpy().astype(np.uint8)
    pil_mask = Image.fromarray(np.squeeze(im_array))
    
    # paste the mask on the original image
    new_im = Image.new("RGBA", pil_mask.size, (0, 0, 0, 0))
    new_im.paste(image_pil, mask=pil_mask)
    
    return new_im

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/remove-bg', methods=['POST'])
def remove_bg():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    img_bytes = file.read()
    image_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    # Process
    result_pil = process_image(image_pil)
    
    # Convert to base64
    buffered = io.BytesIO()
    result_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return jsonify({'image': f"data:image/png;base64,{img_str}"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)