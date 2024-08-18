# from flask import Flask, request, jsonify, send_file, render_template
# from werkzeug.utils import secure_filename
# import os
# from io import BytesIO
# from PIL import Image
# import torch
# from torchvision import transforms
# from web import KBNet_s 

# # Initialize the Flask application
# app = Flask(__name__)

# # Initialize the model
# model = KBNet_s()

# # Load the trained model state dict
# model_path = "/all/cse/uday/image/NAFNet/sidd.pth"
# try:
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#     print("Model loaded successfully.")
# except Exception as e:
#     print(f"Error loading the model: {e}")
#     raise e  # Re-raise the exception after logging

# # Define a transform to convert input image to tensor
# transform = transforms.Compose([
#     transforms.ToTensor()
# ])

# # Allowed extensions
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'})
    
#     file = request.files['file']
    
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'})
    
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file_path = os.path.join('uploads', filename)
#         file.save(file_path)
        
#         # Open the image file
#         img = Image.open(file_path).convert('RGB')
        
#         # Apply the transformation and add batch dimension
#         input_tensor = transform(img).unsqueeze(0)
        
#         # Denoise the image using the model
#         with torch.no_grad():
#             denoised_tensor = model(input_tensor).squeeze(0)
        
#         # Convert the tensor back to image
#         denoised_img = transforms.ToPILImage()(denoised_tensor)
        
#         # Save the denoised image
#         output_path = os.path.join('uploads', 'denoised_' + filename)
#         denoised_img.save(output_path)
        
#         return send_file(output_path, mimetype='image/png')

#     return jsonify({'error': 'Invalid file type'})

# if __name__ == '__main__':
#     if not os.path.exists('uploads'):
#         os.makedirs('uploads')
#     app.run(debug=True)
from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
import os
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, tensor2img, imwrite

# Initialize the Flask application
app = Flask(__name__)

# Load the model using functions from BasicSR
def load_model():
    opt = parse_options(is_train=False)
    opt['num_gpu'] = torch.cuda.device_count()
    model = create_model(opt)
    #model.load_state_dict(torch.load("/all/cse/uday/image/NAFNet/sidd.pth"))
    model.eval()
    return model

model = load_model()

# Define a transform to convert input image to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        
        # Open the image file
        img = Image.open(file_path).convert('RGB')
        
        # Apply the transformation and add batch dimension
        input_tensor = transform(img).unsqueeze(0)
        
        # Run inference using the loaded model
        with torch.no_grad():
            model.feed_data(data={'lq': input_tensor})
            model.test()
            visuals = model.get_current_visuals()
            denoised_img_tensor = visuals['result']
        
        # Convert the tensor back to image
        denoised_img = tensor2img([denoised_img_tensor])
        denoised_img_pil = Image.fromarray(denoised_img)

        # Save the denoised image temporarily
        temp_output_path = os.path.join('uploads', 'denoised_' + filename)
        denoised_img_pil.save(temp_output_path)

        # Return the denoised image as response
        return send_file(temp_output_path, mimetype='image/png')

    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
