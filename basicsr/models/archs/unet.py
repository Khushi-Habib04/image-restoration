'''import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from basicsr.models.archs.kb_utils import KBAFunction, LayerNorm2d, SimpleGate
from basicsr.models.archs.kbnet_s_arch import KBNet_s  # Ensure model.py is in your working directory

def main():
    # Initialize the model
    img_channel = 3
    width = 64
    enc_blks = [2, 2, 4, 6]
    middle_blk_num = 10
    dec_blks = [2, 2, 2, 2]

    net = KBNet_s(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                  enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

    # Load image and preprocess
    image_path = 'image001.png'  # Change to your image path
    save_path = 'processed_image.png'      # Path to save the processed image

    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))  # Resize image to match model input shape
    ])
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Process the image
    net.eval()
    with torch.no_grad():
        output_tensor = net(input_tensor)

    # Convert tensors to images
    input_image = transforms.ToPILImage()(input_tensor.squeeze(0))
    output_image = transforms.ToPILImage()(output_tensor.squeeze(0))

    # Save the images
    input_image.save('input_image.png')
    output_image.save(save_path)

    print(f"Processed image saved at {save_path}")

if __name__ == '__main__':
    main()
'''
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from basicsr.models.archs.kb_utils import KBAFunction, LayerNorm2d, SimpleGate
from basicsr.models.archs.kbnet_s_arch import KBNet_s  # Ensure model.py is in your working directory

def main():
    # Initialize the model
    img_channel = 3
    width = 64
    enc_blks = [2, 2, 4, 6]
    middle_blk_num = 10
    dec_blks = [2, 2, 2, 2]

    net = KBNet_s(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                  enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

    # Load weights from the .pth file
    weights_path = "/all/cse/uday/image/NAFNet/experiments/NAFNet-SIDD-width64_archived_20240523_163537/models/net_g_295000.pth"  # Change to your .pth file path
    state_dict = torch.load(weights_path)

    # Filter out keys that do not match
    model_state_dict = net.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}

    # Check for missing keys
    missing_keys = set(model_state_dict.keys()) - set(filtered_state_dict.keys())
    if missing_keys:
        print(f"Warning: Missing keys in state_dict: {missing_keys}")

    net.load_state_dict(filtered_state_dict, strict=False)
    # Load image and preprocess
    image_path = 'image001.png'  # Change to your image path
    save_path = 'processed_image.png'      # Path to save the processed image

    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))  # Resize image to match model input shape
    ])
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Process the image
    net.eval()
    with torch.no_grad():
        output_tensor = net(input_tensor)

    # Convert tensors to images
    input_image = transforms.ToPILImage()(input_tensor.squeeze(0))
    output_image = transforms.ToPILImage()(output_tensor.squeeze(0))

    # Save the images
    input_image.save('input_image.png')
    output_image.save(save_path)

    print(f"Processed image saved at {save_path}")

if __name__ == '__main__':
    main()
