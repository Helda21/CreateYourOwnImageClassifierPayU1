import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from train import load_model
import json
import argparse

# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, help='Image to predict')
parser.add_argument('--checkpoint', type=str, help='Model checkpoint to use when predicting')
parser.add_argument('--topk', type=int, help='Return top K predictions')
parser.add_argument('--labels', type=str, help='JSON file containing label names')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
args = parser.parse_args()

# Implement the code to predict the class from an image file
def predict(image, checkpoint, topk=5, labels='', gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Load the checkpoint
    checkpoint_dict = torch.load(checkpoint)
    arch = checkpoint_dict['arch']
    num_labels = len(checkpoint_dict['class_to_idx'])
    hidden_units = checkpoint_dict['hidden_units']
    
    model = load_model(arch=arch, num_labels=num_labels, hidden_units=hidden_units)

    # Use GPU if selected and available
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Set the model to evaluation mode
    model.eval()

    # Load and preprocess the image
    img_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    pil_image = Image.open(image)
    image = img_transforms(pil_image).unsqueeze(0).to(device)

    # Perform the forward pass
    with torch.no_grad():
        output = model(image)

    # Get the top predicted classes and their probabilities
    probabilities, indices = torch.topk(output, topk)
    probabilities = torch.nn.functional.softmax(probabilities, dim=1).squeeze().cpu().numpy()
    indices = indices.squeeze().cpu().numpy()

    # Map indices to class labels if provided
    class_labels = []
    if labels:
        with open(labels, 'r') as f:
            class_labels = json.load(f)
        class_labels = [class_labels[str(index)] for index in indices]

    return probabilities, indices, class_labels

# Perform predictions if invoked from the command line
if args.image and args.checkpoint:
    probabilities, indices, class_labels = predict(args.image, args.checkpoint, args.topk, args.labels, args.gpu)
    print('Predictions and probabilities:')
    for prob, index, label in zip(probabilities, indices, class_labels):
        print(f'{label}: {prob:.5f}')