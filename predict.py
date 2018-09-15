import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import models, transforms
from collections import OrderedDict
import os
import argparse
import json

from PIL import Image

def get_input_args():
    """
    Retrieves command line arguments created and defined using
    the argparse module. get_input_args() returns these
    arguments as an ArgumentParser object.
    Parameters:
     None - run get_input_args() to store command line arguments
    Returns:
     parser.parse_args() - data structure that stores command
     line arguments object
    """
    proj_dir = os.getcwd()
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input', type=str,
                        help='image file to be tested')
    
    parser.add_argument('checkpoint', type=str,
                        help='checkpoint file to load trained neural network from')
    
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='json file to help map numerical categories to flower names')
                        
    parser.add_argument('--top_k', type=int, default=3,
                        help='predict.py will return the top k most likely possibilities as its predictions')
    
    parser.add_argument('--gpu', action='store_true',
                        help='include \'--gpu\' in command line to use GPU for training')
    
    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        for i, param in model.named_parameters():
            param.requires_grad = False
    elif checkpoint['arch'] == 'densenet':
        model = models.densenet(pretrained=True)
        for i, param in model.named_parameters():
            param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, checkpoint['hidden_units'])),
                              ('relu1', nn.ReLU()),
                              ('fc2', nn.Linear(checkpoint['hidden_units'], 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    model.classifier = classifier
    
    model.class_to_idx = checkpoint['class_to_idx']
        
    return model
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_pil = Image.open(image)

    # define transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # preprocess the image
    img_tensor = preprocess(img_pil)
    img_tensor.requires_grad_(False)
    
    return np.array(img_tensor)
    
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    arr = process_image(image_path)
    img_tensor=Variable(torch.from_numpy(arr.reshape(1,3,224,224)))
    
    img_tensor = img_tensor.to(device)
        
    output = model.forward(img_tensor)
    
    ps = torch.exp(output)
    
    probs, class_indices = ps.topk(topk)
    
    probs = probs.to(torch.device('cpu'))
    class_indices = class_indices.to(torch.device('cpu'))
    
    probs, class_indices = np.array(probs.detach()), np.array(class_indices.detach())
    
    class_titles = [list(model.class_to_idx.keys())[list(model.class_to_idx.values()).index(x)] for x in class_indices[0]]
    
    return probs[0], class_titles
    
def main():
    in_args = get_input_args()
    
    with open(in_args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    model = load_checkpoint(in_args.checkpoint)
    
    if in_args.gpu:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        
    model = model.to(device)
    
    probs, classes = predict(in_args.input, model, device, topk=in_args.top_k)
    
    for i, (c, prob) in enumerate(zip(classes, probs)):
        print('Prediction {}'.format(i+1))
        print(cat_to_name[c].title())
        print('Probability: {0:.2f}'.format(prob))
        print('================================================')
    
    return

if __name__ == '__main__':
    main()
