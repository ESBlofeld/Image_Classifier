import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
from collections import OrderedDict


def load_checkpoint_file(filepath, device='cpu'):
    '''Function that loads a checkpoint and rebuilds the model
    '''
    checkpoint = torch.load(filepath, map_location=device)
    model = getattr(models, checkpoint['architecture'])(pretrained=True)
    model.class_to_idx = checkpoint['model_class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    with Image.open(image) as im:
        # find the shortest side
        if im.width < im.height:
            short = im.width
        else:
            short = im.height

        # determine ratio to resize to 256
        resize_ratio = 256 / short
        resized_height = int(im.height * resize_ratio)
        resized_width = int(im.width * resize_ratio)
        
        # resize the image
        im_resized = im.resize((resized_width, resized_height))
        
        # find coords of 224x224 crop box
        left = int((resized_width - 224) / 2)
        upper = int((resized_height - 224) / 2)
        right = left + 224
        lower = upper + 224
        
        # crop the image
        im_crop = im_resized.crop((left, upper, right, lower))
        
        # re-encode color channels as floats
        im_encoded = np.array(im_crop) / 255
        
        # normalize image for classification
        im_normalized = (im_encoded - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        
        # re-order dimensions for Pytorch (3rd dimension goes first, then 1st and 2nd in order)
        im_reordered = im_normalized.transpose((2, 0, 1))

        return im_reordered

def predict(image_path, model, proc, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # process image to be classified
    image_proc_np = process_image(image_path)
    #print(image_proc_np.shape)

    # resize numpy array
    image_proc_np = np.resize(image_proc_np, (1,3,224,224))

    # now convert numpy array to a tensor
    image_proc_tensor = torch.from_numpy(image_proc_np).type(torch.FloatTensor)

    
    # Move input to the device (GPU, or CPU if GPU not available)
    image_proc_tensor = image_proc_tensor.to(proc)

    # move model to device (CPU or GPU)
    model.to(proc)
    
    model.eval()
    with torch.no_grad():
        outputs = model(image_proc_tensor)
        
    probabilities = torch.exp(outputs)
        
    # get the top k probabilities and class indexes
    top_ps, top_class = torch.topk(probabilities, dim=1, k=topk)

    # make tensors into numpy arrays
    top_ps, top_class = top_ps.to('cpu').numpy(), top_class.to('cpu').numpy()

    #make numpy arrays into lists, so can get class names from indexes
    top_ps, top_class = top_ps[0].tolist(), top_class[0].tolist()

    # use the index - class mapping to get the class for the top k indexes
    index_to_class = {idx: class_ for class_, idx in model.class_to_idx.items()}
    # reverse the mapping to get top k classes
    classes = [index_to_class[i] for i in top_class]

    return top_ps, classes
    

def main():
    # get the command line args and save into variables
    parser = argparse.ArgumentParser()

    parser.add_argument('image_path', type=str, help='Set path and filename of test image e.g. "./images/dandelion.jpg"')
    parser.add_argument('cat_name_path', type=str, help='Set path and filename of category to names mapping file (json) e.g. "./cat_to_names.json"')
    parser.add_argument('checkpoint_path', type=str, help='Set path and filename to checkpoint file e.g. "./checkpoint.pth"')
    parser.add_argument('--top_k', default=1, type=int, help='Set top k value e.g. 5')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training (if available). If this flag is not set or GPU is unavailable, default is CPU')

    args = parser.parse_args()

    image_path = args.image_path
    checkpoint_path = args.checkpoint_path
    top_k = args.top_k
    cat_name_path = args.cat_name_path
    gpu = args.gpu

    # Use GPU if command line flag is set, and it's available on the host
    device = torch.device("cuda" if (gpu and torch.cuda.is_available()) else "cpu")

    # Load the categories to names mappings into a dictionary
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    # load model from checkpoint file
    model = load_checkpoint_file(checkpoint_path, device)

    # Make prediction on supplied image and return top k probablities and classes
    probabilities, classes = predict(image_path, model, device, top_k)

    # get top k flower names
    flower_names_from_classes = [cat_to_name[i] for i in classes]

    for name, prob in zip(flower_names_from_classes, probabilities):
        print(f"'{name}' has probability of {prob:.3f}")

if __name__ == "__main__":
    main()