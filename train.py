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

# get the command line args and save into variables
parser = argparse.ArgumentParser()

parser.add_argument('data_dir', type=str, help='Set path to training images.')
parser.add_argument('--save_dir', type=str, default='./', help='Set directory to save checkpoints. Default is current directory.')
parser.add_argument('--arch', type=str, default='densenet121', choices={"resnet18",
                                                                        "vgg13",
                                                                        "densenet121"
                                                                        },
                    help='Choose architecture. Default is densenet121')

''' TODO add these model options for --arch
                                                                        "alexnet",
                                                                        "squeezenet1_0",
                                                                        "inception_v3",
                                                                        "googlenet",
                                                                        "shufflenet_v2_x1_0",
                                                                        "mobilenet_v2",
                                                                        "resnext50_32x4d",
                                                                        "wide_resnet50_2",
                                                                        "mnasnet1_0"
'''

parser.add_argument('--learning_rate', default=0.01, type=float, help='Set learning rate. Default is 0.01')
parser.add_argument('--hidden_units', default=512, type=int, help='Set number of hidden units. Default is 512')
parser.add_argument('--epochs', default=5, type=int, help='Set number of training epochs. Default is 5')
parser.add_argument('--gpu', action='store_true', help='Use GPU for training (if available). If this flag is not set or GPU is unavailable, default is CPU')

args = parser.parse_args()

data_dir = args.data_dir
save_dir = args.save_dir
arch = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
gpu = args.gpu

# build paths for training, validation & test data
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define transforms for the training, validation, and testing sets
# train transform includes some augmentation steps
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

# Load the categories to names mappings into a dictionary
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Build network
# Use GPU if command line flag is set, and it's available on the host
device = torch.device("cuda:0" if (gpu and torch.cuda.is_available()) else "cpu")

#Load pre-trained model
match arch:
    case "resnet18":
        model = models.resnet18(pretrained=True)
        classifier_input_units = 512
    case "vgg13":
        model = models.vgg13(pretrained=True)
        classifier_input_units = 25088
    case "densenet121":
        model = models.densenet121(pretrained=True)
        #model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        classifier_input_units = 1024
# TODO allow these other models
'''
    case "alexnet":
        model = models.alexnet(pretrained=True)
    case "squeezenet1_0":
        model = models.squeezenet1_0(pretrained=True)
    case "inception_v3":
        model = models.inception_v3(pretrained=True)
    case "googlenet":
        model = models.googlenet(pretrained=True)
    case "shufflenet_v2_x1_0":
        model = models.shufflenet_v2_x1_0(pretrained=True)
    case "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
    case "resnext50_32x4d":
        model = models.resnext50_32x4d(pretrained=True)
    case "wide_resnet50_2":
        model = models.wide_resnet50_2(pretrained=True)
    case "mnasnet1_0":
        model = models.mnasnet1_0(pretrained=True)
'''

# Turn off gradients
for param in model.parameters():
    param.requires_grad = False

# Replace existing classifier with input layer (1024), hidden layer (512), output layer (102)
# Classifier input layer has 1024 nodes to match densenet classifier 
# Classifier output layer has 102 nodes because there are 102 classes in the cat_to_name.json file
classifier = nn.Sequential(OrderedDict([
                          # hidden layer with relu activation and dropout
                          ('fc1', nn.Linear(classifier_input_units, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(0.2)),

                          # output layer
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

# Add new classifier to slected model
model.classifier = classifier

# Set the criterion to Negative Log Likelihood Loss
criterion = nn.NLLLoss()

# Set the optimizer and learnign rate
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# Move model to GPU if available
model.to(device);

# Initialize training variables
print_every = 20
steps = 0
running_loss = 0

# Train the model
for epoch in range(epochs):
    # training pass
    model.train()
    for inputs, labels in train_loader:
        steps += 1
        
        # Move input and label tensors to the device (GPU, or CPU if GPU not available)
        inputs, labels = inputs.to(device), labels.to(device)

        # clear gradients
        optimizer.zero_grad()
        
        # do the forward prop
        logps = model.forward(inputs)

        # calculate the loss
        loss = criterion(logps, labels)
        
        # do the backwards prop
        loss.backward()
        optimizer.step()

        # add training loss
        running_loss += loss.item()
        
        if steps % print_every == 0:
            with torch.no_grad():
                model.eval()
                validation_loss = 0
                accuracy = 0

                # validation pass
                for inputs, labels in valid_loader:
                    # Move input and label tensors to the device (GPU, or CPU if GPU not available)
                    inputs, labels = inputs.to(device), labels.to(device)
                
                    logps = model(inputs)
                    loss = criterion(logps, labels)
                    validation_loss += loss.item()

                    # calculate validation accuracy
                    ps = torch.exp(logps)
                    top_ps, top_class = ps.topk(1, dim=1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Step: {steps}.. "
                  f"Train loss: {running_loss / print_every:.3f}.. "
                  f"Validation loss: {validation_loss / len(valid_loader):.3f}.. "
                  f"Validation accuracy: {accuracy / len(valid_loader):.3f}")
            
            running_loss = 0
            model.train()

# Save the checkpoint 
# Grab the current date/time so we can build a unique checkpoint filename
from datetime import datetime
ts = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
checkpoint_filename = arch + '_checkpoint_' + ts + '.pth'

# Save the mapping of flower labels to indexes to the model
model.class_to_idx = train_data.class_to_idx

checkpoint = {'architecture': arch,
              'classifier': model.classifier,
              'model_class_to_idx': model.class_to_idx,
              'epochs': epochs,
              'optimizer_state_dict': optimizer.state_dict(),
              'model_state_dict': model.state_dict(),
              'learning_rate': learning_rate,
              'hidden_units': hidden_units}

torch.save(checkpoint, checkpoint_filename)