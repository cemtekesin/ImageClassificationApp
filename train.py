# Imports here
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from workspace_utils import active_session
from PIL import Image
import os, random
from torch.autograd import Variable 
from arguments import get_inputs
import json

# TODO: Define your transforms for the training, validation, and testing sets
#Rotation training set prior resizing increase training performance
#RandomResizedCrop size is output size of each edge, scale is range of size of the origin size croppped, ratio is range of aspect ratio, and default interpolation is 1(PIL)
#Degrees: 60 means -60 to 60 actually, Instead of 30, I used -30,30 with 2 integers
#Color should stay same but I had tested ColorJitter as well, it droped the performance transforms.ColorJitter(brightness=1.0, contrast=1.0, saturation=1.0, hue=0.5),
#train_transforms = transforms.Compose([transforms.RandomRotation(degrees=(-30,30)),
#                                       transforms.RandomResizedCrop(size=224, scale=(0.75, 1.25), ratio=(0.5,1.78), interpolation =1),
#                                       transforms.ColorJitter(brightness=1.0, contrast=1.0, saturation=1.0, hue=0.5),
#                                      transforms.RandomHorizontalFlip(),transforms.ToTensor(),
#                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])



args = get_inputs()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'



train_transforms = transforms.Compose([transforms.RandomRotation(degrees=(-30,30)),
                                       transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform = validation_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
validationloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
testloader = torch.utils.data.DataLoader(test_dataset,batch_size=64)


#load category names
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

#Enable GPU if available and not disabled by user
if (torch.cuda.is_available() and args.gpu):
    device = "cuda"
else:
    device = "cpu"

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Feed Forward Classifier (Remeber to freeze parameters to avoid backprop)

model_architecture = args.arch
model_vgg11 = 'vgg11'
model_vgg13 = 'vgg13'
model_vgg16 = 'vgg16'
model_densenet121 = 'densenet121'
model_densenet169 = 'densenet169'
model_densenet201 = 'densenet201'
model_alexnet = 'alexnet'
model_resnet152 = 'resnet152'

hidden_unit = args.hidden_units
drop_p = args.dropout
learn_rate = args.learn_rate
batch_s = args.batch_size   

#Starting to build model based on selected architecture. Default is VGG16

    
#If selected architecture is VGG16
if (model_architecture == model_vgg16):
    model_new = models.vgg16(pretrained=True)
    input_size = 512*7*7
    for param in model_new.parameters():
        param.requires_grad = False
    
    model_new.classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size,hidden_unit, bias = True)),
                                                ('dropout', nn.Dropout (p = drop_p)),
                                                ('relu1', nn.ReLU()),
                                                ('fc2', nn.Linear(hidden_unit, 102, bias = True)),
                                                ('logsoftmax_output', nn.LogSoftmax(dim=1))]))
#If selected architecture is VGG13
if (model_architecture == model_vgg13):
    model_new = models.vgg13(pretrained=True)
    input_size = 512*7*7
    
    for param in model_new.parameters():
        param.requires_grad = False

    model_new.classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size,hidden_unit, bias = True)),
                                                ('dropout', nn.Dropout (p = drop_p)),
                                                ('relu1', nn.ReLU()),
                                                ('fc2', nn.Linear(hidden_unit, 102, bias = True)),
                                                ('logsoftmax_output', nn.LogSoftmax(dim=1))]))
#If selected architecture is VGG11
if (model_architecture == model_vgg11):
    model_new = models.vgg11(pretrained=True)
    input_size = 512*7*7
    
    for param in model_new.parameters():
        param.requires_grad = False
    
    model_new.classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size,hidden_unit, bias = True)),
                                                ('dropout', nn.Dropout (p = drop_p)),
                                                ('relu1', nn.ReLU()),
                                                ('fc2', nn.Linear(hidden_unit, 102, bias = True)),
                                                ('logsoftmax_output', nn.LogSoftmax(dim=1))]))
        

#If selected architecture is densenet121
if (model_architecture == model_densenet121):
    model_new = models.densenet121(pretrained=True)
    input_size = 1024
    
    for param in model_new.parameters():
        param.requires_grad = False
    
    model_new.classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size,hidden_unit, bias = True)),
                                                ('dropout', nn.Dropout (p = drop_p)),
                                                ('relu1', nn.ReLU()),
                                                ('fc2', nn.Linear(hidden_unit, 102, bias = True)),
                                                ('logsoftmax_output', nn.LogSoftmax(dim=1))]))
#If selected architecture is densenet169
if (model_architecture == model_densenet169):
    model_new = models.densenet169(pretrained=True)
    input_size = 1664
    
    for param in model_new.parameters():
        param.requires_grad = False
        
    model_new.classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size,hidden_unit, bias = True)),
                                                ('dropout', nn.Dropout (p = drop_p)),
                                                ('relu1', nn.ReLU()),
                                                ('fc2', nn.Linear(hidden_unit, 102, bias = True)),
                                                ('logsoftmax_output', nn.LogSoftmax(dim=1))]))
#If selected architecture is densenet201
if (model_architecture == model_densenet201):
    model_new = models.densenet201(pretrained=True)
    input_size = 1920
    
    for param in model_new.parameters():
        param.requires_grad = False
    
    model_new.classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size,hidden_unit, bias = True)),
                                                ('dropout', nn.Dropout (p = drop_p)),
                                                ('relu1', nn.ReLU()),
                                                ('fc2', nn.Linear(hidden_unit, 102, bias = True)),
                                                ('logsoftmax_output', nn.LogSoftmax(dim=1))]))
        
#If selected architecture is alexnet
if (model_architecture == model_alexnet):
    model_new = models.alexnet(pretrained=True)
    input_size = 9216
    
    for param in model_new.parameters():
        param.requires_grad = False
    
    model_new.classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size,hidden_unit, bias = True)),
                                                ('dropout', nn.Dropout (p = drop_p)),
                                                ('relu1', nn.ReLU()),
                                                ('fc2', nn.Linear(hidden_unit, 102, bias = True)),
                                                ('logsoftmax_output', nn.LogSoftmax(dim=1))]))
        
#If selected architecture is resnet152
if (model_architecture == model_resnet152):
    model_new = models.resnet152(pretrained=True)
    input_size = 2048
    
    for param in model_new.parameters():
        param.requires_grad = False
    
    model_new.fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size,hidden_unit, bias = True)),
                                                ('dropout', nn.Dropout (p = drop_p)),
                                                ('relu1', nn.ReLU()),
                                                ('fc2', nn.Linear(hidden_unit, 102, bias = True)),
                                                ('logsoftmax_output', nn.LogSoftmax(dim=1))]))
        
criterion = nn.NLLLoss()

#Because REsnet's last layer is not self.classifier, you need to change it, I assigned it to model_new.fc instead, hence optimizer need to be adapted below as well

if (model_architecture == model_resnet152):
    optimizer = optim.Adam(model_new.fc.parameters(), lr = learn_rate)
else:
    optimizer = optim.Adam(model_new.classifier.parameters(), lr= learn_rate)

model_new.to(device)


#Train Model
print('Selected device is: ',device)
with active_session():
    epochs = args.epoch
    steps = 0
    running_loss = 0
    print_every = 20
    for epoch in range(epochs):
        for images, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            images, labels = images.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model_new.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model_new.eval()
                with torch.no_grad():
                    for inputs, outputs in validationloader:
                        inputs, outputs = inputs.to(device), outputs.to(device)
                        logps = model_new.forward(inputs)
                        batch_loss = criterion(logps, outputs)
                    
                        validation_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == outputs.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        #Validation Loss and Accuracy                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(validationloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validationloader):.3f}")
                running_loss = 0
                model_new.train()

#Saving the Checkpoint
model_new.class_to_idx = train_dataset.class_to_idx

if (model_architecture == model_resnet152):
    checkpoint = {
    'input_size' : input_size,
    'hidden_layers' : [hidden_unit],
    'output_size' : 102,         
    'state_dict' : model_new.state_dict(),
    'optimizer' : optimizer.state_dict(),
    'fc' : model_new.fc,
    'class_to_idx' : model_new.class_to_idx,
    'epochs' : epochs,
    'batch_size' : batch_s,
    'dropout' : drop_p,
    'learning_rate' : learn_rate,
    'architecture' : model_architecture}
else:
    checkpoint = {'input_size' : input_size,
    'hidden_layers' : [hidden_unit],
    'output_size' : 102,         
    'state_dict' : model_new.state_dict(),
    'optimizer' : optimizer.state_dict(),
    'classifier' : model_new.classifier,
    'class_to_idx' : model_new.class_to_idx,
    'epochs' : epochs,
    'batch_size' : batch_s,
    'dropout' : drop_p,
    'learning_rate' : learn_rate,
    'architecture' : model_architecture
}
save_folder = args.save_dir
file_name = args.checkpoint
save_name = save_folder+"/"+file_name
torch.save(checkpoint, save_name)
print("Checkpoint saved as: ", save_name)