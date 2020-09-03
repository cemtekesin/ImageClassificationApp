from time import time
import torch
import torchvision
from torch import nn, optim
import numpy as np
import torch
import json
from PIL import Image
from torch.autograd import Variable 
from arguments_predict import get_inputs

args = get_inputs()
filename = args.checkpoint
load_dir = args.load_dir
image_path = args.image
topk = args.top_k
checkpoint_name = load_dir+"/"+filename
cat_to_name = args.category_names
with open(cat_to_name, 'r') as f:
    categories = json.load(f)
output_size = len(categories)

#Enable GPU if available and not disabled by user
if (torch.cuda.is_available() and args.gpu):
    device = "cuda"
else:
    device = "cpu"

# TODO: Write a function that loads a checkpoint and rebuilds the model

def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    if (checkpoint['architecture'] == 'resnet152'):
        input_size = checkpoint['input_size']
        hidden_layers = checkpoint['hidden_layers']
        output_size = 102
        learn_rate = checkpoint['learning_rate']
        drop_p = checkpoint['dropout']
        model = getattr(torchvision.models, checkpoint['architecture'])(pretrained = True)
        model.fc = checkpoint['fc']
        model.epochs = checkpoint['epochs']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        optimizer = optim.Adam(model.fc.parameters(), learn_rate)
        optimizer.load_state_dict(checkpoint['optimizer'])
        criterion = nn.NLLLoss()
    else:
        input_size = checkpoint['input_size']
        hidden_layers = checkpoint['hidden_layers']
        output_size = 102
        learn_rate = checkpoint['learning_rate']
        drop_p = checkpoint['dropout']
        model = getattr(torchvision.models, checkpoint['architecture'])(pretrained = True)
        model.classifier = checkpoint['classifier']
        model.epochs = checkpoint['epochs']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        optimizer = optim.Adam(model.classifier.parameters(), learn_rate)
        optimizer.load_state_dict(checkpoint['optimizer'])
        criterion = nn.NLLLoss()
        
    return model, optimizer, criterion, model.class_to_idx



model, optimizer, criterion, class_to_idx = load_checkpoint(checkpoint_name)

def process_image(image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        img=Image.open(image)
    
        width, height = img.size
        if width< height:
            img.resize((255, int(height/width)))
        else:
            img.resize((int(255*width/height),255))
    
        width, height = img.size
        left = (width - 224)/2
        top = (height - 224)/2
        right = (width + 224)/2
        bottom = (height + 224)/2
        img = img.crop((left, top, right, bottom))
    
        #Turn image into a numpy array
        img = np.array(img)
    
        #adjust image values between 0 and 1
    
        img = img/255
    
        #Normalize
    
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std

        #Make the color channel dimension first with transpose
        img = img.transpose((2,0,1))
    
        return img

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
        # PyTorch tensors assume the color channel is the first dimension
        # but matplotlib assumes is the third dimension
        image = image.transpose((1, 2, 0))
    
        # Undo preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
      
    
        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)
    
        ax.imshow(image)
    
        return ax

#print(model)
#print(optimizer)
#print(criterion)
#print(optimizer)


def pred(image_path, model, topk):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        #Turn off dropouts and put model into evaluation mode
        model.eval()
        model.cpu()
    
        #Open the image
        image = process_image(image_path)

        #Transfer to tensor
        image = torch.from_numpy(np.array([image])).float()
    
        #Image becomes input with Variable()
        image = Variable(image)

    
    
        with torch.no_grad():
            output = model.forward(image)    
            ps = output.exp()
            probs, classes = torch.topk(ps, topk)
        
            class_to_idx_rev = {model.class_to_idx[k]: k for k in model.class_to_idx}

            labels = []

            for label in classes.numpy()[0]:
                labels.append(class_to_idx_rev[label])

            return probs.numpy()[0], labels

# TODO: Implement the code to predict the class from an image file
prob, classes = pred(image_path, model, topk)
for x, y in zip (prob, [categories[x] for x in classes]):
    print(x, y)
