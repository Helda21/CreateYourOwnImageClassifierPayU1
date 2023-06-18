#!/usr/bin/env python
# coding: utf-8

# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

# In[53]:


# Imports here
import numpy as np
import argparse
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from collections import OrderedDict
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import tensorflow as tf
import numpy as np
import json
import time
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim


# In[54]:


# Configurar la semilla aleatoria para reproducibilidad
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

# In[55]:


# Directorios de los conjuntos de datos
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# In[56]:



# Transformaciones de datos
import torchvision.models as models

# Load the data and define the transformations
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# Define the dataloaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

# Get the number of flower classes
number_of_classes = len(train_data.classes)

# Build and train the model
model = models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, number_of_classes),
    nn.LogSoftmax(dim=1)
)

model.classifier = classifier


# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[57]:


import json

# Load the mapping from category label to category name
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.
# 
# One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to
# GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.
# 
# ## Note for Workspace users: 
# If your network is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. Typically this happens with wide dense layers after the convolutional layers. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with `ls -lh`), you should reduce the size of your hidden layers and train again.

# In[58]:


# TODO: Build and train your network
def get_model():
    model = models.densenet121(pretrained=True)
    return model

def build_model(hidden_layers, class_to_idx):
    model = get_model()
    for param in model.parameters():
        param.requires_grad = False
    
    classifier_input_size = model.classifier.in_features
    print("Input size: ", classifier_input_size)
    output_size = 102

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(classifier_input_size, hidden_layers)),
        ('relu', nn.ReLU()),
        ('dropout1', nn.Dropout(0.5)),
        ('fc2', nn.Linear(hidden_layers, output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    model.class_to_idx = class_to_idx
    return model


# In[59]:


# TODO: Build and train your network
model = torchvision.models.vgg16(pretrained=True)

# freeze model parameters
for param in model.parameters():
    param.requires_grad = False

    
in_features_of_pretrained_model = model.classifier[0].in_features
    
# alter the classifier so that it has 102 out features (i.e. len(cat_to_name.json))
number_of_flower_classes = len(train_data.classes)
classifier = nn.Sequential(nn.Linear(in_features=in_features_of_pretrained_model, out_features=2048, bias=True),
                           nn.ReLU(inplace=True),
                           nn.Dropout(p=0.2),
                           nn.Linear(in_features=2048, out_features=number_of_flower_classes, bias=True),
                           nn.LogSoftmax(dim=1)
                          )

model.classifier = classifier


# ## Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[60]:


import torch
from torch import nn

# Definir el dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definir el criterio de pérdida
criterion = nn.CrossEntropyLoss()

# Definir la función de validación
def validate(model, criterion, dataloader):
    # Configurar el modo de evaluación
    model.eval()

    # Variables para el cálculo de la pérdida y precisión
    test_loss = 0
    accuracy = 0
    total = 0

    # Desactivar el cálculo de gradientes durante la validación
    with torch.no_grad():
        # Iterar sobre el conjunto de datos de validación
        for images, labels in dataloader:
            # Mover los datos a la GPU si está disponible
            images, labels = images.to(device), labels.to(device)

            # Realizar una pasada hacia adelante
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Acumular la pérdida
            test_loss += loss.item()

            # Obtener las clases predichas
            _, predicted = torch.max(outputs.data, 1)

            # Actualizar el contador de precisión
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    # Calcular la pérdida y precisión promedio
    test_loss /= len(dataloader)
    accuracy = 100 * accuracy / total

    return test_loss, accuracy


# ... (Código anterior)

# Definir el modelo
model = torchvision.models.resnet50(pretrained=True)
# Ajustar la última capa para clasificar en 102 categorías
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 102)
model.to(device)

# ... (Código anterior)

# Validar en el conjunto de prueba
test_loss, accuracy = validate(model, criterion, dataloaders['testing'])
print("Val. Accuracy: {:.3f}".format(accuracy))
print("Val. Loss: {:.3f}".format(test_loss))


# In[61]:


def validate(model, criterion, dataloader):
    # Configurar el modo de evaluación
    model.eval()

    # Variables para el cálculo de la pérdida y precisión
    test_loss = 0
    accuracy = 0
    total = 0

    # Desactivar el cálculo de gradientes durante la validación
    with torch.no_grad():
        # Iterar sobre el conjunto de datos de validación
        for images, labels in dataloader:
            # Mover los datos a la GPU si está disponible
            images, labels = images.to(device), labels.to(device)

            # Realizar una pasada hacia adelante
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Acumular la pérdida
            test_loss += loss.item()

            # Obtener las clases predichas
            _, predicted = torch.max(outputs.data, 1)

            # Actualizar el contador de precisión
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    # Calcular la pérdida y precisión promedio
    test_loss /= len(dataloader)
    accuracy = 100 * accuracy / total

    return test_loss, accuracy


# Definir el dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definir el criterio de pérdida
criterion = nn.CrossEntropyLoss()

# Validar en el conjunto de prueba
test_loss, accuracy = validate(model, criterion, dataloaders['testing'])
print("Val. Accuracy: {:.3f}".format(accuracy))
print("Val. Loss: {:.3f}".format(test_loss))


# In[86]:


# TODO: Do validation on the test  set
#todo ok
num_epochs = 10


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[62]:


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# In[63]:


# TODO: Save the checkpoint 
checkpoint = {'model': model,
              'state_dict': model.state_dict(),
              'epochs': epochs,
              'optimizer_state_dict': optimizer.state_dict(),
              'class_to_idx': class_to_idx
             }

torch.save(checkpoint, 'checkpoint.pth')


# In[47]:


#ok


# In[46]:


#ok


# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[64]:


# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    # Load the checkpoint
    checkpoint = torch.load(filepath)

    # Rebuild the model architecture
    model = models.vgg16(pretrained=True)
    model.classifier = checkpoint['classifier']

    # Load the model state
    model.load_state_dict(checkpoint['state_dict'])

    # Load the mapping of classes to indices
    model.class_to_idx = checkpoint['class_to_idx']

    return model


# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# In[65]:


#def process_image(image):
   # ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
     #   returns an Numpy array
    #'''
    
    # TODO: Process a PIL image for use in a PyTorch model
    

from PIL import Image
import numpy as np
import torch

def process_image(image_path):
    # Abrir la imagen usando PIL
    image = Image.open(image_path)

    # Redimensionar la imagen manteniendo la relación de aspecto
    image = image.resize((256, 256))
    width, height = image.size

    # Calcular las dimensiones del recorte
    new_width, new_height = 224, 224
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    # Recortar la imagen
    image = image.crop((left, top, right, bottom))

    # Convertir la imagen a un arreglo de Numpy
    np_image = np.array(image)

    # Normalizar la imagen
    np_image = np_image / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Transponer el canal de color
    np_image = np_image.transpose((2, 0, 1))

    # Convertir el arreglo de Numpy a un tensor de PyTorch
    tensor_image = torch.from_numpy(np_image)

    # Agregar una dimensión de lote (batch dimension)
    tensor_image = tensor_image.unsqueeze(0)

    return tensor_image


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# In[66]:


"""
def imshow(image, ax=None, title=None):
    Imshow for Tensor.
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
"""

def imshow(image, ax=None, title=None):
    
    if ax is None:
        fig, ax = plt.subplots()
    
    # Los tensores de PyTorch asumen que el canal de color es la primera dimensión,
    # pero matplotlib asume que es la tercera dimensión
    image = image.numpy().transpose((1, 2, 0))
    
    # Deshacer la preprocesamiento
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Asegurarse de que la imagen esté entre 0 y 1 para que se vea correctamente
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[67]:


def predict(image_path, model, topk=5):
    # Cargar el punto de control del modelo
    checkpoint = torch.load(model)
    
    # Reconstruir el modelo a partir del punto de control
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    # Preprocesar la imagen
    image = process_image(image_path)
    
    # Convertir la imagen a un tensor
    image_tensor = torch.from_numpy(image).unsqueeze(0)
    
    # Mover el tensor a la GPU si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)
    
    # Pasar la imagen a través del modelo
    model.eval()
    with torch.no_grad():
        output = model.forward(image_tensor)
    
    # Calcular las probabilidades y las clases más probables
    probabilities = torch.exp(output)
    top_probs, top_classes = probabilities.topk(topk)
    
    # Obtener las etiquetas de clase correspondientes a los índices
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_labels = [idx_to_class[class_idx] for class_idx in top_classes.squeeze().tolist()]
    
    return top_probs.squeeze().tolist(), top_labels


# ## Sanity Checking
# 
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
# 
# <img src='assets/inference_example.png' width=300px>
# 
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# In[69]:


# TODO: Display an image along with the top 5 classes
import matplotlib.pyplot as plt

def predict(image_path, model_checkpoint, topk=5):
    # Cargar el punto de control del modelo
    checkpoint = torch.load(model_checkpoint)
    
    # Crear el modelo y cargar los pesos del punto de control
    model = get_model()
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    # Preprocesar la imagen
    image = process_image(image_path)
    image = torch.from_numpy(image).unsqueeze(0)
    
    # Pasar la imagen por el modelo para obtener las probabilidades
    model.eval()
    with torch.no_grad():
        output = model(image)
        probabilities = torch.exp(output)
    
    # Obtener las probabilidades y las clases más probables
    top_probabilities, top_indices = torch.topk(probabilities, topk)
    top_probabilities = top_probabilities.squeeze().tolist()
    top_indices = top_indices.squeeze().tolist()
    
    # Convertir los índices a las clases correspondientes
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[index] for index in top_indices]
    
    return top_probabilities, top_classes


# In[ ]:





# ## Reminder for Workspace users
# If your network becomes very large when saved as a checkpoint, there might be issues with saving backups in your workspace. You should reduce the size of your hidden layers and train again. 
#     
# We strongly encourage you to delete these large interim files and directories before navigating to another page or closing the browser tab.

# In[70]:


# TODO remove .pth files or move it to a temporary `~/opt` directory in this Workspace
import os
import shutil

# Ruta del directorio temporal
temp_dir = '/home/workspace/opt'

# Verificar si el directorio temporal existe, si no, crearlo
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Ruta del directorio actual
current_dir = os.getcwd()

# Obtener la lista de archivos .pth en el directorio actual
pth_files = [file for file in os.listdir(current_dir) if file.endswith('.pth')]

# Mover los archivos .pth al directorio temporal
for file in pth_files:
    shutil.move(os.path.join(current_dir, file), os.path.join(temp_dir, file))

print("Archivos .pth movidos al directorio temporal.")


# In[ ]:




