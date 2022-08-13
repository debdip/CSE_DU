"""

Quickstart
===================
This section runs through the API for common tasks in machine learning. Refer to the links in each section to dive deeper.

Working with data
-----------------
PyTorch has two `primitives to work with data <https://pytorch.org/docs/stable/data.html>`_:
``torch.utils.data.DataLoader`` and ``torch.utils.data.Dataset``.
``Dataset`` stores the samples and their corresponding labels, and ``DataLoader`` wraps an iterable around
the ``Dataset``.

"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

######################################################################
# PyTorch offers domain-specific libraries such as `TorchText <https://pytorch.org/text/stable/index.html>`_,
# `TorchVision <https://pytorch.org/vision/stable/index.html>`_, and `TorchAudio <https://pytorch.org/audio/stable/index.html>`_,
# all of which include datasets. For this tutorial, we  will be using a TorchVision dataset.
#
# The ``torchvision.datasets`` module contains ``Dataset`` objects for many real-world vision data like
# CIFAR, COCO (`full list here <https://pytorch.org/vision/stable/datasets.html>`_). In this tutorial, we
# use the FashionMNIST dataset. Every TorchVision ``Dataset`` includes two arguments: ``transform`` and
# ``target_transform`` to modify the samples and labels respectively.

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

######################################################################
# We pass the ``Dataset`` as an argument to ``DataLoader``. This wraps an iterable over our dataset, and supports
# automatic batching, sampling, shuffling and multiprocess data loading. Here we define a batch size of 64, i.e. each element
# in the dataloader iterable will return a batch of 64 features and labels.

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

######################################################################
# Read more about `loading data in PyTorch <data_tutorial.html>`_.
#

######################################################################
# --------------
#

################################
# Creating Models
# ------------------
# To define a neural network in PyTorch, we create a class that inherits
# from `nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_. We define the layers of the network
# in the ``__init__`` function and specify how data will pass through the network in the ``forward`` function. To accelerate
# operations in the neural network, we move it to the GPU if available.

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

######################################################################
# Read more about `building neural networks in PyTorch <buildmodel_tutorial.html>`_.
#


######################################################################
# --------------
#


#####################################################################
# Optimizing the Model Parameters
# ----------------------------------------
# To train a model, we need a `loss function <https://pytorch.org/docs/stable/nn.html#loss-functions>`_
# and an `optimizer <https://pytorch.org/docs/stable/optim.html>`_.

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


#######################################################################
# In a single training loop, the model makes predictions on the training dataset (fed to it in batches), and
# backpropagates the prediction error to adjust the model's parameters.

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

##############################################################################
# We also check the model's performance against the test dataset to ensure it is learning.

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

##############################################################################
# The training process is conducted over several iterations (*epochs*). During each epoch, the model learns
# parameters to make better predictions. We print the model's accuracy and loss at each epoch; we'd like to see the
# accuracy increase and the loss decrease with every epoch.

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

######################################################################
# Read more about `Training your model <optimization_tutorial.html>`_.
#

######################################################################
# --------------
#

######################################################################
# Saving Models
# -------------
# A common way to save a model is to serialize the internal state dictionary (containing the model parameters).

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")



######################################################################
# Loading Models
# ----------------------------
#
# The process for loading a model includes re-creating the model structure and loading
# the state dictionary into it.

model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))

#############################################################
# This model can now be used to make predictions.

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')


"""


Tensors
==========================

Tensors are a specialized data structure that are very similar to arrays and matrices.
In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model’s parameters.

Tensors are similar to `NumPy’s <https://numpy.org/>`_ ndarrays, except that tensors can run on GPUs or other hardware accelerators. In fact, tensors and
NumPy arrays can often share the same underlying memory, eliminating the need to copy data (see :ref:`bridge-to-np-label`). Tensors
are also optimized for automatic differentiation (we'll see more about that later in the `Autograd <autogradqs_tutorial.html>`__
section). If you’re familiar with ndarrays, you’ll be right at home with the Tensor API. If not, follow along!
"""

import torch
import numpy as np


######################################################################
# Initializing a Tensor
# ~~~~~~~~~~~~~~~~~~~~~
#
# Tensors can be initialized in various ways. Take a look at the following examples:
#
# **Directly from data**
#
# Tensors can be created directly from data. The data type is automatically inferred.

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

######################################################################
# **From a NumPy array**
#
# Tensors can be created from NumPy arrays (and vice versa - see :ref:`bridge-to-np-label`).
np_array = np.array(data)
x_np = torch.from_numpy(np_array)


###############################################################
# **From another tensor:**
#
# The new tensor retains the properties (shape, datatype) of the argument tensor, unless explicitly overridden.

x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")


######################################################################
# **With random or constant values:**
#
# ``shape`` is a tuple of tensor dimensions. In the functions below, it determines the dimensionality of the output tensor.

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")



######################################################################
# --------------
#

######################################################################
# Attributes of a Tensor
# ~~~~~~~~~~~~~~~~~
#
# Tensor attributes describe their shape, datatype, and the device on which they are stored.

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


######################################################################
# --------------
#

######################################################################
# Operations on Tensors
# ~~~~~~~~~~~~~~~~~
#
# Over 100 tensor operations, including arithmetic, linear algebra, matrix manipulation (transposing,
# indexing, slicing), sampling and more are
# comprehensively described `here <https://pytorch.org/docs/stable/torch.html>`__.
#
# Each of these operations can be run on the GPU (at typically higher speeds than on a
# CPU). If you’re using Colab, allocate a GPU by going to Runtime > Change runtime type > GPU.
#
# By default, tensors are created on the CPU. We need to explicitly move tensors to the GPU using
# ``.to`` method (after checking for GPU availability). Keep in mind that copying large tensors
# across devices can be expensive in terms of time and memory!

# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")


######################################################################
# Try out some of the operations from the list.
# If you're familiar with the NumPy API, you'll find the Tensor API a breeze to use.
#

###############################################################
# **Standard numpy-like indexing and slicing:**

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

######################################################################
# **Joining tensors** You can use ``torch.cat`` to concatenate a sequence of tensors along a given dimension.
# See also `torch.stack <https://pytorch.org/docs/stable/generated/torch.stack.html>`__,
# another tensor joining op that is subtly different from ``torch.cat``.
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)


######################################################################
# **Arithmetic operations**

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)


######################################################################
# **Single-element tensors** If you have a one-element tensor, for example by aggregating all
# values of a tensor into one value, you can convert it to a Python
# numerical value using ``item()``:

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))


######################################################################
# **In-place operations**
# Operations that store the result into the operand are called in-place. They are denoted by a ``_`` suffix.
# For example: ``x.copy_(y)``, ``x.t_()``, will change ``x``.

print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

######################################################################
# .. note::
#      In-place operations save some memory, but can be problematic when computing derivatives because of an immediate loss
#      of history. Hence, their use is discouraged.



######################################################################
# --------------
#


######################################################################
# .. _bridge-to-np-label:
#
# Bridge with NumPy
# ~~~~~~~~~~~~~~~~~
# Tensors on the CPU and NumPy arrays can share their underlying memory
# locations, and changing one will change	the other.


######################################################################
# Tensor to NumPy array
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

######################################################################
# A change in the tensor reflects in the NumPy array.

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")


######################################################################
# NumPy array to Tensor
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
n = np.ones(5)
t = torch.from_numpy(n)

######################################################################
# Changes in the NumPy array reflects in the tensor.
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")


"""

Datasets & DataLoaders
===================

"""

#################################################################
# Code for processing data samples can get messy and hard to maintain; we ideally want our dataset code
# to be decoupled from our model training code for better readability and modularity.
# PyTorch provides two data primitives: ``torch.utils.data.DataLoader`` and ``torch.utils.data.Dataset``
# that allow you to use pre-loaded datasets as well as your own data.
# ``Dataset`` stores the samples and their corresponding labels, and ``DataLoader`` wraps an iterable around
# the ``Dataset`` to enable easy access to the samples.
#
# PyTorch domain libraries provide a number of pre-loaded datasets (such as FashionMNIST) that
# subclass ``torch.utils.data.Dataset`` and implement functions specific to the particular data.
# They can be used to prototype and benchmark your model. You can find them
# here: `Image Datasets <https://pytorch.org/vision/stable/datasets.html>`_,
# `Text Datasets  <https://pytorch.org/text/stable/datasets.html>`_, and
# `Audio Datasets <https://pytorch.org/audio/stable/datasets.html>`_
#

############################################################
# Loading a Dataset
# -------------------
#
# Here is an example of how to load the `Fashion-MNIST <https://research.zalando.com/project/fashion_mnist/fashion_mnist/>`_ dataset from TorchVision.
# Fashion-MNIST is a dataset of Zalando’s article images consisting of 60,000 training examples and 10,000 test examples.
# Each example comprises a 28×28 grayscale image and an associated label from one of 10 classes.
#
# We load the `FashionMNIST Dataset <https://pytorch.org/vision/stable/datasets.html#fashion-mnist>`_ with the following parameters:
#  - ``root`` is the path where the train/test data is stored,
#  - ``train`` specifies training or test dataset,
#  - ``download=True`` downloads the data from the internet if it's not available at ``root``.
#  - ``transform`` and ``target_transform`` specify the feature and label transformations


import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


#################################################################
# Iterating and Visualizing the Dataset
# -----------------
#
# We can index ``Datasets`` manually like a list: ``training_data[index]``.
# We use ``matplotlib`` to visualize some samples in our training data.

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

#################################################################
# ..
#  .. figure:: /_static/img/basics/fashion_mnist.png
#    :alt: fashion_mnist


######################################################################
# --------------
#

#################################################################
# Creating a Custom Dataset for your files
# ---------------------------------------------------
#
# A custom Dataset class must implement three functions: `__init__`, `__len__`, and `__getitem__`.
# Take a look at this implementation; the FashionMNIST images are stored
# in a directory ``img_dir``, and their labels are stored separately in a CSV file ``annotations_file``.
#
# In the next sections, we'll break down what's happening in each of these functions.


import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


#################################################################
# __init__
# ^^^^^^^^^^^^^^^^^^^^
#
# The __init__ function is run once when instantiating the Dataset object. We initialize
# the directory containing the images, the annotations file, and both transforms (covered
# in more detail in the next section).
#
# The labels.csv file looks like: ::
#
#     tshirt1.jpg, 0
#     tshirt2.jpg, 0
#     ......
#     ankleboot999.jpg, 9


def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    self.img_labels = pd.read_csv(annotations_file)
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform


#################################################################
# __len__
# ^^^^^^^^^^^^^^^^^^^^
#
# The __len__ function returns the number of samples in our dataset.
#
# Example:


def __len__(self):
    return len(self.img_labels)


#################################################################
# __getitem__
# ^^^^^^^^^^^^^^^^^^^^
#
# The __getitem__ function loads and returns a sample from the dataset at the given index ``idx``.
# Based on the index, it identifies the image's location on disk, converts that to a tensor using ``read_image``, retrieves the
# corresponding label from the csv data in ``self.img_labels``, calls the transform functions on them (if applicable), and returns the
# tensor image and corresponding label in a tuple.

def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    image = read_image(img_path)
    label = self.img_labels.iloc[idx, 1]
    if self.transform:
        image = self.transform(image)
    if self.target_transform:
        label = self.target_transform(label)
    return image, label


######################################################################
# --------------
#


#################################################################
# Preparing your data for training with DataLoaders
# -------------------------------------------------
# The ``Dataset`` retrieves our dataset's features and labels one sample at a time. While training a model, we typically want to
# pass samples in "minibatches", reshuffle the data at every epoch to reduce model overfitting, and use Python's ``multiprocessing`` to
# speed up data retrieval.
#
# ``DataLoader`` is an iterable that abstracts this complexity for us in an easy API.

from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

###########################
# Iterate through the DataLoader
# --------------------------
#
# We have loaded that dataset into the ``DataLoader`` and can iterate through the dataset as needed.
# Each iteration below returns a batch of ``train_features`` and ``train_labels`` (containing ``batch_size=64`` features and labels respectively).
# Because we specified ``shuffle=True``, after we iterate over all batches the data is shuffled (for finer-grained control over
# the data loading order, take a look at `Samplers <https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler>`_).

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

######################################################################
# --------------
#

#################################################################
# Further Reading
# --------------
# - `torch.utils.data API <https://pytorch.org/docs/stable/data.html>`_

"""


Transforms
===================

Data does not always come in its final processed form that is required for
training machine learning algorithms. We use **transforms** to perform some
manipulation of the data and make it suitable for training.

All TorchVision datasets have two parameters -``transform`` to modify the features and
``target_transform`` to modify the labels - that accept callables containing the transformation logic.
The `torchvision.transforms <https://pytorch.org/vision/stable/transforms.html>`_ module offers
several commonly-used transforms out of the box.

The FashionMNIST features are in PIL Image format, and the labels are integers.
For training, we need the features as normalized tensors, and the labels as one-hot encoded tensors.
To make these transformations, we use ``ToTensor`` and ``Lambda``.
"""

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

#################################################
# ToTensor()
# -------------------------------
#
# `ToTensor <https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.ToTensor>`_
# converts a PIL image or NumPy ``ndarray`` into a ``FloatTensor``. and scales
# the image's pixel intensity values in the range [0., 1.]
#

##############################################
# Lambda Transforms
# -------------------------------
#
# Lambda transforms apply any user-defined lambda function. Here, we define a function
# to turn the integer into a one-hot encoded tensor.
# It first creates a zero tensor of size 10 (the number of labels in our dataset) and calls
# `scatter_ <https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html>`_ which assigns a
# ``value=1`` on the index as given by the label ``y``.

target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

######################################################################
# --------------
#

#################################################################
# Further Reading
# ~~~~~~~~~~~~~~~~~
# - `torchvision.transforms API <https://pytorch.org/vision/stable/transforms.html>`_

"""

Build the Neural Network
===================

Neural networks comprise of layers/modules that perform operations on data.
The `torch.nn <https://pytorch.org/docs/stable/nn.html>`_ namespace provides all the building blocks you need to
build your own neural network. Every module in PyTorch subclasses the `nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_.
A neural network is a module itself that consists of other modules (layers). This nested structure allows for
building and managing complex architectures easily.

In the following sections, we'll build a neural network to classify images in the FashionMNIST dataset.

"""

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


#############################################
# Get Device for Training
# -----------------------
# We want to be able to train our model on a hardware accelerator like the GPU,
# if it is available. Let's check to see if
# `torch.cuda <https://pytorch.org/docs/stable/notes/cuda.html>`_ is available, else we
# continue to use the CPU.

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

##############################################
# Define the Class
# -------------------------
# We define our neural network by subclassing ``nn.Module``, and
# initialize the neural network layers in ``__init__``. Every ``nn.Module`` subclass implements
# the operations on input data in the ``forward`` method.

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

##############################################
# We create an instance of ``NeuralNetwork``, and move it to the ``device``, and print
# its structure.

model = NeuralNetwork().to(device)
print(model)


##############################################
# To use the model, we pass it the input data. This executes the model's ``forward``,
# along with some `background operations <https://github.com/pytorch/pytorch/blob/270111b7b611d174967ed204776985cefca9c144/torch/nn/modules/module.py#L866>`_.
# Do not call ``model.forward()`` directly!
#
# Calling the model on the input returns a 10-dimensional tensor with raw predicted values for each class.
# We get the prediction probabilities by passing it through an instance of the ``nn.Softmax`` module.

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")


######################################################################
# --------------
#


##############################################
# Model Layers
# -------------------------
#
# Let's break down the layers in the FashionMNIST model. To illustrate it, we
# will take a sample minibatch of 3 images of size 28x28 and see what happens to it as
# we pass it through the network.

input_image = torch.rand(3,28,28)
print(input_image.size())

##################################################
# nn.Flatten
# ^^^^^^^^^^^^^^^^^^^^^^
# We initialize the `nn.Flatten  <https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html>`_
# layer to convert each 2D 28x28 image into a contiguous array of 784 pixel values (
# the minibatch dimension (at dim=0) is maintained).

flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

##############################################
# nn.Linear
# ^^^^^^^^^^^^^^^^^^^^^^
# The `linear layer <https://pytorch.org/docs/stable/generated/torch.nn.Linear.html>`_
# is a module that applies a linear transformation on the input using its stored weights and biases.
#
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())


#################################################
# nn.ReLU
# ^^^^^^^^^^^^^^^^^^^^^^
# Non-linear activations are what create the complex mappings between the model's inputs and outputs.
# They are applied after linear transformations to introduce *nonlinearity*, helping neural networks
# learn a wide variety of phenomena.
#
# In this model, we use `nn.ReLU <https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html>`_ between our
# linear layers, but there's other activations to introduce non-linearity in your model.

print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")



#################################################
# nn.Sequential
# ^^^^^^^^^^^^^^^^^^^^^^
# `nn.Sequential <https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html>`_ is an ordered
# container of modules. The data is passed through all the modules in the same order as defined. You can use
# sequential containers to put together a quick network like ``seq_modules``.

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

################################################################
# nn.Softmax
# ^^^^^^^^^^^^^^^^^^^^^^
# The last linear layer of the neural network returns `logits` - raw values in [-\infty, \infty] - which are passed to the
# `nn.Softmax <https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html>`_ module. The logits are scaled to values
# [0, 1] representing the model's predicted probabilities for each class. ``dim`` parameter indicates the dimension along
# which the values must sum to 1.

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)


#################################################
# Model Parameters
# -------------------------
# Many layers inside a neural network are *parameterized*, i.e. have associated weights
# and biases that are optimized during training. Subclassing ``nn.Module`` automatically
# tracks all fields defined inside your model object, and makes all parameters
# accessible using your model's ``parameters()`` or ``named_parameters()`` methods.
#
# In this example, we iterate over each parameter, and print its size and a preview of its values.
#


print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

######################################################################
# --------------
#

#################################################################
# Further Reading
# --------------
# - `torch.nn API <https://pytorch.org/docs/stable/nn.html>`_

"""


Automatic Differentiation with ``torch.autograd``
=======================================

When training neural networks, the most frequently used algorithm is
**back propagation**. In this algorithm, parameters (model weights) are
adjusted according to the **gradient** of the loss function with respect
to the given parameter.

To compute those gradients, PyTorch has a built-in differentiation engine
called ``torch.autograd``. It supports automatic computation of gradient for any
computational graph.

Consider the simplest one-layer neural network, with input ``x``,
parameters ``w`` and ``b``, and some loss function. It can be defined in
PyTorch in the following manner:
"""

import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)


######################################################################
# Tensors, Functions and Computational graph
# ------------------------------------------
#
# This code defines the following **computational graph**:
#
# .. figure:: /_static/img/basics/comp-graph.png
#    :alt:
#
# In this network, ``w`` and ``b`` are **parameters**, which we need to
# optimize. Thus, we need to be able to compute the gradients of loss
# function with respect to those variables. In order to do that, we set
# the ``requires_grad`` property of those tensors.

#######################################################################
# .. note:: You can set the value of ``requires_grad`` when creating a
#           tensor, or later by using ``x.requires_grad_(True)`` method.

#######################################################################
# A function that we apply to tensors to construct computational graph is
# in fact an object of class ``Function``. This object knows how to
# compute the function in the *forward* direction, and also how to compute
# its derivative during the *backward propagation* step. A reference to
# the backward propagation function is stored in ``grad_fn`` property of a
# tensor. You can find more information of ``Function`` `in the
# documentation <https://pytorch.org/docs/stable/autograd.html#function>`__.
#

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

######################################################################
# Computing Gradients
# -------------------
#
# To optimize weights of parameters in the neural network, we need to
# compute the derivatives of our loss function with respect to parameters,
# namely, we need :math:`\frac{\partial loss}{\partial w}` and
# :math:`\frac{\partial loss}{\partial b}` under some fixed values of
# ``x`` and ``y``. To compute those derivatives, we call
# ``loss.backward()``, and then retrieve the values from ``w.grad`` and
# ``b.grad``:
#

loss.backward()
print(w.grad)
print(b.grad)


######################################################################
# .. note::
#   - We can only obtain the ``grad`` properties for the leaf
#     nodes of the computational graph, which have ``requires_grad`` property
#     set to ``True``. For all other nodes in our graph, gradients will not be
#     available.
#   - We can only perform gradient calculations using
#     ``backward`` once on a given graph, for performance reasons. If we need
#     to do several ``backward`` calls on the same graph, we need to pass
#     ``retain_graph=True`` to the ``backward`` call.
#


######################################################################
# Disabling Gradient Tracking
# ---------------------------
#
# By default, all tensors with ``requires_grad=True`` are tracking their
# computational history and support gradient computation. However, there
# are some cases when we do not need to do that, for example, when we have
# trained the model and just want to apply it to some input data, i.e. we
# only want to do *forward* computations through the network. We can stop
# tracking computations by surrounding our computation code with
# ``torch.no_grad()`` block:
#

z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)


######################################################################
# Another way to achieve the same result is to use the ``detach()`` method
# on the tensor:
#

z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

######################################################################
# There are reasons you might want to disable gradient tracking:
#   - To mark some parameters in your neural network as **frozen parameters**. This is
#     a very common scenario for
#     `finetuning a pretrained network <https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html>`__
#   - To **speed up computations** when you are only doing forward pass, because computations on tensors that do
#     not track gradients would be more efficient.


######################################################################

######################################################################
# More on Computational Graphs
# ----------------------------
# Conceptually, autograd keeps a record of data (tensors) and all executed
# operations (along with the resulting new tensors) in a directed acyclic
# graph (DAG) consisting of
# `Function <https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function>`__
# objects. In this DAG, leaves are the input tensors, roots are the output
# tensors. By tracing this graph from roots to leaves, you can
# automatically compute the gradients using the chain rule.
#
# In a forward pass, autograd does two things simultaneously:
#
# - run the requested operation to compute a resulting tensor
# - maintain the operation’s *gradient function* in the DAG.
#
# The backward pass kicks off when ``.backward()`` is called on the DAG
# root. ``autograd`` then:
#
# - computes the gradients from each ``.grad_fn``,
# - accumulates them in the respective tensor’s ``.grad`` attribute
# - using the chain rule, propagates all the way to the leaf tensors.
#
# .. note::
#   **DAGs are dynamic in PyTorch**
#   An important thing to note is that the graph is recreated from scratch; after each
#   ``.backward()`` call, autograd starts populating a new graph. This is
#   exactly what allows you to use control flow statements in your model;
#   you can change the shape, size and operations at every iteration if
#   needed.

######################################################################
# Optional Reading: Tensor Gradients and Jacobian Products
# --------------------------------------
#
# In many cases, we have a scalar loss function, and we need to compute
# the gradient with respect to some parameters. However, there are cases
# when the output function is an arbitrary tensor. In this case, PyTorch
# allows you to compute so-called **Jacobian product**, and not the actual
# gradient.
#
# For a vector function :math:`\vec{y}=f(\vec{x})`, where
# :math:`\vec{x}=\langle x_1,\dots,x_n\rangle` and
# :math:`\vec{y}=\langle y_1,\dots,y_m\rangle`, a gradient of
# :math:`\vec{y}` with respect to :math:`\vec{x}` is given by **Jacobian
# matrix**:
#
# .. math::
#
#
#    J=\left(\begin{array}{ccc}
#       \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}}\\
#       \vdots & \ddots & \vdots\\
#       \frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
#       \end{array}\right)
#
# Instead of computing the Jacobian matrix itself, PyTorch allows you to
# compute **Jacobian Product** :math:`v^T\cdot J` for a given input vector
# :math:`v=(v_1 \dots v_m)`. This is achieved by calling ``backward`` with
# :math:`v` as an argument. The size of :math:`v` should be the same as
# the size of the original tensor, with respect to which we want to
# compute the product:
#

inp = torch.eye(5, requires_grad=True)
out = (inp+1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")


######################################################################
# Notice that when we call ``backward`` for the second time with the same
# argument, the value of the gradient is different. This happens because
# when doing ``backward`` propagation, PyTorch **accumulates the
# gradients**, i.e. the value of computed gradients is added to the
# ``grad`` property of all leaf nodes of computational graph. If you want
# to compute the proper gradients, you need to zero out the ``grad``
# property before. In real-life training an *optimizer* helps us to do
# this.

######################################################################
# .. note:: Previously we were calling ``backward()`` function without
#           parameters. This is essentially equivalent to calling
#           ``backward(torch.tensor(1.0))``, which is a useful way to compute the
#           gradients in case of a scalar-valued function, such as loss during
#           neural network training.
#

######################################################################
# --------------
#

#################################################################
# Further Reading
# ~~~~~~~~~~~~~~~~~
# - `Autograd Mechanics <https://pytorch.org/docs/stable/notes/autograd.html>`_

"""

Optimizing Model Parameters
===========================

Now that we have a model and data it's time to train, validate and test our model by optimizing its parameters on
our data. Training a model is an iterative process; in each iteration (called an *epoch*) the model makes a guess about the output, calculates
the error in its guess (*loss*), collects the derivatives of the error with respect to its parameters (as we saw in
the `previous section  <autograd_tutorial.html>`_), and **optimizes** these parameters using gradient descent. For a more
detailed walkthrough of this process, check out this video on `backpropagation from 3Blue1Brown <https://www.youtube.com/watch?v=tIeHLnjs5U8>`__.

Prerequisite Code
-----------------
We load the code from the previous sections on `Datasets & DataLoaders <data_tutorial.html>`_
and `Build Model  <buildmodel_tutorial.html>`_.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()


##############################################
# Hyperparameters
# -----------------
#
# Hyperparameters are adjustable parameters that let you control the model optimization process.
# Different hyperparameter values can impact model training and convergence rates
# (`read more <https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html>`__ about hyperparameter tuning)
#
# We define the following hyperparameters for training:
#  - **Number of Epochs** - the number times to iterate over the dataset
#  - **Batch Size** - the number of data samples propagated through the network before the parameters are updated
#  - **Learning Rate** - how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training.
#

learning_rate = 1e-3
batch_size = 64
epochs = 5



#####################################
# Optimization Loop
# -----------------
#
# Once we set our hyperparameters, we can then train and optimize our model with an optimization loop. Each
# iteration of the optimization loop is called an **epoch**.
#
# Each epoch consists of two main parts:
#  - **The Train Loop** - iterate over the training dataset and try to converge to optimal parameters.
#  - **The Validation/Test Loop** - iterate over the test dataset to check if model performance is improving.
#
# Let's briefly familiarize ourselves with some of the concepts used in the training loop. Jump ahead to
# see the :ref:`full-impl-label` of the optimization loop.
#
# Loss Function
# ~~~~~~~~~~~~~~~~~
#
# When presented with some training data, our untrained network is likely not to give the correct
# answer. **Loss function** measures the degree of dissimilarity of obtained result to the target value,
# and it is the loss function that we want to minimize during training. To calculate the loss we make a
# prediction using the inputs of our given data sample and compare it against the true data label value.
#
# Common loss functions include `nn.MSELoss <https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss>`_ (Mean Square Error) for regression tasks, and
# `nn.NLLLoss <https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss>`_ (Negative Log Likelihood) for classification.
# `nn.CrossEntropyLoss <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss>`_ combines ``nn.LogSoftmax`` and ``nn.NLLLoss``.
#
# We pass our model's output logits to ``nn.CrossEntropyLoss``, which will normalize the logits and compute the prediction error.

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()



#####################################
# Optimizer
# ~~~~~~~~~~~~~~~~~
#
# Optimization is the process of adjusting model parameters to reduce model error in each training step. **Optimization algorithms** define how this process is performed (in this example we use Stochastic Gradient Descent).
# All optimization logic is encapsulated in  the ``optimizer`` object. Here, we use the SGD optimizer; additionally, there are many `different optimizers <https://pytorch.org/docs/stable/optim.html>`_
# available in PyTorch such as ADAM and RMSProp, that work better for different kinds of models and data.
#
# We initialize the optimizer by registering the model's parameters that need to be trained, and passing in the learning rate hyperparameter.

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#####################################
# Inside the training loop, optimization happens in three steps:
#  * Call ``optimizer.zero_grad()`` to reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.
#  * Backpropagate the prediction loss with a call to ``loss.backward()``. PyTorch deposits the gradients of the loss w.r.t. each parameter.
#  * Once we have our gradients, we call ``optimizer.step()`` to adjust the parameters by the gradients collected in the backward pass.


########################################
# .. _full-impl-label:
#
# Full Implementation
# -----------------------
# We define ``train_loop`` that loops over our optimization code, and ``test_loop`` that
# evaluates the model's performance against our test data.

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


########################################
# We initialize the loss function and optimizer, and pass it to ``train_loop`` and ``test_loop``.
# Feel free to increase the number of epochs to track the model's improving performance.

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")



#################################################################
# Further Reading
# -----------------------
# - `Loss Functions <https://pytorch.org/docs/stable/nn.html#loss-functions>`_
# - `torch.optim <https://pytorch.org/docs/stable/optim.html>`_
# - `Warmstart Training a Model <https://pytorch.org/tutorials/recipes/recipes/warmstarting_model_using_parameters_from_a_different_model.html>`_

"""


Save and Load the Model
============================

In this section we will look at how to persist model state with saving, loading and running model predictions.
"""

import torch
import torchvision.models as models


#######################################################################
# Saving and Loading Model Weights
# --------------------------------
# PyTorch models store the learned parameters in an internal
# state dictionary, called ``state_dict``. These can be persisted via the ``torch.save``
# method:

model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')

##########################
# To load model weights, you need to create an instance of the same model first, and then load the parameters
# using ``load_state_dict()`` method.

model = models.vgg16() # we do not specify pretrained=True, i.e. do not load default weights
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

###########################
# .. note:: be sure to call ``model.eval()`` method before inferencing to set the dropout and batch normalization layers to evaluation mode. Failing to do this will yield inconsistent inference results.

#######################################################################
# Saving and Loading Models with Shapes
# -------------------------------------
# When loading model weights, we needed to instantiate the model class first, because the class
# defines the structure of a network. We might want to save the structure of this class together with
# the model, in which case we can pass ``model`` (and not ``model.state_dict()``) to the saving function:

torch.save(model, 'model.pth')

########################
# We can then load the model like this:

model = torch.load('model.pth')

########################
# .. note:: This approach uses Python `pickle <https://docs.python.org/3/library/pickle.html>`_ module when serializing the model, thus it relies on the actual class definition to be available when loading the model.

#######################
# Related Tutorials
# -----------------
# `Saving and Loading a General Checkpoint in PyTorch <https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html>`_
