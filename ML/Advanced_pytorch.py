"""
Neural Transfer Using PyTorch
=============================


Introduction
------------

This tutorial explains how to implement the `Neural-Style algorithm <https://arxiv.org/abs/1508.06576>`__
developed by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge.
Neural-Style, or Neural-Transfer, allows you to take an image and
reproduce it with a new artistic style. The algorithm takes three images,
an input image, a content-image, and a style-image, and changes the input 
to resemble the content of the content-image and the artistic style of the style-image.

 
.. figure:: /_static/img/neural-style/neuralstyle.png
   :alt: content1
"""

######################################################################
# Underlying Principle
# --------------------
# 
# The principle is simple: we define two distances, one for the content
# (:math:`D_C`) and one for the style (:math:`D_S`). :math:`D_C` measures how different the content
# is between two images while :math:`D_S` measures how different the style is
# between two images. Then, we take a third image, the input, and
# transform it to minimize both its content-distance with the
# content-image and its style-distance with the style-image. Now we can
# import the necessary packages and begin the neural transfer.
# 
# Importing Packages and Selecting a Device
# -----------------------------------------
# Below is a  list of the packages needed to implement the neural transfer.
#
# -  ``torch``, ``torch.nn``, ``numpy`` (indispensables packages for
#    neural networks with PyTorch)
# -  ``torch.optim`` (efficient gradient descents)
# -  ``PIL``, ``PIL.Image``, ``matplotlib.pyplot`` (load and display
#    images)
# -  ``torchvision.transforms`` (transform PIL images into tensors)
# -  ``torchvision.models`` (train or load pre-trained models)
# -  ``copy`` (to deep copy the models; system package)

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy


######################################################################
# Next, we need to choose which device to run the network on and import the
# content and style images. Running the neural transfer algorithm on large
# images takes longer and will go much faster when running on a GPU. We can
# use ``torch.cuda.is_available()`` to detect if there is a GPU available.
# Next, we set the ``torch.device`` for use throughout the tutorial. Also the ``.to(device)``
# method is used to move tensors or modules to a desired device. 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################################################
# Loading the Images
# ------------------
#
# Now we will import the style and content images. The original PIL images have values between 0 and 255, but when
# transformed into torch tensors, their values are converted to be between
# 0 and 1. The images also need to be resized to have the same dimensions.
# An important detail to note is that neural networks from the
# torch library are trained with tensor values ranging from 0 to 1. If you
# try to feed the networks with 0 to 255 tensor images, then the activated
# feature maps will be unable to sense the intended content and style.
# However, pre-trained networks from the Caffe library are trained with 0
# to 255 tensor images. 
#
#
# .. Note::
#     Here are links to download the images required to run the tutorial:
#     `picasso.jpg <https://pytorch.org/tutorials/_static/img/neural-style/picasso.jpg>`__ and
#     `dancing.jpg <https://pytorch.org/tutorials/_static/img/neural-style/dancing.jpg>`__.
#     Download these two images and add them to a directory
#     with name ``images`` in your current working directory.

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


style_img = image_loader("./data/images/neural-style/picasso.jpg")
content_img = image_loader("./data/images/neural-style/dancing.jpg")

assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"


######################################################################
# Now, let's create a function that displays an image by reconverting a 
# copy of it to PIL format and displaying the copy using 
# ``plt.imshow``. We will try displaying the content and style images 
# to ensure they were imported correctly.

unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated


plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')

######################################################################
# Loss Functions
# --------------
# Content Loss
# ~~~~~~~~~~~~
# 
# The content loss is a function that represents a weighted version of the
# content distance for an individual layer. The function takes the feature
# maps :math:`F_{XL}` of a layer :math:`L` in a network processing input :math:`X` and returns the
# weighted content distance :math:`w_{CL}.D_C^L(X,C)` between the image :math:`X` and the
# content image :math:`C`. The feature maps of the content image(:math:`F_{CL}`) must be
# known by the function in order to calculate the content distance. We
# implement this function as a torch module with a constructor that takes
# :math:`F_{CL}` as an input. The distance :math:`\|F_{XL} - F_{CL}\|^2` is the mean square error
# between the two sets of feature maps, and can be computed using ``nn.MSELoss``.
# 
# We will add this content loss module directly after the convolution
# layer(s) that are being used to compute the content distance. This way
# each time the network is fed an input image the content losses will be
# computed at the desired layers and because of auto grad, all the
# gradients will be computed. Now, in order to make the content loss layer
# transparent we must define a ``forward`` method that computes the content
# loss and then returns the layer’s input. The computed loss is saved as a
# parameter of the module.
# 

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

######################################################################
# .. Note::
#    **Important detail**: although this module is named ``ContentLoss``, it
#    is not a true PyTorch Loss function. If you want to define your content
#    loss as a PyTorch Loss function, you have to create a PyTorch autograd function 
#    to recompute/implement the gradient manually in the ``backward``
#    method.

######################################################################
# Style Loss
# ~~~~~~~~~~
# 
# The style loss module is implemented similarly to the content loss
# module. It will act as a transparent layer in a
# network that computes the style loss of that layer. In order to
# calculate the style loss, we need to compute the gram matrix :math:`G_{XL}`. A gram
# matrix is the result of multiplying a given matrix by its transposed
# matrix. In this application the given matrix is a reshaped version of
# the feature maps :math:`F_{XL}` of a layer :math:`L`. :math:`F_{XL}` is reshaped to form :math:`\hat{F}_{XL}`, a :math:`K`\ x\ :math:`N`
# matrix, where :math:`K` is the number of feature maps at layer :math:`L` and :math:`N` is the
# length of any vectorized feature map :math:`F_{XL}^k`. For example, the first line
# of :math:`\hat{F}_{XL}` corresponds to the first vectorized feature map :math:`F_{XL}^1`.
# 
# Finally, the gram matrix must be normalized by dividing each element by
# the total number of elements in the matrix. This normalization is to
# counteract the fact that :math:`\hat{F}_{XL}` matrices with a large :math:`N` dimension yield
# larger values in the Gram matrix. These larger values will cause the
# first layers (before pooling layers) to have a larger impact during the
# gradient descent. Style features tend to be in the deeper layers of the
# network so this normalization step is crucial.
# 

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


######################################################################
# Now the style loss module looks almost exactly like the content loss
# module. The style distance is also computed using the mean square
# error between :math:`G_{XL}` and :math:`G_{SL}`.
# 

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


######################################################################
# Importing the Model
# -------------------
# 
# Now we need to import a pre-trained neural network. We will use a 19
# layer VGG network like the one used in the paper.
# 
# PyTorch’s implementation of VGG is a module divided into two child
# ``Sequential`` modules: ``features`` (containing convolution and pooling layers),
# and ``classifier`` (containing fully connected layers). We will use the
# ``features`` module because we need the output of the individual
# convolution layers to measure content and style loss. Some layers have
# different behavior during training than evaluation, so we must set the
# network to evaluation mode using ``.eval()``.
# 

cnn = models.vgg19(pretrained=True).features.to(device).eval()



######################################################################
# Additionally, VGG networks are trained on images with each channel
# normalized by mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
# We will use them to normalize the image before sending it into the network.
# 

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


######################################################################
# A ``Sequential`` module contains an ordered list of child modules. For
# instance, ``vgg19.features`` contains a sequence (Conv2d, ReLU, MaxPool2d,
# Conv2d, ReLU…) aligned in the right order of depth. We need to add our
# content loss and style loss layers immediately after the convolution
# layer they are detecting. To do this we must create a new ``Sequential``
# module that has content loss and style loss modules correctly inserted.
# 

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


######################################################################
# Next, we select the input image. You can use a copy of the content image
# or white noise.
# 

input_img = content_img.clone()
# if you want to use white noise instead uncomment the below line:
# input_img = torch.randn(content_img.data.size(), device=device)

# add the original input image to the figure:
plt.figure()
imshow(input_img, title='Input Image')


######################################################################
# Gradient Descent
# ----------------
# 
# As Leon Gatys, the author of the algorithm, suggested `here <https://discuss.pytorch.org/t/pytorch-tutorial-for-neural-transfert-of-artistic-style/336/20?u=alexis-jacq>`__, we will use
# L-BFGS algorithm to run our gradient descent. Unlike training a network,
# we want to train the input image in order to minimise the content/style
# losses. We will create a PyTorch L-BFGS optimizer ``optim.LBFGS`` and pass
# our image to it as the tensor to optimize.
# 

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img])
    return optimizer


######################################################################
# Finally, we must define a function that performs the neural transfer. For
# each iteration of the networks, it is fed an updated input and computes
# new losses. We will run the ``backward`` methods of each loss module to
# dynamicaly compute their gradients. The optimizer requires a “closure”
# function, which reevaluates the module and returns the loss.
# 
# We still have one final constraint to address. The network may try to
# optimize the input with values that exceed the 0 to 1 tensor range for
# the image. We can address this by correcting the input values to be
# between 0 to 1 each time the network is run.
# 

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


######################################################################
# Finally, we can run the algorithm.
# 

output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

plt.figure()
imshow(output, title='Output Image')

# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()

# -*- coding: utf-8 -*-
"""
Creating Extensions Using numpy and scipy
=========================================
**Author**: `Adam Paszke <https://github.com/apaszke>`_

**Updated by**: `Adam Dziedzic <https://github.com/adam-dziedzic>`_

In this tutorial, we shall go through two tasks:

1. Create a neural network layer with no parameters.

    -  This calls into **numpy** as part of its implementation

2. Create a neural network layer that has learnable weights

    -  This calls into **SciPy** as part of its implementation
"""

import torch
from torch.autograd import Function

###############################################################
# Parameter-less example
# ----------------------
#
# This layer doesn’t particularly do anything useful or mathematically
# correct.
#
# It is aptly named BadFFTFunction
#
# **Layer Implementation**

from numpy.fft import rfft2, irfft2


class BadFFTFunction(Function):
    @staticmethod
    def forward(ctx, input):
        numpy_input = input.detach().numpy()
        result = abs(rfft2(numpy_input))
        return input.new(result)

    @staticmethod
    def backward(ctx, grad_output):
        numpy_go = grad_output.numpy()
        result = irfft2(numpy_go)
        return grad_output.new(result)

# since this layer does not have any parameters, we can
# simply declare this as a function, rather than as an nn.Module class


def incorrect_fft(input):
    return BadFFTFunction.apply(input)

###############################################################
# **Example usage of the created layer:**

input = torch.randn(8, 8, requires_grad=True)
result = incorrect_fft(input)
print(result)
result.backward(torch.randn(result.size()))
print(input)

###############################################################
# Parametrized example
# --------------------
#
# In deep learning literature, this layer is confusingly referred
# to as convolution while the actual operation is cross-correlation
# (the only difference is that filter is flipped for convolution,
# which is not the case for cross-correlation).
#
# Implementation of a layer with learnable weights, where cross-correlation
# has a filter (kernel) that represents weights.
#
# The backward pass computes the gradient wrt the input and the gradient wrt the filter.

from numpy import flip
import numpy as np
from scipy.signal import convolve2d, correlate2d
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class ScipyConv2dFunction(Function):
    @staticmethod
    def forward(ctx, input, filter, bias):
        # detach so we can cast to NumPy
        input, filter, bias = input.detach(), filter.detach(), bias.detach()
        result = correlate2d(input.numpy(), filter.numpy(), mode='valid')
        result += bias.numpy()
        ctx.save_for_backward(input, filter, bias)
        return torch.as_tensor(result, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.detach()
        input, filter, bias = ctx.saved_tensors
        grad_output = grad_output.numpy()
        grad_bias = np.sum(grad_output, keepdims=True)
        grad_input = convolve2d(grad_output, filter.numpy(), mode='full')
        # the previous line can be expressed equivalently as:
        # grad_input = correlate2d(grad_output, flip(flip(filter.numpy(), axis=0), axis=1), mode='full')
        grad_filter = correlate2d(input.numpy(), grad_output, mode='valid')
        return torch.from_numpy(grad_input), torch.from_numpy(grad_filter).to(torch.float), torch.from_numpy(grad_bias).to(torch.float)


class ScipyConv2d(Module):
    def __init__(self, filter_width, filter_height):
        super(ScipyConv2d, self).__init__()
        self.filter = Parameter(torch.randn(filter_width, filter_height))
        self.bias = Parameter(torch.randn(1, 1))

    def forward(self, input):
        return ScipyConv2dFunction.apply(input, self.filter, self.bias)


###############################################################
# **Example usage:**

module = ScipyConv2d(3, 3)
print("Filter and bias: ", list(module.parameters()))
input = torch.randn(10, 10, requires_grad=True)
output = module(input)
print("Output from the convolution: ", output)
output.backward(torch.randn(8, 8))
print("Gradient for the input map: ", input.grad)

###############################################################
# **Check the gradients:**

from torch.autograd.gradcheck import gradcheck

moduleConv = ScipyConv2d(3, 3)

input = [torch.randn(20, 20, dtype=torch.double, requires_grad=True)]
test = gradcheck(moduleConv, input, eps=1e-6, atol=1e-4)
print("Are the gradients correct: ", test)
# -*- coding: utf-8 -*-
"""
Creating Extensions Using numpy and scipy
=========================================


In this tutorial, we shall go through two tasks:

1. Create a neural network layer with no parameters.

    -  This calls into **numpy** as part of its implementation

2. Create a neural network layer that has learnable weights

    -  This calls into **SciPy** as part of its implementation
"""

import torch
from torch.autograd import Function

###############################################################
# Parameter-less example
# ----------------------
#
# This layer doesn’t particularly do anything useful or mathematically
# correct.
#
# It is aptly named BadFFTFunction
#
# **Layer Implementation**

from numpy.fft import rfft2, irfft2


class BadFFTFunction(Function):
    @staticmethod
    def forward(ctx, input):
        numpy_input = input.detach().numpy()
        result = abs(rfft2(numpy_input))
        return input.new(result)

    @staticmethod
    def backward(ctx, grad_output):
        numpy_go = grad_output.numpy()
        result = irfft2(numpy_go)
        return grad_output.new(result)

# since this layer does not have any parameters, we can
# simply declare this as a function, rather than as an nn.Module class


def incorrect_fft(input):
    return BadFFTFunction.apply(input)

###############################################################
# **Example usage of the created layer:**

input = torch.randn(8, 8, requires_grad=True)
result = incorrect_fft(input)
print(result)
result.backward(torch.randn(result.size()))
print(input)

###############################################################
# Parametrized example
# --------------------
#
# In deep learning literature, this layer is confusingly referred
# to as convolution while the actual operation is cross-correlation
# (the only difference is that filter is flipped for convolution,
# which is not the case for cross-correlation).
#
# Implementation of a layer with learnable weights, where cross-correlation
# has a filter (kernel) that represents weights.
#
# The backward pass computes the gradient wrt the input and the gradient wrt the filter.

from numpy import flip
import numpy as np
from scipy.signal import convolve2d, correlate2d
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class ScipyConv2dFunction(Function):
    @staticmethod
    def forward(ctx, input, filter, bias):
        # detach so we can cast to NumPy
        input, filter, bias = input.detach(), filter.detach(), bias.detach()
        result = correlate2d(input.numpy(), filter.numpy(), mode='valid')
        result += bias.numpy()
        ctx.save_for_backward(input, filter, bias)
        return torch.as_tensor(result, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.detach()
        input, filter, bias = ctx.saved_tensors
        grad_output = grad_output.numpy()
        grad_bias = np.sum(grad_output, keepdims=True)
        grad_input = convolve2d(grad_output, filter.numpy(), mode='full')
        # the previous line can be expressed equivalently as:
        # grad_input = correlate2d(grad_output, flip(flip(filter.numpy(), axis=0), axis=1), mode='full')
        grad_filter = correlate2d(input.numpy(), grad_output, mode='valid')
        return torch.from_numpy(grad_input), torch.from_numpy(grad_filter).to(torch.float), torch.from_numpy(grad_bias).to(torch.float)


class ScipyConv2d(Module):
    def __init__(self, filter_width, filter_height):
        super(ScipyConv2d, self).__init__()
        self.filter = Parameter(torch.randn(filter_width, filter_height))
        self.bias = Parameter(torch.randn(1, 1))

    def forward(self, input):
        return ScipyConv2dFunction.apply(input, self.filter, self.bias)


###############################################################
# **Example usage:**

module = ScipyConv2d(3, 3)
print("Filter and bias: ", list(module.parameters()))
input = torch.randn(10, 10, requires_grad=True)
output = module(input)
print("Output from the convolution: ", output)
output.backward(torch.randn(8, 8))
print("Gradient for the input map: ", input.grad)

###############################################################
# **Check the gradients:**

from torch.autograd.gradcheck import gradcheck

moduleConv = ScipyConv2d(3, 3)

input = [torch.randn(20, 20, dtype=torch.double, requires_grad=True)]
test = gradcheck(moduleConv, input, eps=1e-6, atol=1e-4)
print("Are the gradients correct: ", test)


"""
(optional) Exporting a Model from PyTorch to ONNX and Running it using ONNX Runtime
========================================================================

In this tutorial, we describe how to convert a model defined
in PyTorch into the ONNX format and then run it with ONNX Runtime.

ONNX Runtime is a performance-focused engine for ONNX models,
which inferences efficiently across multiple platforms and hardware
(Windows, Linux, and Mac and on both CPUs and GPUs).
ONNX Runtime has proved to considerably increase performance over
multiple models as explained `here
<https://cloudblogs.microsoft.com/opensource/2019/05/22/onnx-runtime-machine-learning-inferencing-0-4-release>`__

For this tutorial, you will need to install `ONNX <https://github.com/onnx/onnx>`__
and `ONNX Runtime <https://github.com/microsoft/onnxruntime>`__.
You can get binary builds of ONNX and ONNX Runtime with
``pip install onnx onnxruntime``.
Note that ONNX Runtime is compatible with Python versions 3.5 to 3.7.

``NOTE``: This tutorial needs PyTorch master branch which can be installed by following
the instructions `here <https://github.com/pytorch/pytorch#from-source>`__

"""

# Some standard imports
import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx


######################################################################
# Super-resolution is a way of increasing the resolution of images, videos
# and is widely used in image processing or video editing. For this
# tutorial, we will use a small super-resolution model.
#
# First, let's create a SuperResolution model in PyTorch.
# This model uses the efficient sub-pixel convolution layer described in
# `"Real-Time Single Image and Video Super-Resolution Using an Efficient
# Sub-Pixel Convolutional Neural Network" - Shi et al <https://arxiv.org/abs/1609.05158>`__
# for increasing the resolution of an image by an upscale factor.
# The model expects the Y component of the YCbCr of an image as an input, and
# outputs the upscaled Y component in super resolution.
#
# `The
# model <https://github.com/pytorch/examples/blob/master/super_resolution/model.py>`__
# comes directly from PyTorch's examples without modification:
#

# Super Resolution model definition in PyTorch
import torch.nn as nn
import torch.nn.init as init


class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

# Create the super-resolution model by using the above model definition.
torch_model = SuperResolutionNet(upscale_factor=3)


######################################################################
# Ordinarily, you would now train this model; however, for this tutorial,
# we will instead download some pre-trained weights. Note that this model
# was not trained fully for good accuracy and is used here for
# demonstration purposes only.
#
# It is important to call ``torch_model.eval()`` or ``torch_model.train(False)``
# before exporting the model, to turn the model to inference mode.
# This is required since operators like dropout or batchnorm behave
# differently in inference and training mode.
#

# Load pretrained model weights
model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
batch_size = 1    # just a random number

# Initialize model with the pretrained weights
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None
torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))

# set the model to inference mode
torch_model.eval()


######################################################################
# Exporting a model in PyTorch works via tracing or scripting. This
# tutorial will use as an example a model exported by tracing.
# To export a model, we call the ``torch.onnx.export()`` function.
# This will execute the model, recording a trace of what operators
# are used to compute the outputs.
# Because ``export`` runs the model, we need to provide an input
# tensor ``x``. The values in this can be random as long as it is the
# right type and size.
# Note that the input size will be fixed in the exported ONNX graph for
# all the input's dimensions, unless specified as a dynamic axes.
# In this example we export the model with an input of batch_size 1,
# but then specify the first dimension as dynamic in the ``dynamic_axes``
# parameter in ``torch.onnx.export()``.
# The exported model will thus accept inputs of size [batch_size, 1, 224, 224]
# where batch_size can be variable.
#
# To learn more details about PyTorch's export interface, check out the
# `torch.onnx documentation <https://pytorch.org/docs/master/onnx.html>`__.
#

# Input to the model
x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
torch_out = torch_model(x)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

######################################################################
# We also computed ``torch_out``, the output after of the model,
# which we will use to verify that the model we exported computes
# the same values when run in ONNX Runtime.
#
# But before verifying the model's output with ONNX Runtime, we will check
# the ONNX model with ONNX's API.
# First, ``onnx.load("super_resolution.onnx")`` will load the saved model and
# will output a onnx.ModelProto structure (a top-level file/container format for bundling a ML model.
# For more information `onnx.proto documentation <https://github.com/onnx/onnx/blob/master/onnx/onnx.proto>`__.).
# Then, ``onnx.checker.check_model(onnx_model)`` will verify the model's structure
# and confirm that the model has a valid schema.
# The validity of the ONNX graph is verified by checking the model's
# version, the graph's structure, as well as the nodes and their inputs
# and outputs.
#

import onnx

onnx_model = onnx.load("super_resolution.onnx")
onnx.checker.check_model(onnx_model)


######################################################################
# Now let's compute the output using ONNX Runtime's Python APIs.
# This part can normally be done in a separate process or on another
# machine, but we will continue in the same process so that we can
# verify that ONNX Runtime and PyTorch are computing the same value
# for the network.
#
# In order to run the model with ONNX Runtime, we need to create an
# inference session for the model with the chosen configuration
# parameters (here we use the default config).
# Once the session is created, we evaluate the model using the run() api.
# The output of this call is a list containing the outputs of the model
# computed by ONNX Runtime.
#

import onnxruntime

ort_session = onnxruntime.InferenceSession("super_resolution.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")


######################################################################
# We should see that the output of PyTorch and ONNX Runtime runs match
# numerically with the given precision (rtol=1e-03 and atol=1e-05).
# As a side-note, if they do not match then there is an issue in the
# ONNX exporter, so please contact us in that case.
#


######################################################################
# Running the model on an image using ONNX Runtime
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# So far we have exported a model from PyTorch and shown how to load it
# and run it in ONNX Runtime with a dummy tensor as an input.

######################################################################
# For this tutorial, we will use a famous cat image used widely which
# looks like below
#
# .. figure:: /_static/img/cat_224x224.jpg
#    :alt: cat
#

######################################################################
# First, let's load the image, pre-process it using standard PIL
# python library. Note that this preprocessing is the standard practice of
# processing data for training/testing neural networks.
#
# We first resize the image to fit the size of the model's input (224x224).
# Then we split the image into its Y, Cb, and Cr components.
# These components represent a greyscale image (Y), and
# the blue-difference (Cb) and red-difference (Cr) chroma components.
# The Y component being more sensitive to the human eye, we are
# interested in this component which we will be transforming.
# After extracting the Y component, we convert it to a tensor which
# will be the input of our model.
#

from PIL import Image
import torchvision.transforms as transforms

img = Image.open("./_static/img/cat.jpg")

resize = transforms.Resize([224, 224])
img = resize(img)

img_ycbcr = img.convert('YCbCr')
img_y, img_cb, img_cr = img_ycbcr.split()

to_tensor = transforms.ToTensor()
img_y = to_tensor(img_y)
img_y.unsqueeze_(0)


######################################################################
# Now, as a next step, let's take the tensor representing the
# greyscale resized cat image and run the super-resolution model in
# ONNX Runtime as explained previously.
#

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
ort_outs = ort_session.run(None, ort_inputs)
img_out_y = ort_outs[0]


######################################################################
# At this point, the output of the model is a tensor.
# Now, we'll process the output of the model to construct back the
# final output image from the output tensor, and save the image.
# The post-processing steps have been adopted from PyTorch
# implementation of super-resolution model
# `here <https://github.com/pytorch/examples/blob/master/super_resolution/super_resolve.py>`__.
#

img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

# get the output image follow post-processing step from PyTorch implementation
final_img = Image.merge(
    "YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, Image.BICUBIC),
        img_cr.resize(img_out_y.size, Image.BICUBIC),
    ]).convert("RGB")

# Save the image, we will compare this with the output image from mobile device
final_img.save("./_static/img/cat_superres_with_ort.jpg")


######################################################################
# .. figure:: /_static/img/cat_superres_with_ort.jpg
#    :alt: output\_cat
#
#
# ONNX Runtime being a cross platform engine, you can run it across
# multiple platforms and on both CPUs and GPUs.
#
# ONNX Runtime can also be deployed to the cloud for model inferencing
# using Azure Machine Learning Services. More information `here <https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-onnx>`__.
#
# More information about ONNX Runtime's performance `here <https://github.com/microsoft/onnxruntime#high-performance>`__.
#
#
# For more information about ONNX Runtime `here <https://github.com/microsoft/onnxruntime>`__.
#
