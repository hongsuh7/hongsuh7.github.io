---
layout: post
title:  "A Simple Introduction to Neural ODEs"
excerpt_separator: <!--more-->
---

You might have heard some hype around [neural ordinary differential equations](https://arxiv.org/pdf/1806.07366.pdf) (Neural ODEs). The idea of using a continuous process to model a discrete one is very fundamental in many parts of mathematics, so I was excited to learn about this stuff. Even though the code from the original paper is available online, I couldn't find a simple high-level explanation + implementation of neural ODEs on a simple dataset. In this post, I'll explain the idea behind and purported advantages of Neural ODEs and create a MNIST classifier using a Neural ODE.

<!--more-->

## Introduction

Recall that a resnet (residual neural network) is simply a neural network with residual blocks in them. To explain residual blocks, let $x_0\in \mathbb{R}^m$ be the input to the network and $f_0,f_1,\ldots, f_{n-1}: \mathbb{R}^m \to \mathbb{R}^m$ be functions whose second argument takes in some parameters $\alpha_i$. Then a residual block is the map $x_0 \mapsto x_n$ below:
++ \begin{cases} x_0&\mapsto x_0 + f_0(x_0, \alpha_0) = x_1, \\\ x_1 &\mapsto x_1 + f_1(x_1, \alpha_1) = x_2, \\\ &\vdots \\\ x_{n-1} &\mapsto x_{n-1} + f_{n-1}(x_{n-1}, \alpha_{n-1}) = x_n. \end{cases}++
Note that a particularity of this structure is that the dimension must remain the same at each step, which is why all the functions $f_i$ must map $\mathbb{R}^m \to \mathbb{R}^m$. 

Rewriting the residual block, we see that it is a discrete difference equation with control $\alpha$:
++ \begin{cases} x_1 - x_0 &= f_0(x_0,\alpha_0), \\\ x_2 - x_1 &= f_1(x_1,\alpha_1), \\\ &\vdots \\\ x_n - x_{n-1} &= f_{n-1}(x_{n-1}, \alpha_{n-1}). \end{cases} ++
When I say that $\alpha$ is the control, I mean that it is a sort of external value you can manipulate to change the behavior of the difference equation. 

If we move the subscripts of $f$ into the third argument of $f$, then we can write the residual block as the map
++ x_0 \mapsto x_n, \text{ where }\quad x_{k+1} - x_k = f(x_k, \alpha_k, k). ++

Writing it this way, we see that a residual block has a natural continuous analogue. Consider $x_0,x_1,\ldots,x_n$ as the evolution of a state through time, where $0,1,\ldots,n$ is considered the passing of time. Then the map from $x_0$ to $x_1$ is simply the evolution of the state $x_0$ for 1 unit of time. Similarly, the map from $x_0$ to $x_n$ is the evolution of the initial state $x_0$ for $n$ units of time. Each line in the residual block can be seen as an approximation of a time-one differential equation map (which is the map taking $x(0)$ to $x(1)$ if $x$ is the solution of an ODE), and we can string them all together into a time-$n$ differential equation map
++ x_0 \mapsto x(n), \text{ where }\quad \begin{cases} \dot{x}(t) &= f(x(t), \alpha(t), t), \\\ x(0) &= x_0. \end{cases} ++

This is the idea of Neural ODEs: replace residual blocks with their continuous analogue. The claim is that there are two advantages.
1. The continuous process might be better at modeling the truth than the discrete process.
2. Increasing the number of layers increases the memory usage of backprop. There is a natural way to obtain the gradient of the loss function with respect to the parameters with ODEs called the *adjoint method* which uses less memory.

### A note on the adjoint method
The idea of the adjoint method is the following. Suppose initial time is zero and final time is $T$. Suppose the state space is $\mathbb{R}^m$.
1. Solve the ODE forward in time to get the solution $x(T)$ at final time. No need to store any information in this forward pass.
2. Compute the backwards adjoint ODE with the terminal state $x(T)$ going back to the initial time $t=0$. The solution is labeled $a(t) \in \mathbb{R}^m, t \in [0,T]$, and we do not need to store this data. To compute $a(t)$ backwards, we simultaneously recompute $x(t)$ backwards.
3. Recall from step 2 that we are not storing the $a(t)$ information. Instead, we can compute the gradient of the loss function with respect to the parameters by adding up some function of $a(t)$ backwards in time, without any need to store all of $a(t), t\in[0,T]$ at the same time.

The difference between the adjoint method and backprop is that backprop stores the results of the intermediate steps $x_0,x_1,\ldots, x_{n-1}$, whereas the adjoint method recomputes them backwards. So the adjoint method consumes less memory (though I don't know enough to say whether this is actually a problem that needs a solution).

## Implementation on MNIST dataset

We will write a simple PyTorch implementation of an ODENet on the MNIST dataset using Ricky Chen's [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq) package. 

First install `torchdiffeq` using pip with the command

```
pip install torchdiffeq
```

Now let's begin the script. Import all the relevant packages and prepare the MNIST dataset.
```
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# enables the adjoint method for training,
# probably doesn't help in this setting
from torchdiffeq import odeint_adjoint as odeint

import matplotlib.pyplot as plt 
import numpy as np
import time

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(0.5, 0.5)]) # normalize to [-1,1]

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=4)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=True, num_workers=4)
```

Then we define a class `MyNet` under which we will write our two models.
```
class MyNet(nn.Module):
	def __init__(self, path):
		super(MyNet, self).__init__()
		self.path = path

	def num_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def load(self):
		self.load_state_dict(torch.load('./' + self.path + '.pth'))
```

Next we define the function $f(x,a,t)$ in the ODE $\dot{x} = f(x,a,t)$. For simplicity, we get rid of the $t$ dependence and make the control also time-independent. Below is the function ++f = \text{gn} \circ \text{conv}(a) \circ \text{relu} \circ \text{gn},++ where $\text{gn}$ is the GroupNorm function, $\text{relu}$ is the ReLU function, $\text{conv}$ is a convolution, and $a$ is the set of parameters of the $\text{conv}$ function. (In order to run this on a personal computer, we need a pretty small dimension so the GroupNorm will just become an InstanceNorm for us.) Notice that every function preserves dimension, which is necessary for the ODE to be defined.

```
class ODEFunc(nn.Module):
	def __init__(self, dim):
		super(ODEFunc, self).__init__()
		self.gn = nn.GroupNorm(min(32, dim), dim)
		self.conv = nn.Conv2d(dim, dim, 3, padding = 1)
		self.nfe = 0 # time counter

	def forward(self, t, x):
		self.nfe += 1
		x = self.gn(x)
		x = F.relu(x)
		x = self.conv(x)
		x = self.gn(x)
		return x
```
Now we must define the ODEBlock which will replace the residual block. What you see below is the map $x_0 \to x(1)$ where $x$ is the solution of the differential equation
++ \dot{x}(t) = f(x(t), a), \quad t\in[0,1], \quad x(0) = x_0 ++ where $a$ is the set of parameters for `conv`. More precisely, 
++ \dot{x}(t) = \text{gn} \circ \text{conv}(a) \circ \text{relu} \circ \text{gn}(x(t)), \quad t\in[0,1], \quad x(0)=x_0 ++ where all the parameters $a$ are coming from the convolution.

```
class ODEBlock(nn.Module):
	def __init__(self, odefunc):
		super(ODEBlock, self).__init__()
		self.odefunc = odefunc
		self.integration_time = torch.tensor([0, 1]).float()

	def forward(self, x):
		out = odeint(self.odefunc, x, self.integration_time, rtol=1e-1, atol=1e-1) # high tolerances for speed

		# first dimension is x(0) and second is x(1),
		# so we just want the second
		return out[1]
```

Now we create a ODENet with this block. There are three parts to this ODENet. 
1. We take our 28-by-28 image and apply a 3-by-3 convolution without padding to it with 6 output channels. Then we apply GroupNorm and ReLU.
2. We apply the ODEBlock.
3. We apply a 2-by-2 average pool and one fully connected linear layer.
I keep track of the image dimensions below.

```
class ODENet(MyNet):
	def __init__(self):
		super(ODENet, self).__init__('mnist_odenet')
		self.conv1 = nn.Conv2d(1, 6, 3)
		self.gn = nn.GroupNorm(6, 6)
		self.odefunc = ODEFunc(6)
		self.odeblock = ODEBlock(self.odefunc)
		self.pool = nn.AvgPool2d(2)
		self.fc = nn.Linear(6 * 13 * 13, 10)

	def forward(self, x):
		# 26 x 26
		x = self.conv1(x)
		x = F.relu(self.gn(x))

		# stays 26 x 26
		x = self.odeblock(x)

		# 13 x 13
		x = self.pool(x)

		# fully connected layer
		x = x.view(-1, 6*13*13)
		x = self.fc(x)

		return x
```

That's all! Now we just define the training and testing methods.
```
def train(net):
	n = 60000 / (5*16) # batch size 16
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

	for epoch in range(10):  
		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			# get the inputs; data is a list of [inputs, labels]
			inputs, labels = data

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			if i % n == n-1:    
				print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / n))
				running_loss = 0.0

	print('Finished Training')
	torch.save(net.state_dict(), './' + net.path + '.pth')

def test(net):
	initial_time = time.time()
	correct = 0
	total = 0
	with torch.no_grad():
		for data in testloader:
			images, labels = data
			batch_size = images.shape[0]
			outputs = net(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	final_time = time.time()
	print('Accuracy of the ' + net.path + ' network on the test set: %.2f %%' % (100 * correct / total))
	print('Time: %.2f seconds' % (final_time - initial_time))
	return(100 * correct / total)
```

Let's train our model. **Note: this is pretty expensive to run on a personal computer. It took me half a day to train. If you're doing this on a personal computer without a GPU, then consider making the ODENet even simpler, say with 2 channels instead of 6.**
```
odenet = ODENet()
train(odenet)
```
At this point, we have saved our parameters so we can load them back up for the test.
```
odenet.load()
test(odenet)

# Output:
# Accuracy of the mnist_odenet network on the test set: 98.62 %
# Time: 44.61 seconds
```
As you can see, this is very expensive just to test on a regular computer. 

## Conclusion

A Neural ODE is the continuous analogue of a resnet. The main advantage is that in this setting, there is an alternative way to compute the parameter gradients using less memory. 

Some thoughts: during my short time reading up on this subject, I am not yet convinced that Neural ODEs provide enough of a performance boost to justify the additional computation required for it. First, I'm sure that there is a discrete analogue of the adjoint method which is almost surely simpler to implement than its continuous version, for applications in which memory usage is critical. Second, I'm not sure that there is a reason why the continuous model is a more accurate reflection of the true classifier than the discrete model. Usually in math, when there is a discrete model and an analogous continuous model, the continuous one is derived from a microscopic model, and the discrete model is a crude approximation of the continuous one. In such a setting, it's clear that the continuous model is closer to the truth than the discrete model. In the Neural ODE setting, it's not so clear.

Below is the full code in one chunk.
```
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# enables the adjoint method for training
from torchdiffeq import odeint_adjoint as odeint

import matplotlib.pyplot as plt 
import numpy as np
import time

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(0.5, 0.5)]) # normalize to [-1,1]

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=4)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=True, num_workers=4)

############################################################################

class ODEFunc(nn.Module):
	def __init__(self, dim):
		super(ODEFunc, self).__init__()
		self.gn = nn.GroupNorm(min(32, dim), dim)
		self.conv = nn.Conv2d(dim, dim, 3, padding = 1)
		self.nfe = 0 # time counter

	def forward(self, t, x):
		self.nfe += 1
		x = self.gn(x)
		x = F.relu(x)
		x = self.conv(x)
		x = self.gn(x)
		return x

############################################################################

class ODEBlock(nn.Module):
	def __init__(self, odefunc):
		super(ODEBlock, self).__init__()
		self.odefunc = odefunc
		self.integration_time = torch.tensor([0, 1]).float()

	def forward(self, x):
		out = odeint(self.odefunc, x, self.integration_time, rtol=1e-1, atol=1e-1) # high tolerances for speed

		# first dimension is x(0) and second is x(1),
		# so we just want the second
		return out[1]

############################################################################

class ODENet(MyNet):
	def __init__(self):
		super(ODENet, self).__init__('mnist_odenet')
		self.conv1 = nn.Conv2d(1, 6, 3)
		self.gn = nn.GroupNorm(6, 6)
		self.odefunc = ODEFunc(6)
		self.odeblock = ODEBlock(self.odefunc)
		self.pool = nn.AvgPool2d(2)
		self.fc = nn.Linear(6 * 13 * 13, 10)

	def forward(self, x):
		# 26 x 26
		x = self.conv1(x)
		x = F.relu(self.gn(x))

		# stays 26 x 26
		x = self.odeblock(x)

		# 13 x 13
		x = self.pool(x)

		# fully connected layer
		x = x.view(-1, 6*13*13)
		x = self.fc(x)

		return x

############################################################################

def train(net):
	n = 60000 / (5*16) # batch size 16
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

	for epoch in range(10):  
		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			# get the inputs; data is a list of [inputs, labels]
			inputs, labels = data

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			if i % n == n-1:    
				print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / n))
				running_loss = 0.0

	print('Finished Training')
	torch.save(net.state_dict(), './' + net.path + '.pth')

def test(net):
	initial_time = time.time()
	correct = 0
	total = 0
	with torch.no_grad():
		for data in testloader:
			images, labels = data
			batch_size = images.shape[0]
			outputs = net(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	final_time = time.time()
	print('Accuracy of the ' + net.path + ' network on the test set: %.2f %%' % (100 * correct / total))
	print('Time: %.2f seconds' % (final_time - initial_time))
	return(100 * correct / total)


## Main ##
odenet = ODENet()
train(odenet)

## Test ##
odenet.load()
test(odenet)

# Output:
# Accuracy of the mnist_odenet network on the test set: 98.62 %
# Time: 44.61 seconds

```

