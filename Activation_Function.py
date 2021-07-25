# There are many activation functions in Torch, but the ones we usually use are these: relu, sigmoid, tanh, softplus. Then let's see what each of them looks like.
import torch
import torch.nn.functional as F  # the activation functions in there
from torch.autograd import Variable

# Use some fake data to watch the image
x = torch.linspace(-5, 5, 200)  # x data (tensor), shape=(100, 1)
x = Variable(x)

x_np = x.data.numpy()  # Change to numpy array, use when outputting the picture

# some commonly used activation functions
y_relu = F.relu(x).data.numpy()
y_sigmoid = F.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()
y_softmax = F.softmax(x)  # softmax is special and can't be displayed directly, but it is about probability and used for classification.

# plot
import matplotlib.pyplot as plt

plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()

