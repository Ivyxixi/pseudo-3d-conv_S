from __future__ import print_function
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy

'''
# Conv1d
m = nn.Conv1d(16, 33, 3, stride=2)
input = autograd.Variable(torch.randn(20, 16, 50))
output = m(input)
print (output.shape)

#Conv2d
# With square kernels and equal stride
m = nn.Conv2d(16, 33, 3, stride=2)
input = autograd.Variable(torch.randn(20, 16, 50, 50))
output = m(input)
print (output)


#Conv3d
# With square kernels and equal stride
m = nn.Conv3d(16, 33, 3, stride=2)
input = autograd.Variable(torch.randn(20, 16, 50, 50, 50))
output = m(input)
print (output)
'''
   

#########################
# conv3d to conv2d
# initial Conv3d
# With square kernels and equal stride
m = nn.Conv3d(16, 33, (1,3,3), stride=(1,2,2))
input = autograd.Variable(torch.randn(20, 16, 8, 50, 50))
output = m(input)
print (output)                       # (20,33,8,24,24)

# transformed conv3d by conv2d
conv2d=nn.Conv2d(16,33,3,stride=2)
deep=input.shape[2]
new_output=torch.Tensor(20,33,8,24,24)
print (input[:,:,0,:,:].shape)      # (20,16,50,50)
for i in range(deep):
    temp2D_input=input[:,:,i,:,:]
    temp2D_output=conv2d(temp2D_input)
    print (temp2D_output.shape)      # (20,33,24,24)
    new_output[:,:,i,:,:]=temp2D_output.data
print (new_output)                  # (20,33,8,24,24)

'''
# use the same weight
# transformed conv3d by conv2d
input = autograd.Variable(torch.randn(20, 1, 8, 50, 50))
conv2d=nn.Conv2d(1,1,3,stride=2)
deep=input.shape[2]
new_output=torch.Tensor(20,1,8,24,24)
for i in range(deep):
    temp2D_input=input[:,:,i,:,:]
    temp2D_output=conv2d(temp2D_input)
    print (conv2d.weight)
    new_output[:,:,i,:,:]=temp2D_output.data
print (new_output.shape)                  # (20,1,8,24,24)
# all the weights are the same
'''