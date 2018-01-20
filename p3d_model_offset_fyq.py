# imports
from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from layers_fyq import ConvOffset2D

__all__ = ['P3D_offset', 'P3D63_offset', 'P3D131_offset','P3D199_offset']

# conv S is 1x3x3
def conv_S(in_planes,out_planes,stride=1,padding=1):
    return nn.Conv3d(in_planes,out_planes,kernel_size=(1,3,3), stride=1, padding=padding,bias=False)

# conv T is 3x1x1
def conv_T(in_planes,out_planes,stride=1,padding=1):
    return nn.Conv3d(in_planes,out_planes,kernel_size=(3,1,1),stride=1, padding=padding,bias=False)

# downsamle
def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1),
                             out.size(2), out.size(3),
                             out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out

# Bottleneck module
# 1. not 3d module:
#       conv2d(1,1)--bn--conv2d(3,3)--bn--conv2d(1,1)--bn--relu
# 2. 3d module (ST_A/ST_B/ST_C)
#       baisc network layers:
#       conv3d(1,1,1)--bn--conv3d(1,3,3)--bn--conv3d(3,1,1)--bn--conv3d(1,1,1)--relu
#       3 variables: ST_A   ST_B  ST_C
# IN A WORD, it defines 4 different bottleneck(with residual x) 
class Bottleneck_offset(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, n_s=0, depth_3d=47,ST_struc=('A','B','C')):
        super(Bottleneck_offset, self).__init__()
        self.downsample = downsample
        self.depth_3d=depth_3d
        self.ST_struc=ST_struc
        self.len_ST=len(self.ST_struc)

        stride_p=stride
        if not self.downsample ==None:
            stride_p=(1,2,2)
        if n_s<self.depth_3d:
            if n_s==0:
                stride_p=1
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False,stride=stride_p)
            self.bn1 = nn.BatchNorm3d(planes)
        else:
            if n_s==self.depth_3d:
                stride_p=2
            else:
                stride_p=1
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,stride=stride_p)
            self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
        #                        padding=1, bias=False)
        self.id=n_s
        self.ST=list(self.ST_struc)[self.id%self.len_ST]
        if self.id<self.depth_3d:
            self.offset=ConvOffset2D(planes)
            self.conv2 = conv_S(planes,planes, stride=1,padding=(0,1,1))
            self.bn2 = nn.BatchNorm3d(planes)
    
            self.conv3 = conv_T(planes,planes, stride=1,padding=(1,0,0))
            self.bn3 = nn.BatchNorm3d(planes)
        else:
            self.conv_normal = nn.Conv2d(planes, planes, kernel_size=3, stride=1,padding=1,bias=False)
            self.bn_normal = nn.BatchNorm2d(planes)

        if n_s<self.depth_3d:
            self.conv4 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
            self.bn4 = nn.BatchNorm3d(planes * 4)
        else:
            self.conv4 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
            self.bn4 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride


    def ConvOffset3D_fyq(self,x):
        deep=x.shape[2]
        new_output=torch.Tensor(x.shape)
        for i in range(deep):
            temp2D_input=x[:,:,i,:,:]
            temp2D_output=self.offset(temp2D_input)
            new_output[:,:,i,:,:]=temp2D_output.data
        print (new_output.shape)                 
        result=new_output.type(torch.cuda.FloatTensor)
        result=Variable(result)
        return result


    def ST_A(self,x):
        x = self.ConvOffset3D_fyq(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x

    def ST_B(self,x):
        x = self.ConvOffset3D_fyq(x)
        tmp_x = self.conv2(x)
        tmp_x = self.bn2(tmp_x)
        tmp_x = self.relu(tmp_x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x+tmp_x

    def ST_C(self,x):
        x = self.ConvOffset3D_fyq(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        tmp_x = self.conv3(x)
        tmp_x = self.bn3(tmp_x)
        tmp_x = self.relu(tmp_x)

        return x+tmp_x

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        if self.id<self.depth_3d: # C3D parts: 

            if self.ST=='A':
                out=self.ST_A(out)
            elif self.ST=='B':
                out=self.ST_B(out)
            elif self.ST=='C':
                out=self.ST_C(out)
        else:
            out = self.conv_normal(out)   # normal is res5 part, C2D all.
            out = self.bn_normal(out)
            out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# P3D network
class P3D_offset(nn.Module):

    def __init__(self, block, layers, modality='RGB',
        shortcut_type='B', num_classes=400,dropout=0.5,ST_struc=('A','B','C')):
        self.inplanes = 64
        super(P3D_offset, self).__init__()
        # self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2),
        #                        padding=(3, 3, 3), bias=False)
        self.input_channel = 3 if modality=='RGB' else 2  # 2 is for flow 
        self.ST_struc=ST_struc

        self.conv1_custom = nn.Conv3d(self.input_channel, 64, kernel_size=(1,7,7), stride=(1,2,2),
                                padding=(0,3,3), bias=False)

        # print (layers) [3,8,36,3]
        self.depth_3d=sum(layers[:3])  # C3D layers are only (res2,res3,res4),  res5 is C2D

        self.bn1 = nn.BatchNorm3d(64) # bn1 is followed by conv1
        self.cnt=0
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=2, padding=0)       # pooling layer for conv1.
        self.maxpool_2 = nn.MaxPool3d(kernel_size=(2,1,1),padding=0,stride=(2,1,1))   # pooling layer for res2, 3, 4.

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=(5, 5), stride=1)                              # pooling layer for res5.
        self.dropout=nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # some private attribute
        self.input_size=(self.input_channel,16,160,160)       # input of the network
        self.input_mean = [0.485, 0.456, 0.406] if modality=='RGB' else [0.5]
        self.input_std = [0.229, 0.224, 0.225] if modality=='RGB' else [np.mean([0.229, 0.224, 0.225])]


    @property
    def scale_size(self):
        return self.input_size[2] * 256 // 160   # asume that raw images are resized (340,256).

    @property
    def temporal_length(self):
        return self.input_size[1]

    @property
    def crop_size(self):
        return self.input_size[2]

    # make layer 
    # blocks means there are how much block in this layer
    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        stride_p=stride #especially for downsample branch.

        if self.cnt<self.depth_3d:
            if self.cnt==0:
                stride_p=1
            else:
                stride_p=(1,2,2)
            if stride != 1 or self.inplanes != planes * block.expansion:
                if shortcut_type == 'A':
                    downsample = partial(downsample_basic_block,
                                         planes=planes * block.expansion,
                                         stride=stride)
                else:
                    downsample = nn.Sequential(
                        nn.Conv3d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=stride_p, bias=False),
                        nn.BatchNorm3d(planes * block.expansion)
                    )

        else:
            if stride != 1 or self.inplanes != planes * block.expansion:
                if shortcut_type == 'A':
                    downsample = partial(downsample_basic_block,
                                         planes=planes * block.expansion,
                                         stride=stride)
                else:
                    downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=2, bias=False),
                        nn.BatchNorm2d(planes * block.expansion)
                    )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,n_s=self.cnt,depth_3d=self.depth_3d,ST_struc=self.ST_struc))
        self.cnt+=1

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,n_s=self.cnt,depth_3d=self.depth_3d,ST_struc=self.ST_struc))
            self.cnt+=1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_custom(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.maxpool_2(self.layer1(x))  #  Part Res2
        x = self.maxpool_2(self.layer2(x))  #  Part Res3
        x = self.maxpool_2(self.layer3(x))  #  Part Res4

        sizes=x.size()
        x = x.view(-1,sizes[1],sizes[3],sizes[4])  #  Part Res5
        x = self.layer4(x)
        x = self.avgpool(x)

        x = x.view(-1,self.fc.in_features)
        x = self.fc(self.dropout(x))

        return x


    def freeze(self, module_classes):
        '''
        freeze modules for finetuning
        '''
        for k, m in self._modules.items():
            if any([type(m) == mc for mc in module_classes]):
                for param in m.parameters():
                    param.requires_grad = False

    def unfreeze(self, module_classes):
        '''
        unfreeze modules
        '''
        for k, m in self._modules.items():
            if any([isinstance(m, mc) for mc in module_classes]):
                for param in m.parameters():
                    param.requires_grad = True

    def parameters(self):
        return filter(lambda p: p.requires_grad, super(P3D_offset, self).parameters())


# 63layers P3D
def P3D63_offset(**kwargs):
    """Construct a P3D63 modelbased on a ResNet-50-3D model.
    """
    model = P3D_offset(Bottleneck_offset, [3, 4, 6, 3], **kwargs)
    return model

#131layers P3D
def P3D131_offset(**kwargs):
    """Construct a P3D131 model based on a ResNet-101-3D model.
    """
    model = P3D_offset(Bottleneck_offset, [3, 4, 23, 3], **kwargs)
    return model

#199layers P3D
def P3D199_offset(pretrained=False,modality='RGB',**kwargs):
    """construct a P3D199 model based on a ResNet-152-3D model.
    """
    model = P3D_offset(Bottleneck_offset, [3, 8, 36, 3], modality=modality,**kwargs)
    if pretrained==True:
        if modality=='RGB':
            pretrained_file='p3d_rgb_199.checkpoint.pth.tar'
        elif modality=='Flow':
            pretrained_file='p3d_flow_199.checkpoint.pth.tar'
        weights=torch.load(pretrained_file)['state_dict']
        model.load_state_dict(weights)
    return model

def get_P3D_offset_fyq(trainable=True, freeze_filter=[nn.Conv3d,nn.Conv2d, nn.Linear],modality='RGB',**kwargs):
    model = P3D_offset(Bottleneck_offset, [3, 8, 36, 3], modality=modality,**kwargs)
    if not trainable:
        model.freeze(freeze_filter)
    return model

# custom operation
def get_optim_policies(model=None,modality='RGB',enable_pbn=True):
    '''
    first conv:         weight --> conv weight
                        bias   --> conv bias
    normal action:      weight --> non-first conv + fc weight
                        bias   --> non-first conv + fc bias
    bn:                 the first bn2, and many all bn3.

    '''
    first_conv_weight = []
    first_conv_bias = []
    normal_weight = []
    normal_bias = []
    bn = []

    if model==None:
        log.l.info('no model!')
        exit()

    conv_cnt = 0
    bn_cnt = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Conv2d):
            ps = list(m.parameters())
            conv_cnt += 1
            if conv_cnt == 1:
                first_conv_weight.append(ps[0])
                if len(ps) == 2:
                    first_conv_bias.append(ps[1])
            else:
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
        elif isinstance(m, torch.nn.Linear):
            ps = list(m.parameters())
            normal_weight.append(ps[0])
            if len(ps) == 2:
                normal_bias.append(ps[1])
              
        elif isinstance(m, torch.nn.BatchNorm3d):
            bn_cnt += 1
            # later BN's are frozen
            if not enable_pbn or bn_cnt == 1:
                bn.extend(list(m.parameters()))
        elif isinstance(m,torch.nn.BatchNorm2d):
            bn.extend(list(m.parameters()))
        elif len(m._modules) == 0:
            if len(list(m.parameters())) > 0:
                raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

    slow_rate=0.7
    n_fore=int(len(normal_weight)*slow_rate)
    slow_feat=normal_weight[:n_fore] # finetune slowly.
    slow_bias=normal_bias[:n_fore] 
    normal_feat=normal_weight[n_fore:]
    normal_bias=normal_bias[n_fore:]

    return [
        {'params': first_conv_weight, 'lr_mult': 5 if modality == 'Flow' else 1, 'decay_mult': 1,
         'name': "first_conv_weight"},
        {'params': first_conv_bias, 'lr_mult': 10 if modality == 'Flow' else 2, 'decay_mult': 0,
         'name': "first_conv_bias"},
        {'params': slow_feat, 'lr_mult': 1, 'decay_mult': 1,
         'name': "slow_feat"},
        {'params': slow_bias, 'lr_mult': 2, 'decay_mult': 0,
         'name': "slow_bias"},
        {'params': normal_feat, 'lr_mult': 1 , 'decay_mult': 1,
         'name': "normal_feat"},
        {'params': normal_bias, 'lr_mult': 2, 'decay_mult':0,
         'name': "normal_bias"},
        {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
         'name': "BN scale/shift"},
    ]



print ("hello")

if __name__ == '__main__':
    # we need the pretrained data: 'p3d_rgb_199.checkpoint.pth.tar'
    # model = P3D199_offset(pretrained=False,num_classes=400)
    # model = model.cuda()
    model=get_P3D_offset()
    model=model.cuda()
    data=torch.autograd.Variable(torch.rand(10,3,16,160,160)).cuda()   # if modality=='Flow', please change the 2nd dimension 3==>2
    out=model(data)
    print (out.size(),out)
