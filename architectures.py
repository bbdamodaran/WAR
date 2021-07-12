# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 17:00:20 2018

@author: damodara
"""
 
import torch
import torch.nn as nn
from torch.autograd import Variable


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x
        
class mnist_featext(nn.Module):
    def __init__(self, input_shape):
        super(mnist_featext, self).__init__()
        self.cnn_features = nn.Sequential(
                            nn.Conv2d(input_shape[0], 32, 3), 
                            nn.ReLU(),
                            nn.MaxPool2d(3),
                            nn.Conv2d(32, 64, 3),
                            nn.ReLU(),
                            nn.MaxPool2d(3))
                
        flt_size = self._get_conv_output(input_shape)
        self.adv_layer = nn.Sequential(nn.Linear(flt_size, 256),
                                       nn.Linear(256, 10))
                                       
    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self.cnn_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size        
        
        
            
    def forward(self, x):
        out = self.cnn_features(x)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity
    
    def embed(self, x):
        out = self.model(x)
        out = out.view(out.shape[0], -1)
        return out

    

class cifar10_featext(nn.Module):
    def __init__(self, input_shape, n_class, drop_out=None):
        self.drop_out = drop_out
        super(cifar10_featext, self).__init__()
                   
        self.cnn_features = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Dropout2d(0.5),

            # nn.Conv2d(64, 128, 3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.Conv2d(128, 128, 3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.MaxPool2d((2, 2), stride=2),
            # nn.Dropout2d(0.5),

            # nn.Conv2d(128, 196, 3, stride=1, padding=1),
            # nn.BatchNorm2d(196),
            # nn.ReLU(),
            # nn.Conv2d(196, 196, 3, stride=1, padding=1),
            # nn.BatchNorm2d(196),
            # nn.ReLU(),
            # nn.MaxPool2d((2, 2), stride=2),
            nn.AdaptiveAvgPool2d((1,1))
            )

#        flt_size = self._get_conv_output(input_shape)
        flt_size = 64
        self.dp =  nn.Dropout2d(0.5)
        self.adv_layer = nn.Sequential(nn.Linear(flt_size, n_class))
                                       
    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self.cnn_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size        
        
        
            
    def forward(self, x):
        out = self.cnn_features(x)
        out = out.view(out.shape[0], -1)
        if self.drop_out is None:
            out = self.dp(out)
        logits = self.adv_layer(out)
        return logits
    
    def embed(self, x):
        out = self.model(x)
        out = out.view(out.shape[0], -1)
        return out

class cnn_coteach(nn.Module):
    def __init__(self, input_shape, nclass, drop_out=None):
        self.drop_out = drop_out
        super(cnn_coteach, self).__init__()

        self.cnn_features = nn.Sequential(
            nn.Conv2d(input_shape[0], 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Dropout2d(0.25),

            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d((2, 2), stride=2),
            nn.AdaptiveAvgPool2d((1,1))
        )

        # flt_size = self._get_conv_output(input_shape)
        flt_size = 128
        print(flt_size)
        self.dp = nn.Dropout2d(0.5)
        self.adv_layer = nn.Sequential(nn.Linear(flt_size, nclass))

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self.cnn_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def forward(self, x):
        out = self.cnn_features(x)
        out = out.view(out.shape[0], -1)
        if self.drop_out is None:
            out = self.dp(out)
        logits = self.adv_layer(out)
        return logits

    def embed(self, x):
        out = self.model(x)
        out = out.view(out.shape[0], -1)
        return out

def torchvisionmodels(model_name ='resnet18', n_classes=10, pretrained=False):
    import torchvision.models as models
    if model_name == 'resnet18':
        model= models.resnet18(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)



    return model

class ResNet(nn.Module):
    def __init__(self, model_name = 'resnet18', n_class=10, pretrained=False, dropout=False):
        super(ResNet, self).__init__()
        self.model_name= model_name
        self.dropout = dropout
        import torchvision.models as models

        if model_name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            num_ftrs = model.fc.in_features

            self.cnn_features = nn.Sequential(*list(model.children())[:-1])
            self.drop_out = nn.Dropout(0.25)
            self.fc = nn.Linear(num_ftrs, n_class)
            del model
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            num_ftrs = model.fc.in_features
            self.cnn_features = nn.Sequential(*list(model.children())[:-1])
            self.drop_out = nn.Dropout(0.25)
            self.fc = nn.Linear(num_ftrs, n_class)
            del model


    def forward(self, x):
        x = self.cnn_features(x)
        x = torch.squeeze(x)
        if self.dropout:
            x = self.drop_out(x)
        x = self.fc(x)
        return  x



class cnn_coteach_avg(nn.Module):
    def __init__(self, input_shape, nclass, drop_out=None):
        self.drop_out = drop_out
        super(cnn_coteach_avg, self).__init__()

        self.cnn_features = nn.Sequential(
            nn.Conv2d(input_shape[0], 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(3,stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Dropout2d(0.25),

            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d((2, 2), stride=2),
            nn.AdaptiveAvgPool2d((1,1))
        )

        # flt_size = self._get_conv_output(input_shape)
        flt_size = 128
        print(flt_size)
        self.dp = nn.Dropout2d(0.5)
        self.adv_layer = nn.Sequential(nn.Linear(flt_size, nclass))

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self.cnn_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def forward(self, x):
        out = self.cnn_features(x)
        out = out.view(out.shape[0], -1)
        if self.drop_out is None:
            out = self.dp(out)
        logits = self.adv_layer(out)
        return logits

    def embed(self, x):
        out = self.model(x)
        out = out.view(out.shape[0], -1)
        return out

    
