# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 10:05:51 2018

@author: damodara
"""


import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import copy
import utils_pytorch as utils
from utils_pytorch import *
import numpy as np

import importlib

import pylab as pl 
import matplotlib.pyplot as plt
 
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
import ot
from sklearn.datasets import  make_blobs
import os
seed=1985
np.random.seed(seed)

#%% generate datasets
data, label = make_blobs(n_samples=1000, n_features=2, centers=[[0, 0.5], [2., 1.2], [-1.0,2.3]], 
                         cluster_std=0.4, random_state=42)

from sklearn.model_selection import train_test_split
data, testdata, label, testlabel = train_test_split(data,
                                            label, test_size=0.2, random_state=42)

#centers=[[0, 0.5], [1.5, 2], [-0.5,2.5]], cluster_std=0.4)
plt.figure()
plt.scatter(data[:,0], data[:,1], c=label)
#plt.figure()
#plt.scatter(testdata[:,0], testdata[:,1], c=testlabel)
batch=5
n_class = len(np.unique(label))
#%%
from simulate_noisylabel import noisify_with_P
P = np.eye(n_class)
asymmetric=0
noise = 0.2
P[0,0], P[0,1]= 1-noise, noise
P[1,1], P[1,0]= 1-noise, noise
if noise>0.0:
    if asymmetric==1:
        label,P = noisify_with_P(label.ravel(), noise=noise, random_state=42)
    elif asymmetric==0:
        label = noisify_with_P(label.ravel(), noise=noise, P=P,random_state=42)        
else:
    label = label
    P = np.eye(n_class)

#n_idx = np.nonzero(label != n_trainlabelf)
plt.figure()
plt.scatter(data[:,0], data[:,1], c=label)
plt.title('noisy label')

#%%
pl.rcParams['text.usetex']=True
pl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += [' ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return ' '.join(rv)

def plot_history(history, option = 'loss', title = 'loss' ):
    if option == 'loss':
        loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
#        val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
        ylabel = 'loss'
    elif option == 'acc':
        loss_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
#        val_loss_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
        ylabel = 'acc'
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    pl.figure()
    for l in loss_list:
        pl.plot(epochs, history.history[l], label=l)
#    for l in val_loss_list:
#        pl.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    pl.title(title)
    pl.xlabel('Epochs')
    pl.ylabel(ylabel)
    pl.legend(loc='best')
    
    
def make_meshgrid(x, y, h=.02,offset=0.5):
    x_min, x_max = x.min()- offset, x.max()+ offset
    y_min, y_max = y.min()- offset, y.max()+ offset
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, f, xx, yy, **params):
    Z = f(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.pcolormesh(xx, yy,Z,edgecolors='face', alpha=0.1,**params)
    out = ax.contour(xx, yy, Z,colors=('darkred',),linewidths=(1,))
    return out
    
def plot_quiver(ax, xx,yy, modell, f):
    mesh_data = np.c_[xx.ravel(), yy.ravel()]
    ul_x= torch.from_numpy(mesh_data).float().to(device)
#    ul_y =torch.from_numpy(slabel).float().to(device)
    ul_y = modell(ul_x)
    new_inputs, new_pred, grad = adv_sample_wat(modell, ul_x, ul_y, M, reg= reg,eps=eps)       
#    outs = model.predict(mesh_data)     
    #grad = normalize_vector_np(grad)
    xgrad, ygrad = grad[:,0].reshape(xx.shape), grad[:,1].reshape(yy.shape)
    out = ax.quiver(xx,yy, xgrad, ygrad)
    return out
    
def plot_color(ax, xx,yy, modell):
    mesh_data = np.c_[xx.ravel(), yy.ravel()]
    ul_x= torch.from_numpy(mesh_data).float().to(device)
#    ul_y =torch.from_numpy(slabel).float().to(device)
    ul_y = modell(ul_x)
    fx = F.softmax(ul_y.detach(), 1)
    out = ax.scatter(xx,yy, c=fx.numpy())
    return out

def plot_streamline(ax, xx,yy, modell, f):
    mesh_data = np.c_[xx.ravel(), yy.ravel()]
    ul_x= torch.from_numpy(mesh_data).float().to(device)
#    ul_y =torch.from_numpy(slabel).float().to(device)
    ul_y = modell(ul_x)
    new_inputs, new_pred, grad = adv_sample_wat(modell, ul_x, ul_y, M, reg= reg, eps=eps)       
    xgrad, ygrad = grad[:,0].reshape(xx.shape), grad[:,1].reshape(yy.shape)
    speed = np.sqrt(xgrad*xgrad + ygrad*ygrad)
    lw = 4.0* speed / speed.max()
    #out = ax.streamplot(xx, yy, xgrad, ygrad, density=1, color='k', linewidth=lw)
    out = ax.streamplot(xx, yy, xgrad, ygrad, density=1, color='k')
    return out

cmap_light = ListedColormap(['antiquewhite', 'lemonchiffon', 'lightcyan'])
cmap_bold = ListedColormap(['orangered', 'gold', 'lightseagreen'])

cmap_all='Blues'
#%% for plotting purpose
xx, yy = make_meshgrid(data[:,0], data[:,1], h=.02)

sxx,syy = make_meshgrid(data[:,0], data[:,1], h=.02)            
gxx,gyy = make_meshgrid(data[:,0], data[:,1], h=.2)    
#%%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 512)
        self.fc2 = nn.Linear(512, 128)
#        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
#        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


steps_epoch = int(data.shape[0]/batch)
def my_gen():
    while True:
        idx = np.random.choice(data.shape[0], batch, replace=False)
        yield data[idx,], label[idx,]
        
def ul_gen():
    r = np.random.RandomState()
    while True:
        for i in range(steps_epoch):
            idx = r.randint(0, data.shape[0], batch)
            yield data[idx,], label[idx,]
            
def ul_gen_test():
    r = np.random.RandomState()
    while True:
        for i in range(steps_epoch):
            idx = r.randint(0, data.shape[0], batch)
            yield data[idx,], label[idx,]
#%%
gen = my_gen()
gen2 = ul_gen()
gen3= ul_gen_test()
torch.manual_seed(1985)
np.random.seed(1985)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(model.parameters(), lr=0.001)
#%% CCE Model

n=1.
reg = 0.5
eps = 0.5

#M = np.array([[0., 0.0002, 0.0002], [1., 0., 1.0],
#[0.0002, 1.0, 0.]])

M = np.array([[0., 50., 1.0], [50., 0., 1.0],
[1.0, 1.0, 0.]])

M = M/50

M = np.array([[0., 5, 1.0], [5., 0., 1.0],
[1.0, 1.0, 0.]])
#M = np.ones((n_class, n_class))
for i in range(n_class):
     M[i,i]=0
#%%   Cross entropy model training

steps_epoch = int(data.shape[0]/batch)
model.train()
epochs=50
lds_loss=[]
cce_loss =[]
tot_loss=[]
for j in range(epochs):   
    for i in tqdm(range(steps_epoch)):   
        # reset
        if 1:
            ce_losses = utils.AverageMeter()
            vat_losses = utils.AverageMeter()
            prec1 = utils.AverageMeter()
        
        x_l, y_l = next(gen2)
        x_l, y_l = torch.from_numpy(x_l).float().to(device), torch.from_numpy(y_l).float().to(device)
        y_l = y_l.long()
        optimizer.zero_grad()
        cross_entropy = nn.CrossEntropyLoss()
        output = model(x_l)
        classification_loss = cross_entropy(output, y_l)         
        classification_loss.backward()
        optimizer.step()
    
        acc = utils.accuracy(output, y_l)
        ce_losses.update(classification_loss.item(), x_l.shape[0])        
        prec1.update(acc.item(), x_l.shape[0])        
        cce_loss.append(classification_loss.item())
        
        if 1:
            print('It: {:d}/{:d},  cce_loss={:.4f}, acc={:.4f}'.
            format(i,steps_epoch, classification_loss.item(),  acc.item()))
        
    print('Epoch: {:d} completed'.format(j))    



#%%
testgen = ul_gen()
model.eval()
correct = 0
count=0
for k in range(steps_epoch):
    x, y = next(testgen)
    count += 1
    print(count)
    with torch.no_grad():
        x, y = torch.from_numpy(x).float().to(device), torch.from_numpy(y).float().to(device)
        y = y.long()            
        outputs = model(x)
    correct += torch.eq(outputs.max(dim=1)[1], y).detach().cpu().float().sum()

test_acc = correct / len(data) * 100.

print('Test Accuracy: {:.4f}%\n'.format(test_acc))


#%%
def pytorch_pred(model, x):
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(x).float().to(device)
        pred_l = model(x).max(dim=1)[1]
#        pred_l = model(x)
        pred_l = pred_l.detach().cpu()
    return pred_l
#tr_pred = pytorch_pred(model, data)
#tr_acc = utils.accuracy(tr_pred, torch.from_numpy(label).float().to(device).long())
    #%% cross entropy decision boundary
# show result
fig = pl.figure(figsize=(6,6))
ax = fig.add_subplot(1, 1, 1)
plot_contours(ax, lambda x: pytorch_pred(model,x), xx, yy,cmap=cmap_light)
pl.scatter(data[:,0],data[:,1],c=label, cmap=cmap_all, edgecolors='k',marker='o',s=50)
plt.title('cce model')
pl.xticks([], [])
pl.yticks([], [])
pl.legend(loc='best')
pl.tight_layout()
#pl.savefig('cce.png')
pl.show()

#%% Adversarial samples and direction computed using WAT with respect to CCE model
label_cat = (label[:,np.newaxis] == np.arange(n_class).reshape(1, n_class)).astype('float64')
#sdata = data[20:30,]
#slabel = label_cat[20:30,]
ul_x= torch.from_numpy(data).float().to(device)
ul_y =torch.from_numpy(label_cat).float().to(device)
#ul_y = model(ul_x)
new_inputs, new_pred, r_grad = adv_sample_wat(model, ul_x, ul_y, M, reg= reg, eps=eps,
                                              pred_label=False, num_iters=1, xi=0.05) 
fx = F.softmax(ul_y.detach(), 1)

#r_grad = normalize_vector_np(r_grad)
fig = pl.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
tex=r'$$M='+bmatrix(M)+'$$'
pl.text(-3, -0.2, tex, fontsize=14)
pl.quiver(data[:,0], data[:,1], r_grad[:,0], r_grad[:,1])
plot_contours(ax, lambda x: pytorch_pred(model,x), xx, yy,cmap=cmap_light)
pl.scatter(data[:,0],data[:,1],c=fx, cmap=cmap_all, edgecolors='k',marker='o',s=50)
#plot_color(ax, xx, yy, model)
pl.scatter(new_inputs[:,0], new_inputs[:,1],c=label, cmap=cmap_all, edgecolors='k',marker='s',s=25 )
#for aa in range(data.shape[0]):
#    pl.plot([data[aa, 0], new_inputs.numpy()[aa,0]],[data[aa,1], new_inputs.numpy()[aa,1]])
pl.title('wat adv direction samples wrt ccemodel reg='+str(reg))
#%%
fig = pl.figure(figsize=(8,8))
#pl.scatter(new_inputs[:,0], new_inputs[:,1],c=label, cmap=cmap_all, edgecolors='k',marker='s',s=25 )
pl.scatter(r_grad[:,0], r_grad[:,1], c=label,edgecolors='k',marker='o',s=25)
#%%
fn = r'./figures'
#%% Ploting streamline and adv direction of WAT wrt CCE model
tex=r'$$M='+bmatrix(M)+'$$'
fig = pl.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
pl.text(-3, -0.2, tex, fontsize=14)
plot_streamline(ax, sxx, syy, model, M)
plot_contours(ax, lambda x: pytorch_pred(model,x), xx, yy,cmap=cmap_light)
pl.scatter(data[:,0],data[:,1],c=label, cmap=cmap_all, edgecolors='k',marker='o',s=50)
pl.xticks([], [])
pl.yticks([], [])
pl.legend(loc='best')
pl.title('WAT grad stream  wrt CCE model reg='+str(reg))
pl.tight_layout()
#        pl.savefig(os.path.join(fn, 'wat_stream_cce_reg_'+str(reg)+'_M_'+str(n)+'.png'))
pl.show()

fig = pl.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
pl.text(-3, -0.2, tex, fontsize=14)
plot_quiver(ax, gxx, gyy, model,M)
plot_contours(ax, lambda x: pytorch_pred(model,x), xx, yy,cmap=cmap_light)
pl.scatter(data[:,0],data[:,1],c=label, cmap=cmap_all, edgecolors='k',marker='o',s=50)
pl.xticks([], [])
pl.yticks([], [])
pl.legend(loc='best')
pl.title('WAT grad wrt CCE model reg='+str(reg))
pl.tight_layout()
#        pl.savefig(os.path.join(fn, 'wat_grad_cce_reg_'+str(reg)+'_M_'+str(n)+'.png'))
pl.show()


#%% WAT model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
wat_model = Net().to(device)
#wat_model = copy.deepcopy(model)
#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(wat_model.parameters(), lr=0.001)  

#%% 
#reg = 0.05
#eps = 0.5


steps_epoch = int(data.shape[0]/batch)
wat_model.train()
epochs=50
wa_loss=[]
cce_loss =[]
tot_loss=[]
for j in range(epochs):   
    for i in tqdm(range(steps_epoch)):   
        # reset
        if 1:
            ce_losses = utils.AverageMeter()
            wat_losses = utils.AverageMeter()
            prec1 = utils.AverageMeter()
        
        x_l, y_l = next(gen)
        x_l, y_l = torch.from_numpy(x_l).float().to(device), torch.from_numpy(y_l).float().to(device)
        y_l = y_l.long()
        
        
        optimizer.zero_grad()
    
        cross_entropy = nn.CrossEntropyLoss()
        output = wat_model(x_l)
        classification_loss = cross_entropy(output, y_l)
        
        wat_lds = wat_loss(wat_model, x_l, output, M, reg=reg, eps=eps)               
        loss = classification_loss +  0.5*wat_lds
        loss.backward()
        optimizer.step()
    
        acc = utils.accuracy(output, y_l)
        ce_losses.update(classification_loss.item(), x_l.shape[0])
        wat_losses.update(wat_lds.item(), x_l.shape[0])
        prec1.update(acc.item(), x_l.shape[0])
        wa_loss.append(wat_lds.item())
        cce_loss.append(classification_loss.item())
        tot_loss.append(loss.item())
        if 1:
            print('It: {:d}/{:d}, loss={:.4f}, cce_loss={:.4f}, wat_loss={:.4f},acc={:.4f}'.
            format(i,steps_epoch, loss.item(), classification_loss.item(), wat_lds.item(), acc.item()))
        
    print('Epoch: {:d} completed'.format(j))       
#%%
testgen = ul_gen()
wat_model.eval()
correct = 0
count=0
for k in range(steps_epoch):
    x, y = next(testgen)
    count += 1
    with torch.no_grad():
        x, y = torch.from_numpy(x).float().to(device), torch.from_numpy(y).float().to(device)
        y = y.long()            
        outputs = wat_model(x)
    correct += torch.eq(outputs.max(dim=1)[1], y).detach().cpu().float().sum()

test_acc = correct / len(data) * 100.

print('Test Accuracy: {:.4f}%\n'.format(test_acc))
#%% WAT model decision boundry

# show result
fig = pl.figure(figsize=(6,6))
ax = fig.add_subplot(1, 1, 1)
plot_contours(ax, lambda x: pytorch_pred(wat_model,x), xx, yy,cmap=cmap_light)
pl.scatter(data[:,0],data[:,1],c=label, cmap=cmap_all, edgecolors='k',marker='o',s=50)
plt.title('wat model')
pl.xticks([], [])
pl.yticks([], [])
pl.legend(loc='best')
pl.tight_layout()
#pl.savefig('cce.png')
pl.show()

#%% wat decision boundary wrt adv samples from cce model
ul_x= torch.from_numpy(data).float().to(device)
#ul_y =torch.from_numpy(slabel).float().to(device)
ul_y = model(ul_x)
new_inputs, new_pred, r_grad = adv_sample_wat(model, ul_x, ul_y, M, reg= reg,eps=eps)       
#r_grad = normalize_vector_np(r_grad)
fig = pl.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
pl.quiver(data[:,0], data[:,1], r_grad[:,0], r_grad[:,1])
#plot_contours(ax, lambda x: pytorch_pred(wat_model,x), xx, yy,cmap=cmap_light)
pl.scatter(data[:,0],data[:,1],c=label, cmap=cmap_all, edgecolors='k',marker='o',s=50)
pl.scatter(new_inputs[:,0], new_inputs[:,1],c=label, cmap=cmap_all, edgecolors='k',marker='s',s=25 )
fx = F.softmax(ul_y.detach(), 1)
#plot_color(ax, xx, yy, wat_model)
pl.scatter(data[:,0],data[:,1],c=fx.numpy(), cmap=cmap_all, edgecolors='k',marker='o',s=50)
#for aa in range(data.shape[0]):
#    pl.plot([data[aa, 0], new_inputs.numpy()[aa,0]],[data[aa,1], new_inputs.numpy()[aa,1]])
#for aa in range(data.shape[0]):
#    pl.plot(data[aa, 0],[data[aa,1], c=ul_y[aa])
pl.title('wat decision bound wrt adv samples from cce model')
#%%
fn = r'./figures'
#%% stream line and adv direction of WAT model using WAT
tex=r'$$M='+bmatrix(M)+'$$'
fig = pl.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
pl.text(-3, -0.2, tex, fontsize=14)
plot_streamline(ax, sxx, syy, wat_model, M)
plot_contours(ax, lambda x: pytorch_pred(wat_model,x), xx, yy,cmap=cmap_light)
pl.scatter(data[:,0],data[:,1],c=label, cmap=cmap_all, edgecolors='k',marker='o',s=50)
pl.xticks([], [])
pl.yticks([], [])
pl.legend(loc='best')
pl.title('WAT grad wrt WAT model reg='+str(reg))
pl.tight_layout()
#        pl.savefig(os.path.join(fn, 'wat_stream_cce_reg_'+str(reg)+'_M_'+str(n)+'.png'))
pl.show()

fig = pl.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
pl.text(-3, -0.2, tex, fontsize=14)
plot_quiver(ax, gxx, gyy, wat_model,M)
plot_contours(ax, lambda x: pytorch_pred(wat_model,x), xx, yy,cmap=cmap_light)
pl.scatter(data[:,0],data[:,1],c=label, cmap=cmap_all, edgecolors='k',marker='o',s=50)
pl.xticks([], [])
pl.yticks([], [])
pl.legend(loc='best')
pl.title('WAT grad wrt WAT model reg='+str(reg))
pl.tight_layout()
#        pl.savefig(os.path.join(fn, 'wat_grad_cce_reg_'+str(reg)+'_M_'+str(n)+'.png'))
pl.show()
