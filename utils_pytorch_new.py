# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 10:12:22 2018

@author: damodara
"""

from collections import OrderedDict
import logging
# import logzero
from pathlib import Path
# from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# most of the functions are extracted from the web 
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, top_k=(1,)):
    """Computes the precision@k for the specified values of k"""
    max_k = max(top_k)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    if len(res) == 1:
        res = res[0]

    return res


def save_checkpoint(model, epoch, filename, optimizer=None):
    if optimizer is None:
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
        }, filename)
    else:
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename)


# did not understand first if statement (true case)
def load_checkpoint(model, path, optimizer=None):
    resume = torch.load(path)

    if ('module' in list(resume['state_dict'].keys())[0]) \
            and not (isinstance(model, torch.nn.DataParallel)):
        new_state_dict = OrderedDict()
        for k, v in resume['state_dict'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(resume['state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(resume['optimizer'])
        return model, optimizer
    else:
        return model


def set_logger(path, loglevel=logging.INFO, tf_board_path=None):
    path_dir = '/'.join(path.split('/')[:-1])
    if not Path(path_dir).exists():
        Path(path_dir).mkdir(parents=True)
    logzero.loglevel(loglevel)
    logzero.formatter(logging.Formatter('[%(asctime)s %(levelname)s] %(message)s'))
    logzero.logfile(path)

    if tf_board_path is not None:
        tb_path_dir = '/'.join(tf_board_path.split('/')[:-1])
        if not Path(tb_path_dir).exists():
            Path(tb_path_dir).mkdir(parents=True)
        writer = SummaryWriter(tf_board_path)

        return writer

def kl_div_with_logit(q_logit, p_logit):

    q = F.softmax(q_logit, dim=1)
    logq = F.log_softmax(q_logit, dim=1)
    logp = F.log_softmax(p_logit, dim=1)

    qlogq = ( q *logq).sum(dim=1).mean(dim=0)
    qlogp = ( q *logp).sum(dim=1).mean(dim=0)
    return qlogq - qlogp


def _l2_normalize_(d):

    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d = d/(torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8) #
    return d

# based on idea of co-teaching 
def small_cce(output, target, remove_rate, beta=5):
    # to minimize the CCE only on the small loss samples
    cce_loss = F.cross_entropy(output, target, reduce=False)
    index = torch.argsort(cce_loss.data)
    num = int(remove_rate * len(index))
    sel_ind = index[:num] # indexes of small loss samples
    loss = F.cross_entropy(output[sel_ind], target[sel_ind])
    return loss, sel_ind, index[num:]

def vat_loss(model, ul_x, ul_y, xi=1e-6, eps=2.5, num_iters=1, adv_img=False):
    # if adv_img = True, returns of t
    d = torch.Tensor(ul_x.size()).normal_()
    for i in range(num_iters):
        d = xi *_l2_normalize_(d)
        d = Variable(d.to(device), requires_grad=True)
        y_hat = model(ul_x + d)
        delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
        delta_kl.backward()

        d = d.grad.data.clone().cpu()
        d.detach()
        model.zero_grad()

    d = _l2_normalize_(d)
    d = Variable(d.to(device))
    r_adv = eps *d
    # compute lds
    adv_img_sample = ul_x + r_adv.detach()
    y_hat = model(adv_img_sample)
    delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
    if adv_img:
        return delta_kl, adv_img_sample
    else:
        return delta_kl


   
def loss_multi_sinkhorn(y_true, y_pred, M, reg=0.05,nbloop=10, return_coupling=False):
    
    r_eps=torch.from_numpy(np.array(1e-18)).float().to(device)
    Km=np.exp(-M/reg)    
    M2=torch.from_numpy(M).float().to(device)    
    Km2=torch.from_numpy(Km).float().to(device)
    
    a=y_pred
    b=y_true
    u0 = torch.ones_like(a).to(device)
    v=b/(torch.mm(u0,Km2)+r_eps)
    u=a/(torch.mm(v,torch.t(Km2))+r_eps)
    
    for i in range(nbloop-1):
        v=b/(torch.mm(u.clone(),Km2)+r_eps)
        u=a/(torch.mm(v.clone(),torch.t(Km2))+r_eps)
#        wloss = torch.sum(u*torch.mm(v, torch.t(Km2*M2)), 1) 
    wloss = torch.einsum('ki,ij,kj',[u,Km2*M2,v])
    if return_coupling ==False:
        return wloss/y_pred.size()[0]
    else:
        return wloss/y_pred.size()[0], u*torch.mm(v, torch.t(Km2))
    
def wat_loss(model, ul_x, l_y, M, reg=0.01, xi=1e-6, eps=2.5, num_iters=1,
             pred_label=False, nbloop=20, adv_img=False, deep_layer_adv=False):
    '''
    if adv_img is True, returns adversarial image and its prediction
    deep_layer_adv - computes adv in the deep layer, before softmax layer in this implementation
    '''

    ul_y = model(ul_x)
    ul_y = F.softmax(ul_y, dim=1)

    if deep_layer_adv:
        ul_x = model.cnn_features(ul_x)
        ul_x = ul_x.view(ul_x.shape[0], -1)

    if pred_label:
        # for computing adv direction wrt current prediction of the model
        d = torch.Tensor(ul_x.size()).normal_()
    else:
        # for computing adv dir wrt the class label, d is the original data (image)
        d = ul_x.clone()

    for i in range(num_iters):
        if pred_label:
            # compute the perturbed input to compute grad (when adv dir wrt prediction)
            # W(f(x), f(x+d)), grad wrt d (dir)
            d = xi *_l2_normalize_(d)
            d = Variable(d.to(device), requires_grad=True)
            if deep_layer_adv:
                y_hat = model.adv_layer(ul_x + d)
            else:
                y_hat = model(ul_x + d)
            y_hat = F.softmax(y_hat, dim=1)
            # wat loss wrt the one-hot label
            delta_kl = loss_multi_sinkhorn(ul_y.detach(), y_hat, M, reg=reg, nbloop=nbloop)
        else:
            # W(y, f(x)), grad wrt to input
            d = Variable(d, requires_grad = True)
            if deep_layer_adv:
                y_hat = model.adv_layer(d)
            else:
                y_hat = model(d)
            y_hat = F.softmax(y_hat, dim=1)
            # wat loss wrt the one-hot label
            delta_kl = loss_multi_sinkhorn(l_y, y_hat, M, reg=reg, nbloop=nbloop)

        delta_kl.backward(retain_graph=True)

        d = d.grad.data.clone().cpu()
        d.detach()
        model.zero_grad()

    # normalize the grad
    d = _l2_normalize_(d)
    d = Variable(d.to(device), requires_grad=False)
    r_adv = eps *d # perturbation

    # compute lds
    wat_advimg = ul_x + r_adv.detach()
    if deep_layer_adv:
        y_hat = model.adv_layer(wat_advimg)
    else:
        y_hat = model(wat_advimg)
    y_hat = F.softmax(y_hat, dim=1)
    delta_kl = loss_multi_sinkhorn(ul_y.detach(), y_hat, M, reg=reg, nbloop=nbloop)
    # delta_kl = loss_multi_sinkhorn(ul_y, y_hat) # update the parameters without fixing
    # delta_kl = loss_multi_sinkhorn(l_y, y_hat) # minimize adv samples wrt label
    # delta_kl = kl_div_with_logit(l_y, y_hat,M, reg=reg, nbloop=nbloop) # minimize KL diver
    if adv_img == False:
        return delta_kl
    elif adv_img == True:
        return delta_kl, wat_advimg, y_hat

def adv_sample_vat(model, ul_x, ul_y, xi=1e-6, eps=2.5, num_iters=1):

    # find r_adv
#    xi=10
    d = torch.Tensor(ul_x.size()).normal_()
    for i in range(num_iters):
        d = xi *_l2_normalize_(d)
        d = Variable(d, requires_grad=True)
        y_hat = model(ul_x + d)
        delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
        delta_kl.backward()

        d = d.grad.data.clone().cpu()
        d.detach()
        model.zero_grad()

    d = _l2_normalize_(d)
    d = Variable(d)
    r_adv = eps *d
    # compute lds
    adv_sample = ul_x + r_adv.detach()
    y_hat = model(adv_sample)
     
    return adv_sample.detach(), y_hat.detach(), d  
    
def adv_sample_wat(model, ul_x, ul_y, M, reg=0.05, xi=1e-6, eps=2.5, num_iters=1,
                   pred_label=True):

    # find r_adv
#    xi=10
    Km=np.exp(-M/reg)
    nbloop=20
    M2=torch.from_numpy(M).float()
    r_eps=torch.from_numpy(np.array(1e-18)).float()
    Km2=torch.from_numpy(Km).float()
    
    def loss_multi_sinkhorn(y_true, y_pred):
        a=y_pred
        b=y_true
        u0 = torch.ones_like(a)
        v=b/(torch.mm(u0,Km2)+r_eps)
        u=a/(torch.mm(v,torch.t(Km2))+r_eps)
        
        for i in range(nbloop-1):
            v=b/(torch.mm(u.clone(),Km2)+r_eps)
            u=a/(torch.mm(v.clone(),torch.t(Km2))+r_eps)
#        wloss = torch.sum(u*torch.mm(v, torch.t(Km2*M2)), 1) 
        wloss = torch.einsum('ki,ij,kj',[u,Km2*M2,v])
        return wloss/y_pred.size()[0]    
    
    
    if pred_label:
        d = torch.Tensor(ul_x.size()).normal_()
        ul_y = F.softmax(ul_y, dim=1)
    else:
        d = ul_x.clone()
        
    for i in range(num_iters):
        if pred_label:
            d = xi *_l2_normalize_(d)
            d = Variable(d, requires_grad=True)
            y_hat = model(ul_x + d)
        else:
            d = Variable(d, requires_grad = True)
            y_hat = model(d)
        y_hat = F.softmax(y_hat, dim=1)
#        y_label = (ul_y.detach().argmax(1).reshape(ul_y.size(0),1) == torch.arange(3).reshape(1, 3).long()).float()
        delta_w = loss_multi_sinkhorn(ul_y.detach(), y_hat)
        delta_w.backward()
        if pred_label:
            d = d.grad.data.clone().cpu()
            d.detach()
        else:
            d  = d.grad.data.clone().cpu()
            d.detach()
            
        model.zero_grad()
    
    d = _l2_normalize_(d)
    d = Variable(d)
    r_adv = eps *d
    # compute lds
    adv_sample = ul_x + r_adv.detach()
    y_hat1 = model(adv_sample)
     
    return adv_sample.detach(), y_hat.detach(), d.detach()



# def img_adv_img_plot(image, adv_image):




def plot_embedding(X, y, d=None, title=None, save_fig=0, pname=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    # Plot colors numbers
    plt.figure(figsize=(10, 10))
    for i in range(X.shape[0]):
        #        plot colored number
        #        plt.text(X[i, 0], X[i, 1], str(y[i]),
        #                 color=plt.cm.bwr(d[i] / 1.),
        #                 fontdict={'weight': 'bold', 'size': 9})
        if d is not None:
            if d[i] == 0:
                c = 'red'
            elif d[i] == 1:
                c = 'green'
            elif d[i] == 2:
                c = 'blue'

        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color='red',
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    #    red_patch = mpatches.Patch(color='red', label='Source data')
    #    green_patch = mpatches.Patch(color='green', label='Target data')
    #    plt.legend(handles=[red_patch, green_patch])
    #    plt.show()
    if title is not None:
        plt.title(title)
    if save_fig:
        fname = title + '_num.png'
        if pname is not None:
            fname = os.path.join(pname, fname)
        plt.savefig(fname)