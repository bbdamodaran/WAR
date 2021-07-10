# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 16:40:20 2019

@author: damodara

B.B.Damodaran et.al, WAR: Wasserstien adversarial regularization for learning with noisy labels
Accepted at IEEE PAMI

"""

import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import utils_pytorch_new as utils
from utils_pytorch_new import *
from CustomLoader import CustomDataset  # custom data loder
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torch.autograd import Variable
# for plotting and saving the plot
import matplotlib as mpl
import matplotlib.pylab as plt



batch_size = 256
data_set = 'cifar10'
method = 'WAR'

# cifar 10 mean and std : mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.2435, 0.2616)
# cifar 100 mean and std : transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))

if data_set == 'cifar10':
    # pre-processing to tensor, and mean subtraction
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (1, 1, 1)),
     ])
    # data generator to load the full training data into memory
    trainset =  datasets.CIFAR10(root='./data/cifar10', train=True,
                                            download=True, transform=transform)
    # test data generator
    testset =  datasets.CIFAR10(root='./data/cifar10', train=False,
                                            download=True,transform=transform)
elif data_set == 'mnist':
    # pre-processing to tensor and [-1, 1] scaling
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5)),
     ])
    # data generator to load the full training data into memory
    trainset = datasets.MNIST(root='./data/mnist', train=True,
                                download=True, transform=transform)
    # test data generator
    testset = datasets.MNIST(root='./data/mnist', train=False,
                               download=True, transform=transform)
elif data_set == 'fashion-mnist':
    # pre-processing to tensor and [-1, 1] scaling
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,)),
     ])
    # data generator to load the full training data into memory
    trainset = datasets.FashionMNIST(root='./data/fmnist', train=True,
                                download=True, transform=transform)
    # test data generator
    testset = datasets.FashionMNIST(root='./data/fmnist', train=False,
                               download=True, transform=transform)
    
elif data_set == 'cifar100':
     # pre-processing to tensor, and mean subtraction
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5071, 0.4865, 0.4409), (1, 1, 1)),
     ])
    # data generator to load the full training data into memory
    trainset =  datasets.CIFAR100(root='./data/cifar100', train=True,
                                            download=True, transform=transform)
    # test data generator
    testset =  datasets.CIFAR100(root='./data/cifar100', train=False,
                                            download=True,transform=transform)
   
    

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=8)
                                            
#%% load all the training data into memory for generating the noisy labels
trainlabel =list()
traindata =[]
testlabel = list()
testdata = []
for data, labels in train_loader:
     trainlabel.extend(labels.numpy())
     traindata.append(data)

traindata = torch.cat(traindata, dim=0) # convert list to tensor
trainlabel = np.array(trainlabel)

nclass = len(np.unique(trainlabel))
steps_epoch = int(traindata.shape[0]/batch_size)

for data, labels in test_loader:
     testlabel.extend(labels.numpy())
     testdata.append(data)
testdata = torch.cat(testdata, dim=0) # convert list to tensor
testlabel = np.array(testlabel)



#%% function to compute the test accuracy
def eval_model(modell):
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = modell(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return  correct/total

# function for initalizing weights for the CNN and FCN layers
# have to re-check whether it is correct
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        print(m.weight.data.shape)
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
            # torch.nn.init.zeros_(m.bias.data)
        # nn.init.kaiming_normal_(m.bias.data)
    elif isinstance(m, nn.Linear):
        print(m.weight.data.shape)
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
            # torch.nn.init.zeros_(m.bias.data)
        # nn.init.kaiming_normal_(m.bias.data)
      

#%% cross entropy training

def train_eval_CCE(model, tr_loader, criterion,epochs=1,fname=None,verbose=1):
    n_iter = 0 # for step decay, intialization

    steps_epoch = int(traindata.shape[0]/batch_size) # number of steps in each epoch

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    hist ={'cce_loss':list(), 'train_acc':list()}
    test_accuracy = list()
    for epoch in range(epochs):  # loop over the dataset multiple times
        model.train()
        
        # setting learning rate
        if epoch >80:
            blr = 0.00001
            beta1 = 0.1
        elif epoch >40:
            blr = 0.0001
        else:
            blr = 0.001
        for param_group in optimizer.param_groups:
            # param_group['lr']=alpha_plan[epoch]
            param_group['lr'] = blr
            # param_group['betas']=(beta1, 0.999) # Only change beta1
        if 1:
            ce_loss = utils.AverageMeter()
            prec = utils.AverageMeter()
            
        running_loss = 0.0
    #    for i in tqdm(range(steps_epoch)):
        for i, (inputs, labels) in enumerate(tr_loader):
  
    
            inputs, labels = inputs.to(device), labels.to(device)
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            acc = utils.accuracy(outputs, labels)
            running_loss += loss.item()
            ce_loss.update(loss.item(), inputs.shape[0])
            prec.update(acc.item(), inputs.shape[0])
            # if (i%100)==99:    # print every 2000 mini-batchesm avg loss
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss /(i+1))
        
        hist['cce_loss'].append(ce_loss.avg)
        hist['train_acc'].append(prec.avg)
        cce_testacc = eval_model(model.eval())
        test_accuracy.append(cce_testacc)

        if verbose:
            print('Epoch : {:d}/{:d}, Train acc: {:}, Test acc : {:}, loss:{:}'.format(epoch, epochs, prec.avg,cce_testacc, ce_loss.avg))
    print('Cross enttropy Finished Training')
    if fname is not None:
        fname1 = fname + '_ep_' + str(epoch) + '.pth'
        utils.save_checkpoint(model, epoch, fname1, optimizer=optimizer)
    return hist, test_accuracy
        
    
#%% VAT training
def train_eval_VAT(model, tr_loader, epochs=1, eps=2.5, fname=None, verbose=0):
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    cross_entropy = nn.CrossEntropyLoss()
    #
    hist ={'tot_loss':list(), 'cce_loss':list(), 'vat_loss':list(), 'train_acc':list()}
    test_accuracy = list()
    for epoch in range(0, epochs):
        model.train()
        beta1 = 0.9
        if epoch >80:
            blr = 0.00001
            # blr = 0.001 # SGD
            beta1 = 0.1        
        elif epoch >40:
            blr = 0.0001
            # blr = 0.01 # SGD       
        else:
            blr = 0.001
            # blr = 0.1 # SGD
    
        for param_group in optimizer.param_groups:
            param_group['lr'] = blr

        # reset
        if 1:
            ce_losses = utils.AverageMeter() 
            vat_losses = utils.AverageMeter()
            prec1 = utils.AverageMeter()
            tt_losses = utils.AverageMeter()
    
        for i, (x_l, y_l) in enumerate(tr_loader):
            
            x_l, y_l = x_l.to(device), y_l.to(device)
            # converting labels into one-hot encoded vector for the wat adv wrt labels
            l_y = (y_l.unsqueeze(1) == torch.arange(10).reshape(1, 10).long().to(device)).float()
            optimizer.zero_grad()
    
            output = model(x_l)
            classification_loss = cross_entropy(output, y_l)
            
            if epoch>=15:
                beta = 5
            else:
                beta = 0
    
            vat_lds = vat_loss(model, x_l, output, eps=eps, adv_img=False)
        
            loss = classification_loss + beta*vat_lds 
            loss.backward()
            optimizer.step()
            
            # keep tracking the loss values, etc
            acc = utils.accuracy(output, y_l)
            ce_losses.update(classification_loss.item(), x_l.shape[0])
            vat_losses.update(vat_lds.item(), x_l.shape[0])
            prec1.update(acc.item(), x_l.shape[0])
            tt_losses.update(loss.item(), x_l.shape[0])
            
        hist['cce_loss'].append(ce_losses.avg)
        hist['tot_loss'].append(tt_losses.avg)
        hist['vat_loss'].append(vat_losses.avg)
        hist['train_acc'].append(prec1.avg)
        if fname is not None and (epoch%20==19 or epoch == epochs-1):
            fname1 = fname+'_ep_'+str(epoch)+'.pth'
            utils.save_checkpoint(model, epoch, fname1, optimizer=optimizer)
        if verbose:
            print('It: {:d}/{:d}, loss={:.4f}, cce_loss={:.4f}, vat_loss={:.4f},acc={:.4f}'.
                  format(i + 1, steps_epoch, tt_losses.avg, ce_losses.avg, vat_losses.avg, prec1.avg))
        test_acc = eval_model(model.eval())
        test_accuracy.append(test_acc)
        if verbose:
            print('Epoch: {:d}/{:d} Test acc:{:}'.format(epoch, epochs, test_acc))
    return hist, test_accuracy

def train_eval_WAT(model, tr_loader,epochs=1, reg=0.05, eps=0.005, beta=10, fname=None,nbloop=5,
                   verbose=1):
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    cross_entropy = nn.CrossEntropyLoss() # cross entropy loss 
    
    # Ground cost of WAT from word2vec model
    #import ot
    if data_set == 'cifar10':
        word2vec_embed = np.load('/word2vec_embed/cifar10label_word_vec.npz')
    elif data_set == 'cifar100':
        # word2vec_embed = np.load('/home/damodara/DeepNetModels/word2vec/fashionmnist_word_vec.npz')
        word2vec_embed = np.load('/word2vec_embed/cifar100label_word_vec.npz') # cifar100
    elif data_set == 'fashion-mnist':
        pass
        
    embed = word2vec_embed['embed']
    M = ot.dist(embed, embed, 'euclidean')
    M = M/M.max()
    M = np.exp(-M)  #  dissimilrity to similarity

    # M[:,:]=1 # for total variation cost
    for i in range(nclass):
        M[i,i]=0 # setting digaonal as zero
    
    hist ={'tot_loss':list(), 'cce_loss':list(), 'wat_loss':list(), 'train_acc':list()}
    test_accuracy = list()
    for epoch in range(0, epochs):
        model.train()
        # beta1 = 0.9
        # if epoch >120: # CIFAR 100
        if epoch > 80:  # CIFAR 10
        # if epoch >40: # FMNIST
            blr = 0.00001
            # blr = 0.001 # SGD
            beta1 = 0.1
            
        # elif epoch >80:  # CIFAR 100
        if epoch > 40:  # CIFAR 10
        # elif epoch >20: # FMNIST

            blr = 0.0001
            # blr = 0.01 # SGD
            
        else:
            blr = 0.001
            # blr = 0.1 # SGD
    
        for param_group in optimizer.param_groups:
            param_group['lr'] = blr

        # reset
        if 1:
            ce_losses = utils.AverageMeter()
            wat_losses = utils.AverageMeter()
            pred = utils.AverageMeter()
            tt_losses = utils.AverageMeter()

        for i, (x_l, y_l) in enumerate(tr_loader):

            x_l, y_l = x_l.to(device), y_l.to(device)
            # converting labels into one-hot encoded vector for the wat adv wrt labels
            l_y = (y_l.unsqueeze(1) == torch.arange(nclass).reshape(1, nclass).long().to(device)).float()
            optimizer.zero_grad()
    
            output = model(x_l)
            classification_loss = cross_entropy(output, y_l)
            
            # wat loss
            if epoch>=15:
                beta_w = beta
            else:
                beta_w =0

            if type(eps).__module__ == np.__name__:
                eps = torch.from_numpy(np.array(eps)).float().to(device)

            wat_lds = wat_loss(model, x_l, l_y, M, reg=reg, eps=eps,
                               pred_label=True, adv_img=False,deep_layer_adv=False,nbloop=nbloop)


            loss = classification_loss + beta_w*wat_lds #+ ent_loss
            loss.backward()
            optimizer.step()
            
            # keep tracking the loss values, etc
            acc = utils.accuracy(output, y_l)
            ce_losses.update(classification_loss.item(), x_l.shape[0])
            wat_losses.update(wat_lds.item(), x_l.shape[0])
            pred.update(acc.item(), x_l.shape[0])
            tt_losses.update(loss.item(), x_l.shape[0])            
            # if (i%100)==99:
            #     print('It: {:d}/{:d}, loss={:.4f}, cce_loss={:.4f}, wat_loss={:.4f},acc={:.4f}'.
            #     format(i+1,steps_epoch, tt_losses.avg, ce_losses.avg, wat_losses.avg, prec1.avg))
        if verbose:
            print('It: {:d}/{:d}, loss={:.4f}, cce_loss={:.4f}, wat_loss={:.4f},acc={:.4f}, beta={}'.
                  format(i + 1, steps_epoch, tt_losses.avg, ce_losses.avg, wat_losses.avg, pred.avg, beta_w))
        hist['cce_loss'].append(ce_losses.avg)
        hist['tot_loss'].append(tt_losses.avg)
        hist['wat_loss'].append(wat_losses.avg)
        hist['train_acc'].append(pred.avg)
        test_acc = eval_model(model.eval())
        test_accuracy.append(test_acc)
        if fname is not None and (epoch%20==19):
            fname1 = fname+'_ep_'+str(epoch)+'.pth'
            utils.save_checkpoint(model, epoch, fname1, optimizer=optimizer)
        if verbose:
            print('Epoch: {:d}/{:d} Test acc:{:}'.format(epoch, epochs, test_acc))
    return hist, test_accuracy

def loss_plot(hist, testacc=None, pname= None, fname=None):
    loss_list = [s for s in hist.keys() if 'loss' in s]
    acc_list = [s for s in hist.keys() if 'acc' in s]

    fig1 = plt.figure(num=1)
    epochs = range(1,len(hist[loss_list[0]]) + 1)
    for l in loss_list:
        plt.plot(epochs, hist[l], label = l)
    if fname is not None:
        plt.title(fname)
        plt.savefig(os.path.join(pname, fname+'_loss.png'))
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.legend(loc='best')
        plt.close()

    fig2 = plt.figure(num=2)
    for l in acc_list:
        plt.plot(epochs, hist[l], label=l)
    if fname is not None:
        plt.title(fname)
        plt.savefig(os.path.join(pname, fname+'_acc.png'))
        plt.xlabel('Epochs')
        plt.ylabel('acc')
        plt.legend(loc='best')
        plt.close()

    if testacc is not None:
        fig3 = plt.figure(num=3)
        plt.plot(epochs, testacc, label = 'testacc')
        if fname is not None:
            plt.title(fname)
            plt.savefig(os.path.join(pname, fname+'_acc.png'))
            plt.xlabel('Epochs')
            plt.ylabel('testacc')
            plt.legend(loc='best')
            plt.close()

# loss function of the 
class SCELoss(torch.nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%%  training the models
noise = np.array([0.0, 0.2, 0.4])
noise = np.array([0.4])
#%% Noisy labels simulation
from simulate_noisylabel import noisify_with_P,mnist_simulate_noisylabel, noisify_cifar10_asymmetric,noisify_cifar100_asymmetric,noisify_fashionmnist_asymmetric
asymmetric=0
random_state=42
num_runs = 1
cross_ent_acc = np.zeros((len(noise), num_runs))
vat_acc = np.zeros((len(noise), num_runs))
wat_acc = np.zeros((len(noise), num_runs))
vat_acc_std = np.zeros((len(noise), num_runs))
print(data_set)
for r in range(num_runs):
    for k in np.arange(0, len(noise)):
        if noise[k]>0.0:
            if asymmetric==1:
                n_trainlabel, P = noisify_with_P(trainlabel.ravel(), noise=noise[k], random_state=random_state)
            elif asymmetric==0:
                if data_set == 'mnist':
                    n_trainlabel,P = mnist_simulate_noisylabel(trainlabel, noise=noise[k],random_state=random_state)
                elif data_set == 'fashion-mnist':
                    n_trainlabel,P = noisify_fashionmnist_asymmetric(trainlabel.ravel(), noise=noise[k],random_state=random_state)
                elif data_set == 'cifar10':
                    n_trainlabel,P = noisify_cifar10_asymmetric(trainlabel.ravel(), noise=noise[k],random_state=random_state)
                elif data_set == 'cifar100':
                    n_trainlabel,P = noisify_cifar100_asymmetric(trainlabel.ravel(), noise=noise[k],random_state=random_state)
                    err = np.sum(trainlabel != n_trainlabel)
                    print("Percent of Noise labels=", err / len(trainlabel))
        else:
            n_trainlabel = trainlabel
            P = np.eye(nclass)

        print(P)
        # convert the noisy label to pytorch tensor
        n_trainlabel = torch.from_numpy(n_trainlabel).float().long()

        # %% Training data generator based on noisy labels
        # trset = CustomDataset(traindata, n_trainlabel, transform = transform_dataaug)
        trset = torch.utils.data.TensorDataset(traindata, n_trainlabel)
        tr_loader = torch.utils.data.DataLoader(trset, batch_size=batch_size,
                                                    shuffle=True, num_workers=8)
        # device name

        #%% model architectures
        from architectures import mnist_featext, cifar10_featext,cnn_coteach, cnn_coteach_avg
        if data_set == 'mnist':
            input_shape = np.array([1, 28, 28])
            drop_out = None
            basemodel = cnn_coteach
        elif data_set == 'cifar10':
            input_shape = np.array([3, 32, 32])
            drop_out= None
            basemodel = cnn_coteach #cifar10_featext
        elif data_set == 'cifar100':
            input_shape = np.array([3, 32, 32])
            basemodel = cnn_coteach
            #cnn_coteach_avg
        elif data_set == 'fashion-mnist':
            input_shape = np.array([1, 28, 28])
            basemodel = cnn_coteach


        #%% cross entropy model
        if method == 'CCE':
            cpname ='results/cifar100_runs3/Sym_CCE_Loss/'
            model_save = True
            file_save = True
            epochs=150
            fn = data_set + '_asym_' + str(noise[k]) +'_cce_'+'run_'+str(r)
            try:
                os.system('mkdir -p %s' % cpname+'n_' + str(noise[k]))
            except OSError:
                pass
    
    
            cpname1 = os.path.join(cpname, 'n_' + str(noise[k]))
            fname = os.path.join(cpname1, fn)
            if model_save is not True:
                fname = None
    
    
            # model definition and moving to device
            model = basemodel(input_shape, nclass,drop_out=None).to(device)
            model.apply(weights_init) # weights initalization
            #
            criterion = nn.CrossEntropyLoss()  # cross entropy loss
            if noise[k]==0.4:
                alpha= 2.0
            else:
                alpha = 6.0
            beta = 0.1
            # criterion= SCELoss(alpha=alpha, beta=beta, num_classes=nclass)
            cce_hist, cce_test_acc = train_eval_CCE(model, tr_loader, criterion, epochs=epochs, fname=fname,verbose=1)
            cross_ent_acc[k, r] = np.mean(cce_test_acc[-10:])
            if file_save:
                fn_s = fname+'_hist_test_acc.npz'
                np.savez(fn_s, cce_hist = cce_hist, cce_test_acc = cce_test_acc)
                # dumb = np.load(fn_s)
            loss_plot(cce_hist, testacc=cce_test_acc, pname=cpname, fname=fn)
            print("CCE Test acc =", cce_test_acc[-1])
            del model
      
        #
        # %% VAT model
        if method == 'VAT':
            cpname = 'results/cifar10/VAT_Sym/'
    
            model_save = True
            file_save = True
            epochs = 120
            fn = data_set + '_asym_' + str(noise[k]) + '_vat_'+'run_'+str(r)
            try:
                os.system('mkdir -p %s' % cpname+'n_' + str(noise[k]))
            except OSError:
                pass
    
            cpname1 = os.path.join(cpname, 'n_' + str(noise[k]))
            fname = os.path.join(cpname1, fn)
    
            vat_model = basemodel(input_shape,nclass).to(device)
            vat_model.apply(weights_init)
            #
            vat_hist, vat_test_acc = train_eval_VAT(vat_model, tr_loader, eps=0.005,epochs=epochs, fname=fname,verbose=1)
            vat_acc[k,r] = np.mean(vat_test_acc[-10:])
            vat_acc_std[k,r]= np.std(vat_test_acc[-10:])
            if file_save:
                fn_s = fname+'_hist_test_acc.npz'
                np.savez(fn_s, vat_hist = vat_hist, vat_test_acc = vat_test_acc)
            loss_plot(vat_hist, testacc=vat_test_acc, pname=cpname, fname=fn)
            print("VAT Test acc =", vat_test_acc[-1])
            print('VAT training finished')
            del vat_model, vat_hist, vat_test_acc
        



        #%% WAT model
        if method == 'WAR':
            pname = './results/cifar10/WAT_SinkIter/tmp/'
            reg = np.array([0.05])
            eps = np.array([0.005])
            sink_iter = np.array([20])
            model_save = False
            file_save = False
            beta = 10
            epochs=120
            wat_acc ={}
      
            fn = data_set + '_asym_'+str(noise[k])+'_reg_'+str(reg)+'_eps_'+str(eps)+'_nrun_'+str(r)
            try:
                os.system('mkdir -p %s' % pname + 'n_'+str(noise[k])+'_reg_'+str(reg)+'_eps_'+str(eps)+'_sinkiter_'+str(sink_iter))
            except OSError:
                pass
    
            pname1 = os.path.join(pname, 'n_'+str(noise[k])+'_reg_'+str(reg)+'_eps_'+str(eps)+'_sinkiter_'+str(sink_iter))
            fname = os.path.join(pname1, fn)
            if model_save is not True:
                fname = None
            wat_model = basemodel(input_shape,nclass).to(device)
            wat_model.apply(weights_init)
            wat_hist, wat_test_acc = train_eval_WAT(wat_model, tr_loader, epochs=epochs, reg=reg, eps=eps, beta=beta,
                                                    fname=fname, nbloop=sink_iter,verbose=1)
            wat_acc[k,r] = np.mean(wat_test_acc[-10:])
            wat_acc[str(sink_iter)] = np.mean(wat_test_acc[-10:])
            fname = os.path.join(pname1, fn)
            fn_acc = fname+'_hist_test_acc.npz'
            if file_save:
                np.savez(fn_acc, wat_hist = wat_hist, wat_test_acc = wat_test_acc)
    
    
            loss_plot(wat_hist, testacc=wat_test_acc, pname=pname1, fname=fn)
            print("WAT Test acc =", wat_test_acc[-1])
            del wat_model, wat_hist, wat_test_acc
            print('Finished noise = {:}, reg = {:}, eps = {:}'.format(noise[k], reg, eps))
            if file_save:
                np.save(fname+'_all_sink_iter_test_acc.npy', wat_acc)


# pn = 'results/cifar10/WAT_test/'
# fn = 'wat_cifar10.npz'
# np.savez(os.path.join(pn,fn), vat_acc = vat_acc, vat_std_acc = vat_acc_std)







