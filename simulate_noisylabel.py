# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 10:13:17 2018

@author: damodara
"""

import numpy as np
from numpy.testing import assert_array_almost_equal


def build_uniform_P(size, noise):
    """ The noise matrix flips any class to any other with probability
    noise / (#class - 1).
    """

    assert(noise >= 0.) and (noise <= 1.)

    P = noise / (size - 1) * np.ones((size, size))
    np.fill_diagonal(P, (1 - noise) * np.ones(size))

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def build_for_cifar100(size, noise):
    """ The noise matrix flips to the "next" class with probability 'noise'.
    """

    assert(noise >= 0.) and (noise <= 1.)

    P = (1. - noise) * np.eye(size)
    for i in np.arange(size - 1):
        P[i, i+1] = noise

    # adjust last row
    P[size-1, 0] = noise

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def row_normalize_P(P, copy=True):

    if copy:
        P_norm = P.copy()
    else:
        P_norm = P

    D = np.sum(P, axis=1)
    for i in np.arange(P_norm.shape[0]):
        P_norm[i, :] /= D[i]
    return P_norm   

def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y

def noisify_with_P(label, noise, P=None, random_state=None):
    nb_classes = len(np.unique(label))
    if noise > 0.0:
        if P is None:
            P = build_uniform_P(nb_classes, noise)
        # seed the random numbers with #run
        label_noisy = multiclass_noisify(label, P=P,
                                           random_state=random_state)
        actual_noise = (label_noisy != label).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        label = label_noisy
#    else:
#        P = np.eye(nb_classes)

    return label, P
    

    

def mnist_simulate_noisylabel(label, noise, random_state=None):
    '''
    simulates the noisy label for mnist data
     mistakes:
        1 <- 7
        2 -> 7
        3 -> 8
        5 <-> 6
    '''
    label = label.ravel()
    n_class = len(np.unique(label))
    P= np.eye(n_class)
    n = noise
    
    if n>0.0:
        # 1<-7
        P[7,7], P[7,1] = 1. - n, n
        
        # 2 ->7
        P[2, 2], P[2, 7] = 1. -n, n
        
        # 5 <-> 6
        P[5,5], P[5,6] = 1. -n, n
        P[6,6], P[6,5] = 1. -n, n
        
        # 3 ->8
        P[3,3], P[3,8] = 1. -n, n
        
        label_noisy = multiclass_noisify(label, P=P,
                                           random_state=random_state)
        actual_noise = (label_noisy != label).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        label = label_noisy
        
    return label, P


def noisify_fashionmnist_asymmetric(y_train, noise, random_state=None):
    """mistakes:
         9 --> 7
         4 <--> 6
         6 -->2
         3 -->0
         5 -->7

    """
    nb_classes = 10
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 9 ->7
        P[9, 9], P[9, 1] = 1. - n, n

        # 5 -> 7
        P[5, 5], P[5, 7] = 1. - n, n
        P[2, 2], P[2, 6] = 1. - n, n

        # 4 <-> 6
        P[4, 4], P[4, 6] = 1. - n, n
        P[6, 6], P[6, 4] = 1. - n, n

        # 3 -> 0
        P[3, 3], P[3, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy

    return y_train, P
        
def noisify_cifar10_asymmetric(y_train, noise, random_state=None):
    """mistakes:
        automobile <- truck
        bird -> airplane
        cat <-> dog
        deer -> horse
    """
    nb_classes = 10
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # automobile <- truck
        P[9, 9], P[9, 1] = 1. - n, n

        # bird -> airplane
        P[2, 2], P[2, 0] = 1. - n, n

        # cat <-> dog
        P[3, 3], P[3, 5] = 1. - n, n
        P[5, 5], P[5, 3] = 1. - n, n

        # automobile -> truck
        P[4, 4], P[4, 7] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy

    return y_train, P

def noisify_cifar100_asymmetric(y_train, noise, random_state=None):
    """mistakes are inside the same superclass of 10 classes, e.g. 'fish'
    """
    nb_classes = 100
    P = np.eye(nb_classes)
    n = noise
    nb_superclasses = 20
    nb_subclasses = 5

    if n > 0.0:
        for i in np.arange(nb_superclasses):
            init, end = i * nb_subclasses, (i+1) * nb_subclasses
            P[init:end, init:end] = build_for_cifar100(nb_subclasses, n)

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy

    return y_train, P      

        
def noisify_rslc_asymmetric(y_train, noise, random_state=None):
    """ mistakes in labelling the land cover classes
    baseballdiam -> dense res, med res
    beach -> river
    dense res <--> med res
    intersection -> freeway
    mob home park <-> dense res
    overpass <--> intersection
    tenniscourt --> med res
    """
    nb_classes = 19
    P = np.eye(nb_classes)
    n = noise
    
    if n>0.0:
        P[1,1], P[1,10] = 1.-n, n
        P[2,2], P[2,14] = 1.-n, n
        P[4,4], P[4,10] = 1.-n, n
        P[9,9], P[9,6] = 1.-n, n
        P[11,11], P[11,4] = 1.-n, n
        P[12, 12], P[12,9] = 1.-n, n
        P[18, 18], P[18, 10] = 1.-n, n
        
        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy

    return y_train
    
def noisify_aid_asymmetric(y_train, noise, random_state=None):
    """ mistakes in labelling the land cover classes in AID dataset
    bareland -> desert
    centre -> stroage tank
    church -> centre, stroage tank
    Dense Res --> Med Res
    Desert -->bareland
    Indust --> med res
    Meadow --> farm land
    Med Res --> Dense Res
    play ground --> meadow, school
    Resort --> Med Res
    school --> Med Res, play ground
    stadium --> play ground
    """
    nb_classes = 30
    P = np.eye(nb_classes)
    n = noise
    
    if n>0.0:
        P[1,1], P[1,9] = 1.-n, n
        P[5,5], P[5,28] = 1.-n, n
        P[6,6], P[6,5], P[6,28] = 1.-n, n//2, n//2 
        P[8,8], P[8,14] = 1.-n, n
        P[9,9], P[9,1] = 1.-n, n
        P[12,12], P[12,14] = 1.-n, n
        P[13,13], P[13, 10] = 1.-n, n
        P[14,14], P[14,8] =  1.-n, n
        P[18,18], P[18,13], P[18, 24] =  1.-n, n//2, n//2 
        P[22,22], P[22,14 ] = 1.-n, n
        P[24,24], P[24,14], P[24, 18] = 1.-n, n//2, n//2 
        P[27,27], P[27,18 ] = 1.-n, n
        
        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy

    return y_train
    
def noisify_patternnet_asymmetric(y_train, noise, random_state=None):
    """ mistakes in labelling the land cover classes in PatternNet dataset
    cemetery -> christmas_tree_fram
    harbor <--> ferry terminal
    Den.Res --> costal home
    overpass <--> intersection
    park.space --> park.lot
    runway_mark --> park.space
    costal home <--> sparse Res
    swimming pool --> costal home
    """
    nb_classes = 38
    P = np.eye(nb_classes)
    n = noise
    
    if n>0.0:
        P[5,5], P[5,7] = 1.-n, n
        P[9,9], P[9,32] =1.-n, n
        P[11,11], P[11,9] = 1.-n, n
        P[17,17], P[17,12] = 1.-n, n
        P[12,12], P[12,17] = 1.-n, n
        P[18,18], P[18,23] = 1.-n, n
        P[23,23], P[23,18] = 1.-n, n
        P[25,25], P[25, 24] = 1.-n, n
        P[29,29], P[29,25] =  1.-n, n
        
        P[32,32], P[32,9] =  1.-n, n
        P[34,34], P[34,9] = 1.-n, n
        
        
        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy

    return y_train
    
def noisify_NWPU45_asymmetric(y_train, noise, random_state=None):
    """ mistakes in labelling the land cover classes in NWPU-RESISC45 dataset
    baseballdiam --> Med Res
    beach --> river
    Den. Res <--> Med Res
    Intersection --> freeway
    Mob home park <--> Den Res
    Overpass <--> Intersection
    tennis court --> Med Res
    runway --> freeway
    thermal pow stat --> cloud
    wetland --> lake
    rect farm land --> meadow
    chruch --> palace
    commerical area --> Den Res
    """
    nb_classes = 45
    P = np.eye(nb_classes)
    n = noise
    
    if n>0.0:
        P[2,2], P[2,23] = 1.-n, n
        P[4,4], P[4,32] =1.-n, n
        P[7,7], P[7,27] =  1.-n, n
        P[10,10], P[10,11] = 1.-n, n
        P[11,11], P[11,23], P[11,24] = 1.-n, n//2, n//2        
        P[23,23], P[23,11] = 1.-n, n        
        P[19,19], P[19,14], P[19,26] = 1.-n, n//2, n//2        
        P[24,24], P[24,11] = 1.-n, n         
        P[26,26], P[26,19] = 1.-n, n              
        P[41,41], P[41,23] = 1.-n, n
        P[34,34], P[34,14] = 1.-n, n
        P[43,43], P[43,9] =  1.-n, n        
        P[44,44], P[44,21] =  1.-n, n
        P[31,31], P[31,22] = 1.-n, n
        
        
        
        
        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy

    return y_train
    
def noisify_ChikuseiHSI_asymmetric(y_train, noise, random_state=None):
    """ mistakes in labelling the land cover classes in Chikusei HSI dataset
    bare soil (park)  --> bare soil (farm)
    bare soil (farm) --> bare soil (park), rowcrops
    weeds --> grass, rowcrops, forest
    forest --> rice(grown), weeds
    grass --> weeds, rowcrops
    rice(grown) --> forest,weeds
    rowcrops --> baresoil(farm), weeds, grass
    plastic home --> Asphalt, manmade(dark)
    manmade(dark) -->plastic house
    paved ground --> baresoil(farm)
    """
    nb_classes = 19
    P = np.eye(nb_classes)
    n = noise
    
    if n>0.0:
        P[2,2], P[2,3] = 1.-n, n
        P[3,3], P[3,2], P[3,10] =1.-n, n//2, n//2
        P[5,5], P[5,7], P[5,10], P[5,6] =  1.-n, n//3, n//3, n//3        
        P[6,6], P[6,8], P[6,5] = 1.-n, n//2, n//2        
        P[7,7], P[7,5], P[7,10] = 1.-n, n//2, n//2         
        P[8,8], P[8,6] ,P[8,5] = 1.-n, n//2, n//2         
        P[10,10], P[10,3], P[10,5], P[10,7] = 1.-n, n//3, n//3         
        P[11,11], P[11,17], P[11,13] = 1.-n, n//2, n//2           
        P[13,13], P[13,11] = 1.-n, n                      
        P[18,18], P[18,3] = 1.-n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy

    return y_train
    
def noisify_Salinas_asymmetric(y_train, noise, random_state=None):
    """ mistakes in labelling the land cover classes in Salinas HSI dataset
    brocoli weed1 <--> brocoli weed2
    fallow <--> fallow smooth
    fallow rough --> fallow smooth
    graphes untrained <--> vinyard untrained
    corn sense weeds --> soil vinyard develop, lettuce romain 4wk, lettuce romain 5wk
    soil vinyard develop --> corn sense weeds
    """
    nb_classes = 16
    P = np.eye(nb_classes)
    n = noise
    
    if n>0.0:
        P[0,0], P[0,1] = 1.-n, n
        P[1,1], P[1,0] = 1.-n, n        
        P[2,2], P[2,4] = 1.-n, n
        P[4,4], P[4,2] = 1.-n, n
        P[3,3], P[3,4] = 1.-n, n
        P[7,7], P[7,14] = 1.-n, n
        P[14,14], P[14,7] = 1.-n, n        
        P[9,9], P[9,8], P[9,10], P[9,11] =  1.-n, n/3, n/3, n/3                                 
        P[8,8], P[8,9] = 1.-n, n
        P[9,9], P[9,8] = 1.-n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy

    return y_train,P
    
def noisify_dfc2018_asymmetric(y_train, noise, random_state=None):
    """ mistakes in labelling the land cover classes in 2018 DFC Houstan HSI dataset
    class 0 --> class 1, class 3
    class 1 --> class 5
    class 3 --> class4, class 7
    class 4 --> class 3, class7, class 10
    class 7 --> class9, class 10, (class15)
    class 8 --> class10
    class 9 --> class12, class10, class11
    class 10 --> class12, class11
    class 11 --> class12
    class 12 --> class13
    class 15 --> class9
    
    """
    nb_classes = 20
    P = np.eye(nb_classes)
    n = noise
    
    if n>0.0:
        P[0,0], P[0,1], P[0,3] = 1.-n, n/2, n/2
        P[1,1], P[1,5] = 1.-n, n        
        P[3,3], P[3,4], P[3,7] = 1.-n, n/2, n/2
        P[4,4], P[4,3], P[4,7], P[4,10] = 1.-n, n/3, n/3, n/3        
        P[7,7], P[7,9], P[7,10] = 1.-n, n/2, n/2
        P[8,8], P[8,10] = 1.-n, n        
        P[9,9], P[9,12], P[9,10], P[9,11] =  1.-n, n/3, n/3, n/3         
        P[10,10], P[10,12], P[10,11]  = 1.-n, n/2, n/2        
        P[11,11], P[11,12] = 1.-n, n  
        P[12,12], P[12,13] = 1.-n, n 
        P[15,15], P[15,9] = 1.-n, n                                     
        
        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy

    return y_train,P
    
        
    