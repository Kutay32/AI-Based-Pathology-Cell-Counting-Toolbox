#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 17:13:26 2022

@author: user01
"""
import cv2
import numpy as np
from scipy import linalg
import numpy as np
from scipy import ndimage as ndi
import tensorflow as tf
from skimage import measure, morphology
from scipy.ndimage import binary_fill_holes
from skimage.segmentation import find_boundaries
from gray2color import gray2color
import cv2
import copy
from gray2color import gray2color
import matplotlib.pyplot as plt

# Normalized optical density (OD) matrix M for H and E.
rgb_from_her = np.array([[0.65, 0.70, 0.29], # H
                         [0.07, 0.99, 0.11], # E
                         [0.00, 0.00, 0.00]])# R
rgb_from_her[2, :] = np.cross(rgb_from_her[0, :], rgb_from_her[1, :])
her_from_rgb = linalg.inv(rgb_from_her)

# lookup tables for bwmorph_thin
G123_LUT = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1,
       0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
       1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
       0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1,
       0, 0, 0], dtype=np.bool_)

G123P_LUT = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
       1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0,
       0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
       1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
       0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0], dtype=np.bool_)

def bwmorph_thin(image, n_iter=None):
    # check parameters
    if n_iter is None:
        n = -1
    elif n_iter <= 0:
        raise ValueError('n_iter must be > 0')
    else:
        n = n_iter
    
    # check that we have a 2d binary image, and convert it
    # to uint8
    skel = np.array(image).astype(np.uint8)
    
    if skel.ndim != 2:
        raise ValueError('2D array required')
    if not np.all(np.in1d(image.flat,(0,1))):
        raise ValueError('Image contains values other than 0 and 1')

    # neighborhood mask
    mask = np.array([[ 8,  4,  2],
                     [16,  0,  1],
                     [32, 64,128]],dtype=np.uint8)

    # iterate either 1) indefinitely or 2) up to iteration limit
    while n != 0:
        before = np.sum(skel) # count points before thinning
        
        # for each subiteration
        for lut in [G123_LUT, G123P_LUT]:
            # correlate image with neighborhood mask
            N = ndi.correlate(skel, mask, mode='constant')
            # take deletion decision from this subiteration's LUT
            D = np.take(lut, N)
            # perform deletion
            skel[D] = 0
            
        after = np.sum(skel) # coint points after thinning
        
        if before == after:  
            # iteration had no effect: finish
            break
            
        # count down to iteration limit (or endlessly negative)
        n -= 1
    skel = skel.astype(np.bool_)
    return skel.astype(np.uint8)

def deconv_stains(rgb, conv_matrix):
    '''
    Parameters
    ----------
    rgb: a 3-channel RGB iamge with channel dim at axis=-1 e.g. (W,H,3) type: uint8/float32
    conv_matrix: Deconvolution matrix D of shape (3,3); type: float32
    Returns
    -------
    image with doconvolved stains, same dimension as input.
    '''
    # change datatype to float64
    rgb = (rgb).astype(np.float64)
    np.maximum(rgb, 1E-6, out=rgb)  # to avoid log artifacts
    log_adjust = np.log(1E-6)  # for compensate the sum above
    x = np.log(rgb)
    stains = (x / log_adjust) @ conv_matrix

    # normalizing and shifting the data distribution to proper pixel values range (i.e., [0,255])
    h = 1 - (stains[:,:,0]-np.min(stains[:,:,0]))/(np.max(stains[:,:,0])-np.min(stains[:,:,0]))
    e = 1 - (stains[:,:,1]-np.min(stains[:,:,1]))/(np.max(stains[:,:,1])-np.min(stains[:,:,1]))
    r = 1 - (stains[:,:,2]-np.min(stains[:,:,2]))/(np.max(stains[:,:,2])-np.min(stains[:,:,2]))

    her = cv2.merge((h,e,r)) * 255

    return her.astype(np.uint8)

def enclose_boundry(sem_mask, instances):
    frame = np.ones(sem_mask.shape)
    frame[2:-2,2:-2] = 0
    # for nuclie who are touching the image boudry
    inst_b = np.multiply(frame, sem_mask)
    inst_b = np.add(instances, inst_b)
    _,inst_b = cv2.threshold(inst_b, 0, 1, cv2.THRESH_BINARY)
    inst_b = inst_b.astype(np.uint8)
    return inst_b
def read_img(img_path, modelip_img_w, modelip_img_h):
    
    img = cv2.imread(img_path, -1) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h = deconv_stains(img, her_from_rgb)
    
    img = cv2.resize(img, (modelip_img_w, modelip_img_h), interpolation=cv2.INTER_LINEAR) 
    h = cv2.resize(h, (modelip_img_w, modelip_img_h), interpolation=cv2.INTER_LINEAR) 
    
    return img, h
    
def Tumor_IO(img_path, sem_mask, inst_mask, modelip_img_w, modelip_img_h):
    '''
    See desdcription of Depth_Data_Generator
    '''
    img = cv2.imread(img_path, -1) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #h = decovn_he(img)
    h = deconv_stains(img, her_from_rgb)

    sem = cv2.imread(sem_mask, -1)
    inst = cv2.imread(inst_mask, -1)
    if len(np.unique(sem)) == 1:# b/c  only BG is present
        sem = sem * 0
        inst = inst * 0
    # b/c the overlayed boundries might contain pixel value > 1
    _,inst = cv2.threshold(inst, 0, 1, cv2.THRESH_BINARY)
    
    # verify boundries enclosement
    # still we need to enclose boundry to be consistent in test and train time
    inst = enclose_boundry(sem, inst)
    
    if img.shape[0] != modelip_img_w:
        img = cv2.resize(img, (modelip_img_w, modelip_img_h), interpolation=cv2.INTER_LINEAR) 
        h = cv2.resize(h, (modelip_img_w, modelip_img_h), interpolation=cv2.INTER_LINEAR) 
    
    # to normalize [0, 255] pixel values to [0, 1]
    # if you are using builtin keras model then dont normalize
    img = img
    h = h 
    inst = inst[:,:, np.newaxis]
    sem = sem[:,:, np.newaxis]
    
    return img, sem, inst, h

def gray2encoded(y_true, num_class):
    '''
    Parameters
    ----------
    y_true : 2D array of shape [H x W] containing unique pixel values for all N classes i.e., [0, 1, ..., N] 
    num_class : int no. of classes inculding BG
    Returns
    -------
    encoded_op : one-hot encoded 3D array of shape [H W N] where N=num_class

    '''
    num_class = num_class
    
    y_true = tf.cast(y_true, 'int32')
    
    encoded_op = tf.one_hot(y_true, num_class, axis = -1)
    
    if tf.executing_eagerly()==False:
        sess1 = tf.compat.v1.Session()
        encoded_op = sess1.run(encoded_op)
    else: 
        encoded_op = encoded_op.numpy()
    return encoded_op

def seprate_instances(sem_mask, instance_boundaries, num_classes, apply_morph=True, kernel_size=3):
    '''

    Parameters
    ----------
    sem_mask : 2D array of shape [H x W] containing unique pixel values for all N classes i.e., [0, 1, ..., N]
    instance_boundaries : 2D array of shape [H x W] bounderies for all N classes i.e., [0->BG, 1->boundry]
    num_classes : no of classes in the sem mask including BG an int
    apply_morph : apply morphological operator so that the edges which were chipped of will be recovered
    Returns
    kernel_size : int kernel size to apply morphological operations (3 default b/c gives best results)
    -------
    op : 3D array containing seperated instances in each channel shape [H x W x N]

    '''
    
    # change datatypt to perform operation
    instances = instance_boundaries.astype(np.float16)
    sem_mask = sem_mask.astype(np.float16)
    instances2 = instances * 6 # bc largest value in sem mask is 5
    
    t = np.subtract(sem_mask, instances2)
    negative_remover = lambda a: (np.abs(a)+a)/2 # one line funstion created by lamda 1 input and 1 output
    t = negative_remover(t).astype(np.uint8)
    # or you can use following line
    #t = np.where(t > 0, t, 0).astype(np.uint8)
    
    # Now as in PanNuke dataset the BG was in 5ht channel and during preprocessing we shifted it to 
    # 0th channel. Now going back so that 0th channel is Neoplastic class and 5th channel is BG as given 
    # in original data description.
    
    if len(np.unique(cv2.fastNlMeansDenoising(t))) == 1:# 1st denoising there might be some noise in the op image
        # if only BG is present than only last channel will be one, do it here
        # b/c the np where conditions wont have any effect on the array if it 
        # only have one class
        tt = np.zeros((t.shape[0], t.shape[1], num_classes))
        tt[:,:,5] = tt[:,:,-1] + 1
        t = tt
    else:# if have atleast one nuclie present/ swaping channels again to match GT
        t = np.where(t == 5, 6, t)
        t = np.where(t == 0, 5, t)
        t = np.where(t == 6, 0, t)
        
        t = gray2encoded(t, num_classes)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size,kernel_size))# before i started main_203 it was 2x2
    op = np.zeros(t.shape)
    for i in range(num_classes):
        # Bc at some place boundry is diagonal and very thin (1px) so measure-label
        # will join two seprate blobs so this will seprate them a little
        t[:,:,i] = cv2.erode(t[:,:,i],kernel,iterations = 1)
        # b/c now 5th channel is BG; still 0 digit represents BG in all channels
        # in 5th channel also the BG of the BG*
        op[:,:,i] = measure.label(t[:,:,i], connectivity=2, background=0)# 2 is ususal
        
    if apply_morph == True:
        #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(10,10))
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        for i in range(num_classes-1):# bc last channel has BG we dont want to change that    
            op[:,:,i] = cv2.dilate(op[:,:,i],kernel,iterations = 1)
            
    op[:,:,5] = np.where(op[:,:,5]>1, 1, op[:,:,5])
     
    return op




def remove_small_obj_n_holes(seg_op, min_area=10, kernel_size=3):
    '''
    Parameters
    ----------
    seg_op :  a 4D array of N channels [1 H W N] where N is number of classses
    min_area : The smallest allowable object size.
    kernel_size : int kernel size to apply morphological operations (3 default b/c gives best results)
    Returns
    -------
    a : 4D array of N channels [1 H W N] with noise removed and holes filled
    '''
    seg_op = copy.deepcopy(seg_op).astype(np.uint8)
    #k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    a = seg_op.squeeze()
    for i in range(a.shape[-1]-1): # iterate over each class seprately
        # need to convert array into boolen type
        b = morphology.remove_small_objects(a[:,:,i+1].astype(bool), min_size=min_area).astype(np.uint8)
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)
        a[:,:,i+1] = b
        #a[:,:,i+1] = morphology.convex_hull_object(b, connectivity=2)
        #a[:,:,i+1] = binary_fill_holes(b).astype(int)
    a = a[np.newaxis,:,:,:]# keep IO size consistant
    
    return a

def assgin_via_majority(seg):
    '''
    Parameters
    ----------
    seg : 2D array containing unique pixel values for each class
    Returns
    -------
    x: 2D array where an instance is assigned to be the class of most frequently
       occuring pixel value (as each unique pixel value represent a class).
    '''
    a = copy.deepcopy(seg).astype(np.uint8)
    # 1st convert to binary mask
    _, th = cv2.threshold(a, 0, 1, cv2.THRESH_BINARY)
    # now measure label
    b = measure.label(th, connectivity=2, background=0)
    # now make n unique channels n= no. of labels measured
    c = gray2encoded(b, len(np.unique(b)))
    
    op = np.zeros(c.shape)
    for i in range(len(np.unique(b))-1):
        temp = np.multiply(c[:,:,i+1], a)# multiply each channel element wise
        mfp = most_frequent_pixel(temp)
        # now convert the range form [0, 1] to [0, mfp]
        _, temp = cv2.threshold(temp, 0, mfp, cv2.THRESH_BINARY)
        op[:,:,i+1] = temp
    x = np.sum(op, axis=2)
    
    return x.astype(np.uint8)

def most_frequent_pixel(img):
    '''
    Parameters
    ----------
    img : 2D array containing unique pixel values for each class
    Returns
    -------
    op : int, most frequently occuring pixel value excluding which has pixel value of 0
    '''
    unq, count = np.unique(img, return_counts=True)
    idx = np.where(count == np.max(count[1:]))
    op = int(unq[idx][0])
    
    return op

def decode_predictions(seg_op, inst_op, thresh=0.5):
    '''
    Parameters
    ----------
    seg_op : Raw logits from CNN output, shape [B, H, W, N]
    inst_op : Raw logits from CNN output, shape [B, H, W, 1]
    thresh : Threshold on pixel confidence a float between [0, 1]
    Returns
    -------
    seg_op : activated and thresholded output of CNN
    inst_op : activated and thresholded output of CNN
    '''
    seg_op = softmax_activation(seg_op)
    seg_op = (seg_op > thresh).astype(np.uint8)
    seg_op = remove_small_obj_n_holes(seg_op, min_area=22, kernel_size=3)
    seg_op = np.argmax(seg_op[0,:,:,:], 2).astype(np.uint8)
    seg_op = assgin_via_majority(seg_op) # assigning instance via majority pixels ((post processing))
    seg_op = (seg_op).astype(np.uint8)
    
    inst_op = sigmoid_activation(inst_op)
    inst_op = (inst_op > thresh).astype(np.uint8)
    inst_op = inst_op.squeeze()
    inst_op = (inst_op).astype(np.uint8)
    inst_op = bwmorph_thin(inst_op)
    
    return seg_op, inst_op

def get_inst_seg(sep_inst, img, blend=True):
    '''
    Parameters
    ----------
    sep_inst : a 3D array of shape [H, W, N] where N is number of classes and in
            each channel all the instances have a unique value.
    img : Original RGB image for overlaying the instance seg results
    blend: wether to project the inst mask over the RGB original image or not
    Returns
    -------
    blend : a 3D array in RGB format [H W 3] in which each instance have of each
            and all classes have a unique RGB value 
            1. overalyed over original image if; blend=True
            2. Raw mask if; blend=False
    '''    
    img = cv2.resize(img, (sep_inst.shape[0], sep_inst.shape[1]), interpolation=cv2.INTER_LINEAR) 
    sep_inst = measure.label(sep_inst[:,:,0:5], connectivity=2, background=0) # ignore BG channel i.e. 6th ch.
    # take element wise sum of all channels so that each instance of each class
    # has a unique value in whole 3D array.
    sep_inst = np.sum(sep_inst, axis=-1) 
    rgb = gray2color(sep_inst.astype(np.uint8), use_pallet='ade20k')
    if blend:
        inv = 1 - cv2.threshold(sep_inst.astype(np.uint8), 0, 1, cv2.THRESH_BINARY)[1]
        inv = cv2.merge((inv, inv, inv))
        blend = np.multiply(img, inv)
        blend = np.add(blend, rgb)
    else:
        blend = rgb
    
    return blend

def get_inst_seg_bdr(sep_inst, img, blend=True):
    '''
    Parameters
    ----------
    sep_inst : a 3D array of shape [H, W, N] where N is number of classes and in
            each channel all the instances have a unique value.
    img : Original RGB image for overlaying the instance seg results
    blend: wether to project the inst mask over the RGB original image or not
    Returns
    -------
    blend : a 3D array in RGB format [H W 3] in which each instance have of each
            and all classes have a unique RGB border. 
            1. overalyed over original image if; blend=True
            2. Raw mask if; blend=False
    ''' 
    img = cv2.resize(img, (sep_inst.shape[0], sep_inst.shape[1]), interpolation=cv2.INTER_LINEAR) 
    sep_inst = measure.label(sep_inst[:,:,0:5], connectivity=2, background=0)# ignore BG channel i.e. 6th ch.
    # take element wise sum of all channels so that each instance of each class
    # has a unique value in whole 3D array.
    sep_inst = np.sum(sep_inst, axis=-1)
    # isolate all instances 
    sep_inst_enc = gray2encoded(sep_inst, num_class=len(np.unique(sep_inst)))
    # as the in encoded output the 0th channel will be BG we don't need it so
    sep_inst_enc = sep_inst_enc[:,:,1:]
    # get boundaries of thest isolated instances
    temp = np.zeros(sep_inst_enc.shape)
    for i in range(sep_inst_enc.shape[2]):
        temp[:,:,i] = find_boundaries(sep_inst_enc[:,:,i], connectivity=1, mode='thick', background=0)
    
    # bc argmax will make the inst at 0 ch zeros so add a dummy channel
    dummy = np.zeros((temp.shape[0], temp.shape[1], 1))
    temp =  np.concatenate((dummy, temp), axis=-1)
    
    sep_inst_bdr = np.argmax(temp, axis=-1)
    sep_inst_bdr_rgb = gray2color(sep_inst_bdr.astype(np.uint8), use_pallet='ade20k')
    if blend:
        inv = 1 - cv2.threshold(sep_inst_bdr.astype(np.uint8), 0, 1, cv2.THRESH_BINARY)[1]
        inv = cv2.merge((inv, inv, inv))
        blend = np.multiply(img, inv)
        blend = np.add(blend, sep_inst_bdr_rgb)
    else:
        blend = sep_inst_bdr_rgb
        
    return blend

def get_sem(sem, img, blend=True):
    '''
    Parameters
    ----------
    sem : a 2D array of shape [H, W] where containing unique value for each class.
    img : Original RGB image for overlaying the semantic seg results
    blend: wether to project the inst mask over the RGB original image or not
    Returns
    -------
    blend : a 3D array in RGB format [H W 3] in which each class have a unique RGB color. 
            1. overalyed over original image if; blend=True
            2. Raw mask if; blend=False
    ''' 
    img = cv2.resize(img, (sem.shape[0], sem.shape[1]), interpolation=cv2.INTER_LINEAR) 
    seg = gray2color(sem.astype(np.uint8), use_pallet='pannuke')
    
    if blend:
        inv = 1 - cv2.threshold(sem.astype(np.uint8), 0, 1, cv2.THRESH_BINARY)[1]
        inv = cv2.merge((inv, inv, inv))
        blend = np.multiply(img, inv)
        blend = np.add(blend, seg)
    else:
        blend = seg
        
    return blend

def get_sem_bdr(sem, img, blend=True):
    '''
    Parameters
    ----------
    sem : a 2D array of shape [H, W] where containing unique value for each class.
    img : Original RGB image for overlaying the semantic seg results
    blend: wether to project the inst mask over the RGB original image or not
    Returns
    -------
    blend : a 3D array in RGB format [H W 3] in which each class have a unique RGB border. 
            1. overalyed over original image if; blend=True
            2. Raw mask if; blend=False
    ''' 
    img = cv2.resize(img, (sem.shape[0], sem.shape[1]), interpolation=cv2.INTER_LINEAR) 
    # 1-hot encode all classes 
    sem_enc = gray2encoded(sem, num_class=6)
    # as the in encoded output the 0th channel will be BG we don't need it so
    sem_enc = sem_enc[:,:,1:]
    # get boundaries of thest isolated instances
    temp = np.zeros(sem_enc.shape)
    for i in range(sem_enc.shape[2]):
        temp[:,:,i] = find_boundaries(sem_enc[:,:,i], connectivity=1, mode='thick', background=0)
    
    dummy = np.zeros((temp.shape[0], temp.shape[1], 1))
    temp =  np.concatenate((dummy, temp), axis=-1)
        
    sem_bdr = np.argmax(temp, axis=-1)
    sem_bdr_rgb = gray2color(sem_bdr.astype(np.uint8), use_pallet='pannuke')
    if blend:
        inv = 1 - cv2.threshold(sem_bdr.astype(np.uint8), 0, 1, cv2.THRESH_BINARY)[1]
        inv = cv2.merge((inv, inv, inv))
        blend = np.multiply(img, inv)
        blend = np.add(blend, sem_bdr_rgb)
    else:
        blend = sem_bdr_rgb
    return blend

def my_argmax(tensor):
    '''
    Fixes the zero channel problem i.e. the class predicted at 0th channel 
    wont go to 0 as it does with usual np.argmax
    Parameters
    ----------
    pred_tensor : 3D/4D array of shape [B, H, W, N] or [H, W, N]
    Returns
    -------
    argmaxed output of shape [B, H, W] or [H, W]]
    '''
    pred_tensor = np.copy(tensor)
    j = 0
    for i in range(pred_tensor.shape[-1]):
        j = i+1
        pred_tensor[:,:,:,i] = pred_tensor[:,:,:,i] * j
    
    pred_tensor = np.sum(pred_tensor, axis=-1)
    return pred_tensor    

def plot_confusion_matrix(cm, class_names, normalize = True, show_text = True, from_clf = False, my_cmap = 'Greens'):
    '''
    Parameters
    ----------
    cm : a nxn dim numpy array.
    class_names: a list of class names (str type)
    normalize: whether to normalize the values
    show_text: whether to show value in each block of the matrix, If matrix is large like 10x10 or 20x20 it's better to set it to false
               because it'll be difficult to read values but you can see the network behaviour via color map.
    show_fpfn: whether to show false positives on GT axis and false negatives on Pred axis. FN -> not detected & FP -> wrong detections
    Returns
    -------
    fig: a plot of confusion matrix along with colorbar
    '''
    if from_clf:
        conf_mat = cm
        x_labels = copy.deepcopy(class_names)
        y_labels = copy.deepcopy(class_names)
    else:
        conf_mat = cm[1:, 1:]
        x_labels = class_names
        y_labels = class_names    
    
    c_m = conf_mat
    
    if normalize:
        row_sums = c_m.sum(axis=1)
        c_m = c_m / row_sums[:, np.newaxis]
        c_m = np.round(c_m, 3)
    
    fig, ax = plt.subplots(figsize=(len(class_names)+3, len(class_names)+3))
    im = ax.imshow(c_m, cmap = my_cmap) 
    
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(y_labels)))
    ax.set_yticks(np.arange(len(x_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(y_labels)
    ax.set_yticklabels(x_labels)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")#ha=right
    
    if show_text:
        for i in range(len(x_labels)):
            for j in range(len(y_labels)):
                text = ax.text(j, i, c_m[i, j], color="k", ha="center", va="center")#color=clr_select(i, j)
    
    ax.set_title("Normalized Confusion Matrix")
    fig.tight_layout()
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    plt.colorbar(sm)
    plt.show() 
    return fig     

def water(img, mask):
    '''
    Parameters
    ----------
    img : 3D array, RGB iamge [H W 3]
    mask : 2D array, semantic/binary segmentaion mask [H W]

    Returns
    -------
    img : RGB image wiht overlayd boundry instances
    new : instacnes boundaries
    '''
    img = (img).astype(np.uint8)
    mask = (mask).astype(np.uint8)
    original_image = np.copy(img)
    
    # apply threshold to converto sem-mask to binary mask
    ret, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # so that BG pixel have 0 value and FG will have 255 value
    thresh = 255 - thresh
    
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # Normalize the distance image for range = {0.0, 1.0}
    # so we can visualize and threshold it 
    dist_transform = cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    _, sure_fg = cv2.threshold(dist_transform, 0.4, 1.0, cv2.THRESH_BINARY)
    #ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    
    # remove bg form the image so that water shed will only focus on cells
    img[thresh==0]=1
    
    markers = markers.astype('int32')
    markers = cv2.watershed(img, markers)
    # draw boundaries on real iamge
    original_image[markers == -1] = [255,0,0]
    # draw boundary on empty convas
    new = np.zeros(img.shape)
    new[markers == -1] = [255, 255, 255]
    new = (new).astype(np.uint8)
    new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
    new = (new/255).astype(np.uint8)
    return original_image, new


def sigmoid_activation(pred):
    pred = tf.convert_to_tensor(pred)
    active_preds = tf.keras.activations.sigmoid(pred)
    if tf.executing_eagerly()==False:
        sess = tf.compat.v1.Session()
        active_preds = sess.run(active_preds)
    else:
        active_preds = active_preds.numpy()
        
    return active_preds

def softmax_activation(pred):
    pred = tf.convert_to_tensor(pred)
    active_preds = tf.keras.activations.softmax(pred, axis=-1)
    if tf.executing_eagerly()==False:
        sess = tf.compat.v1.Session()
        active_preds = sess.run(active_preds)
    else:
        active_preds = active_preds.numpy()
        
    return active_preds