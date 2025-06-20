import numpy as np
import cv2, imgviz
import matplotlib.pyplot as plt
from scipy import linalg
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

'''
Keys:
H-> Hematoxylin
E-> Eosin
DAB-> Diaminobenzidine
R-> Residual
'''
# Normalized optical density (OD) matrix M for H, E and DAB. 
rgb_from_hed = np.array([[0.65, 0.70, 0.29], # H
                         [0.07, 0.99, 0.11], # E
                         [0.27, 0.57, 0.78]])# D 
hed_from_rgb = linalg.inv(rgb_from_hed)       

# Normalized optical density (OD) matrix M for H and E.
rgb_from_her = np.array([[0.65, 0.70, 0.29], # H
                         [0.07, 0.99, 0.11], # E
                         [0.00, 0.00, 0.00]])# R
rgb_from_her[2, :] = np.cross(rgb_from_her[0, :], rgb_from_her[1, :])
her_from_rgb = linalg.inv(rgb_from_her)

# Normalized optical density (OD) matrix M for H and DAB
rgb_from_hdr = np.array([[0.650, 0.704, 0.286], # H
                         [0.268, 0.570, 0.776], # D
                         [0.000, 0.000, 0.000]])# R
# calculating deconvolution matrix (D)
rgb_from_hdr[2, :] = np.cross(rgb_from_hdr[0, :], rgb_from_hdr[1, :])
hdr_from_rgb = linalg.inv(rgb_from_hdr)


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
