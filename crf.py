import time
import numpy as np
import torch
from torch.autograd import Variable
import os

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax


def run_crf(images, outputs):

    img = images  # shape: (3, height, width)
    prob = outputs  # shape: (1, height, width)
    masks = crf(img, prob)

    masks = Variable(torch.from_numpy(masks).float(), volatile=True)

    return masks

def crf(img, prob):

    func_start = time.time()

    img = np.swapaxes(img, 0, 2)
    # img.shape: (width, height, num of channels)

    num_iter = 5

    prob = np.swapaxes(prob, 1, 2)  # shape: (1, width, height)

    # preprocess prob to (num_classes, width, height) since we have 2 classes: car and background.
    num_classes = 2
    probs = np.tile(prob, (num_classes, 1, 1))  # shape: (2, width, height)
    probs[0] = np.subtract(1, prob) # class 0 is background
    probs[1] = prob                 # class 1 is car

    d = dcrf.DenseCRF(img.shape[0] * img.shape[1], num_classes)

    unary = unary_from_softmax(probs)  # shape: (num_classes, width * height)
    unary = np.ascontiguousarray(unary)
    d.setUnaryEnergy(unary)

    # This potential penalizes small pieces of segmentation that are
    # spatially isolated -- enforces more spatially consistent segmentations
    feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    # Note that this potential is not dependent on the image itself.

    # This creates the color-dependent features --
    # because the segmentation that we get from CNN are too coarse
    # and we can use local color features to refine them
    feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                      img=img, chdim=2)

    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)


    Q = d.inference(num_iter)  # set the number of iterations
    res = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))
    # res.shape: (width, height)

    res = np.swapaxes(res, 0, 1)  # res.shape:    (height, width)
    res = res[np.newaxis, :, :]   # res.shape: (1, height, width)

    func_end = time.time()
    # print('{:.2f} sec spent on CRF with {} iterations'.format(func_end - func_start, num_iter))
    # about 2 sec for a 1280 * 960 image with 5 iterations
    return res