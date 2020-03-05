#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time
from collections import namedtuple
import cv2
from PIL import Image
import networkx as nx
import numpy as np
from skimage import segmentation as sg

from PATH import *
from utils import *
import solver_prob as solver

import torch
import torch.nn as nn
from torch.nn import functional as F
import models
from matplotlib import pyplot as plt
import sys

# ignore warnings
import warnings
warnings.filterwarnings("ignore")


class FeatureOut(nn.Module):
    def __init__(self, model, extracted_layer):
        super(FeatureOut, self).__init__()
        self.features = nn.Sequential(
            *list(model.module.base.children())[0],
            *list(model.module.base.children())[1]
        )[:extracted_layer]
    def forward(self, x):
        x = self.features(x)
        return x

def set_model(checkpoint):
    """
    set the model
    """
    model = models.DeepLabV2_ResNet101_MSC(21)
    state_dict = torch.load(checkpoint)
    print("    Init:", checkpoint)
    for m in model.base.state_dict().keys():
        if m not in state_dict.keys():
            print("    Skip init:", m)
    model.base.load_state_dict(state_dict, strict=False)  # to skip ASPP
    model = torch.nn.DataParallel(model)
    model.cuda(0)
    model.eval()
    return model


def mask_show(image, mask, groups, name="image"):
    """
    show image with mask
    """
    img = cv2.addWeighted(image, 0.4, mask, 0.6, 0)
    img = sg.mark_boundaries(img, groups, color=(1,1,1))
    cv2.imshow(name, img)
    cv2.waitKey(0)


def mask_to_label(mask):
    """
    convert mask image into label martix
    """
    # get the image size
    h, w, _ = mask.shape

    # build a color to label map
    color_to_idx = {}
    for label in class_info:
        color_to_idx[class_info[label].color] = class_info[label].id

    # generate label matrix
    label = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            b, g, r = mask[y, x]
            color = (r, g, b)
            label[y, x] = color_to_idx[color]

    return label


def label_to_mask(labels):
    """
    convert label martix into mask image
    """
    # get the image size
    h, w = labels.shape

    # build a color to label map
    idx_to_color = {}
    for label in class_info:
        idx_to_color[class_info[label].id] = class_info[label].color

    # generate label matrix
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            id = labels[y, x]
            r, g, b = idx_to_color[id]
            mask[y, x] = np.array([b, g, r])

    return mask


def transform(image):

    # BGR to RGB
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # resize
    img = cv2.resize(img, (513, 513), interpolation=cv2.INTER_LINEAR).astype(int)
    mean = np.array([104, 117, 123], dtype=int)
    img -= mean
    #std = np.array([0.229, 0.224, 0.225])
    #img /= std
    # to tensor
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).float()
    img = img.view(1, 3, 513, 513)
    img.cuda(0)

    return img


def get_map(image, model, softmax=False):
    """
    get feature map and probability map from cnn
    """
    # transform image
    img = transform(image).cuda(0)

    # run inference
    with torch.no_grad():
        feat_map = model(img)

    feat_map = F.interpolate(feat_map, size=(image.shape[0], image.shape[1]), mode='bilinear')
    if softmax:
        feat_map = F.softmax(feat_map, dim=1)
    feat_map = feat_map.data.cpu().numpy()[0]

    return feat_map

def show_feat(feat_map):
    """
    visualize each feature map
    """
    for i in range(feat_map.shape[0]):
        plt.imshow(feat_map[i])
        plt.show()


if __name__ == "__main__":

    intersections = np.zeros((21))
    unions = np.zeros((21))

    path = DATA_PATH
    data_generator = data_loader.load_cityscapes(path, "scribbles")
    prob_path = PROB_PATH

    # create folder
    if not os.path.isdir("./experiments_eccv"):
        os.mkdir("./experiments_eccv")
    if not os.path.isdir("./experiments_eccv/prob_heur_4000/"):
        os.mkdir("./experiments_eccv/prob_heur_4000")

    cnt = 0
    ssegs = []
    preds = []

    tick = time.time()

    algo_time = 0

    # lambda is the growing parameter that acts finally as the regularization parameter beta, psi is  weights color, phi is weights on fea(prob.)
    # wi * wj * (psi * np.linalg.norm(Yi - Yj) ** 2 + phi * np.linalg.norm(Zi - Zj) ** 2)<= beta * cij * (wi + wj):
    # beta = (iter / iterations) ** gamma * lambd
    lambd = 0.1
    psi = 0.0
    phi = 0.3


    try:
        for filename, image, sseg, inst, scribbles in data_generator:
            cnt += 1
            height, width = image.shape[:2]
            if scribbles is not None:
                print("{}: Generating ground truth approach for image {}...".format(cnt, filename))
                # BGR to RGB
                scribbles = cv2.cvtColor(scribbles, cv2.COLOR_BGR2RGB)
                # ignore region
                scribbles[:,:,1] = np.where(scribbles[:,:,1]==255, 128, scribbles[:,:,1])
            else:
                # skip image which does not have annotation
                print("{}: Skipping image {} because it does not have annotation...".format(cnt, filename))
                cnt -= 1
                continue

            # skip existed gt
            if os.path.isfile("./experiments_eccv/prob_heur/" + filename + "_gtFine_instanceIds.png"):
                print("Annotation exists, skip {}".format(filename))
                cnt -= 1
                continue

            # generate superpixels
            # superpixels = superpixel.get(image)
            #print(path + "/graphs/" + filename)
            graph = nx.read_gpickle(path + "/graphs_4000/" + filename + ".gpickle")

            superpixels = graph.get_superpixels_map()
            # split by annotation
            superpixels = superpixel.split(superpixels, scribbles)

            # build graph
            graph = to_graph.to_superpixel_graph(image, scribbles, superpixels)

            # get prob map
            prob = np.load(prob_path + filename + "_leftImg8bit.npy")[0].astype("float")
            #show_feat(prob)
            graph.load_feat_map(prob, attr="prob")

            tick1 = time.time()
            heuristic_graph = solver.heuristic.solve(graph.copy(), lambd, psi, phi, attr="prob")
            # convert into mask
            algo_time +=  time.time() - tick1
            #print("Average algo time: {}".format(time.time() - tick1))
            mask, pred = to_image.graph_to_image(heuristic_graph, height, width, scribbles)
            # mask_show(image, mask, pred, name="heur")
            # cv2.destroyAllWindows()

            # get formatted sseg and inst
            sseg_pred, inst_pred = to_image.format(pred)
            # save annotation
            Image.fromarray(sseg_pred).save("./experiments_eccv/prob_heur_4000/"  + filename + "_gtFine_labelIds.png")
            Image.fromarray(inst_pred).save("./experiments_eccv/prob_heur_4000/" + filename + "_gtFine_instanceIds.png")
            cv2.imwrite("./experiments_eccv/prob_heur_4000/" + filename + "_gtFine_color.png", mask)

            # store for score
            preds += list(pred%21)
            ssegs += list(sseg)

            # visualize
            # mask_show(image, mask, inst_pred, name="image")
            # cv2.destroyAllWindows()

            # terminate with iteration limit
            #if cnt > 1:
            #    break
    except KeyboardInterrupt:
        pass


    # show paramters
    print(lambd, psi, phi)

    print("Real used average time: {}".format((time.time() - tick) / cnt))
    print("Average algo time: {}".format(algo_time / cnt))

    # calculate MIoU
    print("Score for origin scribbles:")
    print(metrics.scores(ssegs, preds, 19))
