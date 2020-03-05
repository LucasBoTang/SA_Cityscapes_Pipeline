#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time
from collections import namedtuple
import argparse
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

    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', type=str, default="prob")
    parser.add_argument('--fdr', type=str, default="prob_ilp")
    parser.add_argument('--scribbles', type=str, default="scribbles")
    args = parser.parse_args()

    # check dir
    assert os.path.isdir("./experiments_eccv/" + args.fdr), "./experiments_eccv/" + args.fdr + " does not exist!"

    if args.feature == "feat":
        # load cnn model
        model = set_model("./models/checkpoints/deeplabv1_resnet101-coco.pth")
        feature_out = FeatureOut(model, 4)
        print(feature_out)

    # parameters
    if args.feature == "feat":
        lambd=0.1
        psi=0.0
        phi=0.001
    elif args.feature == "prob":
        lambd = 0.1
        psi = 0.0
        phi = 0.3

    # data path
    path = DATA_PATH
    prob_path = PROB_PATH
    data_generator = data_loader.load_cityscapes(path, "arti_scribbles")

    cnt = 0
    for filename, image, sseg, inst, scribbles in data_generator:

        height, width = image.shape[:2]
        if scribbles is not None:
            print("{}: Generating instance label for image {}...".format(cnt, filename))
            # covert into standard
            scribbles = data_loader.scribbles_reformat(scribbles)
        else:
            # skip image which does not have annotation
            print("Skipping image {} because it does not have annotation...".format(filename))
            continue

        # skip existed gt
        if os.path.isfile("./experiments_eccv/" + args.fdr + "/" + filename + "_gtFine_instanceIds.png"):
            print("Annotation exists, skip {}".format(filename))
            continue

        # skp no sseg
        if not os.path.isfile("./experiments_eccv/" + args.fdr + "/" + filename + "_gtFine_labelIds.png"):
            print("No labelIds, skip {}".format(filename))
            continue

        # generate superpixels
        # superpixels = superpixel.get(image)
        # load superpixel form graphs
        graph = nx.read_gpickle(path + "/graphs/" + filename + ".gpickle")
        superpixels = graph.get_superpixels_map()
        # split by annotation
        superpixels = superpixel.split(superpixels, scribbles)

        # inst scribbles
        inst_scribbles = np.zeros_like(scribbles)
        inst_scribbles[:,:,0] = scribbles[:,:,0]
        inst_scribbles[:,:,1] = scribbles[:,:,2] * (scribbles[:,:,0] == 255)

        # feature map
        # load features
        if args.feature == "prob":
            # probability map
            feat = np.load(prob_path + filename + "_leftImg8bit.npy")[0].astype("float")
        elif args.feature == "feat":
            # feature map
            feat = get_map(image, feature_out)

        # get sseg
        sseg = np.array(Image.open("./experiments_eccv/" + args.fdr + "/" + filename + "_gtFine_labelIds.png"))
        sseg = data_loader.scribble_convert(sseg)

        # get sseg id
        for l in np.unique(sseg[:,:,1]):
            # if has instances
            if l in data.instanceTrainId:
                region = sseg[:,:,1] == l
                print("Segment for {}...".format(data.label_map[l]))
                # build graph
                inst_scribbles[:,:,2] = region * 255
                graph = to_graph.to_superpixel_graph(image, inst_scribbles, superpixels*region, slabel="inst")
                graph.load_feat_map(feat, attr="feat")
                # init pred
                pred = np.zeros((height, width), dtype=np.uint16)
                # solve
                heuristic_graph = solver.heuristic.solve(graph.copy(), lambd, psi, phi, attr="feat")
                # new part
                for group in heuristic_graph.nodes:
                    # get the color of current label
                    label = heuristic_graph.nodes[group]["label"]
                    if not label:
                        continue
                    label = int(label)
                    # get pixels in current graph
                    pixels = heuristic_graph.nodes[group]["pixels"]
                    # assign pixels with color
                    for x, y in pixels:
                        pred[y, x] = label
                # add instance
                sseg[:,:,2] += (pred * region).astype(np.uint16)
                # print(np.unique(sseg[:,:,2]))
                # from matplotlib import pyplot as plt
                # plt.imshow(sseg[:,:,2])
                # plt.show()

        # to instanceIds
        inst = np.zeros((height, width), dtype=int)
        for trainid, id in data.train2id.items():
            if trainid in data.instanceTrainId:
                inst += id * (sseg[:,:,1] == trainid) * 1000
            else:
                inst += id * (sseg[:,:,1] == trainid)
        inst += sseg[:,:,2]

        cnt += 1
        # save
        Image.fromarray(inst.astype(np.uint16)).save("./experiments_eccv/" + args.fdr + "/" + filename + "_gtFine_instanceIds.png")
