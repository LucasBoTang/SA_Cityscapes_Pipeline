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
import argparse

from PATH import *
from utils import *

import torch
import torch.nn as nn
from torch.nn import functional as F
import models
from matplotlib import pyplot as plt


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
    parser.add_argument('--dir', type=str, default="./experiments_eccv")
    parser.add_argument('--feature', type=str, default="prob")
    parser.add_argument('--solver', type=str, default="heur")
    parser.add_argument('--scribbles', type=str, default="scribbles")
    parser.add_argument('--graph', type=str, default="graphs")
    parser.add_argument('--param', type=int, default=0.01)
    parser.add_argument('--timelimit', type=int, default=10)
    args = parser.parse_args()

    # path
    path = DATA_PATH
    prob_path = PROB_PATH
    out_path = args.dir + "/" + args.feature + "_" + args.solver
    if args.scribbles == "arti_scribbles":
        out_path += "_arti"
    if args.graph != "graphs":
        out_path += "_" + args.graph
    print("Save images into {}...".format(out_path))

    # check args
    assert args.feature in ["RGB", "feat", "prob"], "Feature should be 'RGB' 'feat' or 'prob'"
    assert args.feature != "RGB", "RGB features is not implemented"
    assert args.solver in ["heur", "ilp"], "Solver should be 'heur' or 'ilp'"
    assert args.scribbles in ["scribbles", "arti_scribbles"], "Scribbles should be 'scribbles' or 'arti_scribbles'"
    assert os.path.isdir(path + "/" + args.graph), path + "/" + args.graph + " does not exist"

    # load part of DCNN for feature map
    if args.feature == "feat":
        # load cnn model
        model = set_model("./models/checkpoints/deeplabv1_resnet101-coco.pth")
        feature_out = FeatureOut(model, 4)
        print(feature_out)

    # parameters
    if args.feature == "feat":
        import solver
        lambd=0.1
        psi=0.0
        phi=0.001
    elif args.feature == "prob":
        import solver_prob as solver
        lambd = 0.1
        psi = 0.0
        phi = 0.3

    # get heuristic result
    if args.solver == "ilp":
        heur_fdr = out_path.replace("ilp", "heur")
        assert os.path.isdir(heur_fdr), "Heuristic results do not exist, please run experiment for heuristic first!"

    # create folder
    if not os.path.isdir(args.dir):
        os.mkdir(args.dir)
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    # init for calculate mIoU
    intersections = np.zeros((21))
    unions = np.zeros((21))

    # data loader
    cnt = 0
    total_time = 0
    ssegs = []
    preds = []
    data_generator = data_loader.load_cityscapes(path, args.scribbles)
    for filename, image, sseg, inst, scribbles in data_generator:

        # process scribbles
        if scribbles is not None:
            print("Generating ground truth approach for image {}...".format(filename))
            # covert into standard
            scribbles = data_loader.scribbles_reformat(scribbles)
        else:
            # skip image which does not have annotation
            print("Skipping image {} because it does not have annotation...".format(filename))

        # skip existed gt
        if os.path.isfile(out_path + "/" + filename + "_gtFine_labelIds.png"):
            print("Annotation exists, skip {}".format(filename))
            print()
            continue

        # generate superpixels
        # superpixels = superpixel.get(image)
        # load superpixel form graphs
        if not os.path.isfile(path + "/" + args.graph + "/" + filename + ".gpickle"):
            print("{} does not have graph file...".format(filename))
            print()
            continue
        graph = nx.read_gpickle(path + "/" + args.graph + "/" + filename + ".gpickle")
        superpixels = graph.get_superpixels_map()
        # split by annotation
        superpixels = superpixel.split(superpixels, scribbles)

        if args.solver == "ilp":
            # load heuristic result directly
            pred_id = np.array(Image.open(heur_fdr + "/" + filename + "_gtFine_labelIds.png"))
            pred = np.zeros_like(pred_id, dtype=np.uint8)
            for trainid in np.unique(pred_id):
                id = data.id2train[trainid]
                if id == -1 or id == 255:
                    id = 20
                pred += id * (pred_id == trainid).astype(np.uint8)
            # mark ignore
            ignore = (pred_id == 0)
            scribbles[:,:,0] = scribbles[:,:,0] * (1 - ignore) + 255 * ignore
            scribbles[:,:,1] = scribbles[:,:,1] * (1 - ignore) + 128 * ignore
            # get superpixels map
            superpixels = superpixels * (pred != 20)

        # build graph
        graph = to_graph.to_superpixel_graph(image, scribbles, superpixels)

        # load features
        if args.feature == "prob":
            # probability map
            prob = np.load(prob_path + filename + "_leftImg8bit.npy")[0].astype("float")
            graph.load_feat_map(prob, attr="feat")
        elif args.feature == "feat":
            # feature map
            feat_map = get_map(image, feature_out)
            graph.load_feat_map(feat_map, attr="feat")

        # solve
        tick = time.time()
        height, width = image.shape[:2]
        if args.solver == "heur":
            # solve
            heuristic_graph = solver.heuristic.solve(graph.copy(), lambd, psi, phi, attr="feat")
            # convert into mask
            mask, pred = to_image.graph_to_image(heuristic_graph, height, width, scribbles)

        elif args.solver == "ilp":
            # drop instance id
            for i in graph.nodes:
                label = graph.nodes[i]["label"]
                if label:
                    graph.nodes[i]["label"] = label.split("_")[0]
            # merge nodes with same label
            graph.add_pseudo_edge()
            # build ilp
            ilp = solver.ilp.build_model(graph, args.param)
            # get superpixels map
            superpixels = graph.get_superpixels_map()
            # warm start
            solver.ilp.warm_start(ilp, pred%21, superpixels)
            # set time limit
            timelimit = args.timelimit
            ilp.parameters.timelimit.set(timelimit)
            # solve
            ilp.solve()
            mask, pred = to_image.ilp_to_image(graph, ilp, height, width, scribbles)

        # record time
        tock = time.time()
        total_time += tock - tick
        cnt += 1

        # Cityscapes format
        sseg_pred, inst_pred = to_image.format(pred)

        # save annotation
        Image.fromarray(sseg_pred).save(out_path + "/"  + filename + "_gtFine_labelIds.png")
        if args.solver == "heur":
            Image.fromarray(inst_pred).save(out_path + "/" + filename + "_gtFine_instanceIds.png")
        cv2.imwrite(out_path + "/" + filename + "_gtFine_color.png", mask)

        # store for score
        preds += list(pred%21)
        ssegs += list(sseg)
        print()

        # visualize
        mask_show(image, mask, inst_pred, name="image")
        cv2.destroyAllWindows()

        # terminate with iteration limit
        #if cnt > 1:
        #    break

    # time consuming
    print("Average algorithms time: {}".format(total_time / cnt))

    # calculate MIoU
    print("Score for origin scribbles:")
    print(metrics.scores(ssegs, preds, 19))
