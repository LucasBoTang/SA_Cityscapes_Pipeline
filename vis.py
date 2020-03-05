#!/usr/bin/env python
# coding: utf-8

import os
import sys
import cv2
from PIL import Image
import networkx as nx
import numpy as np
from skimage import segmentation as sg
import argparse

from PATH import *
from utils import *

def vis_superpixels(image, groups):
    """
    get image with superpixel
    """
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = sg.mark_boundaries(img, groups, color=(1,1,1))
    img = (img * 255).astype(np.uint8)

    return img

def vis_scribbles(image, mask, annotated):
    """
    get image with scribbles
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    annotated = np.expand_dims(annotated == 255, axis=2)
    img = image * (1 - annotated) + mask * annotated

    return img.astype(np.uint8)


if __name__ == "__main__":

    path = DATA_PATH

    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--scribbles', type=str, default="")
    parser.add_argument('--superpixels', dest='superpixels', action='store_true')
    parser.set_defaults(superpixels=False)
    parser.add_argument('--graph', type=str, default="graphs")
    args = parser.parse_args()

    # output folder
    out_path = "./experiments_eccv/vis/"
    if args.scribbles == "scribbles":
        out_path += "scri"
    elif args.scribbles == "arti_scribbles":
        out_path += "arti"
    elif args.scribbles == "modi_scribbles":
        out_path += "modi"

    if args.superpixels:
        if args.graph == "graphs":
            out_path += "2000"
        else:
            out_path += args.graph.split("_")[-1]

    # create folder
    if not os.path.isdir("./experiments_eccv/vis/"):
        os.mkdir("./experiments_eccv/vis")
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    # biuld data loader
    cnt = 0
    data_generator = data_loader.load_cityscapes(path, args.scribbles)
    for filename, image, sseg, inst, scribbles in data_generator:

        # superpixel
        if args.superpixels:
            gfile = path + "/" + args.graph + "/" + filename + ".gpickle"
            if os.path.isfile(gfile):
                print("{}: Visualizing superpixels for image {}...".format(cnt, filename))
                # get superpixels
                graph = nx.read_gpickle(gfile)
                superpixels = graph.get_superpixels_map()
                # visualize
                img = vis_superpixels(image, superpixels)
            else:
                print("{} does not have graph file...".format(filename))
                continue
        else:
            img = image.copy()

        if args.scribbles:
            if scribbles is not None:
                print("{}: Visualizing artificial scribbles for image {}...".format(cnt, filename))
                # reformat scribbles
                scribbles = data_loader.scribbles_reformat(scribbles)
                # get mask
                mask, annotated = to_image.get_mask(scribbles, erode=True)
                # visualize
                img = vis_scribbles(img, mask, annotated)
            else:
                print("{}: Skipping image {} because it does not have annotation...".format(cnt, filename))
                continue

        # show image
        cv2.imshow("vis", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # write image
        cnt += 1
        cv2.imwrite(out_path + "/" + filename + ".png", img)
