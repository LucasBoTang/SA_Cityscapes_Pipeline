#!/usr/bin/env python
# coding: utf-8

import os
import sys

import glob
import cv2
from PIL import Image
import networkx as nx
import numpy as np
from skimage import segmentation as sg
import argparse

from PATH import *
from utils import *

if __name__ == "__main__":

    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--fdr', type=str, default="prob_heur")
    args = parser.parse_args()

    # search files
    search = os.path.join("./experiments_eccv/" + args.fdr, "*_instanceIds.png")
    files = glob.glob(search)
    files.sort()

    for file in files:
        # output name
        output_name = file.split("\\")[-1].replace("_gtFine_instanceIds", "_panoColor")
        print("Drawing {}...".format(output_name))

        # load inst id
        ids = np.array(Image.open(file))
        # convert
        anno = data_loader.scribble_convert(ids)

        # init
        pano = np.zeros_like(anno)
        # get sseg id
        for l in np.unique(anno[:,:,1]):
            # get label
            label = data.label_map[l]
            # get inst id
            for i in np.unique(anno[:,:,2] * (anno[:,:,1] == l)):
                # init
                inst = np.zeros_like(anno)
                # inst region
                region = (anno[:,:,1] == l) & (anno[:,:,2] == i)
                # get sseg color
                r, g, b = data.class_info[label].color
                # change color
                if i > 0:
                    r += np.random.randint(-80, 80)
                    g += np.random.randint(-80, 80)
                    b += np.random.randint(-80, 80)
                    r = min(max(0, r), 255)
                    g = min(max(0, g), 255)
                    b = min(max(0, b), 255)
                # add color
                pano[region] = [b, g, r]

        # add boundaires
        pano = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)
        pano = sg.mark_boundaries(pano, ids, color=(1,1,1))
        pano = (pano * 255).astype(np.uint8)
        pano = cv2.cvtColor(pano, cv2.COLOR_RGB2BGR)

        # show image
        # cv2.imshow("vis", pano)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.imwrite("./experiments_eccv/" + args.fdr + "/" + output_name, pano)
