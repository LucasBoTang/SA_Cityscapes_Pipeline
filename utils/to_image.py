#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import networkx as nx
from .data import *

def graph_to_image(graph, height, width, scribbles):
    """
    convert NetworkX undirected graph into OpenCV BGR color image
    """
    # create empty images
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    pred = np.zeros((height, width), dtype=int)

    # new part
    for group in graph.nodes:
        # get the color of current label
        label = graph.nodes[group]["label"]
        try:
            label, inst = label.split("_")[0], int(label.split("_")[1])
        except:
            inst = 0
        if label:
            r, g, b = class_info[label].color
            label_color = [b, g, r]
            index = class_info[label].id + 21 * inst
        else:
            label_color = [0, 0, 0]
            index = 20
        # get pixels in current graph
        pixels = graph.nodes[group]["pixels"]
        # assign pixels with color
        for x, y in pixels:
            mask[y, x] = label_color
            pred[y, x] = index

    # orignal part
    for y in range(height):
        for x in range(width):
            annotated, trainid, inst = scribbles[y, x]
            # annotated
            if annotated:
                if trainid == 128:
                    mask[y, x] = [0, 0, 0]
                    pred[y, x] = 20
                else:
                   label = label_map[trainid]
                   r, g, b = class_info[label].color
                   mask[y, x] = [b, g, r]
                   pred[y, x] = class_info[label].id + 21 * inst


    return mask, pred


def ilp_to_image(graph, ilp, height, width, scribbles):
    """
    convert CPLEX MIP solution into OpenCV BGR color image
    """
    names = ilp.variables.get_names()
    values = ilp.solution.get_values()

    # get annotation
    graph = label_graph(graph, names, values)
    # get mask
    mask, pred = graph_to_image(graph, height, width, scribbles)

    return mask, pred


def format(pred):
    """
    get groud truth in Cityscapes format
    """
    sseg = np.zeros_like(pred, dtype=np.uint8)
    inst = np.zeros_like(pred, dtype=np.uint16)

    for l in labels:
        sseg = sseg + (pred % 21 == l.trainId) * l.id
        if l.hasInstances:
            inst = inst + (pred % 21 == l.trainId) * (l.id * 1000 + pred // 21)
        else:
            inst = inst + (pred % 21 == l.trainId) * l.id

    sseg[pred == 20] = 0
    inst[pred == 20] = 0

    return sseg.astype(np.uint8), inst.astype(np.uint16)


def label_graph(graph, names, values):
    """
    use ilp solution to label graph
    """
    # map id to label
    to_label = {}
    for label in class_info:
        ind = class_info[label].id
        to_label[ind] = label

    for i, name in enumerate(names):
        # only get binary var
        if name[0] != "x":
            break
        value = values[i]
        # make assignment
        if value:
            # get i and l
            _, i, l = name.split('_')
            i = int(i)
            l = int(l) % 21
            # get label name
            label = to_label[l]
            # label graph
            graph.nodes[i]["label"] = label
    return graph

def get_mask(annotations, erode=False):
    """
    visualize annotation
    """
    height, width, _ = annotations.shape

    # init
    annos = np.zeros((height, width), dtype=np.uint8)
    mask = np.zeros((height, width, 3), dtype=np.uint8)

    # orignal part
    for y in range(height):
        for x in range(width):
            annotated, trainid, inst = annotations[y, x]
            # annotated
            if annotated:
                annos[y, x] = 255
                if trainid == 128:
                    mask[y, x] = [0, 0, 0]
                else:
                   label = label_map[trainid]
                   r, g, b = class_info[label].color
                   mask[y, x] = [b, g, r]

    if erode:
        annos = annos - cv2.erode(annos, None, iterations=2)

    return mask.astype(np.uint8), annos.astype(np.uint8)
