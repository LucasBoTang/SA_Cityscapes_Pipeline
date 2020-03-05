#!/usr/bin/env python
# coding: utf-8

import cv2
import networkx as nx
import numpy as np
from skimage import segmentation as sg
from .data import class_info, label_map


def to_superpixel_graph(image, mask, superpixels, slabel="sseg"):
    """
    convert OpenCV BGR color image into NetworkX undirected graph
    edges are bettween grid neighbors (left, right, up, down)

    node attributes:
        mean_color: normalized average color / channel
        label: instance label
        pixels: list of pixels
        weight: number of pixels
        superpixel: label of superpixel

    edge attributes:
        connection: number of connections
    """
    print("Building graph based on superpixels...")

    # avoid to put all pixels into graph, dilate unlabelled area
    selected = (1 - mask[:, :, 0] // 255).reshape((image.shape[0], image.shape[1], 1))
    # dilate
    kernel = np.ones((5, 5), np.uint8)
    selected = cv2.dilate(selected, kernel, iterations=2)

    # int to float
    image = image / 255

    # build graph
    h, w = image.shape[:2]
    graph = Graph(h, w)

    # get nodes
    for y in range(h):
        for x in range(w):
            # skip unselected pixels
            if not selected[y, x]:
                continue
            # skip ignored
            if mask[y, x, 1] == 128:
                continue
            ind = superpixels[y, x]
            pixel = (x, y)
            color = image[y, x].copy()
            # get label
            annotated, label_id, scri_id = mask[y, x]
            if not annotated:
                label = None
            else:
                if slabel == "sseg":
                    label = label_map[label_id] + "_" + str(scri_id)
                else:
                    label = label_id
            # add new superpixel node
            if ind not in graph:
                graph.add_node(ind, mean_color=color, label=label, pixels=[pixel], weight=1)
            # add pixel to current node
            else:
                # just sum togetehr, calculate average later
                graph.nodes[ind]["mean_color"] += color
                graph.nodes[ind]["label"] = label if label else graph.nodes[ind]["label"]
                graph.nodes[ind]["pixels"].append(pixel)
                graph.nodes[ind]["weight"] += 1

    # calculate average color
    for ind in graph.nodes:
        graph.nodes[ind]["mean_color"] = graph.nodes[ind]["mean_color"] / graph.nodes[ind]["weight"]

    # get edges by sliding window
    for y in range(h):
        for x in range(w):
            # skip unselected pixels
            if not selected[y, x]:
                continue
            # skip ignored
            if mask[y, x, 1] == 128:
                continue
            i = superpixels[y, x]
            # since it is an undirected graph, only need to add edges bettween left and up neighbors
            # left neighbor
            if x != 0:
                j = superpixels[y, x - 1]
                if i != j and selected[y, x - 1] and mask[y, x - 1, 1] != 128:
                    # add new edge
                    if not graph.has_edge(i, j):
                        graph.add_edge(i, j, connections=1)
                    # increase connection
                    else:
                        graph[i][j]["connections"] += 1
            # up neighbor
            if y != 0:
                j = superpixels[y - 1, x]
                if i != j and selected[y - 1, x] and mask[y - 1, x, 1] != 128:
                    # add new edge
                    if not graph.has_edge(i, j):
                        graph.add_edge(i, j, connections=1)
                    # increase connection
                    else:
                        graph[i][j]["connections"] += 1

    return graph


class Graph(nx.Graph):

    def __init__(self, height=None, width=None):
        super(Graph, self).__init__()
        self.height = height
        self.width = width

    def __copy__(self):
        graph = super(Graph, self).copy()
        graph.__class__ = Graph
        graph.height, graph.width = self.height, self.width

        return graph

    def load_feat_map(self, feat_map, attr="feat"):
        """
        add feature map to the graph
        """
        # transpose to height, width, channel
        feat_map = feat_map.transpose(1, 2, 0)

        for i in self.nodes:
            # initialize feature map
            self.nodes[i][attr] = np.zeros((feat_map.shape[2]))
            pixels = self.nodes[i]["pixels"]
            # average feature on each superpixel
            for x, y in pixels:
                self.nodes[i][attr] += feat_map[y][x]
            self.nodes[i][attr] /= self.nodes[i]["weight"]

    def add_scribble(self, scribble, label):
        """
        add new scribble to the graph
        """
        superpixels = self.get_superpixels_map()
        for i in range(self.height):
            for j in range(self.width):
                # if scribble on the pixel
                if scribble[i, j]:
                    # assign to superpixel
                    node = superpixels[i, j]
                    # update label
                    self.nodes[node]["label"] = label

    def contract(self, i, j):
        """
        contract node i and node j into node i
        """
        # attributes of node i
        wi = self.nodes[i]["weight"]
        Yi = self.nodes[i]["mean_color"]
        Gi = self.nodes[i]["pixels"]
        label_i = self.nodes[i]["label"]
        # attributes of node j
        wj = self.nodes[j]["weight"]
        Yj = self.nodes[j]["mean_color"]
        Gj = self.nodes[j]["pixels"]
        label_j = self.nodes[j]["label"]

        # region fusion
        # merge node attributes
        self.nodes[i]["pixels"] = Gi + Gj
        self.nodes[i]["mean_color"] = (wi * Yi + wj * Yj) / (wi + wj)
        self.nodes[i]["weight"] = wi + wj
        self.nodes[i]["label"] = label_i if label_i else label_j
        if "feat" in self.nodes[i].keys():
            self.nodes[i]["feat"] = (wi * self.nodes[i]["feat"] + wj * self.nodes[j]["feat"]) / (wi + wj)
        # combine edges
        if j in self.neighbors(i):
            self.remove_edge(i, j)
        for k in self.neighbors(j):
            if k in self.neighbors(i):
                self.edges[i, k]["connections"] += self.edges[j, k]["connections"]
            else:
                self.add_edge(i, k, connections=self.edges[j, k]["connections"])

        # remove node j
        self.remove_node(j)

    def label_merge(self, shortcut=True):
        """
        merge neighbors which have same label
        """
        print("Merge neighors have same label...")

        # loop through all nodes
        for i in list(self.nodes):
            # skip contracted node
            if i not in self.nodes:
                continue
            # skip unlabelled node
            if not self.nodes[i]["label"]:
                continue
            # get label
            label = self.nodes[i]["label"]
            # merge neighbors which have same label
            while True:
                # terminate when no neighnors
                if len(list(self.neighbors(i))) == 0:
                    break
                # loop through neighbors
                merged = False
                for j in list(self.neighbors(i)):
                    # merge neighbors who are in same superpixel
                    if self.nodes[j]["label"] == label:
                        self.contract(i, j)
                        merged = True
                # terminate when no neighbors belongs to same superpixel
                if not merged:
                    break

        # get rid off inner unlabelled
        for i in list(self.nodes):
            # skip contracted node
            if i not in self.nodes:
                continue
            if len(list(self.neighbors(i))) == 1 and not self.nodes[i]["label"]:
                j = list(self.neighbors(i))[0]
                self.contract(j, i)

        if shortcut:
            # add shortcut between seperations
            root_nodes = {}
            for i in list(self.nodes):
                # skip conctratced node
                if i not in self.nodes:
                    continue
                # skip unlabelled node
                if not self.nodes[i]["label"]:
                    continue
                # get label
                label = self.nodes[i]["label"]
                # add new label
                if label not in root_nodes:
                    root_nodes[label] = i
                # add shortcut
                else:
                    print("add shortcut for {}".format(label))
                    root = root_nodes[label]
                    self.add_edge(root, i, connections=0)
                    self.contract(root, i)

        else:
            print("Add dummy labels...")
            label_cnt = {}
            for i in list(self.nodes):
                label = self.nodes[i]["label"]
                # skip unlabelled node
                if not label:
                    continue
                label_cnt[label] = label_cnt.get(label, 0) + 1
                # rename label
                new_label = label + "_" + str(label_cnt[label])
                self.nodes[i]["label"] = new_label

    def add_pseudo_edge(self):
        """
        add pseudo edge
        """
        labels = set()
        for i in self.nodes:
            label = self.nodes[i]["label"]
            if label:
                labels.add(label)

        for label in labels:
            nodes = []
            for i in self.nodes:
                if self.nodes[i]["label"] == label:
                    nodes.append(i)
            # build subgraph
            sub_graph = self.subgraph(nodes)
            # separate component
            root = nodes[0]
            for comp in nx.connected_components(sub_graph):
                if root not in comp:
                    seg = list(comp)[0]
                    print("add shortcut for {}".format(label))
                    self.add_edge(root, seg, connections=0)

    def get_superpixels_map(self):
        """
        get superpixels map from graph
        """
        # initialize
        superpixels = np.zeros((self.height, self.width), dtype=int)
        for i in self.nodes:
            # update superpixels
            for x, y in self.nodes[i]["pixels"]:
                superpixels[y][x] = i

        return superpixels
